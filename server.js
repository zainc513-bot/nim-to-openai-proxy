// server.js — OpenAI-compatible proxy for NVIDIA NIM
// Reasoning payload logic: reasoning.js. Tool-call leak recovery: tools.js.

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');
const { SHOW_REASONING, shapeRequestForModel, getReasoningPayload, resolveEffectiveThinking, StreamNormalizer, normalizeNonStreamChoice } = require('./reasoning');
const { extractLeakedToolCalls, ToolCallStreamRecovery } = require('./tools');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ──────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

// Verbose logging: reasoning payload per fallback attempt, full request/error bodies on failure.
const DEBUG_MODE = process.env.DEBUG_MODE === 'true';

// Asks NIM to include token usage in streamed responses so it can be logged.
// The usage chunk is stripped before forwarding unless the client asked for it.
// Set LOG_STREAM_USAGE=false if a backend rejects stream_options.
const LOG_STREAM_USAGE = process.env.LOG_STREAM_USAGE !== 'false';

// Backend model IDs (comma-separated, or "all") whose reasoning is streamed
// inline in <thinking> tags without the client needing to send the
// x-reasoning-format header. Requires SHOW_REASONING=true. Example:
//   INLINE_REASONING_MODELS=moonshotai/kimi-k3.1-pro
const INLINE_REASONING_MODELS = new Set(
  (process.env.INLINE_REASONING_MODELS || '').split(',').map(s => s.trim()).filter(Boolean)
);

// Heartbeat: while streaming, every HEARTBEAT_INTERVAL_MS of upstream silence
// (e.g. a model reading a huge prompt) the proxy sends an SSE comment line so
// hosts, proxies and clients don't treat the connection as dead, and logs a
// [HEARTBEAT] line confirming the request is still being waited on.
// Set HEARTBEAT_INTERVAL_MS=0 to disable.
const HEARTBEAT_INTERVAL_MS = process.env.HEARTBEAT_INTERVAL_MS !== undefined
  ? Number(process.env.HEARTBEAT_INTERVAL_MS)
  : 10000;

// For streaming requests, open the response immediately and heartbeat while
// waiting for NIM to accept the request, so the client doesn't time out during
// long queues or fallbacks. Set EARLY_STREAM_HEADERS=false to disable.
const EARLY_STREAM_HEADERS = process.env.EARLY_STREAM_HEADERS !== 'false';

// Hedging: if the first model hasn't answered after this many ms, also start
// the next model in the chain and use whichever responds first. NIM can queue
// requests for minutes, so this caps how long one slow model can hold you up.
// Costs extra requests (and rate limit) only when the first model is slow, and
// the reply may come from a different model than the one you picked.
// Set HEDGE_AFTER_MS=0 to disable (strictly one model at a time).
const HEDGE_AFTER_MS = process.env.HEDGE_AFTER_MS !== undefined
  ? Number(process.env.HEDGE_AFTER_MS)
  : 30000;

// Optional watchdog: if the upstream sends nothing at all for this long while
// streaming, give up with a stream error instead of hanging. 0 = off (default),
// because a very long prompt can legitimately take a minute or more to start.
const STALL_TIMEOUT_MS = Number(process.env.STALL_TIMEOUT_MS) || 0;

// Optional per-model speed limits, "model:limit,model:limit". The key can be
// the backend model ID or the client alias (e.g. glm-flash).
//   PROMPT_TOKEN_LIMITS=z-ai/glm-5.3-flash:48000  -> drop oldest chat messages
//     until the estimated prompt fits (system messages and the last 2 messages
//     are always kept). Long prompts are the main cause of slow first tokens.
//   MAX_TOKENS_CAPS=z-ai/glm-5.3-flash:700         -> cap reply length.
// Both are off unless set.
const PROMPT_TOKEN_LIMITS = parseModelLimits(process.env.PROMPT_TOKEN_LIMITS);
const MAX_TOKENS_CAPS = parseModelLimits(process.env.MAX_TOKENS_CAPS);

function parseModelLimits(raw) {
  const map = new Map();
  for (const part of (raw || '').split(',')) {
    const trimmed = part.trim();
    const sep = trimmed.lastIndexOf(':');
    if (sep <= 0) continue;
    const key = trimmed.slice(0, sep).trim();
    const value = Number(trimmed.slice(sep + 1));
    if (key && Number.isFinite(value) && value > 0) map.set(key, value);
  }
  return map;
}

const MAX_TOKENS_LIMIT = 65536;

// Shared keep-alive agent for connection reuse on long-lived deployments.
const keepAliveAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10,
  noDelay: true // send small SSE frames immediately instead of coalescing (disables Nagle)
});

const nim = axios.create({
  baseURL: NIM_API_BASE,
  httpsAgent: keepAliveAgent
});

// Per-attempt timeout before falling back to the next model. Reasoning
// models get a longer window since thinking delays first-token latency.
// Check these against your platform's own request duration limit.
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS) || 180000;
const REASONING_REQUEST_TIMEOUT_MS = Number(process.env.REASONING_REQUEST_TIMEOUT_MS) || 480000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');
if (DEBUG_MODE) console.log('[CONFIG] Debug mode: ENABLED (verbose reasoning + fallback logging)');

// ─── Config validation ──────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}
validateConfig();

// ─── Model Mapping ───────────────────────────────────────────────────────

// Aliases are periodically re-checked against NIM's live catalog (see
// validateModels() and GET /v1/models?live=true). Comments note the prior
// backend model ID where an entry was swapped out for a dead catalog entry.
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'gpt-4-turbo': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'claude-3-opus': 'google/diffusiongemma-26b-a4b-it',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.1-nemotron-70b-instruct', 
  'gemini-turbo': 'nvidia/llama3-chatqa-1.5-70b',
  'gpt-3.5o': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4o': 'stepfun-ai/step-5-preview',
  'mistral': 'mistralai/mistral-large-2-instruct', 
  'mistral-turbo': 'nv-mistralai/mistral-nemo-12b-instruct',
  'mistral-pro': 'mistralai/mistral-7b-instruct-v0.3', 
  'mistral-fast': 'nvidia/mistral-nemo-minitron-8b-8k-instruct', 
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'meta/muse-glimmer-30b',
  'google-lighter': 'poolside/laguna-xs-2.1',
  'glm-flash': 'z-ai/glm-5.3-flash',
  'm3.1': 'minimaxai/minimax-m3.1-flash',
  'kimi': 'moonshotai/kimi-k3.1-pro',
  'mistral-large': 'mistralai/mistral-large-4-1t-a52b', // always the damn comma
};

// Tried right after the alias's own model and BEFORE the generic
// FALLBACK_MODELS, so a failing alias degrades to its closest sibling instead
// of an unrelated model. moonshotai/kimi-k3 was this proxy's Kimi backend
// until Oct 2026 (see git history); K3.1 Pro has been unavailable on NIM's API.
// If a listed model isn't in the catalog the attempt just fails fast and the
// chain moves on.
const ALIAS_FALLBACKS = {
  'kimi': ['moonshotai/kimi-k3']
};

// Used when an unrecognized alias is requested. Must point at a live model.
const DEFAULT_MODEL = 'google/gemma-4-31b-it';

// Ordered by observed reliability/speed — an early failing model delays every fallback behind it.
// diffusiongemma-26b-a4b-it (HTTP 410 Gone) and nemotron-3-super-120b-a12b were
// removed on 2026-11-24 after startup validation reported them missing from
// NIM's catalog. Startup validation now checks this list too.
const FALLBACK_MODELS = [
  'google/gemma-4-31b-it',
  'nvidia/nemotron-3.5-lightning-30b-a3b',
  'meta/muse-glimmer-30b',
  'mistralai/mistral-nemotron'
];

// ─── Middleware ─────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Malformed JSON body -> clean OpenAI-style error instead of Express's default HTML error page.
app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({
      error: {
        message: 'Invalid JSON in request body',
        type: 'invalid_request_error',
        code: 400
      }
    });
  }
  next(err);
});

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const trimmed = authHeader.trim();
  if (!trimmed.startsWith('Bearer ')) return null;
  const token = trimmed.slice('Bearer '.length).trim();
  return token || null;
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models' || req.path === '/') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);
  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────

// Shared by startup validateModels() and GET /v1/models?live=true.
async function fetchLiveModelIds() {
  const response = await nim.get('/models', {
    headers: {
      Authorization: `Bearer ${NIM_API_KEY}`,
      'Content-Type': 'application/json'
    },
    timeout: VALIDATION_TIMEOUT_MS
  });

  return new Set((response.data.data || []).map(m => m.id));
}

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');
  try {
    const availableModels = await fetchLiveModelIds();

    const invalid = [];
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    for (const fb of new Set([DEFAULT_MODEL, ...FALLBACK_MODELS])) {
      if (availableModels.has(fb)) {
        console.log(`[VALIDATION] ✓ (fallback) ${fb}`);
      } else {
        console.warn(`[VALIDATION] ✗ (fallback) ${fb} (not in catalog)`);
        invalid.push({ alias: '(fallback)', nimId: fb, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }
  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
    console.warn('[VALIDATION] Set SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.');
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000 });
    console.log('[DISCORD] Alert sent.');
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Helper: Safe Stream Writing ─────────────────────────────────────────

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// Opens the SSE response (headers + first bytes). Safe to call more than once.
function openSseStream(res) {
  if (res.headersSent) return;
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform'); // no-transform stops intermediaries from recompressing
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // tells nginx-style reverse proxies not to buffer

  // Send each SSE frame as its own packet
  if (res.socket) {
    res.socket.setNoDelay(true);
    res.socket.setTimeout(0);
  }

  // Send headers now so the client sees the stream open immediately
  res.flushHeaders();
}

// ─── Helper: Completion Logging ──────────────────────────────────────────

// Rough token estimate (about 3.5 characters per token for English prose, plus
// a few tokens of per-message overhead). Only used for the optional trimming.
function estimateMessageTokens(message) {
  const content = message?.content;
  let chars = 0;
  if (typeof content === 'string') chars = content.length;
  else if (Array.isArray(content)) {
    for (const part of content) if (typeof part?.text === 'string') chars += part.text.length;
  }
  return Math.ceil(chars / 3.5) + 4;
}

// Drops the oldest non-system messages until the estimated prompt fits the
// budget. System messages and the last `keepTail` messages are never dropped.
function trimMessagesToBudget(messages, budgetTokens, keepTail = 2) {
  if (!Array.isArray(messages) || !budgetTokens) return { messages, dropped: 0 };

  let total = messages.reduce((sum, m) => sum + estimateMessageTokens(m), 0);
  const before = total;
  if (total <= budgetTokens) return { messages, dropped: 0, before, after: total };

  const result = [...messages];
  let dropped = 0;
  for (let i = 0; i < result.length - keepTail && total > budgetTokens;) {
    if (result[i].role === 'system') { i++; continue; }
    total -= estimateMessageTokens(result[i]);
    result.splice(i, 1);
    dropped++;
  }
  return { messages: result, dropped, before, after: total };
}

// Summarizes the reasoning setting actually sent upstream for the model that
// answered. "default" means no reasoning field was sent, so the model used
// its own built-in default (which for GLM 5.3 Flash is max).
function describeReasoning(payload) {
  if (!payload || Object.keys(payload).length === 0) return 'default';
  const kw = payload.chat_template_kwargs || {};
  if (payload.reasoning_effort) return String(payload.reasoning_effort);
  if (kw.reasoning_effort) return String(kw.reasoning_effort);
  if (kw.thinking_mode) return String(kw.thinking_mode);
  if (payload.reasoning_budget !== undefined) return `budget:${payload.reasoning_budget}`;
  if (kw.low_effort) return 'low';
  if (kw.thinking === false || kw.enable_thinking === false) return 'off';
  if (kw.thinking === true || kw.enable_thinking === true) return 'on';
  return 'custom';
}

// One line per finished request: timing and token counts.
// ttft = time to first visible token (content or reasoning), total = full
// request time including any fallback attempts, tok_per_s = completion
// tokens over the generation window (first token to finish).
function logCompletion({ reqId, model, requestedModel, effort, stream, outcome = 'completed', startedAt, firstTokenAt, usage, stats }) {
  const now = Date.now();
  const totalMs = now - startedAt;
  const parts = [`model=${model}`];
  // Present only when a different model than the alias's own answered.
  if (requestedModel && requestedModel !== model) parts.push(`FELL_BACK_FROM=${requestedModel}`);
  parts.push(`effort=${effort ?? 'default'}`, `stream=${stream}`, `outcome=${outcome}`);

  if (firstTokenAt) parts.push(`ttft=${firstTokenAt - startedAt}ms`);
  parts.push(`total=${totalMs}ms`);

  // Streaming diagnostics: chunks = upstream SSE events received, max_gap =
  // longest silence between two of them (a big gap with few chunks means the
  // reply arrived in bursts), heartbeats = keep-alive comments we had to send.
  if (stats) {
    parts.push(`chunks=${stats.chunks}`, `max_gap=${stats.maxGapMs}ms`);
    if (stats.heartbeats) parts.push(`heartbeats=${stats.heartbeats}`);
  }

  if (usage) {
    const completion = usage.completion_tokens;
    const reasoning = usage.completion_tokens_details?.reasoning_tokens;
    parts.push(`prompt_tokens=${usage.prompt_tokens ?? '?'}`);
    parts.push(`completion_tokens=${completion ?? '?'}`);
    if (reasoning) parts.push(`reasoning_tokens=${reasoning}`);
    parts.push(`total_tokens=${usage.total_tokens ?? '?'}`);

    const genMs = now - (firstTokenAt || startedAt);
    if (completion && genMs > 0) {
      parts.push(`tok_per_s=${(completion / (genMs / 1000)).toFixed(1)}`);
    }
  } else {
    parts.push('usage=unavailable');
  }

  console.log(`[USAGE${reqId ? ` #${reqId}` : ''}] ${parts.join(' ')}`);
}

// ─── Helper: Fallback Chain ──────────────────────────────────────────────

// Per-model cooldown after a 403/429, in-memory (resets on restart, not shared across instances).
const modelCooldowns = new Map(); // model -> timestamp (ms) until which to skip it

const RATE_LIMIT_COOLDOWN_MS = Number(process.env.RATE_LIMIT_COOLDOWN_MS) || 30000;
const ACCESS_DENIED_COOLDOWN_MS = Number(process.env.ACCESS_DENIED_COOLDOWN_MS) || 300000;

function isInCooldown(model) {
  const until = modelCooldowns.get(model);
  return typeof until === 'number' && Date.now() < until;
}

function setCooldown(model, ms) {
  modelCooldowns.set(model, Date.now() + ms);
}

// NVIDIA's errors come back as problem+json ({ title, detail, requestId, ... })
// rather than OpenAI's { error: { message } }, and for streaming requests the
// error body arrives as an unread stream, so a plain err.message only says
// "Request failed with status code 4xx". This reads the body (max ~4 KB, 2 s)
// and returns a readable one-line reason, e.g. a 422 naming the bad field.
async function describeUpstreamError(err) {
  const data = err?.response?.data;
  if (!data) return err?.message || 'unknown error';

  let body = data;
  if (typeof data.on === 'function') {
    body = await new Promise(resolve => {
      let text = '';
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        try { data.destroy(); } catch { /* ignore */ }
        resolve(text);
      };
      const timer = setTimeout(finish, 2000);
      data.on('data', chunk => {
        text += chunk.toString('utf8');
        if (text.length > 4096) finish();
      });
      data.on('end', finish);
      data.on('error', finish);
    });
  }

  let parsed = body;
  if (typeof body === 'string') {
    try { parsed = JSON.parse(body); } catch { /* keep raw text */ }
  }

  if (parsed && typeof parsed === 'object') err.upstreamBody = parsed;

  let reason;
  if (parsed && typeof parsed === 'object') {
    const d = parsed.detail;
    const detailText = Array.isArray(d)
      ? d.map(x => `${Array.isArray(x?.loc) ? x.loc.join('.') : '?'}: ${x?.msg || x?.type || ''}`).join('; ')
      : d;
    reason =
      detailText || parsed.error?.message || parsed.message || parsed.title ||
      JSON.stringify(parsed).slice(0, 300);
    if (typeof reason !== 'string') reason = JSON.stringify(reason).slice(0, 300);
    if (parsed.requestId) reason += ` (requestId ${parsed.requestId})`;
  } else {
    reason = String(parsed || '').trim().slice(0, 300);
  }
  return reason ? `${err.message}: ${reason}` : err.message;
}

// Logs once per model and change set when native request shaping removes
// fields or clamps values, so you can see what was adjusted without flooding
// the log.
const loggedNativeDrops = new Set();
function logNativeDrops(model, dropped, clamped = []) {
  const parts = [];
  if (dropped && dropped.length) parts.push(`not sending fields outside NVIDIA's documented schema: ${dropped.join(', ')}`);
  if (clamped && clamped.length) parts.push(`clamped to NVIDIA's documented limits: ${clamped.join(', ')}`);
  if (parts.length === 0) return;
  const key = `${model}|${parts.join('|')}`;
  if (loggedNativeDrops.has(key)) return;
  loggedNativeDrops.add(key);
  console.log(`[NATIVE] ${model}: ${parts.join('; ')}`);
}

// Fields a model has rejected with a 422 during this process's lifetime, so
// later requests don't send them again. Never includes the core fields.
const PROTECTED_FIELDS = new Set(['messages', 'model', 'stream', 'max_tokens']);
const rejectedFields = new Map(); // model -> Set(field)
function rememberRejectedFields(model, fields) {
  const set = rejectedFields.get(model) || new Set();
  for (const f of fields) set.add(f);
  rejectedFields.set(model, set);
}

// Works out which request fields a 422 is complaining about. Handles the
// FastAPI style { detail: [{ loc: ['body', 'field'], msg }] } and, as a
// fallback, any of our field names mentioned in the error text.
function findRejectedFields(err, request) {
  const body = err?.upstreamBody;
  const found = new Set();
  const detail = body?.detail;
  if (Array.isArray(detail)) {
    for (const item of detail) {
      const loc = Array.isArray(item?.loc) ? item.loc : [];
      const key = loc[0] === 'body' ? loc[1] : loc[0];
      if (typeof key === 'string') found.add(key);
    }
  } else {
    const text = typeof detail === 'string' ? detail : (body?.message || body?.error?.message || '');
    if (text) {
      for (const key of Object.keys(request)) {
        if (new RegExp(`\\b${key}\\b`).test(text)) found.add(key);
      }
    }
  }
  return [...found].filter(k => k in request && !PROTECTED_FIELDS.has(k));
}

// Runs attempts against `models` in order and resolves with the first success.
//
// - If an attempt fails, the next model starts immediately (classic fallback).
// - If hedgeAfterMs > 0 and an attempt is still pending after that long, the
//   next model is started IN PARALLEL ("hedging"). Whichever answers first
//   wins and the others are cancelled. NIM can queue a request for minutes, so
//   this stops one slow model from holding up the whole reply.
// - hedgeAfterMs = 0 means no hedging: strictly one model at a time.
// An error with `fatalForChain` set (e.g. a 401) aborts everything at once.
function raceAttempts(models, attemptFn, hedgeAfterMs, reqId) {
  return new Promise((resolve, reject) => {
    let nextIdx = 0;
    let active = 0;
    let settled = false;
    let lastError = null;
    let hedgeTimer = null;
    const controllers = new Map(); // model -> AbortController
    const tag = reqId ? ` #${reqId}` : '';

    const cancelOthers = (exceptModel) => {
      let cancelled = 0;
      for (const [m, c] of controllers) {
        if (m !== exceptModel) { c.abort(); cancelled++; }
      }
      return cancelled;
    };

    const armHedge = () => {
      clearTimeout(hedgeTimer);
      if (settled || hedgeAfterMs <= 0 || nextIdx >= models.length) return;
      hedgeTimer = setTimeout(() => {
        if (settled || nextIdx >= models.length) return;
        console.log(
          `[HEDGE${tag}] no answer after ${Math.round(hedgeAfterMs / 1000)}s, ` +
          `also trying ${models[nextIdx]} (still waiting on ${active} other attempt${active === 1 ? '' : 's'})`
        );
        launch();
      }, hedgeAfterMs);
      if (hedgeTimer.unref) hedgeTimer.unref();
    };

    const launch = () => {
      if (settled || nextIdx >= models.length) return;
      const model = models[nextIdx++];
      const controller = new AbortController();
      controllers.set(model, controller);
      active++;

      attemptFn(model, controller.signal).then(result => {
        active--;
        controllers.delete(model);
        if (settled) {
          // Lost the race but finished anyway: don't leave its stream open.
          try { result.response?.data?.destroy?.(); } catch { /* ignore */ }
          return;
        }
        settled = true;
        clearTimeout(hedgeTimer);
        const cancelled = cancelOthers(model);
        if (cancelled > 0) {
          console.log(`[HEDGE${tag}] ${model} answered first, cancelled ${cancelled} other attempt${cancelled === 1 ? '' : 's'}`);
        }
        resolve(result);
      }).catch(err => {
        active--;
        controllers.delete(model);
        if (settled) return; // cancelled loser or late failure: ignore
        lastError = err;

        if (err && err.fatalForChain) {
          settled = true;
          clearTimeout(hedgeTimer);
          cancelOthers(null);
          reject(err);
          return;
        }

        if (active === 0) {
          if (nextIdx < models.length) {
            launch(); // nothing running: move to the next model right away
          } else {
            settled = true;
            clearTimeout(hedgeTimer);
            reject(lastError || new Error('All models failed'));
          }
        }
      });

      armHedge();
    };

    launch();
  });
}

async function callWithFallback(baseRequest, models, enableThinking, clientReasoningEffort, hasTools, reqId) {
  const timeoutMs = resolveEffectiveThinking(enableThinking, clientReasoningEffort)
    ? REASONING_REQUEST_TIMEOUT_MS
    : REQUEST_TIMEOUT_MS;

  // Skip cooling-down models; fall back to the full list if that empties the chain.
  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  // Streaming requests ask for uncompressed SSE: gzip/brotli buffer output
  // into blocks, which defeats per-token delivery. Non-streaming requests
  // keep default compression.
  const upstreamHeaders = {
    Authorization: `Bearer ${NIM_API_KEY}`,
    'Content-Type': 'application/json',
    ...(baseRequest.stream && {
      'Accept': 'text/event-stream',
      'Accept-Encoding': 'identity'
    })
  };

  // One attempt against one model.
  const attempt = async (model, signal) => {
    const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools);
    const shaped = shapeRequestForModel(
      model,
      { ...baseRequest, model, ...reasoningPayload }
    );
    let fullRequest = shaped.request;
    logNativeDrops(model, shaped.dropped, shaped.clamped);

    // Fields this model already rejected earlier in this process's lifetime.
    const learned = rejectedFields.get(model);
    if (learned && learned.size > 0) {
      fullRequest = { ...fullRequest };
      for (const f of learned) delete fullRequest[f];
    }

    if (DEBUG_MODE) {
      console.log(`[DEBUG] Attempting ${model} with reasoning payload:`, JSON.stringify(reasoningPayload), `(timeout: ${timeoutMs}ms)`);
    }

    const attemptStartedAt = Date.now();
    for (let retries = 0; ; retries++) {
      try {
        const res = await nim.post(
          '/chat/completions',
          fullRequest,
          {
            headers: upstreamHeaders,
            responseType: baseRequest.stream ? 'stream' : 'json',
            timeout: timeoutMs,
            signal
          }
        );
        console.log(`[ATTEMPT] ${model} accepted the request after ${Date.now() - attemptStartedAt}ms (${res.status})`);
        return { response: res, model, reasoningPayload };
      } catch (err) {
        // Cancelled because another model won the race: not a real failure.
        if (axios.isCancel(err)) throw err;

        const status = err.response?.status;
        const errorText = await describeUpstreamError(err);

        // Self-heal: a 422 that names the request fields it dislikes is
        // retried on the SAME model without them (at most twice), and the
        // fields are remembered so later requests skip them. This keeps the
        // request on the model you picked instead of falling back.
        if (status === 422 && retries < 2) {
          const bad = findRejectedFields(err, fullRequest);
          if (bad.length > 0) {
            fullRequest = { ...fullRequest };
            for (const f of bad) delete fullRequest[f];
            rememberRejectedFields(model, bad);
            console.warn(`[SELF-HEAL] ${model} rejected ${bad.join(', ')} (422: ${errorText}); retrying without it and skipping it for this model from now on`);
            continue;
          }
        }

        console.warn(
          `[FALLBACK] Model failed: ${model} after ${Date.now() - attemptStartedAt}ms`,
          status,
          errorText
        );
        if (DEBUG_MODE) {
          console.log(`[DEBUG] Full request body sent to ${model}:`, JSON.stringify(fullRequest));
          console.log(`[DEBUG] Full upstream error response:`, errorText);
        }

        // Same key for every attempt: a 401 means every remaining model would fail identically.
        if (status === 401) {
          err.fatalForChain = true;
        }

        // 403: likely an access-tier issue, not a dead key — cooldown just this model.
        if (status === 403) {
          setCooldown(model, ACCESS_DENIED_COOLDOWN_MS);
        }

        // 429: rate limited — short cooldown.
        if (status === 429) {
          setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
        }

        // Timeouts and upstream 5xx: skip this model briefly so following
        // requests don't each wait out the same failure before falling back.
        const timedOut = !status && (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT');
        if (timedOut || (typeof status === 'number' && status >= 500)) {
          setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
        }

        throw err;
      }
    }
  };

  return raceAttempts(attemptOrder, attempt, HEDGE_AFTER_MS, reqId);
}

// ─── Routes ───────────────────────────────────────────────────────────────

// Prevents a bare 403 (from the auth middleware) on root URL hits from a browser.
app.get('/', (req, res) => {
  res.type('html').send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nim-to-openai-proxy</title>
<style>
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0b0f14;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    text-align: center;
    padding: 24px;
  }
  .card { max-width: 480px; }
  h1 { font-size: 1.3rem; margin: 0 0 0.75rem; }
  p { color: #9aa7b2; line-height: 1.55; margin: 0.5rem 0; }
  code { background: #161b22; padding: 2px 6px; border-radius: 4px; color: #7ee787; }
</style>
</head>
<body>
  <div class="card">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" style="margin: 0 auto 14px; display: block;">
      <circle cx="12" cy="12" r="11" stroke="#7ee787" stroke-width="1.5"/>
      <path d="M7 12.5l3 3 6-6.5" stroke="#7ee787" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>
    <h1>it's up.</h1>
    <p>this proxies OpenAI-format chat requests to NVIDIA NIM. point any OpenAI-compatible client at it, pick a model with a plain alias (<code>gpt-4</code>, <code>mistral</code>, etc), and it handles model fallback, streaming, and each backend's own reasoning/thinking quirks for you.</p>
    <p>it's an API, not a website: nothing lives at this root path.</p>
    <p>send requests to <code>/v1/chat/completions</code> with an <code>Authorization: Bearer &lt;token&gt;</code> header.</p>
    <p>status check, no token needed: <code>/health</code></p>
    <p>bugs / questions: <a href="https://github.com/skywalker14017/nim-to-openai-proxy" style="color:#7ee787;">open an issue on GitHub</a> (docs are there too), or hit me up on Discord (i'll be faster there): <code>skywalker_1401</code></p>
  </div>
</body>
</html>`);
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.6.0' });
});

app.get('/v1/models', async (req, res) => {
  if (req.query.live !== 'true') {
    return res.json({
      object: 'list',
      data: Object.keys(MODEL_MAPPING).map(id => ({
        id,
        object: 'model',
        created: Math.floor(Date.now() / 1000), // OpenAI spec expects Unix seconds
        owned_by: 'nim-proxy'
      }))
    });
  }

  // Cross-checks every alias against NIM's live catalog on demand.
  try {
    const availableModels = await fetchLiveModelIds();
    const data = Object.entries(MODEL_MAPPING).map(([id, backend]) => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nim-proxy',
      backend,
      available: availableModels.has(backend)
    }));

    res.json({
      object: 'list',
      data,
      live_check: {
        checked_at: new Date().toISOString(),
        unavailable_aliases: data.filter(m => !m.available).map(m => m.id)
      }
    });
  } catch (err) {
    console.warn(`[MODELS] Live check failed: ${err.message}`);
    res.status(502).json({
      error: {
        message: `Live model check against NIM failed: ${err.message}`,
        type: 'live_check_error',
        code: 502
      }
    });
  }
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;
  const requestStartedAt = Date.now();
  const reqId = Math.random().toString(36).slice(2, 8);

  try {
    const { model, max_tokens, temperature, stream, reasoning_effort } = req.body;

    let primaryModel = MODEL_MAPPING[model];
    if (!primaryModel) {
      console.warn(`[PROXY] Unknown model alias "${model}", falling back to default: ${DEFAULT_MODEL}`);
      primaryModel = DEFAULT_MODEL;
    }

    // De-dupe: avoids retrying the same model twice if it's also in FALLBACK_MODELS.
    const modelChain = [...new Set([primaryModel, ...(ALIAS_FALLBACKS[model] || []), ...FALLBACK_MODELS])];

    // Forward all client fields except model (replaced per-attempt) and
    // reasoning_effort (translated per-model by getReasoningPayload).
    const { model: _droppedModel, reasoning_effort: _droppedReasoningEffort, ...forwardedFields } = req.body;

    // If the client didn't ask for streamed usage, we request it ourselves for
    // logging and strip it again before forwarding.
    const clientWantsUsage = !!req.body.stream_options?.include_usage;

    // Optional speed limits (see PROMPT_TOKEN_LIMITS / MAX_TOKENS_CAPS). Skipped
    // for tool-calling requests, where dropping messages could orphan a tool
    // result from its call.
    let outgoingMessages = forwardedFields.messages;
    const promptLimit = PROMPT_TOKEN_LIMITS.get(primaryModel) ?? PROMPT_TOKEN_LIMITS.get(model);
    if (promptLimit && !req.body.tools) {
      const trimmed = trimMessagesToBudget(outgoingMessages, promptLimit);
      if (trimmed.dropped > 0) {
        outgoingMessages = trimmed.messages;
        console.log(
          `[TRIM #${reqId}] dropped ${trimmed.dropped} oldest messages ` +
          `(~${trimmed.before} -> ~${trimmed.after} est. tokens, limit ${promptLimit})`
        );
      }
    }
    const tokenCap = MAX_TOKENS_CAPS.get(primaryModel) ?? MAX_TOKENS_CAPS.get(model);

    const baseRequest = {
      ...forwardedFields,
      ...(outgoingMessages !== undefined && { messages: outgoingMessages }),
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT, tokenCap || Infinity),
      stream: stream || false,
      ...(stream && LOG_STREAM_USAGE && {
        stream_options: { ...forwardedFields.stream_options, include_usage: true }
      })
    };

    console.log(
      `[REQ #${reqId}] alias=${model} -> ${primaryModel} stream=${!!stream} ` +
      `messages=${Array.isArray(req.body.messages) ? req.body.messages.length : '?'} ` +
      `max_tokens=${baseRequest.max_tokens} reasoning_effort=${reasoning_effort ?? 'unset'} ` +
      `tools=${!!req.body.tools} chain=${modelChain.length}`
    );

    // NIM can queue a request for minutes before it accepts it (and a failed
    // model adds its own timeout on top). Browsers/frontends like JanitorAI
    // give up on a response that sends no bytes, which shows up as "network
    // error". So for streams, open the SSE response now and send heartbeat
    // comments until the upstream accepts. Trade-off: a failure after this
    // point reaches the client as an SSE error event rather than an HTTP error
    // status. Set EARLY_STREAM_HEADERS=false to go back to the old behavior.
    let waitBeat = null;
    if (stream && EARLY_STREAM_HEADERS) {
      openSseStream(res);
      safeWrite(res, ': connected\n\n');
      if (HEARTBEAT_INTERVAL_MS > 0) {
        const waitStartedAt = Date.now();
        waitBeat = setInterval(() => {
          if (res.writableEnded || res.destroyed) {
            clearInterval(waitBeat);
            return;
          }
          if (safeWrite(res, ': heartbeat\n\n')) {
            console.log(`[HEARTBEAT #${reqId}] still waiting for NIM to accept the request (${Math.round((Date.now() - waitStartedAt) / 1000)}s)`);
          }
        }, HEARTBEAT_INTERVAL_MS);
        waitBeat.unref();
      }
    }

    let callResult;
    try {
      callResult = await callWithFallback(
        baseRequest,
        modelChain,
        ENABLE_THINKING_MODE,
        reasoning_effort,
        !!req.body.tools,
        reqId
      );
    } finally {
      if (waitBeat) clearInterval(waitBeat);
    }
    const { response, model: usedModel, reasoningPayload: usedReasoningPayload } = callResult;

    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);
    const reasoningLabel = describeReasoning(usedReasoningPayload);

    const inlineReasoning =
      req.headers['x-reasoning-format'] === 'inline' ||
      INLINE_REASONING_MODELS.has('all') ||
      INLINE_REASONING_MODELS.has(usedModel);

    if (stream) {
      openSseStream(res); // already open if EARLY_STREAM_HEADERS sent it before the upstream call

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;
      const normalizer = new StreamNormalizer(usedModel);
      const toolRecovery = new ToolCallStreamRecovery();
      let firstTokenAt = null;
      let streamUsage = null;
      let completionLogged = false;

      // Stream diagnostics, reported in the [USAGE] line.
      let chunkCount = 0;
      let maxGapMs = 0;
      let lastChunkAt = null;
      let heartbeats = 0;
      const streamOpenedAt = Date.now();

      // Heartbeat: if nothing has arrived from upstream for a full interval,
      // send an SSE comment so the connection isn't dropped as idle.
      let lastBeatAt = 0;
      const heartbeat = (HEARTBEAT_INTERVAL_MS > 0 || STALL_TIMEOUT_MS > 0)
        ? setInterval(() => {
            const now = Date.now();
            const idleMs = now - (lastChunkAt ?? streamOpenedAt);

            // Watchdog: upstream has said nothing for too long, give up cleanly.
            if (STALL_TIMEOUT_MS > 0 && idleMs >= STALL_TIMEOUT_MS) {
              console.error(`[HEARTBEAT #${reqId}] upstream silent for ${Math.round(idleMs / 1000)}s (limit ${Math.round(STALL_TIMEOUT_MS / 1000)}s), aborting`);
              safeWrite(res, `data: ${JSON.stringify({
                error: { message: 'Upstream stalled', type: 'stream_stalled' }
              })}\n\n`);
              safeWrite(res, 'data: [DONE]\n\n');
              logStreamCompletion('stalled');
              if (!res.writableEnded) res.end();
              if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
              cleanup();
              return;
            }

            // Liveness beat: one every HEARTBEAT_INTERVAL_MS of silence.
            const quietSince = Math.max(lastChunkAt ?? streamOpenedAt, lastBeatAt);
            if (HEARTBEAT_INTERVAL_MS > 0 && now - quietSince >= HEARTBEAT_INTERVAL_MS) {
              if (safeWrite(res, ': heartbeat\n\n')) {
                heartbeats++;
                lastBeatAt = now;
                console.log(
                  `[HEARTBEAT #${reqId}] alive, upstream quiet ${Math.round(idleMs / 1000)}s ` +
                  `(elapsed ${Math.round((now - requestStartedAt) / 1000)}s, chunks=${chunkCount}, beats=${heartbeats})`
                );
              }
            }
          }, 1000)
        : null;
      if (heartbeat) heartbeat.unref();

      const logStreamCompletion = (outcome) => {
        if (completionLogged) return;
        completionLogged = true;
        logCompletion({
          reqId,
          stats: { chunks: chunkCount, maxGapMs, heartbeats },
          model: usedModel,
          requestedModel: primaryModel,
          effort: reasoningLabel,
          stream: true,
          outcome,
          startedAt: requestStartedAt,
          firstTokenAt,
          usage: streamUsage
        });
      };

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (heartbeat) clearInterval(heartbeat);
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        // Exact match avoids false-positiving on model output that happens
        // to contain the literal substring "[DONE]".
        if (line.trim() === 'data: [DONE]') {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        // Fast path: when a chunk needs no rewriting, forward it as-is and skip
        // the JSON.parse/stringify round trip. Safe only if there is no
        // reasoning field to strip or reformat, no '<' that could be part of a
        // <tool_call> tag, no delimiter parser for this model, and no
        // half-buffered tag or open <thinking> block from earlier chunks.
        if (
          !normalizer.parser &&
          !reasoningOpen &&
          toolRecovery.buffer === '' &&
          !line.includes('reasoning') &&
          !line.includes('<') &&
          !line.includes('"usage"')
        ) {
          if (firstTokenAt === null && /"content"\s*:\s*"[^"]/.test(line)) {
            firstTokenAt = Date.now();
          }
          safeWrite(res, `${line.trimEnd()}\n\n`);
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          if (data.usage) streamUsage = data.usage;
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            const normalizedDelta = normalizer.processDelta(delta);
            if (firstTokenAt === null && (normalizedDelta.content || normalizedDelta.reasoning)) {
              firstTokenAt = Date.now();
            }
            let clientContent = '';

            if (SHOW_REASONING && inlineReasoning) {
              if (normalizedDelta.reasoning && !reasoningOpen) {
                clientContent += `<thinking>\n${normalizedDelta.reasoning}`;
                reasoningOpen = true;
              } else if (normalizedDelta.reasoning) {
                clientContent += normalizedDelta.reasoning;
              }

              if (normalizedDelta.content && reasoningOpen) {
                clientContent += `\n</thinking>\n\n${normalizedDelta.content}`;
                reasoningOpen = false;
              } else if (normalizedDelta.content) {
                clientContent += normalizedDelta.content;
              }
            } else {
              clientContent = normalizedDelta.content || '';
            }

            const { content: recoveredContent, toolCallDeltas } = toolRecovery.process(clientContent);
            clientContent = recoveredContent;
            if (toolCallDeltas.length > 0) {
              delta.tool_calls = toolCallDeltas;
              if (data.choices[0]) data.choices[0].finish_reason = 'tool_calls';
            }

            delta.content = clientContent;

            if (SHOW_REASONING && normalizedDelta.reasoning) {
              delta.reasoning = normalizedDelta.reasoning;
              delta.reasoning_content = normalizedDelta.reasoning;
            } else {
              delete delta.reasoning;
              delete delta.reasoning_content;
            }
          }

          // Usage was requested by the proxy for logging; don't leak it to
          // clients that didn't ask for it (drop the usage-only chunk entirely).
          if (data.usage && !clientWantsUsage) {
            delete data.usage;
            if (!data.choices || data.choices.length === 0) return;
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
        } catch {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Upstream sent malformed chunk',
              type: 'stream_parse_error',
              details: line.slice(0, 100)
            }
          })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        const chunkAt = Date.now();
        if (lastChunkAt !== null) maxGapMs = Math.max(maxGapMs, chunkAt - lastChunkAt);
        lastChunkAt = chunkAt;
        chunkCount++;

        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream buffer overflow',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();
        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        const flushedDelta = normalizer.flush();

        const toolRecoveryLeftover = toolRecovery.flush();
        if (toolRecoveryLeftover) {
          console.warn('[TOOL_CALL_RECOVERY] Stream ended mid <tool_call> tag; flushing raw text instead of dropping it.');
          flushedDelta.content = (flushedDelta.content || '') + toolRecoveryLeftover;
        }

        if (flushedDelta.content || flushedDelta.reasoning) {
          let clientContent = '';

          if (SHOW_REASONING && inlineReasoning) {
            if (flushedDelta.reasoning && !reasoningOpen) {
              clientContent += `<thinking>\n${flushedDelta.reasoning}`;
              reasoningOpen = true;
            } else if (flushedDelta.reasoning) {
              clientContent += flushedDelta.reasoning;
            }

            if (flushedDelta.content && reasoningOpen) {
              clientContent += `\n</thinking>\n\n${flushedDelta.content}`;
              reasoningOpen = false;
            } else if (flushedDelta.content) {
              clientContent += flushedDelta.content;
            }
          } else {
            clientContent = flushedDelta.content || '';
          }

          const finalChunk = { choices: [{ delta: {} }] };
          if (clientContent) finalChunk.choices[0].delta.content = clientContent;

          if (SHOW_REASONING && !inlineReasoning && flushedDelta.reasoning) {
            finalChunk.choices[0].delta.reasoning = flushedDelta.reasoning;
            finalChunk.choices[0].delta.reasoning_content = flushedDelta.reasoning;
          }

          if (Object.keys(finalChunk.choices[0].delta).length > 0) {
            safeWrite(res, `data: ${JSON.stringify(finalChunk)}\n\n`);
          }
        }

        // Close an inline <thinking> tag left open if the model was cut off mid-reasoning.
        if (SHOW_REASONING && inlineReasoning && reasoningOpen) {
          safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { content: '\n</thinking>\n' } }] })}\n\n`);
          reasoningOpen = false;
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }
        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        logStreamCompletion('completed');
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        logStreamCompletion('upstream_error');
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;
        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
          logStreamCompletion('client_disconnected');
        }
        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        model: usedModel, // actual model that answered, may differ from the requested alias
        created: Math.floor(Date.now() / 1000),
        choices: (response.data.choices || []).map((choice, i) => {
          const normalizedChoice = normalizeNonStreamChoice(choice, usedModel);
          let content = normalizedChoice.message?.content || '';
          const reasoning = normalizedChoice.message?.reasoning || '';

          const { content: cleanedContent, toolCalls: recoveredToolCalls } = extractLeakedToolCalls(content);
          content = cleanedContent;

          if (SHOW_REASONING && inlineReasoning && reasoning) {
            content = `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
          }

          const finalMessage = { ...normalizedChoice.message, content };

          if (recoveredToolCalls.length > 0) {
            finalMessage.tool_calls = [
              ...(normalizedChoice.message?.tool_calls || []),
              ...recoveredToolCalls
            ];
            // null content on tool-call turns matches real OpenAI responses
            if (!finalMessage.content || !finalMessage.content.trim()) {
              finalMessage.content = null;
            }
          }

          if (SHOW_REASONING && reasoning) {
            finalMessage.reasoning = reasoning;
            finalMessage.reasoning_content = reasoning;
          } else {
            delete finalMessage.reasoning;
            delete finalMessage.reasoning_content;
          }

          const finalChoice = {
            ...normalizedChoice,
            index: i,
            message: finalMessage,
            ...(recoveredToolCalls.length > 0 && { finish_reason: 'tool_calls' })
          };
          return finalChoice;
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);

      logCompletion({
        reqId,
        model: usedModel,
        requestedModel: primaryModel,
        effort: reasoningLabel,
        stream: false,
        startedAt: requestStartedAt,
        firstTokenAt: null,
        usage: response.data.usage
      });
    }
  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    console.error('[PROXY] NIM response:', error.response?.data);

    if (!res.headersSent) {
      // Express only sets Content-Type if unset; force JSON in case the
      // streaming branch already set text/event-stream before failing.
      res.set('Content-Type', 'application/json');
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ──────────────────────────────────────────────────────────────

// Keeps a pooled TLS connection to NIM alive so the first request after an
// idle period doesn't pay for a fresh TCP + TLS handshake. Set
// KEEP_WARM_INTERVAL_MS=0 to disable.
const KEEP_WARM_INTERVAL_MS = process.env.KEEP_WARM_INTERVAL_MS !== undefined
  ? Number(process.env.KEEP_WARM_INTERVAL_MS)
  : 45000;

function startKeepWarm() {
  if (!KEEP_WARM_INTERVAL_MS || KEEP_WARM_INTERVAL_MS < 0) return;
  const timer = setInterval(() => {
    nim.head('/models', {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` },
      timeout: 5000,
      validateStatus: () => true // any response keeps the socket warm
    }).catch(() => {});
  }, KEEP_WARM_INTERVAL_MS);
  timer.unref();
}

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
  startKeepWarm();
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
