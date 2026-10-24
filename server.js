// server.js — OpenAI-compatible proxy for NVIDIA NIM
// Reasoning payload logic: reasoning.js. Tool-call leak recovery: tools.js.

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');
const { SHOW_REASONING, getReasoningPayload, resolveEffectiveThinking, StreamNormalizer, normalizeNonStreamChoice } = require('./reasoning');
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

const MAX_TOKENS_LIMIT = 65536;

// How often to send an SSE comment while waiting on upstream, so proxies/clients
// don't drop an idle streaming connection during long reasoning delays.
const HEARTBEAT_INTERVAL_MS = Number(process.env.HEARTBEAT_INTERVAL_MS) || 10000;

// Shared keep-alive agent for connection reuse on long-lived deployments.
const keepAliveAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10
});

const nim = axios.create({
  baseURL: NIM_API_BASE,
  httpsAgent: keepAliveAgent
});

// Per-attempt timeout before falling back to the next model. Reasoning
// models get a longer window since thinking delays first-token latency.
// Check these against your platform's own request duration limit.
// This is how long to wait for NIM to START answering (response headers). Keep it well
// under the client's own limit (JanitorAI gives up around 2 minutes), otherwise a hung
// model eats the whole window and the fallback never gets a chance to reply in time.
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS) || 60000;
// Same idea when thinking is on: this is the wait for NIM to start answering, not for the
// whole reply. Streaming models start within ~20s even when thinking, so a long wait here
// only delays the fallback past the client's own limit.
const REASONING_REQUEST_TIMEOUT_MS = Number(process.env.REASONING_REQUEST_TIMEOUT_MS) || 90000;
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
  // DeepSeek v4.1 Flash was timing out on NIM even for a tiny request (checked via /test-model).
  // To restore it, swap this back to 'deepseek-ai/deepseek-v4.1-flash'.
  'gpt-4-flash': 'z-ai/glm-5.3-flash',
  'deepseek': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4o': 'stepfun-ai/step-5-preview',
  'mistral': 'mistralai/mistral-large-2-instruct', 
  'mistral-turbo': 'nv-mistralai/mistral-nemo-12b-instruct',
  'mistral-pro': 'mistralai/mistral-7b-instruct-v0.3', 
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'nvidia/mistral-nemo-minitron-8b-8k-instruct', 
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'meta/muse-glimmer-30b',
  'google-lighter': 'poolside/laguna-xs-2.1',
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-flash': 'z-ai/glm-5.3-flash',

  // Vision-Modelle — nur Modelle, die erfolgreich auf Chat-Completions antworten.
  // Getestet am: 2026-09-17 — meta/llama-3.2-11b-vision-instruct: HTTP 200 ✓
  // meta/llama-3.2-90b-vision-instruct: Timeout (HTTP 000) — nicht hinzugefügt.
  // microsoft/phi-3-vision-128k-instruct: HTTP 404 (nicht in diesem Account) — nicht hinzugefügt.
  // nvidia/neva-22b: HTTP 404 — nicht hinzugefügt.
  // nvidia/vila: HTTP 404 — nicht hinzugefügt.
  'llama-vision': 'meta/llama-3.2-11b-vision-instruct',
  'vision': 'meta/llama-3.2-11b-vision-instruct',
  'llama-vision-11b': 'meta/llama-3.2-11b-vision-instruct',
  'llama-3.2-vision': 'meta/llama-3.2-11b-vision-instruct'
};

// Used when an unrecognized alias is requested. Must point at a live model.
const DEFAULT_MODEL = 'google/diffusiongemma-26b-a4b-it';

// Ordered by observed reliability/speed — an early failing model delays every fallback behind it.
const FALLBACK_MODELS = [
  'z-ai/glm-5.3',
  'z-ai/glm-5.3-flash',
  // 'deepseek-ai/deepseek-v4.1-flash', // removed: not responding on NIM, see MODEL_MAPPING note
  'nvidia/nemotron-3-super-120b-a12b'
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
  if (req.path === '/health' || req.path === '/v1/models' || req.path === '/' || req.path === '/test' || req.path === '/test-model') {
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

async function callWithFallback(baseRequest, models, enableThinking, clientReasoningEffort, hasTools, signal) {
  let lastError = null;
  const timeoutMs = resolveEffectiveThinking(enableThinking, clientReasoningEffort)
    ? REASONING_REQUEST_TIMEOUT_MS
    : REQUEST_TIMEOUT_MS;

  // Skip cooling-down models; fall back to the full list if that empties the chain.
  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    // Client already gone: don't start more upstream calls nobody will read.
    if (signal && signal.aborted) {
      throw lastError || new Error('Client disconnected');
    }

    const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools);
    const fullRequest = { ...baseRequest, model, ...reasoningPayload };

    if (DEBUG_MODE) {
      console.log(`[DEBUG] Attempting ${model} with reasoning payload:`, JSON.stringify(reasoningPayload), `(timeout: ${timeoutMs}ms)`);
    }

    try {
      const res = await nim.post(
        '/chat/completions',
        fullRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: timeoutMs,
          signal
        }
      );
      return { response: res, model };
    } catch (err) {
      lastError = err;

      // Aborted because the client disconnected, not a model failure.
      if (signal && signal.aborted) {
        throw err;
      }

      const status = err.response?.status;

      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        status,
        err.response?.data?.error?.message || err.message
      );
      if (DEBUG_MODE) {
        console.log(`[DEBUG] Full request body sent to ${model}:`, JSON.stringify(fullRequest));
        console.log(`[DEBUG] Full upstream error response:`, JSON.stringify(err.response?.data || null));
      }

      // Same key for every attempt: a 401 means every remaining model would fail identically.
      if (status === 401) {
        throw err;
      }

      // 403: likely an access-tier issue, not a dead key — cooldown just this model.
      if (status === 403) {
        setCooldown(model, ACCESS_DENIED_COOLDOWN_MS);
      }

      // 429: rate limited — short cooldown.
      if (status === 429) {
        setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
      }
    }
  }

  throw lastError || new Error('All models failed');
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
  let heartbeat = null;
  let streamStats = null;
  const abortController = new AbortController();

  try {
    // For streaming requests, open the SSE response immediately and send
    // periodic comment lines while waiting on upstream. Without this, slow
    // reasoning models (e.g. GLM-5.3) leave the client connection silent long
    // enough for the platform edge / browser to drop it ("load failed").
    if (req.body && req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders();
      safeWrite(res, ': connected\n\n');
      heartbeat = setInterval(() => safeWrite(res, ': keep-alive\n\n'), HEARTBEAT_INTERVAL_MS);
      res.on('close', () => {
        clearInterval(heartbeat);
        if (!streamEndedCleanly) abortController.abort();
        if (!streamEndedCleanly) {
          const s = streamStats;
          console.warn(
            '[STREAM] Client connection closed before stream finished' +
            (s ? ` (after ${Math.round((Date.now() - s.start) / 1000)}s of streaming: ${s.chunks} chunks, ${s.contentChars} content chars, ${s.reasoningChars} reasoning chars)` : ' (no upstream stream yet)')
          );
        }
        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
      });
    }

    const { model, max_tokens, temperature, stream, reasoning_effort } = req.body;

    let primaryModel = MODEL_MAPPING[model];
    if (!primaryModel) {
      console.warn(`[PROXY] Unknown model alias "${model}", falling back to default: ${DEFAULT_MODEL}`);
      primaryModel = DEFAULT_MODEL;
    }

    // De-dupe: avoids retrying the same model twice if it's also in FALLBACK_MODELS.
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];

    // Forward all client fields except model (replaced per-attempt) and
    // reasoning_effort (translated per-model by getReasoningPayload).
    const { model: _droppedModel, reasoning_effort: _droppedReasoningEffort, ...forwardedFields } = req.body;

    const baseRequest = {
      ...forwardedFields,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false
    };

    const { response, model: usedModel } = await callWithFallback(
      baseRequest,
      modelChain,
      ENABLE_THINKING_MODE,
      reasoning_effort,
      !!req.body.tools,
      abortController.signal
    );

    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    const inlineReasoning = req.headers['x-reasoning-format'] === 'inline';

    if (stream) {
      // Headers were already sent above (SSE headers + heartbeat).
      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;
      const normalizer = new StreamNormalizer(usedModel);
      const toolRecovery = new ToolCallStreamRecovery();
      const stats = streamStats = { chunks: 0, contentChars: 0, reasoningChars: 0, finish: null, start: Date.now(), raw: 0, ignored: 0 };

      // Progress log every 10s so a stalled/slow upstream is visible in the logs.
      const progress = setInterval(() => {
        console.log(
          `[STREAM] Progress (${usedModel}): ${Math.round((Date.now() - stats.start) / 1000)}s, ` +
          `${stats.chunks} chunks, ${stats.contentChars} content chars, ${stats.reasoningChars} reasoning chars`
        );
      }, 10000);

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        clearInterval(progress);
        if (heartbeat) clearInterval(heartbeat);
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) {
          if (DEBUG_MODE && line.trim() && stats.ignored++ < 5) {
            console.log('[DEBUG] Ignored non-"data: " upstream line:', line.slice(0, 200));
          }
          return;
        }
        if (DEBUG_MODE && stats.raw++ < 3) {
          console.log('[DEBUG] Upstream sample line:', line.slice(0, 300));
        }

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

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;
          stats.chunks++;
          if (stats.chunks === 1) {
            console.log(`[STREAM] First upstream chunk after ${Date.now() - stats.start}ms`);
          }
          if (data.choices?.[0]?.finish_reason) stats.finish = data.choices[0].finish_reason;

          if (delta) {
            const normalizedDelta = normalizer.processDelta(delta);
            stats.contentChars += (normalizedDelta.content || '').length;
            stats.reasoningChars += (normalizedDelta.reasoning || '').length;
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
        console.log(
          `[STREAM] Finished (${usedModel}): ${stats.chunks} chunks, ` +
          `${stats.contentChars} content chars, ${stats.reasoningChars} reasoning chars, ` +
          `finish_reason=${stats.finish}, ${Date.now() - stats.start}ms`
        );
        if (stats.contentChars === 0) {
          console.warn('[STREAM] Upstream produced NO visible content (reasoning may have used the whole token budget).');
        }
        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
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
    }
  } catch (error) {
    if (heartbeat) clearInterval(heartbeat);

    if (abortController.signal.aborted) {
      console.warn('[PROXY] Client left before any model answered; abandoned the fallback chain.');
      return;
    }

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

// ─── Test page ──────────────────────────────────────────────────────────
// Served at /test. The page itself exposes nothing; every call it makes uses your
// key from the browser (stored in localStorage only, never in the URL).
// NOTE: the client script below must not contain backticks or ${...}, since it lives
// inside a JS template string.
const TEST_PAGE_HTML = String.raw`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proxy tester</title>
<link rel="icon" href="data:,">
<style>
  :root { --bg:#0b0f14; --card:#11161d; --line:#222b36; --text:#e6edf3; --dim:#9aa7b2; --ok:#7ee787; --bad:#ff7b72; --warn:#e3b341; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; line-height:1.5; }
  main { max-width:760px; margin:0 auto; padding:20px 16px 60px; }
  h1 { font-size:1.3rem; margin:0 0 4px; }
  h2 { font-size:1rem; margin:0 0 4px; }
  .sub { color:var(--dim); margin:0 0 14px; font-size:.9rem; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:16px; }
  label { display:block; font-size:.8rem; color:var(--dim); margin:10px 0 4px; }
  input, select, textarea { width:100%; background:#0b0f14; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:10px; font-size:16px; font-family:inherit; }
  textarea { min-height:90px; resize:vertical; }
  .row { display:flex; gap:12px; flex-wrap:wrap; }
  .row > div { flex:1 1 160px; }
  button { background:var(--ok); color:#06210d; border:0; border-radius:8px; padding:10px 16px; font-size:16px; font-weight:600; cursor:pointer; margin:12px 8px 0 0; }
  button.secondary { background:transparent; color:var(--text); border:1px solid var(--line); }
  button:disabled { opacity:.5; cursor:default; }
  pre.out { white-space:pre-wrap; word-wrap:break-word; background:#0b0f14; border:1px solid var(--line); border-radius:8px; padding:12px; min-height:56px; margin:12px 0 0; font-family:inherit; font-size:.95rem; }
  .stats { color:var(--dim); font-size:.85rem; margin-top:8px; word-wrap:break-word; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; margin-top:12px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; word-break:break-word; }
  th { color:var(--dim); font-weight:500; }
  .ok { color:var(--ok); } .bad { color:var(--bad); } .warn { color:var(--warn); }
  details { margin-top:10px; color:var(--dim); }
  summary { cursor:pointer; }
</style>
</head>
<body>
<main>
  <h1>Proxy tester</h1>
  <p class="sub">Try models through your proxy. Your key is kept only in this browser.</p>

  <div class="card">
    <h2>Connection <span id="health" class="stats"></span></h2>
    <label for="key">CLIENT_AUTH_KEY</label>
    <input id="key" type="password" autocomplete="off" placeholder="Paste your key once">
  </div>

  <div class="card">
    <h2>Chat test</h2>
    <p class="sub">Streams a real reply the way Janitor does, including model fallback.</p>
    <div class="row">
      <div><label for="model">Model</label><select id="model"></select></div>
      <div><label for="effort">Reasoning</label>
        <select id="effort">
          <option value="">Proxy default</option>
          <option value="off">off</option>
          <option value="on">on</option>
          <option value="low">low</option>
          <option value="high">high</option>
          <option value="max">max</option>
        </select>
      </div>
      <div><label for="maxtok">Max tokens</label><input id="maxtok" type="number" value="1024" min="1" max="65536"></div>
    </div>
    <label for="prompt">Message</label>
    <textarea id="prompt">Write a short two-sentence story about a lighthouse keeper.</textarea>
    <button id="send">Send</button><button id="stop" class="secondary" disabled>Stop</button>
    <div id="stats" class="stats"></div>
    <pre id="reply" class="out"></pre>
    <details id="rdet" hidden><summary>Reasoning</summary><pre id="reasoning" class="out"></pre></details>
  </div>

  <div class="card">
    <h2>Model check</h2>
    <p class="sub">Sends a tiny message straight to NIM with no fallback, so a broken model can't hide.</p>
    <label for="checkModel">Full NIM model ID (optional, otherwise uses the model chosen above)</label>
    <input id="checkModel" placeholder="e.g. z-ai/glm-5.3">
    <button id="checkOne">Check one model</button><button id="checkAll" class="secondary">Check all models</button>
    <div id="checkOut" class="stats"></div>
  </div>
</main>

<script>
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var KEY_STORE = 'proxyTesterKey';
  var MODEL_STORE = 'proxyTesterModel';

  function lsGet(k) { try { return localStorage.getItem(k) || ''; } catch (e) { return ''; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function key() { return $('key').value.trim(); }
  function fmt(ms) { return (ms / 1000).toFixed(1) + 's'; }
  function authHeaders(extra) {
    var h = { 'Authorization': 'Bearer ' + key() };
    for (var k in extra) { h[k] = extra[k]; }
    return h;
  }
  function setStats(el, text, cls) { el.textContent = text; el.className = 'stats' + (cls ? ' ' + cls : ''); }

  $('key').value = lsGet(KEY_STORE);
  $('key').addEventListener('input', function () { lsSet(KEY_STORE, key()); });

  fetch('/health').then(function (r) { return r.json(); }).then(function (j) {
    setStats($('health'), '· server ok (v' + j.version + ')', 'ok');
  }).catch(function () {
    setStats($('health'), '· server not reachable (it may be waking up, retry in a minute)', 'warn');
  });

  fetch('/v1/models').then(function (r) { return r.json(); }).then(function (j) {
    var sel = $('model');
    (j.data || []).forEach(function (m) {
      var o = document.createElement('option');
      o.value = m.id; o.textContent = m.id; sel.appendChild(o);
    });
    var pref = lsGet(MODEL_STORE) || 'glm-5.3';
    sel.value = pref;
    if (sel.value !== pref) { sel.selectedIndex = 0; }
  });
  $('model').addEventListener('change', function () { lsSet(MODEL_STORE, $('model').value); });

  /* ---------- Chat test ---------- */
  var controller = null;
  function setBusy(b) { $('send').disabled = b; $('stop').disabled = !b; }
  $('stop').addEventListener('click', function () { if (controller) { controller.abort(); } });

  $('send').addEventListener('click', function () {
    if (!key()) { setStats($('stats'), 'Enter your key first.', 'bad'); return; }

    var body = {
      model: $('model').value,
      messages: [{ role: 'user', content: $('prompt').value }],
      stream: true,
      max_tokens: Number($('maxtok').value) || 1024
    };
    var eff = $('effort').value;
    if (eff) { body.reasoning_effort = eff; }

    var t0 = performance.now(), tFirst = null;
    var content = '', reasoning = '', chunks = 0, used = '', finish = '', gotError = '';
    var finished = false;

    $('reply').textContent = '';
    $('reasoning').textContent = '';
    $('rdet').hidden = true;
    controller = new AbortController();
    setBusy(true);

    var timer = setInterval(function () {
      var label = tFirst === null ? (reasoning ? 'Model is reasoning… ' : 'Waiting for first text… ') : 'Streaming… ';
      setStats($('stats'), label + fmt(performance.now() - t0));
    }, 250);

    function done(note) {
      if (finished) { return; }
      finished = true;
      clearInterval(timer);
      setBusy(false);
      controller = null;
      var s = 'Model: ' + (used || '?') +
        ' · first text: ' + (tFirst === null ? 'none' : fmt(tFirst)) +
        ' · total: ' + fmt(performance.now() - t0) +
        ' · ' + chunks + ' chunks · ' + content.length + ' chars' +
        (finish ? ' · finish: ' + finish : '');
      var cls = 'ok';
      if (!chunks && gotError && !note) { s = 'Request failed: ' + gotError; cls = 'bad'; }
      else if (note) { s += ' · ' + note; cls = 'bad'; }
      else if (gotError) { s += ' · error: ' + gotError; cls = 'bad'; }
      else if (!content) { s += ' · No visible text arrived (the model may have used its whole token budget reasoning).'; cls = 'warn'; }
      setStats($('stats'), s, cls);
    }

    function onLine(line) {
      line = line.trim();
      if (line.indexOf('data: ') !== 0) { return; }
      var p = line.slice(6);
      if (p === '[DONE]') { return; }
      var j;
      try { j = JSON.parse(p); } catch (e) { return; }
      if (j.error) {
        gotError = j.error.message || 'unknown error';
        $('reply').textContent += '\n[error: ' + gotError + ']';
        return;
      }
      if (j.model) { used = j.model; }
      var c = j.choices && j.choices[0];
      if (!c) { return; }
      if (c.finish_reason) { finish = c.finish_reason; }
      var d = c.delta || {};
      chunks++;
      var r = d.reasoning_content || d.reasoning;
      if (r) {
        reasoning += r;
        $('reasoning').textContent = reasoning;
        $('rdet').hidden = false;
      }
      if (d.content) {
        if (tFirst === null) { tFirst = performance.now() - t0; }
        content += d.content;
        $('reply').textContent = content;
      }
    }

    fetch('/v1/chat/completions', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
      signal: controller.signal
    }).then(function (res) {
      if (!res.ok) {
        return res.text().then(function (t) {
          gotError = 'HTTP ' + res.status + ' ' + t.slice(0, 300);
          done('');
        });
      }
      var reader = res.body.getReader();
      var dec = new TextDecoder();
      var buf = '';
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) { if (buf) { onLine(buf); } done(''); return; }
          buf += dec.decode(r.value, { stream: true });
          var lines = buf.split('\n');
          buf = lines.pop();
          lines.forEach(onLine);
          return pump();
        });
      }
      return pump();
    }).catch(function (e) {
      done(e && e.name === 'AbortError' ? 'stopped by you' : 'network error: ' + (e && e.message));
    });
  });

  /* ---------- Model check ---------- */
  function runCheck(model, timeoutSec) {
    var url = '/test-model?model=' + encodeURIComponent(model) + '&timeout=' + timeoutSec;
    return fetch(url, { headers: authHeaders() }).then(function (r) {
      if (r.status === 403) { throw new Error('Wrong or missing key'); }
      return r.json();
    });
  }

  function describe(j) {
    if (j.ok) {
      var reply = j.reply ? String(j.reply).replace(/\s+/g, ' ').slice(0, 60) : '';
      var t = 'OK in ' + j.seconds + 's' +
        (j.reasoning_chars ? ' · ' + j.reasoning_chars + ' reasoning chars' : '') +
        (reply ? ' · "' + reply + '"' : ' · empty reply');
      return { cls: reply ? 'ok' : 'warn', text: t };
    }
    var e = j.error;
    if (e && typeof e === 'object') { e = e.detail || e.message || JSON.stringify(e); }
    return {
      cls: 'bad',
      text: 'Failed after ' + j.seconds + 's' + (j.http_status ? ' (HTTP ' + j.http_status + ')' : '') + ': ' + String(e).slice(0, 200)
    };
  }

  $('checkOne').addEventListener('click', function () {
    if (!key()) { setStats($('checkOut'), 'Enter your key first.', 'bad'); return; }
    var m = $('checkModel').value.trim() || $('model').value;
    var btn = $('checkOne');
    btn.disabled = true;
    setStats($('checkOut'), 'Testing ' + m + ' (up to 60s)…');
    runCheck(m, 60).then(function (j) {
      var d = describe(j);
      setStats($('checkOut'), j.model + ' — ' + d.text, d.cls);
    }).catch(function (e) {
      setStats($('checkOut'), e.message, 'bad');
    }).then(function () { btn.disabled = false; });
  });

  $('checkAll').addEventListener('click', function () {
    if (!key()) { setStats($('checkOut'), 'Enter your key first.', 'bad'); return; }
    var btn = $('checkAll');
    var out = $('checkOut');
    btn.disabled = true;
    setStats(out, 'Loading model list…');

    fetch('/v1/models?live=true').then(function (r) { return r.json(); }).then(function (j) {
      if (!j.data) { throw new Error((j.error && j.error.message) || 'Could not load the model list'); }

      var seen = {}, items = [];
      j.data.forEach(function (m) {
        if (seen[m.backend]) { seen[m.backend].aliases.push(m.id); }
        else {
          seen[m.backend] = { backend: m.backend, aliases: [m.id], available: m.available };
          items.push(seen[m.backend]);
        }
      });

      out.textContent = '';
      out.className = '';
      var table = document.createElement('table');
      var head = document.createElement('tr');
      ['Alias', 'NIM model', 'Result'].forEach(function (h) {
        var th = document.createElement('th'); th.textContent = h; head.appendChild(th);
      });
      table.appendChild(head);
      items.forEach(function (it) {
        var tr = document.createElement('tr');
        var a = document.createElement('td'); a.textContent = it.aliases.join(', ');
        var b = document.createElement('td'); b.textContent = it.backend;
        it.cell = document.createElement('td');
        it.cell.textContent = 'queued';
        tr.appendChild(a); tr.appendChild(b); tr.appendChild(it.cell);
        table.appendChild(tr);
      });
      out.appendChild(table);

      var idx = 0, active = 0;
      function next() {
        if (idx >= items.length) { if (active === 0) { btn.disabled = false; } return; }
        var it = items[idx++];
        if (!it.available) {
          it.cell.textContent = 'Not in NIM catalog';
          it.cell.className = 'bad';
          return next();
        }
        active++;
        it.cell.textContent = 'testing…';
        runCheck(it.aliases[0], 30).then(function (r) {
          var d = describe(r);
          it.cell.textContent = d.text;
          it.cell.className = d.cls;
        }).catch(function (e) {
          it.cell.textContent = e.message;
          it.cell.className = 'bad';
        }).then(function () { active--; next(); });
      }
      for (var i = 0; i < 3; i++) { next(); }
    }).catch(function (e) {
      setStats(out, e.message, 'bad');
      btn.disabled = false;
    });
  });
})();
</script>
</body>
</html>`;

app.get('/test', (req, res) => {
  res.type('html').send(TEST_PAGE_HTML);
});

// Quick diagnostic: sends one tiny message to a model and reports timing + result.
// Open in a browser: /test-model?key=YOUR_CLIENT_AUTH_KEY&model=gpt-4-flash
// (model can be an alias like glm-5.3, or a full NIM id). Optional: &timeout=60
app.get('/test-model', async (req, res) => {
  // Key can come from the Authorization header (used by the /test page) or ?key=.
  const key = extractBearerToken(req.headers.authorization) ||
    (typeof req.query.key === 'string' ? req.query.key : '');
  if (!CLIENT_AUTH_KEY || !safeTimingEqual(key, CLIENT_AUTH_KEY)) {
    return res.status(403).json({ error: 'Missing or wrong key (send it as a Bearer token, or add ?key=YOUR_CLIENT_AUTH_KEY)' });
  }

  const requested = String(req.query.model || 'gpt-4-flash');
  const modelId = MODEL_MAPPING[requested] || requested;
  const timeoutMs = Math.min(Number(req.query.timeout) || 60, 120) * 1000;
  const body = {
    model: modelId,
    messages: [{ role: 'user', content: 'Say hi in five words.' }],
    max_tokens: 200,
    stream: false,
    ...getReasoningPayload(modelId, ENABLE_THINKING_MODE, undefined, false)
  };

  const started = Date.now();
  try {
    const r = await nim.post('/chat/completions', body, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      timeout: timeoutMs
    });
    const msg = r.data?.choices?.[0]?.message || {};
    res.json({
      ok: true,
      model: modelId,
      seconds: ((Date.now() - started) / 1000).toFixed(1),
      reply: msg.content || null,
      reasoning_chars: (msg.reasoning_content || msg.reasoning || '').length,
      finish_reason: r.data?.choices?.[0]?.finish_reason,
      sent: body
    });
  } catch (err) {
    res.json({
      ok: false,
      model: modelId,
      seconds: ((Date.now() - started) / 1000).toFixed(1),
      http_status: err.response?.status || null,
      error: err.response?.data || err.message,
      sent: body
    });
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

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
