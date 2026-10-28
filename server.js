// server.js — OpenAI-compatible proxy for NVIDIA NIM (v6.1)

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual, createHash } = require('crypto');
const { SHOW_REASONING, getReasoningPayload, StreamNormalizer, normalizeNonStreamChoice } = require('./reasoning');
const { extractLeakedToolCalls, ToolCallStreamRecovery } = require('./tools');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───
const num = (v, d) => { const n = Number(v); return Number.isFinite(n) && n > 0 ? n : d; };

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';

const MAX_TOKENS_LIMIT = 65536;
const DEFAULT_MAX_TOKENS = num(process.env.DEFAULT_MAX_TOKENS, 4096);
const HEARTBEAT_INTERVAL_MS = 10000;

// Speed controls. The important one is FIRST_TOKEN_TIMEOUT_MS: if a model has not produced
// anything by then, we give up on it and move to the next model in the chain.
const FIRST_TOKEN_TIMEOUT_MS = num(process.env.FIRST_TOKEN_TIMEOUT_MS, 30000); // streaming: wait for first chunk
const STREAM_IDLE_TIMEOUT_MS = num(process.env.STREAM_IDLE_TIMEOUT_MS, 60000); // streaming: max silence mid-reply
const NON_STREAM_TIMEOUT_MS = num(process.env.REQUEST_TIMEOUT_MS, 90000);      // non-streaming: whole reply
const TOTAL_BUDGET_MS = num(process.env.TOTAL_BUDGET_MS, 120000);              // all fallbacks combined

const keepAliveAgent = new https.Agent({ keepAlive: true, maxSockets: 50, keepAliveMsecs: 10000 });
const nim = axios.create({ baseURL: NIM_API_BASE, httpsAgent: keepAliveAgent });

if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

if (!NIM_API_KEY) { console.error('[FATAL] NIM_API_KEY is required.'); process.exit(1); }
if (!CLIENT_AUTH_KEY) console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected.');

// ─── Model Mapping (Verified NIM IDs) ───
const MODEL_MAPPING = {
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4-flash': 'z-ai/glm-5.3-flash',
  'glm-flash': 'z-ai/glm-5.3-flash',
  'glm-5.3': 'z-ai/glm-5.3',
  'deepseek': 'deepseek-ai/deepseek-v4.1-flash',
  'deepseek-flash': 'deepseek-ai/deepseek-v4-flash-0731',
  'llama-3.1-8b': 'meta/llama-3.1-8b-instruct',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'm3': 'minimaxai/minimax-m3'
};

// Optional: MODEL_MAP_JSON='{"gpt-4o":"deepseek-ai/deepseek-v4-pro-0813"}' adds/overrides aliases without editing code.
try {
  if (process.env.MODEL_MAP_JSON) Object.assign(MODEL_MAPPING, JSON.parse(process.env.MODEL_MAP_JSON));
} catch (e) { console.warn(`[CONFIG] MODEL_MAP_JSON ignored: ${e.message}`); }

const DEFAULT_MODEL = 'meta/llama-3.1-8b-instruct';
const FALLBACK_MODELS = process.env.FALLBACK_MODELS
  ? process.env.FALLBACK_MODELS.split(',').map(s => s.trim()).filter(Boolean)
  : ['meta/llama-3.1-8b-instruct', 'z-ai/glm-5.3-flash'];

function resolveModelId(name) {
  if (typeof name !== 'string') return null;
  if (MODEL_MAPPING[name]) return MODEL_MAPPING[name];
  if (/^[\w.-]+\/[\w.:-]+$/.test(name)) return name;
  console.warn(`[MODEL] Unknown alias "${name}", using default ${DEFAULT_MODEL}`);
  return null;
}

// Only forward standard OpenAI fields. Unknown extras from clients are a common cause of 400s.
const PASSTHROUGH = new Set([
  'messages', 'temperature', 'top_p', 'stop', 'frequency_penalty', 'presence_penalty',
  'seed', 'response_format', 'tools', 'tool_choice',
  ...(process.env.EXTRA_PARAMS || '').split(',').map(s => s.trim()).filter(Boolean)
]);

function buildUpstreamRequest(body, isStream) {
  const out = {};
  for (const k of PASSTHROUGH) if (body[k] !== undefined && body[k] !== null) out[k] = body[k];
  if (!Array.isArray(out.tools) || out.tools.length === 0) { delete out.tools; delete out.tool_choice; }
  out.temperature = body.temperature ?? 0.7;
  const requested = Number(body.max_tokens ?? body.max_completion_tokens);
  out.max_tokens = Math.min(requested > 0 ? requested : DEFAULT_MAX_TOKENS, MAX_TOKENS_LIMIT);
  out.stream = isStream;
  return out;
}

// ─── Middleware ───
app.use(cors());
app.use(express.json({ limit: '10mb' }));

function safeTimingEqual(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string' || !a || !b) return false;
  return timingSafeEqual(createHash('sha256').update(a).digest(), createHash('sha256').update(b).digest());
}

app.use((req, res, next) => {
  if (req.method === 'OPTIONS') return next();
  if (req.path === '/health' || req.path === '/' || req.path === '/v1/models') return next();
  const token = (req.headers.authorization || '').replace(/^Bearer\s+/i, '').trim();
  if (!token || !CLIENT_AUTH_KEY || !safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({ error: { message: 'Forbidden', type: 'authentication_error', code: 403 } });
  }
  next();
});

// ─── Helpers ───
function safeWrite(res, data) {
  try { if (!res.writableEnded && !res.destroyed) res.write(data); } catch {}
}

function finishStreamWithNotice(res, text) {
  if (text) safeWrite(res, `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: text }, finish_reason: null }] })}\n\n`);
  safeWrite(res, `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`);
  safeWrite(res, 'data: [DONE]\n\n');
  if (!res.writableEnded) { try { res.end(); } catch {} }
}

function getErrorMessage(err) {
  if (!err) return 'Unknown error';
  const d = err.response?.data;
  const msg = typeof d === 'string' ? d : d?.error?.message || d?.message || d?.detail;
  return String(msg || err.message || 'Request failed').slice(0, 300);
}

// When we ask axios for a stream and NIM answers 4xx/5xx, the error body is a stream. Read it so errors are readable.
async function readErrorBody(err) {
  const d = err?.response?.data;
  if (!d || typeof d.on !== 'function') return err;
  try {
    const chunks = [];
    let size = 0;
    await new Promise((resolve) => {
      const t = setTimeout(resolve, 2000);
      const done = () => { clearTimeout(t); resolve(); };
      d.on('data', c => { if (size < 4096) { chunks.push(c); size += c.length; } });
      d.on('end', done);
      d.on('error', done);
    });
    if (!d.destroyed) d.destroy();
    const text = Buffer.concat(chunks).toString('utf8');
    try { err.response.data = JSON.parse(text); } catch { err.response.data = text; }
  } catch {}
  return err;
}

/**
 * One upstream attempt.
 *  - non-stream: resolves { data } once the whole body has arrived (timeout covers the whole reply)
 *  - stream:     resolves { stream, firstChunk } as soon as the model emits its FIRST chunk.
 *                The stream is left paused so the caller can attach listeners without losing data.
 */
async function openOnce(payload, isStream, signal, timeoutMs) {
  const ac = new AbortController();
  let timedOut = false;
  const onAbort = () => ac.abort();
  if (signal) { if (signal.aborted) ac.abort(); else signal.addEventListener('abort', onAbort, { once: true }); }
  const timer = setTimeout(() => { timedOut = true; ac.abort(); }, timeoutMs);
  const disarm = () => { clearTimeout(timer); signal?.removeEventListener('abort', onAbort); };
  const toError = (err) => {
    disarm();
    if (timedOut) {
      const e = new Error(`Model produced no output within ${Math.round(timeoutMs / 1000)}s`);
      e.timedOut = true;
      return e;
    }
    return err;
  };

  let res;
  try {
    res = await nim.post('/chat/completions', payload, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        ...(isStream ? { Accept: 'text/event-stream', 'Accept-Encoding': 'identity' } : {})
      },
      responseType: isStream ? 'stream' : 'json',
      timeout: 0,
      signal: ac.signal
    });
  } catch (err) {
    if (isStream) await readErrorBody(err);
    throw toError(err);
  }

  if (!isStream) { disarm(); return { data: res.data }; }

  const stream = res.data;
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      stream.removeListener('data', onData);
      stream.removeListener('error', onError);
      stream.removeListener('end', onEmpty);
      stream.removeListener('close', onEmpty);
    };
    const onData = (chunk) => {
      if (settled) return;
      settled = true; cleanup(); stream.pause(); disarm();
      stream.on('error', () => {}); // never let a late error crash the process
      resolve({ stream, firstChunk: chunk });
    };
    const onError = (err) => {
      if (settled) return;
      settled = true; cleanup();
      reject(toError(err));
    };
    const onEmpty = () => {
      if (settled) return;
      settled = true; cleanup();
      const e = toError(new Error('Model closed the stream without output'));
      reject(e);
    };
    stream.on('data', onData);
    stream.on('error', onError);
    stream.on('end', onEmpty);
    stream.on('close', onEmpty);
  });
}

// Keep-alive sockets can be closed by the server while idle; retry once on a reset.
async function openUpstream(payload, isStream, signal, timeoutMs) {
  try {
    return await openOnce(payload, isStream, signal, timeoutMs);
  } catch (err) {
    if (['ECONNRESET', 'EPIPE'].includes(err.code) && !signal?.aborted) {
      return openOnce(payload, isStream, signal, timeoutMs);
    }
    throw err;
  }
}

const cooldowns = new Map();

async function callWithFallback({ baseReq, models, effort, isStream, signal }) {
  const started = Date.now();
  const active = models.filter(m => (cooldowns.get(m) || 0) <= started);
  const order = active.length ? active : models;
  let lastErr = null;

  for (const model of order) {
    if (signal?.aborted) throw lastErr || new Error('Client aborted');
    const remaining = TOTAL_BUDGET_MS - (Date.now() - started);
    if (lastErr && remaining < 3000) break;
    const perAttempt = isStream ? FIRST_TOKEN_TIMEOUT_MS : NON_STREAM_TIMEOUT_MS;
    const timeoutMs = Math.max(3000, Math.min(perAttempt, remaining));

    // First try with reasoning params; if NIM rejects them (400/422), retry the same model without.
    const reasoning = getReasoningPayload(model, ENABLE_THINKING_MODE, effort);
    const variants = Object.keys(reasoning).length ? [reasoning, {}] : [{}];

    for (let v = 0; v < variants.length; v++) {
      const payload = { ...baseReq, model, ...variants[v] };
      const t0 = Date.now();
      console.log(`[NIM] -> ${model}${v ? ' (without reasoning params)' : ''} ${JSON.stringify(variants[v])}`);
      try {
        const opened = await openUpstream(payload, isStream, signal, timeoutMs);
        return { ...opened, model, waitedMs: Date.now() - t0 };
      } catch (err) {
        lastErr = err;
        const status = err.response?.status;
        console.warn(`[FALLBACK] ${model} failed after ${Date.now() - t0}ms | ${status || (err.timedOut ? 'timeout' : err.code || 'error')} | ${getErrorMessage(err)}`);
        if (signal?.aborted) throw err;
        if (status === 401) throw err; // bad NIM key: no point trying other models
        if ((status === 400 || status === 422) && v < variants.length - 1) continue;
        if (status === 429) cooldowns.set(model, Date.now() + 30000);
        else if (status === 404) cooldowns.set(model, Date.now() + 5 * 60000);
        else if (status === 503 || status === 504 || err.timedOut) cooldowns.set(model, Date.now() + 60000);
        break;
      }
    }
  }
  throw lastErr || new Error('All models failed');
}

// ─── Routes ───
app.get('/', (req, res) => res.send('Proxy is running.'));
app.get('/health', (req, res) => res.json({ status: 'ok' }));
app.get('/v1/models', (req, res) => {
  res.json({ object: 'list', data: Object.keys(MODEL_MAPPING).map(id => ({ id, object: 'model', owned_by: 'nim-proxy' })) });
});

app.post('/v1/chat/completions', async (req, res) => {
  const body = req.body || {};
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return res.status(400).json({ error: { message: '`messages` must be a non-empty array', type: 'invalid_request_error', code: 400 } });
  }

  const isStream = !!body.stream;
  const inline = SHOW_REASONING && req.headers['x-reasoning-format'] === 'inline';
  const reqStart = Date.now();
  const abortController = new AbortController();
  let upstream = null, heartbeat = null, idleTimer = null, finished = false;

  const teardown = () => {
    if (heartbeat) clearInterval(heartbeat);
    if (idleTimer) clearTimeout(idleTimer);
    if (upstream && !upstream.destroyed) upstream.destroy();
  };
  res.on('close', () => {
    if (!res.writableFinished) abortController.abort();
    finished = true;
    teardown();
  });

  try {
    if (isStream) {
      res.status(200);
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // stop nginx-style proxies from buffering the stream
      res.flushHeaders();
      try { res.socket?.setNoDelay(true); } catch {}
      safeWrite(res, ': connected\n\n');
      heartbeat = setInterval(() => safeWrite(res, ': keep-alive\n\n'), HEARTBEAT_INTERVAL_MS);
    }

    const primary = resolveModelId(body.model) || DEFAULT_MODEL;
    const chain = [...new Set([primary, ...FALLBACK_MODELS])];
    const baseReq = buildUpstreamRequest(body, isStream);
    const hasTools = !!baseReq.tools;

    const opened = await callWithFallback({
      baseReq, models: chain, effort: body.reasoning_effort, isStream, signal: abortController.signal
    });
    const usedModel = opened.model;
    console.log(`[NIM] ${usedModel} answered after ${opened.waitedMs}ms (${isStream ? 'first chunk' : 'full reply'})`);

    // ─── Non-Streaming Path ───
    if (!isStream) {
      const data = opened.data;
      const choices = (data.choices || []).map((c, i) => {
        const norm = normalizeNonStreamChoice(c, usedModel);
        const msg = { ...norm.message };
        let content = msg.content || '';
        const reasoning = msg.reasoning || '';
        let finish = norm.finish_reason;

        if (hasTools && content) {
          const r = extractLeakedToolCalls(content);
          content = r.content;
          if (r.toolCalls.length) {
            msg.tool_calls = [...(msg.tool_calls || []), ...r.toolCalls];
            finish = 'tool_calls';
          }
        }
        msg.content = content || (msg.tool_calls?.length ? null : '');

        delete msg.reasoning; delete msg.reasoning_content;
        if (SHOW_REASONING && reasoning) {
          if (inline) msg.content = `<thinking>\n${reasoning}\n</thinking>\n\n${msg.content || ''}`;
          else { msg.reasoning = reasoning; msg.reasoning_content = reasoning; }
        }
        console.log(`[REASONING] ${usedModel} ${reasoning ? `thought ~${Math.round(reasoning.length / 4)} tokens` : 'no reasoning output'}`);
        return { ...norm, index: i, message: msg, finish_reason: finish };
      });
      finished = true;
      console.log(`[DONE] ${usedModel} non-stream total ${Date.now() - reqStart}ms`);
      return res.json({ ...data, model: usedModel, choices });
    }

    // ─── Streaming Path ───
    upstream = opened.stream;
    const decoder = new StringDecoder('utf8');
    const normalizer = new StreamNormalizer(usedModel);
    const toolRecovery = hasTools ? new ToolCallStreamRecovery() : null;

    const st = {
      id: `chatcmpl-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`,
      created: Math.floor(Date.now() / 1000),
      finishReason: null, recoveredTool: false, reasoningOpen: false,
      visible: 0, reasoned: 0, firstVisibleAt: null
    };
    let buffer = '';

    const emit = (obj) => safeWrite(res, `data: ${JSON.stringify(obj)}\n\n`);
    const emitDelta = (delta, finish = null) => emit({
      id: st.id, object: 'chat.completion.chunk', created: st.created, model: usedModel,
      choices: [{ index: 0, delta, finish_reason: finish }]
    });

    // Turns (reasoning, content) into what we actually send to the client
    const render = (reasoning, content) => {
      if (!SHOW_REASONING) return { content: content || '' };
      if (!inline) return { content: content || '', reasoning: reasoning || '' };
      let out = '';
      if (reasoning) {
        if (!st.reasoningOpen) { out += '<thinking>\n'; st.reasoningOpen = true; }
        out += reasoning;
      }
      if (content) {
        if (st.reasoningOpen) { out += '\n</thinking>\n\n'; st.reasoningOpen = false; }
        out += content;
      }
      return { content: out };
    };

    const handleData = (data) => {
      if (data.error && !data.choices) {
        emitDelta({ content: `\n\n[Upstream error: ${String(data.error.message || data.error).slice(0, 200)}]` });
        return;
      }
      const choice = data.choices?.[0];
      if (!choice) { emit(data); return; } // e.g. usage-only chunk

      // Hold the finish_reason back: it is sent once at the very end, after everything has been flushed.
      if (choice.finish_reason) { st.finishReason = choice.finish_reason; choice.finish_reason = null; }

      const norm = normalizer.processDelta(choice.delta || {});
      let content = norm.content || '';
      const reasoning = norm.reasoning || '';
      let recovered = [];

      if (toolRecovery && content) {
        const r = toolRecovery.process(content);
        content = r.content; recovered = r.toolCallDeltas;
      }

      const out = { ...norm };
      delete out.content; delete out.reasoning; delete out.reasoning_content;
      const r = render(reasoning, content);
      if (r.content) out.content = r.content;
      if (r.reasoning) { out.reasoning = r.reasoning; out.reasoning_content = r.reasoning; }
      if (recovered.length) {
        out.tool_calls = [...(out.tool_calls || []), ...recovered];
        st.recoveredTool = true;
      }

      st.visible += content.length;
      st.reasoned += reasoning.length;
      if (content && !st.firstVisibleAt) st.firstVisibleAt = Date.now() - reqStart;

      if (Object.keys(out).length === 0 && !data.usage) return; // nothing to forward
      data.choices[0] = { ...choice, delta: out };
      emit(data);
    };

    const processLine = (raw) => {
      const line = raw.trim();
      if (!line.startsWith('data:')) return;
      const payload = line.slice(5).trim();
      if (!payload || payload === '[DONE]') return;
      let data;
      try { data = JSON.parse(payload); } catch { return; }
      try { handleData(data); } catch (e) { console.warn(`[STREAM] chunk error: ${e.message}`); }
    };

    const finalize = () => {
      if (finished) return;
      finished = true;
      if (idleTimer) clearTimeout(idleTimer);

      buffer += decoder.end();
      if (buffer.trim()) { buffer.split('\n').forEach(processLine); buffer = ''; }

      const flushed = normalizer.flush();
      let content = flushed.content || '';
      const reasoning = flushed.reasoning || '';
      let tools = [];
      if (toolRecovery) {
        if (content) { const r = toolRecovery.process(content); content = r.content; tools = r.toolCallDeltas; }
        content += toolRecovery.flush();
      }
      st.visible += content.length; st.reasoned += reasoning.length;

      const r = render(reasoning, content);
      let tail = r.content || '';
      if (st.reasoningOpen) { tail += '\n</thinking>\n\n'; st.reasoningOpen = false; }
      const out = {};
      if (tail) out.content = tail;
      if (r.reasoning) { out.reasoning = r.reasoning; out.reasoning_content = r.reasoning; }
      if (tools.length) { out.tool_calls = tools; st.recoveredTool = true; }
      if (Object.keys(out).length) emitDelta(out);

      emitDelta({}, st.recoveredTool ? 'tool_calls' : (st.finishReason || 'stop'));
      console.log(`[REASONING] ${usedModel} ${st.reasoned ? `thought ~${Math.round(st.reasoned / 4)} tokens` : 'no reasoning output'}`);
      console.log(`[DONE] ${usedModel} stream total ${Date.now() - reqStart}ms | first text at ${st.firstVisibleAt ?? 'n/a'}ms | ${st.visible} chars`);
      safeWrite(res, 'data: [DONE]\n\n');
      if (!res.writableEnded) { try { res.end(); } catch {} }
    };

    const armIdle = () => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        if (finished) return;
        finished = true;
        console.warn(`[STREAM] ${usedModel} went silent for ${STREAM_IDLE_TIMEOUT_MS / 1000}s, aborting`);
        if (upstream && !upstream.destroyed) upstream.destroy();
        finishStreamWithNotice(res, `\n\n[Proxy: model stopped responding for ${Math.round(STREAM_IDLE_TIMEOUT_MS / 1000)}s]`);
      }, STREAM_IDLE_TIMEOUT_MS);
    };

    const onChunk = (chunk) => {
      if (finished) return;
      armIdle();
      buffer += decoder.write(chunk);
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) processLine(line);
    };

    upstream.on('end', finalize);
    upstream.on('close', finalize);
    upstream.on('error', (err) => {
      if (finished) return;
      finished = true;
      finishStreamWithNotice(res, `\n\n[Stream Error: ${err.message}]`);
    });

    // Process the chunk we already consumed to detect "model is alive", then continue streaming.
    onChunk(opened.firstChunk);
    upstream.on('data', onChunk);
    upstream.resume();

  } catch (err) {
    teardown();
    if (abortController.signal.aborted) return;

    const msg = getErrorMessage(err);
    console.error(`[PROXY] Request failed after ${Date.now() - reqStart}ms: ${msg}`);

    // A 401 here means OUR NIM key was rejected; don't make the client think its own key is wrong.
    let status = err.response?.status || (err.timedOut ? 504 : 500);
    if (status === 401) status = 502;

    if (!res.headersSent) {
      res.status(status).json({ error: { message: msg, type: 'invalid_request_error', code: status } });
    } else {
      finished = true;
      finishStreamWithNotice(res, `\n\n[Proxy Error: ${msg}]`);
    }
  }
});

// JSON parse errors etc.
app.use((err, req, res, next) => {
  if (res.headersSent) return next(err);
  const status = err.status || err.statusCode || 400;
  res.status(status).json({ error: { message: err.message || 'Bad request', type: 'invalid_request_error', code: status } });
});

const server = app.listen(PORT, () => {
  console.log(`[PROXY] Server running on port ${PORT}`);
  if (!SKIP_VALIDATION) {
    nim.get('/models', { headers: { Authorization: `Bearer ${NIM_API_KEY}` }, timeout: 15000 })
      .then(r => console.log(`[VALIDATION] Connected to NIM. Catalog size: ${r.data.data?.length || 0}`))
      .catch(e => console.warn(`[VALIDATION] Failed to reach NIM: ${e.message}`));
  }
});
// Longer than most platform load balancers' idle timeout, avoids random 502s on reused connections
server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
