// server.js — OpenAI-compatible proxy for NVIDIA NIM

'use strict';

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual, createHash } = require('crypto');

const { config, log } = require('./config');
const { MODEL_MAPPING, DEFAULT_MODEL, isBlockedModel, resolveModelId } = require('./models');
const { getReasoningPayload, StreamNormalizer, normalizeNonStreamChoice } = require('./reasoning');
const { extractLeakedToolCalls, ToolCallStreamRecovery } = require('./tools');

if (!config.nimApiKey) {
  log.error('[FATAL] NIM_API_KEY is required.');
  process.exit(1);
}
if (!config.clientAuthKey) log.warn('[WARN] CLIENT_AUTH_KEY not set. All protected routes will be rejected.');
if (config.enableThinking) log.info('[CONFIG] Thinking mode: ENABLED');
if (config.showReasoning) log.info('[CONFIG] Reasoning display: ENABLED');

const nim = axios.create({
  baseURL: config.nimApiBase,
  httpsAgent: new https.Agent({ keepAlive: true, maxSockets: 50 }),
  headers: { Authorization: `Bearer ${config.nimApiKey}`, 'Content-Type': 'application/json' },
});

// ─── Helpers ───

const sse = (data) => `data: ${JSON.stringify(data)}\n\n`;

function safeWrite(res, chunk) {
  if (res.writableEnded || res.destroyed) return;
  try { res.write(chunk); } catch { /* client went away */ }
}

/** Optionally emits a notice, then terminates an SSE stream cleanly. */
function endStream(res, notice) {
  if (notice) safeWrite(res, sse({ choices: [{ index: 0, delta: { content: notice }, finish_reason: null }] }));
  safeWrite(res, sse({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
  safeWrite(res, 'data: [DONE]\n\n');
  if (!res.writableEnded && !res.destroyed) res.end();
}

const sendError = (res, status, message) =>
  res.status(status).json({ error: { message, type: 'invalid_request_error', code: status } });

/** Tracks timers for one request so they can all be cleared together. */
function createTimers() {
  const active = new Set();
  const track = (t) => { active.add(t); return t; };
  return {
    interval: (fn, ms) => track(setInterval(fn, ms)),
    timeout: (fn, ms) => track(setTimeout(fn, ms)),
    clear() {
      for (const t of active) { clearTimeout(t); clearInterval(t); }
      active.clear();
    },
  };
}

/** With `responseType: 'stream'` axios errors carry an unread stream; turn it into text/JSON. */
async function readErrorBody(err) {
  const body = err.response?.data;
  if (!body || typeof body.on !== 'function') return;

  const chunks = [];
  let size = 0;
  try {
    for await (const chunk of body) {
      size += chunk.length;
      if (size > 65536) break;
      chunks.push(chunk);
    }
  } catch { /* stream already broken */ }

  const text = Buffer.concat(chunks).toString('utf8');
  try { err.response.data = JSON.parse(text); } catch { err.response.data = text; }
}

function getErrorMessage(err) {
  const data = err?.response?.data;
  const detail = typeof data === 'string' ? data : data?.error?.message || data?.message || data?.detail;
  const msg = String(detail || err?.message || 'Request failed').slice(0, 300);
  return err?.response?.status ? `Upstream NIM error (${err.response.status}): ${msg}` : msg;
}

async function postToNim(payload, signal) {
  const ac = new AbortController();
  let timedOut = false;

  const onAbort = () => ac.abort();
  if (signal.aborted) ac.abort(); else signal.addEventListener('abort', onAbort, { once: true });
  const timer = setTimeout(() => { timedOut = true; ac.abort(); }, config.requestTimeoutMs);

  try {
    return await nim.post('/chat/completions', payload, {
      responseType: payload.stream ? 'stream' : 'json',
      timeout: 0,
      signal: ac.signal,
    });
  } catch (err) {
    if (timedOut) throw Object.assign(new Error(`Upstream timeout after ${config.requestTimeoutMs}ms`), { timedOut: true });
    throw err;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', onAbort);
  }
}

// ─── Request / response shaping ───

function buildPayload(body, modelId, isStream, hasTools) {
  const { model: _model, reasoning_effort, max_tokens, max_completion_tokens, temperature, ...rest } = body;
  const requested = Number(max_tokens ?? max_completion_tokens);

  const payload = {
    ...rest,
    model: modelId,
    temperature: temperature ?? 0.7,
    max_tokens: Math.min(requested > 0 ? requested : config.defaultMaxTokens, config.maxTokensLimit),
    stream: isStream,
    ...getReasoningPayload(modelId, config.enableThinking, reasoning_effort),
  };

  if (!hasTools) {
    delete payload.tools;
    delete payload.tool_choice;
    if (!config.keepStreamUsage) delete payload.stream_options;
  }
  if (!isStream) delete payload.stream_options;

  return payload;
}

function buildNonStreamChoices(data, modelId, hasTools) {
  return (data.choices || []).map((choice, index) => {
    const norm = normalizeNonStreamChoice(choice, modelId);
    const message = { ...norm.message };
    let finishReason = norm.finish_reason;

    log.debug(`[REASONING] ${modelId} reasoning chars: ${message.reasoning?.length || 0}`);

    if (hasTools) {
      const recovered = extractLeakedToolCalls(message.content || '');
      message.content = recovered.content || null;
      if (recovered.toolCalls.length) {
        message.tool_calls = [...(message.tool_calls || []), ...recovered.toolCalls];
        finishReason = 'tool_calls';
      }
    }

    if (config.showReasoning) {
      if (message.reasoning) message.reasoning_content = message.reasoning;
    } else {
      delete message.reasoning;
      delete message.reasoning_content;
    }

    return { ...norm, index, finish_reason: finishReason, message };
  });
}

/**
 * Pipes an upstream SSE stream to the client, normalizing reasoning,
 * recovering leaked tool calls and guaranteeing a clean termination.
 */
function pipeStream({ res, upstream, modelId, hasTools, inline, startTime, timers, signal }) {
  const decoder = new StringDecoder('utf8');
  const normalizer = new StreamNormalizer(modelId);
  const recovery = hasTools ? new ToolCallStreamRecovery() : null;
  const stats = { content: 0, reasoning: 0 };
  const meta = {};

  let buffer = '';
  let reasoningOpen = false;
  let sawTool = false;
  let recovered = false;
  let finishSent = false;
  let done = false;

  const setField = (obj, key, value) => { if (value) obj[key] = value; else delete obj[key]; };

  // Inline mode: reasoning is streamed inside <thinking> tags in the content.
  const wrapInline = (reasoning, content) => {
    let out = '';
    if (reasoning) {
      out += reasoningOpen ? reasoning : `<thinking>\n${reasoning}`;
      reasoningOpen = true;
    }
    if (content) {
      out += reasoningOpen ? `\n</thinking>\n\n${content}` : content;
      reasoningOpen = false;
    }
    return out;
  };

  const sendChunk = (delta, finishReason = null) =>
    safeWrite(res, sse({
      ...meta,
      object: 'chat.completion.chunk',
      model: modelId,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    }));

  /** Flushes anything the parsers are still holding back. */
  const drain = () => {
    const flushed = normalizer.flush();
    let content = flushed.content || '';
    const reasoning = flushed.reasoning || '';
    let tools = [];
    stats.reasoning += reasoning.length;

    if (recovery) {
      const r = recovery.process(content);
      tools = r.toolCallDeltas;
      content = r.content + recovery.flush();
      if (tools.length) sawTool = recovered = true;
    }

    let text = content;
    if (inline) {
      text = wrapInline(reasoning, content);
      if (reasoningOpen) { text += '\n</thinking>\n\n'; reasoningOpen = false; }
    }
    return { text, reasoning: inline ? '' : reasoning, tools };
  };

  const mergeTail = (delta, tail) => {
    if (tail.text) delta.content = (delta.content || '') + tail.text;
    if (tail.tools.length) delta.tool_calls = [...(delta.tool_calls || []), ...tail.tools];
    if (config.showReasoning && tail.reasoning) {
      delta.reasoning = (delta.reasoning || '') + tail.reasoning;
      delta.reasoning_content = delta.reasoning;
    }
  };

  const handleChunk = (data) => {
    if (data.id) meta.id = data.id;
    if (data.created) meta.created = data.created;

    const choice = data.choices?.[0];
    const delta = choice?.delta;
    if (!delta) { safeWrite(res, sse(data)); return; } // e.g. usage-only chunk

    if (delta.tool_calls?.length) {
      sawTool = true;
      recovery?.reserve(Math.max(...delta.tool_calls.map((t) => (t.index ?? 0) + 1)));
    }

    const norm = normalizer.processDelta(delta);
    let content = norm.content || '';
    const reasoning = norm.reasoning || '';
    stats.content += content.length;
    stats.reasoning += reasoning.length;

    if (recovery) {
      const r = recovery.process(content);
      content = r.content;
      if (r.toolCallDeltas.length) {
        delta.tool_calls = [...(delta.tool_calls || []), ...r.toolCallDeltas];
        sawTool = recovered = true;
      }
    }

    setField(delta, 'content', inline ? wrapInline(reasoning, content) : content);
    delete delta.reasoning;
    delete delta.reasoning_content;
    if (config.showReasoning && !inline && reasoning) {
      delta.reasoning = reasoning;
      delta.reasoning_content = reasoning;
    }

    // The finish chunk is the last point at which buffered text can still be delivered in order.
    if (choice.finish_reason) {
      mergeTail(delta, drain());
      finishSent = true;
      if (recovered && choice.finish_reason === 'stop') choice.finish_reason = 'tool_calls';
    }

    if (!choice.finish_reason && !data.usage && Object.keys(delta).length === 0) return;
    safeWrite(res, sse(data));
  };

  const processLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return;
    const payload = trimmed.slice(5).trim();
    if (payload === '[DONE]') return;

    let data;
    try { data = JSON.parse(payload); } catch (err) {
      log.debug(`Skipping malformed SSE chunk: ${err.message}`);
      return;
    }
    handleChunk(data);
  };

  const complete = () => {
    if (done) return;
    done = true;
    timers.clear();

    buffer += decoder.end();
    if (buffer.trim()) processLine(buffer);

    if (!finishSent) { // upstream ended without a finish_reason
      const delta = {};
      mergeTail(delta, drain());
      sendChunk(delta, recovered ? 'tool_calls' : 'stop');
    }

    log.info(`[DONE] ${modelId} stream finished in ${((Date.now() - startTime) / 1000).toFixed(1)}s`
      + (stats.reasoning ? ` (~${Math.round(stats.reasoning / 4)} reasoning tokens)` : ''));

    safeWrite(res, 'data: [DONE]\n\n');
    if (!res.writableEnded) res.end();
  };

  const abort = (notice) => {
    if (done) return;
    done = true;
    timers.clear();
    endStream(res, notice);
  };

  upstream.on('data', (chunk) => {
    buffer += decoder.write(chunk);
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) processLine(line);
  });
  upstream.on('end', complete);
  upstream.on('error', (err) => {
    if (done || signal.aborted) return;
    log.warn(`[STREAM] Upstream error: ${err.message}`);
    abort(`\n\n[Stream Error: ${err.message}]`);
  });
  upstream.on('close', () => {
    if (done || signal.aborted) return;
    abort('\n\n[Proxy: upstream closed unexpectedly]');
  });

  if (config.debug) {
    timers.interval(() => {
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      log.debug(`[PROGRESS] ${modelId} ${elapsed}s | reasoning ~${Math.round(stats.reasoning / 4)} | content ~${Math.round(stats.content / 4)} tokens`);
    }, 5000);
  }

  timers.timeout(() => {
    if (stats.content > 0 || stats.reasoning > 0 || sawTool) return;
    log.warn(`[SPEED] ${modelId} produced no output within ${config.thinkingDeadlineMs / 1000}s. Aborting.`);
    abort(`\n\n[Proxy: no output after ${config.thinkingDeadlineMs / 1000}s. The model is too slow.]`);
    upstream.destroy();
  }, config.thinkingDeadlineMs);
}

// ─── App ───

const app = express();
app.disable('x-powered-by');

// Auth runs before body parsing so unauthenticated requests never cost a 10 MB parse.
const PUBLIC_PATHS = new Set(['/', '/health', '/v1/models']);
const digest = (s) => createHash('sha256').update(s).digest();
const clientKeyDigest = config.clientAuthKey ? digest(config.clientAuthKey) : null;

function authenticate(req, res, next) {
  if (PUBLIC_PATHS.has(req.path)) return next();
  const header = req.headers.authorization || '';
  const token = header.replace(/^Bearer\s+/i, '').trim();
  if (token && clientKeyDigest && timingSafeEqual(digest(token), clientKeyDigest)) return next();

  // Log why (never the key itself) so a 403 is diagnosable from the server logs.
  const reason = !clientKeyDigest ? 'CLIENT_AUTH_KEY is not set on the server'
    : !header ? 'no Authorization header'
      : `key mismatch (received ${token.length} chars, expected ${config.clientAuthKey.length})`;
  log.warn(`[AUTH] Rejected ${req.method} ${req.path}: ${reason}`);

  return res.status(403).json({
    error: { message: 'Forbidden: Invalid or missing authentication', type: 'authentication_error', code: 403 },
  });
}

app.use(cors());
app.use(authenticate);
app.use(express.json({ limit: '10mb' }));

app.get('/', (req, res) => res.send('Proxy is running.'));
app.get('/health', (req, res) => res.json({ status: 'ok' }));
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map((id) => ({ id, object: 'model', owned_by: 'nim-proxy' })),
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  const body = req.body || {};

  if (isBlockedModel(body.model)) return sendError(res, 400, 'Model not available');
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return sendError(res, 400, '`messages` must be a non-empty array');
  }

  const startTime = Date.now();
  const isStream = Boolean(body.stream);
  const modelId = resolveModelId(body.model) || DEFAULT_MODEL;
  const hasTools = Array.isArray(body.tools) && body.tools.length > 0;

  const timers = createTimers();
  const abort = new AbortController();
  let upstream = null;

  res.on('close', () => {
    timers.clear();
    if (res.writableFinished) return;
    abort.abort();
    upstream?.destroy();
  });

  try {
    if (isStream) {
      // Headers go out immediately so heartbeats can keep slow "thinking" requests alive.
      res.set({
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no',
      });
      res.flushHeaders();
      safeWrite(res, ': connected\n\n');
      timers.interval(() => {
        safeWrite(res, ': keep-alive\n\n');
        log.debug(`[HEARTBEAT] ${Math.round((Date.now() - startTime) / 1000)}s elapsed`);
      }, config.heartbeatMs);
    }

    const payload = buildPayload(body, modelId, isStream, hasTools);
    log.info(`[NIM] → ${modelId}${isStream ? ' (stream)' : ''}`);
    log.debug('Reasoning fields:', JSON.stringify({
      chat_template_kwargs: payload.chat_template_kwargs,
      reasoning_effort: payload.reasoning_effort,
    }));

    const response = await postToNim(payload, abort.signal);
    if (abort.signal.aborted) { response.data?.destroy?.(); return; }

    if (!isStream) {
      res.setHeader('X-Response-Time', `${Date.now() - startTime}ms`);
      return res.json({
        ...response.data,
        model: modelId,
        choices: buildNonStreamChoices(response.data, modelId, hasTools),
      });
    }

    upstream = response.data;
    pipeStream({
      res,
      upstream,
      modelId,
      hasTools,
      inline: config.showReasoning && req.headers['x-reasoning-format'] === 'inline',
      startTime,
      timers,
      signal: abort.signal,
    });
  } catch (err) {
    timers.clear();
    if (abort.signal.aborted) return;

    await readErrorBody(err);
    const msg = getErrorMessage(err);
    log.error(`[PROXY] Request failed: ${msg}`);

    if (res.headersSent) return endStream(res, `\n\n[Proxy Error: ${msg}]`);

    const status = err.response?.status || (err.timedOut ? 504 : 500);
    return res.status(status).json({ error: { message: msg, type: 'invalid_request_error', code: status } });
  }
});

app.use((req, res) => sendError(res, 404, `Not found: ${req.method} ${req.path}`));

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  if (res.headersSent) return;
  const status = err.status || err.statusCode || 500;
  sendError(res, status, status >= 500 ? 'Internal server error' : err.message);
});

const server = app.listen(config.port, () => {
  log.info(`[PROXY] Server running on port ${config.port}`);
  if (config.skipValidation) return;

  nim.get('/models')
    .then((r) => log.info(`[VALIDATION] Connected to NIM. Catalog size: ${r.data.data?.length || 0}`))
    .catch((e) => log.warn(`[VALIDATION] Failed to reach NIM: ${e.message}`));
});

for (const signal of ['SIGTERM', 'SIGINT']) {
  process.on(signal, () => {
    log.info(`[PROXY] ${signal} received, shutting down.`);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 5000).unref();
  });
}
