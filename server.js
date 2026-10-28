// server.js — Clean OpenAI-compatible proxy for NVIDIA NIM

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
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';

const MAX_TOKENS_LIMIT = 65536;
const DEFAULT_MAX_TOKENS = Number(process.env.DEFAULT_MAX_TOKENS) || 4096;
const HEARTBEAT_INTERVAL_MS = 10000;

const REQUEST_TIMEOUT_MS = 80000;
const THINKING_DEADLINE_MS = 80000;

const keepAliveAgent = new https.Agent({ keepAlive: true, maxSockets: 50 });
const nim = axios.create({ baseURL: NIM_API_BASE, httpsAgent: keepAliveAgent });

if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

function validateConfig() {
  if (!NIM_API_KEY) { console.error('[FATAL] NIM_API_KEY is required.'); process.exit(1); }
  if (!CLIENT_AUTH_KEY) console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected.');
}
validateConfig();

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

const DEFAULT_MODEL = 'meta/llama-3.1-8b-instruct';
const FALLBACK_MODELS = ['meta/llama-3.1-8b-instruct', 'z-ai/glm-5.3-flash'];

function resolveModelId(name) {
  if (typeof name !== 'string') return null;
  if (MODEL_MAPPING[name]) return MODEL_MAPPING[name];
  if (/^[\w.-]+\/[\w.:-]+$/.test(name)) return name;
  return null;
}

// ─── Middleware ───
app.use(cors());
app.use(express.json({ limit: '10mb' }));

function safeTimingEqual(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string' || !a || !b) return false;
  return timingSafeEqual(createHash('sha256').update(a).digest(), createHash('sha256').update(b).digest());
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/' || req.path === '/v1/models') return next();
  const token = req.headers.authorization?.replace('Bearer ', '').trim();
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
  if (text) {
    safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { content: text }, finish_reason: null }] })}\n\n`);
  }
  safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n`);
  safeWrite(res, 'data: [DONE]\n\n');
  if (!res.writableEnded) res.end();
}

function getErrorMessage(err) {
  if (!err) return 'Unknown error';
  const d = err.response?.data;
  let msg = typeof d === 'string' ? d : d?.error?.message || d?.message || d?.detail;
  return String(msg || err.message || 'Request failed').slice(0, 300);
}

async function postToNim(payload, signal) {
  const ac = new AbortController();
  let timedOut = false;
  const onAbort = () => ac.abort();
  if (signal) { if (signal.aborted) ac.abort(); else signal.addEventListener('abort', onAbort); }
  
  const timer = setTimeout(() => { timedOut = true; ac.abort(); }, REQUEST_TIMEOUT_MS);

  try {
    const res = await nim.post('/chat/completions', payload, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: payload.stream ? 'stream' : 'json',
      timeout: 0, signal: ac.signal
    });
    return res;
  } catch (err) {
    if (timedOut) { const e = new Error(`Timeout after ${REQUEST_TIMEOUT_MS}ms`); e.timedOut = true; throw e; }
    throw err;
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

const cooldowns = new Map();
async function callWithFallback(baseReq, models, enableThinking, reasoningEffort, hasTools, signal) {
  let lastErr = null;
  const active = models.filter(m => !cooldowns.has(m) || cooldowns.get(m) < Date.now());
  const order = active.length ? active : models;

  for (const model of order) {
    if (signal?.aborted) throw lastErr || new Error('Client aborted');

    const payload = { ...baseReq, model, ...getReasoningPayload(model, enableThinking, reasoningEffort) };
    console.log(`[NIM] Attempting ${model} | payload: ${JSON.stringify(payload.chat_template_kwargs || {})}`);

    try {
      const res = await postToNim(payload, signal);
      return { response: res, model };
    } catch (err) {
      lastErr = err;
      const status = err.response?.status;
      console.warn(`[FALLBACK] ${model} failed | Status: ${status || 'timeout'} | ${getErrorMessage(err)}`);

      if (status === 401) throw err; // Bad API key, no point continuing
      if (status === 429) cooldowns.set(model, Date.now() + 30000);
      if (status === 503 || err.timedOut) cooldowns.set(model, Date.now() + 60000);
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
  let upstreamStream = null;
  let heartbeat = null;
  let cleanup = null;
  const abortController = new AbortController();

  res.on('close', () => {
    if (heartbeat) clearInterval(heartbeat);
    if (res.writableFinished) return;
    abortController.abort();
    if (cleanup) cleanup();
    if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
  });

  try {
    const isStream = !!req.body.stream;
    if (isStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders();
      safeWrite(res, ': connected\n\n');
      heartbeat = setInterval(() => safeWrite(res, ': keep-alive\n\n'), HEARTBEAT_INTERVAL_MS);
    }

    const { model, max_tokens, temperature, reasoning_effort, tools } = req.body;
    const primary = resolveModelId(model) || DEFAULT_MODEL;
    const chain = [...new Set([primary, ...FALLBACK_MODELS])];
    
    const baseReq = {
      ...req.body,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? DEFAULT_MAX_TOKENS, MAX_TOKENS_LIMIT),
      stream: isStream
    };
    delete baseReq.model; delete baseReq.reasoning_effort;

    const hasTools = !!tools;
    const { response, model: usedModel } = await callWithFallback(
      baseReq, chain, ENABLE_THINKING_MODE, reasoning_effort, hasTools, abortController.signal
    );

    if (!isStream) {
      // ─── Non-Streaming Path ───
      const data = response.data;
      const choices = (data.choices || []).map((c, i) => {
        const norm = normalizeNonStreamChoice(c, usedModel);
        let content = norm.message?.content || '';
        let reasoning = norm.message?.reasoning || '';
        
        if (reasoning.length > 0) {
          console.log(`[REASONING] ${usedModel} thought for ~${Math.round(reasoning.length / 4)} tokens (${reasoning.length} chars)`);
        } else {
          console.log(`[REASONING] ${usedModel} finished without reasoning output`);
        }

        let recovered = [];
        if (hasTools) {
          const r = extractLeakedToolCalls(content);
          content = r.content; recovered = r.toolCalls;
        }

        const finalMsg = { ...norm.message, content };
        const allTools = [...(norm.message.tool_calls || []), ...recovered];
        
        if (allTools.length > 0) {
          finalMsg.tool_calls = allTools;
          if (recovered.length > 0) norm.finish_reason = 'tool_calls';
          if (!finalMsg.content) finalMsg.content = null;
        }

        if (!SHOW_REASONING) { delete finalMsg.reasoning; delete finalMsg.reasoning_content; }
        else if (finalMsg.reasoning) finalMsg.reasoning_content = finalMsg.reasoning;

        return { ...norm, index: i, message: finalMsg };
      });

      return res.json({ ...data, model: usedModel, choices });
    }

    // ─── Streaming Path ───
    upstreamStream = response.data;
    const decoder = new StringDecoder('utf8');
    const normalizer = new StreamNormalizer(usedModel);
    const toolRecovery = hasTools ? new ToolCallStreamRecovery() : null;
    
    let buffer = '', reasoningOpen = false, stats = { chars: 0, reason: 0 }, sawTool = false;

    const processLine = (line) => {
      if (!line.startsWith('data: ')) return;
      if (line.trim() === 'data: [DONE]') return;

      try {
        const data = JSON.parse(line.slice(6));
        const delta = data.choices?.[0]?.delta;
        if (delta && delta.tool_calls) sawTool = true;

        if (delta) {
          const norm = normalizer.processDelta(delta);
          stats.chars += (norm.content || '').length;
          if (norm.reasoning) stats.reason += norm.reasoning.length;

          let outContent = '';
          if (SHOW_REASONING && req.headers['x-reasoning-format'] === 'inline') {
            if (norm.reasoning && !reasoningOpen) { outContent += `<thinking>\n${norm.reasoning}`; reasoningOpen = true; }
            else if (norm.reasoning) outContent += norm.reasoning;

            if (norm.content && reasoningOpen) { outContent += `\n</thinking>\n\n${norm.content}`; reasoningOpen = false; }
            else if (norm.content) outContent += norm.content;
          } else {
            outContent = norm.content || '';
          }

          if (toolRecovery) {
            const r = toolRecovery.process(outContent);
            outContent = r.content;
            if (r.toolCallDeltas.length > 0) {
              sawTool = true;
              delta.tool_calls = [...(delta.tool_calls || []), ...r.toolCallDeltas];
              if (data.choices[0]) data.choices[0].finish_reason = 'tool_calls';
            }
          }

          if (outContent) delta.content = outContent; else delete delta.content;
          
          if (SHOW_REASONING && norm.reasoning) {
            delta.reasoning = norm.reasoning;
            delta.reasoning_content = norm.reasoning;
          } else {
            delete delta.reasoning;
            delete delta.reasoning_content;
          }
        }
        safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
      } catch (err) {
        // Ignore malformed chunks
      }
    };

    const onEnd = () => {
      buffer += decoder.end();
      if (buffer.trim()) processLine(buffer);

      const flushed = normalizer.flush();
      let outContent = flushed.content || '';
      const outReasoning = flushed.reasoning || '';
      let flushedTools = [];

      if (outReasoning) stats.reason += outReasoning.length;

      if (toolRecovery) {
        const r = toolRecovery.process(outContent);
        outContent = r.content; flushedTools = r.toolCallDeltas;
        const left = toolRecovery.flush();
        if (left) outContent += left;
      }

      if (flushedTools.length > 0) {
        safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: flushedTools }, finish_reason: 'tool_calls' }] })}\n\n`);
      }

      let finalContent = '';
      if (SHOW_REASONING && req.headers['x-reasoning-format'] === 'inline') {
        if (outReasoning) {
          if (!reasoningOpen) { finalContent = `<thinking>\n${outReasoning}`; reasoningOpen = true; }
          else finalContent = outReasoning;
        }
        if (outContent) {
          if (reasoningOpen) { finalContent += `\n</thinking>\n\n${outContent}`; reasoningOpen = false; }
          else finalContent += outContent;
        }
        if (reasoningOpen) finalContent += '\n</thinking>\n\n';
      } else {
        finalContent = outContent;
      }

      if (finalContent) {
        safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { content: finalContent }, finish_reason: null }] })}\n\n`);
      }

      if (SHOW_REASONING && !req.headers['x-reasoning-format'] && outReasoning) {
        safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { reasoning: outReasoning, reasoning_content: outReasoning }, finish_reason: null }] })}\n\n`);
      }

      // ─── Reasoning Stats Log ───
      if (stats.reason > 0) {
        console.log(`[REASONING] ${usedModel} thought for ~${Math.round(stats.reason / 4)} tokens (${stats.reason} chars)`);
      } else {
        console.log(`[REASONING] ${usedModel} finished without reasoning output`);
      }

      safeWrite(res, 'data: [DONE]\n\n');
      if (!res.writableEnded) res.end();
    };

    upstreamStream.on('data', chunk => {
      buffer += decoder.write(chunk);
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) processLine(line);
    });

    upstreamStream.on('end', onEnd);
    upstreamStream.on('error', (err) => {
      if (!res.writableEnded) finishStreamWithNotice(res, `\n\n[Stream Error: ${err.message}]`);
    });

    const deadline = setTimeout(() => {
      if (stats.chars > 0 || sawTool) return;
      console.log(`[REASONING] ${usedModel} reached deadline without visible output. Aborting.`);
      if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
      finishStreamWithNotice(res, `\n\n[Proxy: Thinking deadline (80s) reached. The model took too long.]`);
    }, THINKING_DEADLINE_MS);

    cleanup = () => { clearTimeout(deadline); upstreamStream.removeAllListeners(); upstreamStream.on('error', () => {}); };

  } catch (err) {
    if (heartbeat) clearInterval(heartbeat);
    if (cleanup) cleanup();
    if (abortController.signal.aborted) return;

    const msg = getErrorMessage(err);
    console.error(`[PROXY] Request failed: ${msg}`);

    if (!res.headersSent) {
      res.status(err.response?.status || (err.timedOut ? 504 : 500)).json({
        error: { message: msg, type: 'invalid_request_error', code: err.response?.status || 500 }
      });
    } else {
      finishStreamWithNotice(res, `\n\n[Proxy Error: ${msg}]`);
    }
  }
});

app.listen(PORT, () => {
  console.log(`[PROXY] Server running on port ${PORT}`);
  if (!SKIP_VALIDATION) {
    nim.get('/models', { headers: { Authorization: `Bearer ${NIM_API_KEY}` } })
      .then(r => console.log(`[VALIDATION] Successfully connected to NIM. Catalog size: ${r.data.data?.length || 0}`))
      .catch(e => console.warn(`[VALIDATION] Failed to reach NIM: ${e.message}`));
  }
});
