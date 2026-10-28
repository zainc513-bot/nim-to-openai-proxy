// server.js
console.log("🚀 [BOOT] STARTING SERVER...");

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual, createHash } = require('crypto');
const { SHOW_REASONING, getReasoningPayload, resolveEffectiveThinking, StreamNormalizer, normalizeNonStreamChoice } = require('./reasoning');
const { extractLeakedToolCalls, ToolCallStreamRecovery } = require('./tools');

const app = express();
const PORT = process.env.PORT || 3000;

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DEBUG_MODE = process.env.DEBUG_MODE === 'true';

const MAX_TOKENS_LIMIT = 65536;
const DEFAULT_MAX_TOKENS = Number(process.env.DEFAULT_MAX_TOKENS) || 4096;
const HEARTBEAT_INTERVAL_MS = Number(process.env.HEARTBEAT_INTERVAL_MS) || 10000;

const keepAliveAgent = new https.Agent({ keepAlive: true, maxSockets: 50, maxFreeSockets: 10 });
const nim = axios.create({ baseURL: NIM_API_BASE, httpsAgent: keepAliveAgent });

// INCREASED TO 2 MINUTES (120000ms) FOR LARGE ROLEPLAY PROMPTS
const REQUEST_TIMEOUT_MS = 120000;
const STREAM_IDLE_TIMEOUT_MS = 120000;
const THINKING_DEADLINE_MS = 120000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024;

if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');
if (DEBUG_MODE) console.log('[CONFIG] Debug mode: ENABLED');

function validateConfig() {
  if (!NIM_API_KEY) console.error('❌ [FATAL ERROR] NIM_API_KEY is missing from Render environment variables!');
  if (!CLIENT_AUTH_KEY) console.warn('⚠️ [WARN] CLIENT_AUTH_KEY is missing from Render environment variables.');
}
validateConfig();

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama3-chatqa-1.5-70b',
  'gpt-4': 'minimaxai/minimax-m3.1-flash',
  'gpt-3.5': 'nvidia/llama3-chatqa-1.5-70b',
  'gpt-4-turbo': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'claude-3-opus': 'google/diffusiongemma-26b-a4b-it',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.1-nemotron-70b-instruct', 
  'gemini-turbo': 'nvidia/llama3-chatqa-1.5-70b',
  'gpt-3.5o': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'gpt-4-flash': 'z-ai/glm-5.3-flash',
  'deepseek': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4o': 'stepfun-ai/step-5-preview',
  'mistral': 'nvidia/llama3-chatqa-1.5-70b', 
  'mistral-turbo': 'nv-mistralai/mistral-nemo-12b-instruct',
  'mistral-pro': 'nvidia/llama3-chatqa-1.5-70b', 
  'mistral-fast': 'nvidia/mistral-nemo-minitron-8b-8k-instruct', 
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'meta/muse-glimmer-30b',
  'google-lighter': 'poolside/laguna-xs-2.1',
  'm3.1': 'minimaxai/minimax-m3.1-flash',
  'glm-flash': 'z-ai/glm-5.3-flash',
  'llama-vision': 'meta/llama-3.2-11b-vision-instruct',
  'vision': 'meta/llama-3.2-11b-vision-instruct',
  'llama-vision-11b': 'meta/llama-3.2-11b-vision-instruct',
  'llama-3.2-vision': 'meta/llama-3.2-11b-vision-instruct'
};

const DEFAULT_MODEL = 'minimaxai/minimax-m3.1-flash';

const FALLBACK_MODELS = [
  'minimaxai/minimax-m3.1-flash',
  'nvidia/llama3-chatqa-1.5-70b',
  'openai/gpt-oss-20b'
];

function resolveModelId(name) {
  if (typeof name !== 'string') return null;
  if (MODEL_MAPPING[name]) return MODEL_MAPPING[name];
  if (/^[\w.-]+\/[\w.:-]+$/.test(name)) return name;
  return null;
}

app.use(cors());

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const trimmed = authHeader.trim();
  if (!trimmed.startsWith('Bearer ')) return null;
  const token = trimmed.slice('Bearer '.length).trim();
  return token || null;
}

function safeTimingEqual(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string' || !a || !b) return false;
  const ha = createHash('sha256').update(a).digest();
  const hb = createHash('sha256').update(b).digest();
  return timingSafeEqual(ha, hb);
}

app.use((req, res, next) => {
  const isPublicModelList = req.path === '/v1/models' && req.query.live !== 'true';
  if (req.path === '/health' || req.path === '/' || req.path === '/test' || isPublicModelList) {
    return next();
  }
  const token = extractBearerToken(req.headers.authorization);
  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({ error: { message: 'Forbidden: Missing auth', type: 'authentication_error', code: 403 } });
  }
  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({ error: { message: 'Forbidden: Invalid credentials', type: 'authentication_error', code: 403 } });
  }
  next();
});

app.use(express.json({ limit: '10mb' }));

app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: { message: 'Invalid JSON in request body', type: 'invalid_request_error', code: 400 } });
  }
  next(err);
});

function upstreamErrorText(err) {
  if (!err) return 'Unknown error';
  if (typeof err === 'string') return err;
  const d = err.response?.data;
  let text = '';
  if (d && typeof d === 'object') {
    text = (typeof d.error === 'string' ? d.error : d.error?.message) || d.detail || d.message || d.title || '';
  } else if (typeof d === 'string') {
    text = d;
  }
  return String(text || err.message || 'Unknown error').slice(0, 300);
}

async function fetchLiveModelIds() {
  const response = await nim.get('/models', {
    headers: { Authorization: `Bearer ${NIM_API_KEY}` },
    timeout: VALIDATION_TIMEOUT_MS
  });
  return new Set((response.data.data || []).map(m => m.id));
}

async function validateModels() {
  if (SKIP_VALIDATION) return;
  console.log('[VALIDATION] Checking model availability...');
  try {
    const available = await fetchLiveModelIds();
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (available.has(nimId)) console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      else console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
    }
  } catch (err) {
    console.warn(`[VALIDATION] Skipped: ${err.message}`);
  }
}

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {}
  return false;
}

function writeNoticeChunk(res, text) {
  safeWrite(res, `data: ${JSON.stringify({
    id: 'chatcmpl-notice', object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000),
    model: 'nim-proxy', choices: [{ index: 0, delta: { content: text }, finish_reason: null }]
  })}\n\n`);
}

function finishStreamWithNotice(res, text) {
  if (text) writeNoticeChunk(res, text);
  safeWrite(res, `data: ${JSON.stringify({
    id: 'chatcmpl-notice', object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000),
    model: 'nim-proxy', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
  })}\n\n`);
  safeWrite(res, 'data: [DONE]\n\n');
  if (!res.writableEnded) res.end();
}

async function postToNim(fullRequest, timeoutMs, clientSignal) {
  const ac = new AbortController();
  let timedOut = false;
  const onClientAbort = () => ac.abort();
  
  if (clientSignal) {
    if (clientSignal.aborted) ac.abort();
    else clientSignal.addEventListener('abort', onClientAbort, { once: true });
  }
  
  const timer = setTimeout(() => { timedOut = true; ac.abort(); }, timeoutMs);

  try {
    return await nim.post('/chat/completions', fullRequest, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: fullRequest.stream ? 'stream' : 'json',
      timeout: 0,
      signal: ac.signal
    });
  } catch (err) {
    if (timedOut) {
      const e = new Error(`timeout of ${timeoutMs}ms exceeded`);
      e.timedOut = true;
      throw e;
    }
    throw err;
  } finally {
    clearTimeout(timer);
    if (clientSignal) clientSignal.removeEventListener('abort', onClientAbort);
  }
}

const modelCooldowns = new Map();
const RATE_LIMIT_COOLDOWN_MS = 30000;
const ACCESS_DENIED_COOLDOWN_MS = 300000;
const UPSTREAM_DOWN_COOLDOWN_MS = 60000;

function isInCooldown(model) {
  const until = modelCooldowns.get(model);
  return typeof until === 'number' && Date.now() < until;
}

function setCooldown(model, ms) {
  modelCooldowns.set(model, Date.now() + ms);
}

async function callWithFallback(baseRequest, models, enableThinking, clientReasoningEffort, hasTools, signal) {
  let lastError = null;
  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    if (signal && signal.aborted) throw lastError || new Error('Client disconnected');

    const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools);
    const fullRequest = { ...baseRequest, model, ...reasoningPayload };

    if (DEBUG_MODE) {
      console.log(`[DEBUG] Attempting ${model} with reasoning payload: ${JSON.stringify(reasoningPayload)}`);
    }

    try {
      const res = await postToNim(fullRequest, REQUEST_TIMEOUT_MS, signal);
      return { response: res, model };
    } catch (err) {
      lastError = err;
      if (signal && signal.aborted) throw err;

      const status = err.response?.status;
      console.warn(`[FALLBACK] Model failed: ${model} | Status: ${status || 'timeout'} | Reason: ${upstreamErrorText(err)}`);

      if (status === 401) throw err;
      if (status === 403) setCooldown(model, ACCESS_DENIED_COOLDOWN_MS);
      if (status === 429) setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
      if (status === 502 || status === 503 || status === 504 || err.timedOut) setCooldown(model, UPSTREAM_DOWN_COOLDOWN_MS);
    }
  }
  throw lastError || new Error('All models failed');
}

app.get('/', (req, res) => res.send("Proxy is up."));
app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let upstreamStream = null;
  let heartbeat = null;
  let activeCleanup = null;
  const abortController = new AbortController();

  res.on('close', () => {
    if (heartbeat) clearInterval(heartbeat);
    if (res.writableFinished) return;
    abortController.abort();
    if (activeCleanup) activeCleanup();
    if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
  });

  try {
    if (req.body && req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders();
      safeWrite(res, ': connected\n\n');
      heartbeat = setInterval(() => safeWrite(res, ': keep-alive\n\n'), HEARTBEAT_INTERVAL_MS);
    }

    const { model, max_tokens, temperature, stream, reasoning_effort } = req.body;

    let primaryModel = resolveModelId(model) || DEFAULT_MODEL;
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];
    
    const { model: _droppedModel, reasoning_effort: _droppedReasoning, ...forwardedFields } = req.body;

    const baseRequest = {
      ...forwardedFields,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? DEFAULT_MAX_TOKENS, MAX_TOKENS_LIMIT),
      stream: stream || false
    };

    const hasTools = !!req.body.tools;
    const { response, model: firstModel } = await callWithFallback(
      baseRequest, modelChain, ENABLE_THINKING_MODE, reasoning_effort, hasTools, abortController.signal
    );

    const inlineReasoning = req.headers['x-reasoning-format'] === 'inline';

    const pipeStream = (response, usedModel, allowDowngrade) => {
      const up = response.data;
      upstreamStream = up;
      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let cleanedUp = false;
      const normalizer = new StreamNormalizer(usedModel);
      const toolRecovery = hasTools ? new ToolCallStreamRecovery() : null;
      const stats = { contentChars: 0 };
      let sawToolCall = false;
      let lastChunkAt = Date.now();

      const progress = setInterval(() => {
        if (Date.now() - lastChunkAt > STREAM_IDLE_TIMEOUT_MS) {
          cleanup();
          if (up && !up.destroyed) up.destroy();
          finishStreamWithNotice(res, `\n\n[Proxy notice: stream idle timeout reached.]`);
        }
      }, 10000);

      let deadlineTimer = null;
      const cleanup = (keepHeartbeat) => {
        if (cleanedUp) return;
        cleanedUp = true;
        clearInterval(progress);
        if (deadlineTimer) clearTimeout(deadlineTimer);
        if (heartbeat && !keepHeartbeat) clearInterval(heartbeat);
        up.removeAllListeners();
        up.on('error', () => {});
      };
      activeCleanup = cleanup;

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;
        if (line.trim() === 'data: [DONE]') return;

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;
          if (delta && delta.tool_calls) sawToolCall = true;

          if (delta) {
            const normalizedDelta = normalizer.processDelta(delta);
            stats.contentChars += (normalizedDelta.content || '').length;
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

            if (toolRecovery) {
              const recovered = toolRecovery.process(clientContent);
              clientContent = recovered.content;
              if (recovered.toolCallDeltas.length > 0) {
                sawToolCall = true;
                delta.tool_calls = [...(delta.tool_calls || []), ...recovered.toolCallDeltas];
                if (data.choices[0]) data.choices[0].finish_reason = 'tool_calls';
              }
            }

            if (clientContent) delta.content = clientContent;
            else delete delta.content;

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
          safeWrite(res, `data: ${JSON.stringify({ error: { message: 'Upstream sent malformed chunk', type: 'stream_parse_error' } })}\n\n`);
        }
      };

      up.on('data', chunk => {
        lastChunkAt = Date.now();
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          finishStreamWithNotice(res, '\n\n[Proxy error: buffer overflow.]');
          up.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) processLine(line);
      });

      up.on('end', () => {
        buffer += decoder.end();
        if (buffer.trim()) for (const line of buffer.split('\n')) processLine(line);

        const flushedDelta = normalizer.flush();
        let flushedContent = flushedDelta.content || '';
        const flushedReasoning = flushedDelta.reasoning || '';
        let flushedToolCalls = [];

        if (toolRecovery) {
          const recovered = toolRecovery.process(flushedContent);
          flushedContent = recovered.content;
          flushedToolCalls = recovered.toolCallDeltas;
          const leftover = toolRecovery.flush();
          if (leftover) flushedContent += leftover;
        }

        if (flushedToolCalls.length > 0) {
          sawToolCall = true;
          safeWrite(res, `data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: flushedToolCalls }, finish_reason: 'tool_calls' }] })}\n\n`);
        }

        let clientContent = '';
        if (SHOW_REASONING && inlineReasoning) {
          if (flushedReasoning) {
            if (!reasoningOpen) { clientContent = `<thinking>\n${flushedReasoning}`; reasoningOpen = true; }
            else clientContent = flushedReasoning;
          }
          if (flushedContent) {
            if (reasoningOpen) { clientContent += `\n</thinking>\n\n${flushedContent}`; reasoningOpen = false; }
            else clientContent += flushedContent;
          }
          if (reasoningOpen) { clientContent += '\n</thinking>\n\n'; reasoningOpen = false; }
        } else {
          clientContent = flushedContent;
        }

        if (clientContent) {
          safeWrite(res, `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: clientContent }, finish_reason: null }] })}\n\n`);
        }

        if (SHOW_REASONING && !inlineReasoning && flushedReasoning) {
          safeWrite(res, `data: ${JSON.stringify({ choices: [{ index: 0, delta: { reasoning: flushedReasoning, reasoning_content: flushedReasoning }, finish_reason: null }] })}\n\n`);
        }

        safeWrite(res, 'data: [DONE]\n\n');
        if (!res.writableEnded) res.end();
        cleanup();
      });

      up.on('error', err => {
        if (!res.writableEnded) finishStreamWithNotice(res, `\n\n[Proxy error: connection dropped (${err.message}).]`);
        cleanup();
      });

      const canDowngrade = allowDowngrade &&
        JSON.stringify(getReasoningPayload(usedModel, ENABLE_THINKING_MODE, reasoning_effort, hasTools)) !==
        JSON.stringify(getReasoningPayload(usedModel, ENABLE_THINKING_MODE, 'off', hasTools));

      const onDeadline = async () => {
        if (cleanedUp || stats.contentChars > 0 || sawToolCall || res.writableEnded) return;

        if (SHOW_REASONING && inlineReasoning && reasoningOpen) {
          safeWrite(res, `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: '\n</thinking>\n\n' }, finish_reason: null }] })}\n\n`);
          reasoningOpen = false;
        }

        cleanup(true);
        up.destroy();
        try {
          const retryChain = [usedModel, ...modelChain.filter(m => m !== usedModel)];
          const next = await callWithFallback(baseRequest, retryChain, ENABLE_THINKING_MODE, 'off', hasTools, abortController.signal);
          if (abortController.signal.aborted || res.writableEnded) {
            if (next.response.data && typeof next.response.data.destroy === 'function') next.response.data.destroy();
            return;
          }
          pipeStream(next.response, next.model, false);
        } catch (err) {
          if (abortController.signal.aborted) return;
          if (heartbeat) clearInterval(heartbeat);
          finishStreamWithNotice(res, `[Proxy error: thinking deadline reached and retry failed.]`);
        }
      };

      if (canDowngrade) {
        deadlineTimer = setTimeout(onDeadline, THINKING_DEADLINE_MS);
      }
    };

    if (stream) {
      pipeStream(response, firstModel, true);
    } else {
      const usedModel = firstModel;
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`, object: 'chat.completion', model: usedModel, created: Math.floor(Date.now() / 1000),
        choices: (response.data.choices || []).map((choice, i) => {
          const normalizedChoice = normalizeNonStreamChoice(choice, usedModel);
          const originalToolCalls = normalizedChoice.message?.tool_calls || [];
          let content = normalizedChoice.message?.content || '';

          let recoveredToolCalls = [];
          if (hasTools) {
            const result = extractLeakedToolCalls(content);
            content = result.content;
            recoveredToolCalls = result.toolCalls;
          }

          const finalMessage = { ...normalizedChoice.message, content };
          const allToolCalls = [...originalToolCalls, ...recoveredToolCalls];

          if (allToolCalls.length > 0) {
            finalMessage.tool_calls = allToolCalls;
            if (recoveredToolCalls.length > 0) normalizedChoice.finish_reason = 'tool_calls';
            if (!finalMessage.content) finalMessage.content = null;
          }

          if (!SHOW_REASONING) {
            delete finalMessage.reasoning;
            delete finalMessage.reasoning_content;
          } else if (finalMessage.reasoning) {
            finalMessage.reasoning_content = finalMessage.reasoning;
          }

          return { ...normalizedChoice, index: i, message: finalMessage };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    if (heartbeat) clearInterval(heartbeat);
    if (activeCleanup) activeCleanup();
    if (abortController.signal.aborted) return;

    const reason = upstreamErrorText(error);
    console.error('[PROXY] Request failed:', reason);

    if (!res.headersSent) {
      const status = error.response?.status || (error.timedOut ? 504 : 500);
      res.status(status).json({ error: { message: reason, type: 'invalid_request_error', code: status } });
    } else if (!res.writableEnded) {
      finishStreamWithNotice(res, `[Proxy error: ${reason}]`);
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

app.listen(PORT, () => {
  console.log(`[PROXY] Server running on port ${PORT}`);
  validateModels().catch(err => console.error('[VALIDATION] Startup check failed:', err.message));
});
