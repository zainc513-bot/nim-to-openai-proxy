// reasoning.js — NIM-specific reasoning payloads and reasoning-stream normalization

'use strict';

const { log } = require('./config');
const { partialLen } = require('./utils');

const MINIMAX = 'minimaxai/minimax-m3';

// Valid `reasoning_effort` values per model. Models not listed accept any value.
const EFFORT_ENUMS = {
  'openai/gpt-oss-120b': ['low', 'medium', 'high'],
  'openai/gpt-oss-20b': ['low', 'medium', 'high'],
  'nvidia/nemotron-3-ultra-550b-a55b': ['none', 'medium', 'high'],
  'minimaxai/minimax-m3': ['adaptive'],
  'moonshotai/kimi-k3': ['low', 'high', 'max'],
  'moonshotai/kimi-k3.1-pro': ['low', 'high', 'max'],
  'meta/muse-glimmer-30b': ['none', 'minimal', 'low', 'medium', 'high', 'max'],
};

// Models that take a plain top-level `reasoning_effort`.
const OPENAI_STYLE = new Set([
  'openai/gpt-oss-120b',
  'openai/gpt-oss-20b',
  'moonshotai/kimi-k3',
  'moonshotai/kimi-k3.1-pro',
  'mistralai/mistral-large-4-1t-a52b',
  'meta/muse-glimmer-30b',
]);

/** Client `reasoning_effort` → validated effort string (or undefined). */
function pickEffort(model, clientEffort) {
  if (typeof clientEffort !== 'string' || !clientEffort) return undefined;
  if (clientEffort === 'on' || clientEffort === 'off') return undefined;

  const allowed = EFFORT_ENUMS[model];
  if (allowed && !allowed.includes(clientEffort)) {
    log.warn(`[REASONING] Dropping invalid effort "${clientEffort}" for ${model}`);
    return undefined;
  }
  return clientEffort;
}

function isThinkingOn(enableThinking, clientEffort) {
  if (clientEffort === 'off' || clientEffort === 'none') return false;
  if (clientEffort === 'on') return true;
  return enableThinking;
}

/** Builds the model-specific request fields that switch reasoning on/off. */
function getReasoningPayload(model, enableThinking, clientEffort) {
  const thinkingOn = isThinkingOn(enableThinking, clientEffort);
  const effort = pickEffort(model, clientEffort);

  if (OPENAI_STYLE.has(model)) {
    if (effort) return { reasoning_effort: effort };
    return thinkingOn ? { reasoning_effort: 'high' } : {};
  }

  switch (model) {
    // GLM: send nothing (matches the NVIDIA Playground).
    case 'z-ai/glm-5.3':
    case 'z-ai/glm-5.3-flash':
      return {};

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      const on = thinkingOn || (effort && effort !== 'none');
      if (!on) return { reasoning_effort: 'none' };
      return { reasoning_effort: effort || 'high', reasoning_budget: 16384 };
    }

    case 'nvidia/nemotron-3-super-120b-a12b':
      return thinkingOn ? { chat_template_kwargs: { enable_thinking: true } } : {};

    case MINIMAX:
      return { chat_template_kwargs: { thinking_mode: thinkingOn ? 'enabled' : 'disabled' } };

    default:
      return thinkingOn ? { chat_template_kwargs: { enable_thinking: true } } : {};
  }
}

// ─── Stream normalization ───

/** Splits a text stream into content / reasoning around open/close delimiters. */
class DelimiterParser {
  constructor(open, close) {
    this.open = open;
    this.close = close;
    this.inside = false;
    this.buf = '';
  }

  process(chunk) {
    this.buf += chunk;
    let content = '';
    let reasoning = '';
    const emit = (text) => { if (this.inside) reasoning += text; else content += text; };

    for (;;) {
      const tag = this.inside ? this.close : this.open;
      const idx = this.buf.indexOf(tag);

      if (idx !== -1) {
        emit(this.buf.slice(0, idx));
        this.inside = !this.inside;
        this.buf = this.buf.slice(idx + tag.length);
        continue;
      }

      // Hold back a possible partial tag at the end of the buffer.
      const keep = partialLen(this.buf, tag);
      emit(this.buf.slice(0, this.buf.length - keep));
      this.buf = this.buf.slice(this.buf.length - keep);
      return { content, reasoning };
    }
  }

  flush() {
    const out = { content: '', reasoning: '' };
    if (this.buf) {
      if (this.inside) out.reasoning = this.buf; else out.content = this.buf;
      this.buf = '';
    }
    return out;
  }
}

const newMiniMaxParser = () => new DelimiterParser('<mm:think>', '</mm:think>');

/** Normalizes streamed deltas so reasoning always lives in `delta.reasoning`. */
class StreamNormalizer {
  constructor(model) {
    this.parser = model === MINIMAX ? newMiniMaxParser() : null;
  }

  processDelta(delta) {
    const out = { ...delta };
    let reasoning = out.reasoning || out.reasoning_content || '';
    let content = out.content || '';

    if (!reasoning && content && this.parser) {
      ({ content, reasoning } = this.parser.process(content));
    }

    if (content) out.content = content; else delete out.content;
    if (reasoning) out.reasoning = reasoning; else delete out.reasoning;
    delete out.reasoning_content;
    return out;
  }

  flush() {
    return this.parser ? this.parser.flush() : { content: '', reasoning: '' };
  }
}

/** Non-stream equivalent: moves inline reasoning into `message.reasoning`. */
function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;

  const msg = choice.message || {};
  let reasoning = msg.reasoning || msg.reasoning_content || '';
  let content = msg.content || '';

  if (!reasoning && content && model === MINIMAX) {
    const parser = newMiniMaxParser();
    const parsed = parser.process(content);
    const rest = parser.flush();
    content = parsed.content + rest.content;
    reasoning = parsed.reasoning + rest.reasoning;
  }

  const message = { ...msg, content: content || null };
  if (reasoning) message.reasoning = reasoning;
  delete message.reasoning_content;
  return { ...choice, message };
}

module.exports = { getReasoningPayload, StreamNormalizer, normalizeNonStreamChoice };
