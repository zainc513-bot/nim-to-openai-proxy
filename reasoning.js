// reasoning.js — NIM-specific reasoning payload translation

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');

// Model-specific valid effort values based on NIM docs
const REASONING_EFFORT_ENUMS = {
  'openai/gpt-oss-120b': ['low', 'medium', 'high'],
  'openai/gpt-oss-20b': ['low', 'medium', 'high'],
  'deepseek-ai/deepseek-v4.1-flash': ['low', 'high', 'max'],
  'deepseek-ai/deepseek-v4-flash-0731': ['low', 'high', 'max'],
  'nvidia/nemotron-3-ultra-550b-a55b': ['none', 'medium', 'high'],
  'z-ai/glm-5.3': ['low', 'high', 'max'],
  'z-ai/glm-5.3-flash': ['low', 'high', 'max'],
  'minimaxai/minimax-m3': ['adaptive'],
  'moonshotai/kimi-k3': ['low', 'high', 'max'],
  'meta/muse-glimmer-30b': ['none', 'minimal', 'low', 'medium', 'high', 'max']
};

function resolveEffectiveThinking(enableThinking, clientReasoningEffort) {
  if (clientReasoningEffort === 'off' || clientReasoningEffort === 'none') return false;
  if (clientReasoningEffort === 'on') return true;
  return enableThinking;
}

function getReasoningPayload(model, enableThinking, clientReasoningEffort) {
  const thinkingOn = resolveEffectiveThinking(enableThinking, clientReasoningEffort);
  
  // Use raw effort for deepseek logic, validated effort for payloads
  const rawEffort = (clientReasoningEffort === 'off' || clientReasoningEffort === 'on') 
    ? undefined : clientReasoningEffort;
  
  let effort = rawEffort;
  const allowed = REASONING_EFFORT_ENUMS[model];
  if (allowed && effort && !allowed.includes(effort)) {
    console.warn(`[REASONING] Dropping invalid effort "${effort}" for ${model}`);
    effort = undefined;
  }

  switch (model) {
    // ─── DeepSeek (Requires chat_template_kwargs + thinking flag) ───
    case 'deepseek-ai/deepseek-v4.1-flash':
    case 'deepseek-ai/deepseek-v4-flash-0731': {
      const wantsThinking = clientReasoningEffort === 'on' || !!rawEffort || thinkingOn;
      if (!wantsThinking) return { chat_template_kwargs: { thinking: false } };
      return { 
        chat_template_kwargs: { 
          thinking: true, 
          reasoning_effort: effort || 'high' 
        } 
      };
    }

    // ─── GLM 5.3 (Defaults to max in NIM, force low to prevent timeouts) ───
    case 'z-ai/glm-5.3':
    case 'z-ai/glm-5.3-flash': {
      const level = effort || 'low'; 
      return {
        reasoning_effort: level,
        chat_template_kwargs: { reasoning_effort: level, clear_thinking: true }
      };
    }

    // ─── Nemotron 3 Ultra (Uses reasoning_budget + effort) ───
    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!thinkingOn) return { reasoning_effort: 'none' };
      return { 
        reasoning_effort: effort || 'high',
        reasoning_budget: 16384 // NIM docs default
      };
    }

    // ─── Nemotron 3 Super (Standard enable_thinking) ───
    case 'nvidia/nemotron-3-super-120b-a12b': {
      if (!thinkingOn) return {};
      return { chat_template_kwargs: { enable_thinking: true } };
    }

    // ─── Standard OpenAI-style reasoning_effort ───
    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b':
    case 'moonshotai/kimi-k3': {
      if (effort) return { reasoning_effort: effort };
      if (thinkingOn) return { reasoning_effort: 'high' };
      return {};
    }

    // ─── Models with custom thinking modes ───
    case 'minimaxai/minimax-m3': {
      return { chat_template_kwargs: { thinking_mode: thinkingOn ? 'enabled' : 'disabled' } };
    }

    default:
      if (thinkingOn) return { chat_template_kwargs: { enable_thinking: true } };
      return {};
  }
}

// ─── Stream Normalizer ───
class DelimiterParser {
  constructor(open, close) { this.open = open; this.close = close; this.in = false; this.buf = ''; }
  process(chunk) {
    this.buf += chunk;
    let content = '', reasoning = '';
    while (true) {
      const target = this.in ? this.close : this.open;
      const idx = this.buf.indexOf(target);
      if (idx !== -1) {
        const text = this.buf.substring(0, idx);
        if (this.in) reasoning += text; else content += text;
        this.in = !this.in;
        this.buf = this.buf.substring(idx + target.length);
      } else {
        let part = 0;
        const max = Math.min(this.buf.length, target.length - 1);
        for (let i = max; i > 0; i--) {
          if (target.startsWith(this.buf.substring(this.buf.length - i))) { part = i; break; }
        }
        const text = this.buf.substring(0, this.buf.length - part);
        if (this.in) reasoning += text; else content += text;
        this.buf = this.buf.substring(this.buf.length - part);
        break;
      }
    }
    return { content, reasoning };
  }
  flush() {
    let out = { content: '', reasoning: '' };
    if (this.buf) { if (this.in) out.reasoning = this.buf; else out.content = this.buf; this.buf = ''; }
    return out;
  }
}

class StreamNormalizer {
  constructor(model) {
    this.parser = model === 'minimaxai/minimax-m3' ? new DelimiterParser('<mm:think>', '</mm:think>') : null;
  }
  processDelta(delta) {
    const out = { ...delta };
    let r = out.reasoning || out.reasoning_content || '';
    let c = out.content || '';
    if (!r && c && this.parser) {
      const parsed = this.parser.process(c);
      r = parsed.reasoning; c = parsed.content;
    }
    if (c) out.content = c; else delete out.content;
    if (r) out.reasoning = r; else delete out.reasoning;
    delete out.reasoning_content;
    return out;
  }
  flush() { return this.parser ? this.parser.flush() : { content: '', reasoning: '' }; }
}

function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;
  const msg = choice.message || {};
  let r = msg.reasoning || msg.reasoning_content || '';
  let c = msg.content || '';
  if (!r && c && model === 'minimaxai/minimax-m3') {
    const p = new DelimiterParser('<mm:think>', '</mm:think>');
    const parsed = p.process(c);
    const flushed = p.flush();
    c = (parsed.content || '') + (flushed.content || '');
    r = (parsed.reasoning || '') + (flushed.reasoning || '');
  }
  const newMsg = { ...msg };
  if (c) newMsg.content = c; else newMsg.content = null;
  if (r) newMsg.reasoning = r;
  delete newMsg.reasoning_content;
  return { ...choice, message: newMsg };
}

module.exports = { SHOW_REASONING, getReasoningPayload, StreamNormalizer, normalizeNonStreamChoice };
