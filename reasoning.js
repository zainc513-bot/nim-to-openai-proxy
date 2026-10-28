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

// Values some clients send that mean "no preference"
const NO_PREFERENCE = new Set(['', 'auto', 'default']);

function normalizeEffort(model, raw) {
  if (!raw || raw === 'on' || raw === 'off' || NO_PREFERENCE.has(raw)) return undefined;
  const allowed = REASONING_EFFORT_ENUMS[model];
  if (allowed && !allowed.includes(raw)) {
    console.warn(`[REASONING] Dropping invalid effort "${raw}" for ${model}`);
    return undefined;
  }
  return raw;
}

/**
 * Returns the extra request fields that turn thinking on/off for a model.
 * Thinking is ON when: client sent a valid effort or "on", or ENABLE_THINKING_MODE=true.
 * Thinking is OFF when: client sent "off"/"none", or nothing was requested and the env flag is false.
 * When it is OFF we actively tell the model so, because several NIM models think by default,
 * and that is the #1 reason replies feel slow.
 */
// Decides whether thinking is on, the effort, and whether this is a "heavy" request
// (real thinking at medium/high/max effort), which is what deserves a long wait.
function resolveThinking(model, enableThinking, clientReasoningEffort) {
  const raw = typeof clientReasoningEffort === 'string' ? clientReasoningEffort.toLowerCase().trim() : undefined;
  const effort = normalizeEffort(model, raw);

  let thinking;
  if (raw === 'off' || raw === 'none') thinking = false;
  else if (raw === 'on' || effort) thinking = true;
  else thinking = !!enableThinking;

  // GLM is forced to "low" when no effort is given; everything else defaults to high.
  const effective = effort || (model.startsWith('z-ai/glm') ? 'low' : 'high');
  const heavy = thinking && !['low', 'minimal'].includes(effective);
  return { thinking, effort, heavy };
}

function getReasoningPayload(model, enableThinking, clientReasoningEffort) {
  const { thinking, effort } = resolveThinking(model, enableThinking, clientReasoningEffort);

  switch (model) {
    // ─── DeepSeek ───
    case 'deepseek-ai/deepseek-v4.1-flash':
    case 'deepseek-ai/deepseek-v4-flash-0731': {
      if (!thinking) return { chat_template_kwargs: { thinking: false } };
      return { chat_template_kwargs: { thinking: true, reasoning_effort: effort || 'high' } };
    }

    // ─── GLM 5.3 (NIM default is max effort, which is very slow) ───
    case 'z-ai/glm-5.3':
    case 'z-ai/glm-5.3-flash': {
      if (!thinking) {
        return {
          reasoning_effort: 'low',
          chat_template_kwargs: { enable_thinking: false, reasoning_effort: 'low', clear_thinking: true }
        };
      }
      const level = effort || 'low';
      return { reasoning_effort: level, chat_template_kwargs: { reasoning_effort: level, clear_thinking: true } };
    }

    // ─── Nemotron 3 Ultra ───
    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!thinking) return { reasoning_effort: 'none' };
      return { reasoning_effort: effort || 'high', reasoning_budget: 16384 };
    }

    // ─── Nemotron 3 Super (thinks by default unless told otherwise) ───
    case 'nvidia/nemotron-3-super-120b-a12b': {
      return { chat_template_kwargs: { enable_thinking: thinking } };
    }

    // ─── gpt-oss (cannot fully disable reasoning, so use the cheapest level) ───
    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      if (effort) return { reasoning_effort: effort };
      return { reasoning_effort: thinking ? 'high' : 'low' };
    }

    case 'moonshotai/kimi-k3': {
      if (effort) return { reasoning_effort: effort };
      if (thinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'minimaxai/minimax-m3': {
      return { chat_template_kwargs: { thinking_mode: thinking ? 'enabled' : 'disabled' } };
    }

    default:
      if (thinking) return { chat_template_kwargs: { enable_thinking: true } };
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
    const out = { content: '', reasoning: '' };
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

module.exports = { SHOW_REASONING, getReasoningPayload, resolveThinking, StreamNormalizer, normalizeNonStreamChoice };
