// reasoning.js
// Backend-specific "thinking" control.

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');

const CONTENT_DELIMITER_TAGS = {
  'minimaxai/minimax-m3': ['<mm:think>', '</mm:think>']
};

class DelimiterParser {
  constructor(openTag, closeTag) {
    this.openTag = openTag;
    this.closeTag = closeTag;
    this.inThinking = false;
    this.buffer = '';
  }

  processChunk(chunk) {
    this.buffer += chunk;
    let content = '';
    let reasoning = '';

    while (true) {
      const targetTag = this.inThinking ? this.closeTag : this.openTag;
      const tagIndex = this.buffer.indexOf(targetTag);

      if (tagIndex !== -1) {
        const textBefore = this.buffer.substring(0, tagIndex);
        if (this.inThinking) reasoning += textBefore;
        else content += textBefore;
        this.inThinking = !this.inThinking;
        this.buffer = this.buffer.substring(tagIndex + targetTag.length);
      } else {
        let partialLen = 0;
        const maxLen = Math.min(this.buffer.length, targetTag.length - 1);
        for (let i = maxLen; i > 0; i--) {
          if (targetTag.startsWith(this.buffer.substring(this.buffer.length - i))) {
            partialLen = i;
            break;
          }
        }
        const textBefore = this.buffer.substring(0, this.buffer.length - partialLen);
        if (this.inThinking) reasoning += textBefore;
        else content += textBefore;
        this.buffer = this.buffer.substring(this.buffer.length - partialLen);
        break;
      }
    }

    return { content, reasoning };
  }

  flush() {
    let content = '';
    let reasoning = '';
    if (this.buffer) {
      if (this.inThinking) reasoning += this.buffer;
      else content += this.buffer;
      this.buffer = '';
    }
    return { content, reasoning };
  }
}

class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = null;
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) this.parser = new DelimiterParser(tags[0], tags[1]);
  }

  processDelta(delta) {
    const normalizedDelta = { ...delta };
    let reasoning = normalizedDelta.reasoning || normalizedDelta.reasoning_content || '';
    let content = normalizedDelta.content || '';

    if (!reasoning && content && this.parser) {
      const parsed = this.parser.processChunk(content);
      reasoning = parsed.reasoning;
      content = parsed.content;
    }

    if (content) normalizedDelta.content = content;
    else delete normalizedDelta.content;

    if (reasoning) normalizedDelta.reasoning = reasoning;
    else delete normalizedDelta.reasoning;

    delete normalizedDelta.reasoning_content;
    return normalizedDelta;
  }

  flush() {
    if (!this.parser) return { content: '', reasoning: '' };
    return this.parser.flush();
  }
}

function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;
  const message = choice.message || {};
  let reasoning = message.reasoning || message.reasoning_content || '';
  let content = message.content || '';

  if (!reasoning && content) {
    let parser = null;
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) parser = new DelimiterParser(tags[0], tags[1]);
    if (parser) {
      const parsed = parser.processChunk(content);
      const flushed = parser.flush();
      content = (parsed.content || '') + (flushed.content || '');
      reasoning = (parsed.reasoning || '') + (flushed.reasoning || '');
    }
  }

  const newMessage = { ...message };
  if (content) newMessage.content = content;
  if (reasoning) newMessage.reasoning = reasoning;
  delete newMessage.reasoning_content;

  return { ...choice, message: newMessage };
}

const REASONING_EFFORT_ENUMS = {
  'openai/gpt-oss-120b': ['low', 'medium', 'high'],
  'openai/gpt-oss-20b': ['low', 'medium', 'high'],
  'deepseek-ai/deepseek-v4.1-flash': ['low', 'high', 'max'],
  'deepseek-ai/deepseek-v4-flash-0731': ['low', 'high', 'max'],
  'deepseek-ai/deepseek-v4-pro-0813': ['low', 'high', 'max'],
  'nvidia/nemotron-3-super-120b-a12b': ['low'],
  'nvidia/nemotron-3-ultra-550b-a55b': ['low'],
  'minimaxai/minimax-m3': ['adaptive'],
  'moonshotai/kimi-k3': ['low', 'high', 'max'],
  'z-ai/glm-5.3': ['low', 'high', 'max'],
  'z-ai/glm-5.3-flash': ['low', 'high', 'max'],
  'meta/muse-glimmer-30b': ['none', 'minimal', 'low', 'medium', 'high', 'max']
};

function validReasoningEffort(model, effort) {
  const allowed = REASONING_EFFORT_ENUMS[model];
  if (!allowed) return effort;
  if (allowed.includes(effort)) return effort;
  return undefined;
}

function resolveEffectiveThinking(enableThinking, clientReasoningEffort) {
  if (clientReasoningEffort === 'off') return false;
  if (clientReasoningEffort === 'on') return true;
  return enableThinking;
}

const NEMOTRON_LIGHTNING_BUDGET_MAP = { low: 2048, medium: 8192, high: 16384, max: -1 };

function getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools) {
  enableThinking = resolveEffectiveThinking(enableThinking, clientReasoningEffort);

  const rawEffort = (clientReasoningEffort === 'off' || clientReasoningEffort === 'on') ? undefined : clientReasoningEffort;
  const effort = validReasoningEffort(model, rawEffort);

  switch (model) {
    case 'nvidia/nemotron-3-super-120b-a12b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (effort === 'low') payload.chat_template_kwargs.low_effort = true;
      return payload;
    }

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (effort === 'low') payload.chat_template_kwargs.low_effort = true;
      if (hasTools) payload.chat_template_kwargs.force_nonempty_content = true;
      return payload;
    }

    case 'nvidia/nemotron-3.5-lightning-30b-a3b': {
      if (!enableThinking) return { reasoning_budget: 0 };
      return { reasoning_budget: NEMOTRON_LIGHTNING_BUDGET_MAP[effort] ?? 16384 };
    }

    case 'deepseek-ai/deepseek-v4.1-flash':
    case 'deepseek-ai/deepseek-v4-flash-0731':
    case 'deepseek-ai/deepseek-v4-pro-0813': {
      // Fix: Use rawEffort so 'medium' doesn't silently disable thinking.
      const wantsThinking = clientReasoningEffort === 'on' || !!rawEffort;
      if (!wantsThinking) {
        return { chat_template_kwargs: { thinking: false } };
      }
      return {
        chat_template_kwargs: { thinking: true, reasoning_effort: effort || 'high' }
      };
    }

    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      if (effort) return { reasoning_effort: effort };
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'z-ai/glm-5.3':
    case 'z-ai/glm-5.3-flash': {
      // Fix: Default to 'low' to prevent 80s timeouts on glm-flash.
      const level = effort || 'low';
      return {
        reasoning_effort: level,
        chat_template_kwargs: { reasoning_effort: level }
      };
    }

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) return {};
      return { chat_template_kwargs: { enable_thinking: true }, include_reasoning: SHOW_REASONING };
    }

    case 'meta/muse-glimmer-30b': {
      if (effort) return { reasoning_effort: effort };
      return { reasoning_effort: enableThinking ? 'high' : 'none' };
    }

    case 'minimaxai/minimax-m3': {
      const thinkingMode = effort === 'adaptive' ? 'adaptive' : (enableThinking ? 'enabled' : 'disabled');
      return { chat_template_kwargs: { thinking_mode: thinkingMode } };
    }

    case 'moonshotai/kimi-k3': {
      if (effort) return { reasoning_effort: effort };
      return { reasoning_effort: enableThinking ? 'high' : 'low' };
    }

    default:
      return {};
  }
}

module.exports = {
  SHOW_REASONING,
  getReasoningPayload,
  resolveEffectiveThinking,
  StreamNormalizer,
  normalizeNonStreamChoice
};
