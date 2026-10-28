// tools.js
// Tool-call leak recovery.

const TOOL_CALL_OPEN = '<tool_call>';
const TOOL_CALL_CLOSE = '</tool_call>';

function generateToolCallId() {
  return `call_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function extractLeakedToolCalls(content) {
  if (!content || !content.includes(TOOL_CALL_OPEN)) {
    return { content, toolCalls: [] };
  }

  const toolCalls = [];
  let result = '';
  let cursor = 0;

  while (true) {
    const openIdx = content.indexOf(TOOL_CALL_OPEN, cursor);
    if (openIdx === -1) {
      result += content.slice(cursor);
      break;
    }
    const closeIdx = content.indexOf(TOOL_CALL_CLOSE, openIdx);
    if (closeIdx === -1) {
      result += content.slice(cursor);
      break;
    }

    result += content.slice(cursor, openIdx);
    const inner = content.slice(openIdx + TOOL_CALL_OPEN.length, closeIdx).trim();

    try {
      const parsed = JSON.parse(inner);
      if (parsed && typeof parsed.name === 'string') {
        toolCalls.push({
          id: generateToolCallId(),
          type: 'function',
          function: { name: parsed.name, arguments: JSON.stringify(parsed.arguments ?? {}) }
        });
      } else {
        result += content.slice(openIdx, closeIdx + TOOL_CALL_CLOSE.length);
      }
    } catch {
      result += content.slice(openIdx, closeIdx + TOOL_CALL_CLOSE.length);
    }

    cursor = closeIdx + TOOL_CALL_CLOSE.length;
  }

  return { content: result, toolCalls };
}

function partialTagSuffixLength(str, tag) {
  const maxLen = Math.min(str.length, tag.length - 1);
  for (let i = maxLen; i > 0; i--) {
    if (tag.startsWith(str.slice(str.length - i))) return i;
  }
  return 0;
}

class ToolCallStreamRecovery {
  constructor() {
    this.buffer = '';
    this.toolCallIndex = 0;
  }

  process(text) {
    this.buffer += text;

    const toolCallDeltas = [];
    let outContent = '';
    let cursor = 0;

    while (true) {
      const openIdx = this.buffer.indexOf(TOOL_CALL_OPEN, cursor);

      if (openIdx === -1) {
        const remaining = this.buffer.slice(cursor);
        const partialLen = partialTagSuffixLength(remaining, TOOL_CALL_OPEN);
        const safeLen = remaining.length - partialLen;
        outContent += remaining.slice(0, safeLen);
        this.buffer = remaining.slice(safeLen);
        break;
      }

      const closeIdx = this.buffer.indexOf(TOOL_CALL_CLOSE, openIdx);
      if (closeIdx === -1) {
        outContent += this.buffer.slice(cursor, openIdx);
        this.buffer = this.buffer.slice(openIdx);
        break;
      }

      outContent += this.buffer.slice(cursor, openIdx);
      const inner = this.buffer.slice(openIdx + TOOL_CALL_OPEN.length, closeIdx).trim();

      try {
        const parsed = JSON.parse(inner);
        if (parsed && typeof parsed.name === 'string') {
          toolCallDeltas.push({
            index: this.toolCallIndex++,
            id: generateToolCallId(),
            type: 'function',
            function: { name: parsed.name, arguments: JSON.stringify(parsed.arguments ?? {}) }
          });
        } else {
          outContent += this.buffer.slice(openIdx, closeIdx + TOOL_CALL_CLOSE.length);
        }
      } catch {
        outContent += this.buffer.slice(openIdx, closeIdx + TOOL_CALL_CLOSE.length);
      }

      cursor = closeIdx + TOOL_CALL_CLOSE.length;
    }

    return { content: outContent, toolCallDeltas };
  }

  flush() {
    const leftover = this.buffer;
    this.buffer = '';
    return leftover;
  }
}

module.exports = { extractLeakedToolCalls, ToolCallStreamRecovery };
