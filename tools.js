// tools.js — Recovery for leaked <tool_call> tags in content

const OPEN = '<tool_call>';
const CLOSE = '</tool_call>';

function genId() {
  return `call_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function extractLeakedToolCalls(content) {
  if (!content || !content.includes(OPEN)) return { content, toolCalls: [] };
  const toolCalls = [];
  let result = '', cursor = 0;

  while (true) {
    const openIdx = content.indexOf(OPEN, cursor);
    if (openIdx === -1) { result += content.slice(cursor); break; }
    const closeIdx = content.indexOf(CLOSE, openIdx);
    if (closeIdx === -1) { result += content.slice(cursor); break; }

    result += content.slice(cursor, openIdx);
    const inner = content.slice(openIdx + OPEN.length, closeIdx).trim();

    try {
      const parsed = JSON.parse(inner);
      if (parsed && typeof parsed.name === 'string') {
        toolCalls.push({
          id: genId(), type: 'function',
          function: { name: parsed.name, arguments: JSON.stringify(parsed.arguments ?? {}) }
        });
      } else { result += content.slice(openIdx, closeIdx + CLOSE.length); }
    } catch {
      result += content.slice(openIdx, closeIdx + CLOSE.length);
    }
    cursor = closeIdx + CLOSE.length;
  }
  return { content: result, toolCalls };
}

function partialLen(str, tag) {
  const max = Math.min(str.length, tag.length - 1);
  for (let i = max; i > 0; i--) {
    if (tag.startsWith(str.slice(str.length - i))) return i;
  }
  return 0;
}

class ToolCallStreamRecovery {
  constructor() { this.buf = ''; this.idx = 0; }
  process(text) {
    this.buf += text;
    const deltas = [];
    let out = '', cursor = 0;

    while (true) {
      const o = this.buf.indexOf(OPEN, cursor);
      if (o === -1) {
        const rem = this.buf.slice(cursor);
        const p = partialLen(rem, OPEN);
        out += rem.slice(0, rem.length - p);
        this.buf = rem.slice(rem.length - p);
        break;
      }
      const c = this.buf.indexOf(CLOSE, o);
      if (c === -1) {
        out += this.buf.slice(cursor, o);
        this.buf = this.buf.slice(o);
        break;
      }
      out += this.buf.slice(cursor, o);
      const inner = this.buf.slice(o + OPEN.length, c).trim();

      try {
        const parsed = JSON.parse(inner);
        if (parsed && typeof parsed.name === 'string') {
          deltas.push({
            index: this.idx++, id: genId(), type: 'function',
            function: { name: parsed.name, arguments: JSON.stringify(parsed.arguments ?? {}) }
          });
        } else { out += this.buf.slice(o, c + CLOSE.length); }
      } catch { out += this.buf.slice(o, c + CLOSE.length); }
      cursor = c + CLOSE.length;
    }
    return { content: out, toolCallDeltas: deltas };
  }
  flush() { const left = this.buf; this.buf = ''; return left; }
}

module.exports = { extractLeakedToolCalls, ToolCallStreamRecovery };
