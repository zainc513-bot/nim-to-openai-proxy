// tools.js — recovery of <tool_call>…</tool_call> blocks that leak into message content

'use strict';

const { partialLen } = require('./utils');

const OPEN = '<tool_call>';
const CLOSE = '</tool_call>';

const genId = () => `call_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;

/** Parses the JSON inside a tool_call block. Returns a `function` object, or null if invalid. */
function parseToolCall(inner) {
  try {
    const parsed = JSON.parse(inner);
    if (parsed && typeof parsed.name === 'string') {
      const args = parsed.arguments ?? {};
      return { name: parsed.name, arguments: typeof args === 'string' ? args : JSON.stringify(args) };
    }
  } catch { /* not JSON — leave the text untouched */ }
  return null;
}

/** Non-stream: pulls leaked tool calls out of `content`. */
function extractLeakedToolCalls(content) {
  if (!content || !content.includes(OPEN)) return { content, toolCalls: [] };

  const toolCalls = [];
  let result = '';
  let cursor = 0;

  for (;;) {
    const open = content.indexOf(OPEN, cursor);
    const close = open === -1 ? -1 : content.indexOf(CLOSE, open);
    if (open === -1 || close === -1) {
      result += content.slice(cursor);
      break;
    }

    result += content.slice(cursor, open);
    const fn = parseToolCall(content.slice(open + OPEN.length, close).trim());
    if (fn) toolCalls.push({ id: genId(), type: 'function', function: fn });
    else result += content.slice(open, close + CLOSE.length);

    cursor = close + CLOSE.length;
  }

  return { content: result, toolCalls };
}

/** Streaming: strips leaked tool calls from content chunks and emits them as tool_call deltas. */
class ToolCallStreamRecovery {
  constructor() {
    this.buf = '';
    this.idx = 0;
  }

  /** Make sure recovered tool calls don't reuse indexes the model already used natively. */
  reserve(nextIndex) {
    this.idx = Math.max(this.idx, nextIndex);
  }

  process(text) {
    this.buf += text;
    const deltas = [];
    let out = '';
    let cursor = 0;

    for (;;) {
      const open = this.buf.indexOf(OPEN, cursor);

      if (open === -1) {
        const rest = this.buf.slice(cursor);
        const keep = partialLen(rest, OPEN);
        out += rest.slice(0, rest.length - keep);
        this.buf = rest.slice(rest.length - keep);
        break;
      }

      const close = this.buf.indexOf(CLOSE, open);
      if (close === -1) {
        out += this.buf.slice(cursor, open);
        this.buf = this.buf.slice(open);
        break;
      }

      out += this.buf.slice(cursor, open);
      const fn = parseToolCall(this.buf.slice(open + OPEN.length, close).trim());
      if (fn) deltas.push({ index: this.idx++, id: genId(), type: 'function', function: fn });
      else out += this.buf.slice(open, close + CLOSE.length);

      cursor = close + CLOSE.length;
    }

    return { content: out, toolCallDeltas: deltas };
  }

  /** Returns any text still held back (an unterminated tag) and resets the buffer. */
  flush() {
    const left = this.buf;
    this.buf = '';
    return left;
  }
}

module.exports = { extractLeakedToolCalls, ToolCallStreamRecovery };
