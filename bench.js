#!/usr/bin/env node
// bench.js — measures time-to-first-token (TTFT) and throughput for models behind the proxy.
//
//   CLIENT_AUTH_KEY=... node bench.js
//   CLIENT_AUTH_KEY=... node bench.js --models gpt-4,kimi-k3 --runs 5 --reasoning on
//
// Options (all optional):
//   --url         Proxy base URL            (default: $PROXY_URL or http://localhost:3000)
//   --models      Comma-separated aliases   (default: every alias from /v1/models)
//   --runs        Runs per model            (default: 3)
//   --max-tokens  Output cap per run        (default: 256)
//   --reasoning   "off" | "on"              (default: off)
//   --prompt      Prompt text               (default: a short explanation task)
//
// Notes:
//   - Runs are sequential so models don't compete with each other.
//   - If SHOW_REASONING is off on the proxy, TTFT is the time to the first *visible* token,
//     which includes any thinking time. That is what a chat user actually feels.
//   - Token counts are estimated as characters / 4.

'use strict';

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i !== -1 && args[i + 1] ? args[i + 1] : fallback;
};

const BASE = opt('url', process.env.PROXY_URL || 'http://localhost:3000').replace(/\/$/, '');
const KEY = process.env.CLIENT_AUTH_KEY;
const RUNS = Number(opt('runs', 3));
const MAX_TOKENS = Number(opt('max-tokens', 256));
const REASONING = opt('reasoning', 'off');
const PROMPT = opt('prompt', 'Explain in three short paragraphs how a hash map works.');

if (!KEY) {
  console.error('Set CLIENT_AUTH_KEY in the environment.');
  process.exit(1);
}

const median = (xs) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

const fmt = (n, digits = 1) => (Number.isFinite(n) ? n.toFixed(digits) : '—');

async function runOnce(model) {
  const t0 = performance.now();
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: true,
      max_tokens: MAX_TOKENS,
      temperature: 0.2,
      reasoning_effort: REASONING,
      messages: [{ role: 'user', content: PROMPT }],
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);

  const decoder = new TextDecoder();
  let buf = '';
  let ttft = null;
  let chars = 0;

  for await (const chunk of res.body) {
    buf += decoder.decode(chunk, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();

    for (const line of lines) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === '[DONE]') continue;

      let data;
      try { data = JSON.parse(payload); } catch { continue; }
      const delta = data.choices?.[0]?.delta;
      if (!delta) continue;

      const text = (delta.content || '') + (delta.reasoning_content || '');
      if (text) {
        if (ttft === null) ttft = performance.now() - t0;
        chars += text.length;
      }
    }
  }

  const total = performance.now() - t0;
  if (ttft === null) throw new Error('no output received');
  const genSeconds = (total - ttft) / 1000;
  return {
    ttft: ttft / 1000,
    total: total / 1000,
    tokPerSec: genSeconds > 0.05 ? chars / 4 / genSeconds : NaN,
  };
}

async function listModels() {
  const res = await fetch(`${BASE}/v1/models`);
  if (!res.ok) throw new Error(`Could not list models: HTTP ${res.status}`);
  return (await res.json()).data.map((m) => m.id);
}

(async () => {
  const models = opt('models', '') ? opt('models', '').split(',').map((s) => s.trim()).filter(Boolean) : await listModels();

  console.log(`Proxy: ${BASE} | runs: ${RUNS} | max_tokens: ${MAX_TOKENS} | reasoning: ${REASONING}\n`);
  console.log('| Alias | OK runs | TTFT (s) | Total (s) | ~tokens/s | Notes |');
  console.log('|---|---|---|---|---|---|');

  for (const model of models) {
    const ok = [];
    let lastError = '';

    for (let i = 0; i < RUNS; i++) {
      try { ok.push(await runOnce(model)); } catch (err) { lastError = err.message; }
    }

    console.log(
      `| \`${model}\` | ${ok.length}/${RUNS} | ${fmt(median(ok.map((r) => r.ttft)), 2)} | `
      + `${fmt(median(ok.map((r) => r.total)))} | ${fmt(median(ok.map((r) => r.tokPerSec)), 0)} | `
      + `${lastError ? lastError.replace(/\|/g, '/') : ''} |`,
    );
  }
})();
