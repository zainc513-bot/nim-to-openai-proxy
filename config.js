// config.js — environment configuration and logger

'use strict';

const flag = (name) => process.env[name] === 'true';

const int = (name, fallback) => {
  const n = Number.parseInt(process.env[name], 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
};

const config = Object.freeze({
  port: int('PORT', 3000),

  // Upstream
  nimApiBase: process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1',
  nimApiKey: process.env.NIM_API_KEY,

  // Client-facing auth
  clientAuthKey: process.env.CLIENT_AUTH_KEY,

  // Behaviour flags
  enableThinking: flag('ENABLE_THINKING_MODE'),
  showReasoning: flag('SHOW_REASONING'),
  skipValidation: flag('SKIP_VALIDATION'),
  keepStreamUsage: flag('KEEP_STREAM_USAGE'),
  debug: flag('DEBUG_MODE'),

  // Limits and timing
  defaultMaxTokens: int('DEFAULT_MAX_TOKENS', 4096),
  maxTokensLimit: 65536,
  heartbeatMs: 10_000,
  requestTimeoutMs: int('REQUEST_TIMEOUT_MS', 120_000),
  thinkingDeadlineMs: int('THINKING_DEADLINE_MS', 120_000),
});

const log = {
  info: (...args) => console.log(...args),
  warn: (...args) => console.warn(...args),
  error: (...args) => console.error(...args),
  debug: (...args) => { if (config.debug) console.log('[DEBUG]', ...args); },
};

module.exports = { config, log };
