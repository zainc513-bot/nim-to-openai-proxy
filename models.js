// models.js — alias → NVIDIA NIM model mapping

'use strict';

const MODEL_MAPPING = Object.freeze({
  // NVIDIA Nemotron
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'nemotron-lightning': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'llama-3.1-8b': 'nvidia/nemotron-3.5-lightning-30b-a3b', // legacy alias, kept for existing clients

  // Moonshot Kimi
  'kimi-k3': 'moonshotai/kimi-k3',
  'kimi-k3.1-pro': 'moonshotai/kimi-k3.1-pro',

  // Z.ai GLM
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-flash': 'z-ai/glm-5.3-flash',
  'gpt-4-flash': 'z-ai/glm-5.3-flash',

  // OpenAI open-weight
  'claude-3-sonnet': 'openai/gpt-oss-20b',

  // MiniMax
  'm3': 'minimaxai/minimax-m3',
});

const DEFAULT_MODEL = 'nvidia/nemotron-3.5-lightning-30b-a3b';

// Models that must never be reachable, whether by alias or raw ID.
const BLOCKED_MODEL = /deepseek/i;

const RAW_MODEL_ID = /^[\w.-]+\/[\w.:-]+$/;

function isBlockedModel(name) {
  return typeof name === 'string' && BLOCKED_MODEL.test(name);
}

/** Returns the NIM model ID for an alias or raw `vendor/model` ID, or null if unknown. */
function resolveModelId(name) {
  if (typeof name !== 'string' || isBlockedModel(name)) return null;
  if (Object.hasOwn(MODEL_MAPPING, name)) return MODEL_MAPPING[name];
  return RAW_MODEL_ID.test(name) ? name : null;
}

module.exports = { MODEL_MAPPING, DEFAULT_MODEL, isBlockedModel, resolveModelId };
