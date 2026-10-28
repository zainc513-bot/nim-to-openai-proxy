// ─── Model Mapping (Fixed typos, removed dead models) ────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'mistralai/mistral-large-2-instruct',
  'gpt-4': 'mistralai/mistral-large-2-instruct',
  'gpt-3.5': 'mistralai/mistral-7b-instruct-v0.3',
  'gpt-4-turbo': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'claude-3-opus': 'google/diffusiongemma-26b-a4b-it',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.1-nemotron-70b-instruct', 
  'gemini-turbo': 'nvidia/llama3-chatqa-1.5-70b',
  'gpt-3.5o': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'gpt-4-flash': 'z-ai/glm-5.3-flash',
  'deepseek': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4o': 'stepfun-ai/step-5-preview',
  'mistral': 'mistralai/mistral-large-2-instruct', 
  'mistral-turbo': 'nv-mistralai/mistral-nemo-12b-instruct',
  'mistral-pro': 'mistralai/mistral-7b-instruct-v0.3', 
  'mistral-fast': 'nvidia/mistral-nemo-minitron-8b-8k-instruct', 
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'meta/muse-glimmer-30b',
  'google-lighter': 'poolside/laguna-xs-2.1',
  'm3.1': 'minimaxai/minimax-m3.1-flash',
  'glm-flash': 'z-ai/glm-5.3-flash',
  'llama-vision': 'meta/llama-3.2-11b-vision-instruct',
  'vision': 'meta/llama-3.2-11b-vision-instruct',
  'llama-vision-11b': 'meta/llama-3.2-11b-vision-instruct',
  'llama-3.2-vision': 'meta/llama-3.2-11b-vision-instruct'
};

const DEFAULT_MODEL = 'mistralai/mistral-large-2-instruct';

function resolveModelId(name) {
  if (typeof name !== 'string') return null;
  if (MODEL_MAPPING[name]) return MODEL_MAPPING[name];
  if (/^[\w.-]+\/[\w.:-]+$/.test(name)) return name;
  return null;
}

// Safest models first so fallback always succeeds. 
// We are using Mistral and Gemma because your log showed they are alive (✓).
const FALLBACK_MODELS = [
  'mistralai/mistral-large-2-instruct',
  'google/gemma-4-31b-it',
  'mistralai/mistral-7b-instruct-v0.3'
];