# Model Guide

What each model behind the proxy is good at, how fast to expect it to be, and how to control its reasoning.

Specs below come from NVIDIA's model pages as of November 2026 and will drift as NVIDIA updates the catalog. Anything marked *(maintainer's note)* is subjective experience from the README, not a benchmark.

## A note on response time

**The speed ratings in this guide are expectations, not measurements.** NVIDIA doesn't publish latency for its free endpoint, and the endpoint is a shared trial service, so real numbers swing with load and time of day. What I can tell you with confidence:

- **Active parameters set the baseline.** These are all mixture-of-experts models, so decode speed tracks the *active* parameters per token, not the headline size. A 3B-active model is much faster than a 104B-active one.
- **Reasoning dominates wall-clock time.** With thinking on, time to first visible token can run to tens of seconds for any model, because the model reasons before it answers. Turning reasoning off (or setting `reasoning_effort` to `low`) is the biggest speed lever you have.
- **Serving tricks help.** NVIDIA serves several of these with FP4 quantization and speculative decoding (multi-token prediction), which is why a large model can still feel fast.
- **The proxy adds almost nothing.** It parses and forwards the stream. Heartbeats keep the connection alive during long thinking, but they don't make anything faster.

One independent data point: a third-party test of NVIDIA's free API ran the same multi-step reasoning task on three Nemotron models and saw roughly 4 s for Nano (30B-A3B), 16 s for Super and 17 s for Ultra, while a one-sentence reply from Ultra came back in under a second ([source](https://dev.classmethod.jp/en/articles/dgx-spark-nemotron3-ultra-nvidia-api/)). Treat it as an anecdote, not a benchmark.

To get real numbers for *your* deployment, run the bundled script (see [Measure it yourself](#measure-it-yourself)).

## At a glance

| Alias | NIM model | Active / total params | Context | Input | Expected speed | Filters *(maintainer's note)* |
|---|---|---|---|---|---|---|
| `nemotron-lightning` (legacy `llama-3.1-8b`) | `nvidia/nemotron-3.5-lightning-30b-a3b` | 3B / 30B | 1M | Text | **Very fast** | Low–Medium |
| `claude-3-sonnet` | `openai/gpt-oss-20b` | ~3.6B / ~21B | ~128K | Text | **Very fast** | Low–Medium |
| `gpt-3.5-turbo` | `nvidia/nemotron-3-super-120b-a12b` | 12B / 120B | — | Text | **Fast** | Low |
| `glm-flash`, `gpt-4-flash` | `z-ai/glm-5.3-flash` | 18B / 320B | 1M | Text, image | **Fast** | — |
| `m3` | `minimaxai/minimax-m3` | ~22–23B / 428B | 1M | Text, image, video | **Medium** | — |
| `gpt-4` | `nvidia/nemotron-3-ultra-550b-a55b` | 55B / 550B | 1M | Text | **Medium–fast** *(maintainer rates it Fast)* | Low |
| `mistral-large-4` | `mistralai/mistral-large-4-1t-a52b` | 52B / 1,050B | 262K | Text, image | **Medium** | — |
| `glm-5.3` | `z-ai/glm-5.3` | ~40B / 753B | 1M | Text | **Medium–slow** | — |
| `kimi-k3` | `moonshotai/kimi-k3` | 104B / 2.8T | 1M | Text, image | **Slow** | Medium–High |
| `kimi-k3.1-pro` | `moonshotai/kimi-k3.1-pro` | — / 2.8T | 1M | Text, image | Slow (expected) | — |

"—" means unknown or not checked. `gpt-oss-20b` figures are from general knowledge rather than NVIDIA's page.

> **`kimi-k3.1-pro` is not usable yet.** NVIDIA's page says it is published for playground use only and that no API key authorizes programmatic access. Calls will fail until that changes.

## Which model should I use?

| If you want… | Try | Why |
|---|---|---|
| Fast replies, everyday chat | `nemotron-lightning` | 3B active; NVIDIA bills it as the fastest 30B-A3B MoE |
| Immersive roleplay with few refusals | `gpt-4` | Maintainer's pick for RP with low filters |
| The strongest coding and agent work | `kimi-k3`, `glm-5.3`, `gpt-4` | All aimed at long-horizon coding and tool use |
| A fast model that still handles images | `glm-flash` | Multimodal with 18B active and speculative decoding |
| Image understanding and structured output | `mistral-large-4` | Text + image in, reasoning, function calling, structured outputs |
| Video understanding | `m3` | The only one here that takes video (up to ~30 min) |
| Huge documents or codebases | `gpt-4`, `kimi-k3`, `glm-5.3`, `nemotron-lightning` | All advertise ~1M-token context |
| Cheapest, lightest tasks | `claude-3-sonnet`, `nemotron-lightning` | Smallest active parameter counts |

## Model notes

### `gpt-4` → Nemotron 3 Ultra (550B-A55B)
NVIDIA's flagship open model, a hybrid Mamba-Transformer MoE with a 1M-token context. NVIDIA positions it for agentic reasoning, coding, planning and tool calling, plus long-context work over large documents and codebases. Speed is helped by multi-token prediction. Reasoning levels: `none`, `medium`, `high`.

### `gpt-3.5-turbo` → Nemotron 3 Super (120B-A12B)
The mid-size Nemotron: far fewer active parameters than Ultra, so noticeably quicker on multi-step tasks. The maintainer uses it for lightweight work. Reasoning is a simple on/off.

### `nemotron-lightning` → Nemotron 3.5 Lightning (30B-A3B)
Released August 2026 and billed by NVIDIA as its fastest 30B-A3B model, aimed at sub-agent and long-running-agent workloads. On NVIDIA's card the NVFP4 build scores about 81.6 on MMLU Pro and 75.6 on GPQA Diamond. It is also the **default model** the proxy falls back to for unknown names. Reasoning is on/off.

### `kimi-k3` → Kimi K3 (104B active / 2.8T)
Moonshot's open-weight, natively multimodal model for long-horizon coding, agentic knowledge work and document understanding. It is the slowest model here. **Multi-turn gotcha:** NVIDIA's card says Kimi K3 was trained with preserved thinking history, so apps doing multi-turn chat or tool calls must send back the complete previous assistant message, including `reasoning_content` and `tool_calls`. To make that possible, set `SHOW_REASONING=true` on the proxy and make sure your client echoes the field back. Reasoning levels: `low`, `high`, `max`.

### `kimi-k3.1-pro` → Kimi K3.1 Pro (preview)
A 2.8T-parameter model with a 1M-token context and text + image input, targeted at agentic coding, 3D modeling and finance. Preview only, as noted above. The proxy assumes the same effort levels as `kimi-k3`, which is unverified.

### `mistral-large-4` → Mistral Large 4 (52B active / 1,050B)
A granular MoE with text and image input, reasoning, function calling and structured outputs, served in NVFP4 with speculative decoding. Context is 262K, a quarter of the other big models, so it is not the pick for whole-codebase prompts. NVIDIA's example sends `reasoning_effort: "high"`; the proxy passes any effort value through unvalidated because the accepted set isn't documented on the page.

### `glm-5.3` → GLM-5.3 (~40B active / 753B)
Z.ai's text-only model for complex coding and long-horizon agent work, with a 1M context. NVIDIA's card notes that chat-style use should pass `clear_thinking=true` in the chat template. The proxy sends no reasoning parameters for GLM (to match NVIDIA's playground), so a client can add `"chat_template_kwargs": {"clear_thinking": true}` to its request if it wants that. I haven't verified this against the live endpoint.

### `glm-flash` / `gpt-4-flash` → GLM-5.3-Flash (18B active / 320B)
Z.ai's first natively multimodal GLM, built for high-throughput, cost-sensitive serving. It takes text and images and is aimed at screenshot and document understanding, coding agents and long context. NVIDIA serves it with multi-token-prediction speculative decoding. Reasoning comes back separately from the answer.

### `m3` → MiniMax-M3 (~22–23B active / 428B)
Multimodal with text, image and video input, 1M context, aimed at long videos (up to 30 minutes), long coding tasks and creative or design work. **License:** NVIDIA lists it under a *Non-Commercial* MiniMax license. A newer MiniMax-M3.1-Flash (428B / 23B active, five reasoning levels) appeared on NVIDIA in October 2026. You can call it by raw ID (`minimaxai/minimax-m3.1-flash`), but the proxy has no reasoning controls wired for it yet.

### `claude-3-sonnet` → gpt-oss-20b
OpenAI's small open-weight reasoning model. Very quick, with three reasoning levels (`low`, `medium`, `high`). Good for short or lightweight turns, not for deep long-context work.

## Controlling reasoning

| Model | How to control it |
|---|---|
| `gpt-oss-20b`, `kimi-k3`, `kimi-k3.1-pro`, `mistral-large-4`, `meta/muse-glimmer-30b` | `reasoning_effort` is sent directly. Valid levels per model are listed in the README |
| `gpt-4` (Ultra) | `reasoning_effort`: `none`, `medium`, `high`. `none` turns thinking off |
| `gpt-3.5-turbo` (Super) | On/off only |
| `m3` | On/off only |
| `glm-5.3`, `glm-flash` | Nothing is sent |
| Any other raw ID | `ENABLE_THINKING_MODE=true` or `reasoning_effort: "on"` sends `enable_thinking` |

For speed, send `"reasoning_effort": "off"` (or `"low"` where supported). For hard problems, send the highest level the model allows.

## Licensing

All access here is through NVIDIA's free **trial** endpoint, governed by the NVIDIA API Trial Terms of Service. That is fine for personal use and prototyping but not a production guarantee. Individual models add their own terms:

| Model | License (per NVIDIA's page) |
|---|---|
| Nemotron 3 family | OpenMDW 1.1, commercial use permitted |
| Kimi K3 | Modified MIT |
| GLM-5.3 | MIT-like, with a security-review condition for very large model-as-a-service operators |
| MiniMax-M3 | **Non-commercial** |

Check each model's page before using it commercially.

## Measure it yourself

`bench.js` streams a short prompt through your proxy and reports time to first token and rough tokens per second, as a Markdown table.

```bash
export CLIENT_AUTH_KEY="your-proxy-key"

# Every alias, 3 runs each, reasoning off
node bench.js --url https://your-proxy.example.com

# Specific models, more runs, reasoning on
node bench.js --models gpt-4,kimi-k3,mistral-large-4 --runs 5 --reasoning on
```

Tips for reading the results:

- Run it at the times you actually use the proxy; shared endpoints are slower at peak hours.
- Compare `--reasoning off` and `--reasoning on` for the same model to see how much thinking costs you.
- If `SHOW_REASONING` is off on the proxy, TTFT includes thinking time, which is what a chat user feels.
- Token counts are estimated at 4 characters per token, so compare models against each other rather than against NVIDIA's numbers.

Paste your results here so the guide reflects your setup:

| Alias | TTFT (s) | Total (s) | ~tokens/s | Date |
|---|---|---|---|---|
| | | | | |
