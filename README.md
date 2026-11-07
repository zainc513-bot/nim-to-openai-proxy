# NIM → OpenAI Proxy

A small, self-hosted translation layer that exposes **NVIDIA NIM** models through an **OpenAI-compatible API**. Point JanitorAI, SillyTavern (with Lorebary), or any other OpenAI-style client at it and use NIM models without changing the client.

> **Maintenance notice:** The original author, Jontte, can no longer maintain this project. It is now maintained by a community member on a best-effort basis.
>
> **Security notice:** If you forked before **June 7, 2026**, pull the latest version. Older versions had an auth bypass and a startup DDoS vulnerability.

## Features

- OpenAI-compatible `/v1/chat/completions` (streaming and non-streaming) and `/v1/models`
- Friendly aliases (`gpt-4`, `kimi-k3.1-pro`, …) mapped to NIM models, plus raw `vendor/model` IDs
- Reasoning support: effort levels, on/off switching, and optional `<thinking>` output
- Recovery of tool calls that models leak into message text as `<tool_call>…</tool_call>`
- Bearer-token auth so your NVIDIA key never reaches the client
- SSE heartbeats so slow "thinking" models don't hit proxy or platform timeouts
- Clean stream termination (always ends with `[DONE]`), even if the upstream drops

## Quick start

**Requirements:** Node.js 20+ and an NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/) (profile icon → *API keys*).

```bash
git clone <your-fork-url>
cd nim-openai-proxy
npm install

export NIM_API_KEY="nvapi-..."
export CLIENT_AUTH_KEY="$(openssl rand -hex 24)"   # this is the "API key" your client will use
npm start
```

Test it:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $CLIENT_AUTH_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello!"}]}'
```

For local development, `.env.example` lists every variable. Load it with your shell or `node --env-file=.env server.js`.

## Deploying

Any host that runs Node works (Railway, Render, Fly.io, a VPS). Railway is the simplest:

1. **Fork this repo first**, then deploy your fork. This keeps your deployment private to you.
2. Wait for the first build to finish, then open the **Variables** tab and set `NIM_API_KEY` and `CLIENT_AUTH_KEY`.
3. In **Settings → Networking**, generate a public URL.
4. In your client, set the proxy URL to `https://<your-url>/v1/chat/completions`, the API key to your `CLIENT_AUTH_KEY`, and the model to an alias from the table below.

Environment variable changes require a restart or redeploy to take effect.

## Authentication

Clients authenticate with `Authorization: Bearer <CLIENT_AUTH_KEY>`. Use a long random string (32+ characters):

```bash
openssl rand -hex 24
```

This is deliberately separate from `NIM_API_KEY`: the client only ever holds the proxy key, and you can rotate it without touching your NVIDIA key. `/`, `/health`, and `/v1/models` are public so clients can check connectivity before auth is configured. If `CLIENT_AUTH_KEY` is unset, every protected route is rejected.

## Models

Any alias below works, and so does any raw NIM ID such as `meta/muse-glimmer-30b`. Unknown names fall back to `nvidia/nemotron-3.5-lightning-30b-a3b`.

| Alias | NIM model | Best for | Speed | Filters |
|---|---|---|---|---|
| `gpt-4` | `nvidia/nemotron-3-ultra-550b-a55b` | Immersive RP | Fast | Low |
| `gpt-3.5-turbo` | `nvidia/nemotron-3-super-120b-a12b` | Lightweight tasks | Fast | Low |
| `nemotron-lightning` (legacy: `llama-3.1-8b`) | `nvidia/nemotron-3.5-lightning-30b-a3b` | General chat, quick tasks | Very fast | Low–Medium |
| `kimi-k3` | `moonshotai/kimi-k3` | Deep RP and coding | Slow | Medium–High |
| `kimi-k3.1-pro` | `moonshotai/kimi-k3.1-pro` | Agentic coding, 1M-token context | — | — |
| `glm-5.3` | `z-ai/glm-5.3` | General use | — | — |
| `glm-flash`, `gpt-4-flash` | `z-ai/glm-5.3-flash` | Fast general use | — | — |
| `claude-3-sonnet` | `openai/gpt-oss-20b` | Fast, distinct voice | Fast | Low–Medium |
| `m3` | `minimaxai/minimax-m3` | Experimental | Medium–High | — |

Speed and filter ratings are the maintainer's subjective notes, not benchmarks. "—" means untested.

> **`kimi-k3.1-pro` is preview-only on NVIDIA's side.** At the time of writing, the model page says it is published for playground use and that no API key authorizes programmatic access yet. Requests will fail until NVIDIA opens the endpoint. The alias is ready for when they do.

DeepSeek models are intentionally blocked, whether requested by alias or raw ID (`400 Model not available`).

To add or change an alias, edit `MODEL_MAPPING` in `models.js`. If the model has special reasoning parameters, add a case in `reasoning.js`.

## Reasoning

Reasoning is controlled by environment variables and per-request fields.

| Setting | Where | Effect |
|---|---|---|
| `ENABLE_THINKING_MODE=true` | env | Turns reasoning on by default for models that support it |
| `SHOW_REASONING=true` | env | Returns reasoning to the client (`reasoning` and `reasoning_content` fields) |
| `reasoning_effort` | request body | `"off"`/`"none"` disables, `"on"` enables, or a model-specific level |
| `x-reasoning-format: inline` | request header | Streams reasoning inside the content as `<thinking>…</thinking>` (needs `SHOW_REASONING=true`) |

Supported effort levels per model (invalid values are dropped with a warning):

| Model | Levels |
|---|---|
| `openai/gpt-oss-*` | `low`, `medium`, `high` |
| `nvidia/nemotron-3-ultra-550b-a55b` | `none`, `medium`, `high` |
| `moonshotai/kimi-k3`, `moonshotai/kimi-k3.1-pro` | `low`, `high`, `max` |
| `meta/muse-glimmer-30b` | `none`, `minimal`, `low`, `medium`, `high`, `max` |
| `minimaxai/minimax-m3` | on/off only |
| GLM 5.3 | no parameters are sent |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NIM_API_KEY` | — | **Required.** Your NVIDIA API key |
| `CLIENT_AUTH_KEY` | — | **Required.** Bearer token clients must send |
| `PORT` | `3000` | Port to listen on |
| `NIM_API_BASE` | `https://integrate.api.nvidia.com/v1` | Upstream base URL |
| `ENABLE_THINKING_MODE` | `false` | Enable reasoning by default |
| `SHOW_REASONING` | `false` | Return reasoning to clients |
| `DEFAULT_MAX_TOKENS` | `4096` | Used when the client sends no `max_tokens` (hard cap: 65536) |
| `REQUEST_TIMEOUT_MS` | `120000` | Max wait for NIM to start responding |
| `THINKING_DEADLINE_MS` | `120000` | Abort a stream that produces no output at all within this time |
| `SKIP_VALIDATION` | `false` | Skip the startup connectivity check |
| `DEBUG_MODE` | `false` | Verbose logs (heartbeats, progress, skipped chunks) |
| `KEEP_STREAM_USAGE` | `false` | Keep `stream_options` on requests without tools |

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | No | Liveness text |
| `GET` | `/health` | No | `{"status":"ok"}` |
| `GET` | `/v1/models` | No | List of aliases |
| `POST` | `/v1/chat/completions` | Yes | Chat completions (set `"stream": true` for SSE) |

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `403 Forbidden: Invalid or missing authentication` | Client key doesn't match `CLIENT_AUTH_KEY` | Make the client's API key identical to the env var, with no extra spaces or quotes. Regenerating the key and redeploying often clears stale values |
| `Upstream NIM error (401/403)` | `NIM_API_KEY` is invalid or expired | Regenerate it at build.nvidia.com. The error is from NVIDIA, not the proxy |
| `Upstream NIM error (404)` | Model was renamed or retired | Check the [NIM catalog](https://build.nvidia.com/) and update `models.js` |
| `kimi-k3.1-pro` fails | NVIDIA hasn't opened API access yet | Use `kimi-k3` until the endpoint is public |
| `400 Model not available` | The model is blocked (DeepSeek) | Pick another model |
| Very slow responses | Large reasoning models at peak hours | Use a lighter model or set `reasoning_effort` to `low`/`off` |
| `[Proxy: no output after 120s]` | Model produced nothing | Retry, switch models, or raise `THINKING_DEADLINE_MS` |
| "Failed to fetch" in JanitorAI | Cached old proxy config | Reload the page after changing URL or model |

Turn on `DEBUG_MODE=true` to see heartbeats, stream progress, and malformed upstream chunks.

## Project layout

```
server.js      HTTP server, auth, request/response shaping, streaming
config.js      Environment configuration and logger
models.js      Alias mapping, model resolution, blocklist
reasoning.js   Per-model reasoning payloads and reasoning stream normalization
tools.js       Recovery of leaked <tool_call> blocks (stream and non-stream)
utils.js       Shared helpers
```

## Contributing

Issues and PRs are welcome, especially for model mapping updates (NIM deprecates things often), bug fixes, and docs. Large architectural changes and features that add complexity without clear benefit are less likely to be merged.

When reporting a bug, include the model alias, whether streaming was on, the error message, and your hosting platform. Check Troubleshooting and the NIM catalog first.

Contact: Discord `Skywalker_1401`.

## Credits and license

Originally created by Jontte, building on a script from a Reddit guide. Released under the [MIT License](LICENSE).

This is a hobby project and is not affiliated with NVIDIA. You need your own NVIDIA API key and are subject to NVIDIA's terms and rate limits.
