# Experimental Codex Gateway for Headroom

This gateway adds an opt-in LiteLLM experiment layer in front of an existing local Headroom proxy:

```text
Codex CLI -> LiteLLM on 127.0.0.1:4000 -> Headroom on 127.0.0.1:8787 -> ChatGPT Codex endpoint
```

The direct Codex-to-Headroom configuration remains the default and rollback path. Headroom remains responsible for
OAuth routing, selecting the final ChatGPT endpoint, and context compression. The gateway does not route directly to
OpenAI, refresh or cache OAuth credentials, retry authentication failures, rewrite rate limits, or compress requests.

This experiment is based on LiteLLM PR #34678 at commit
`01cf5748d6e7feb27c62af5f404fd379ff81d3f3`, whose parent is
`998a372417654f5ec18554e17b8cbe1854d9c683`.

## Start the chain

Start Headroom first:

```bash
/home/neil-king/.local/bin/headroom proxy --port 8787
```

Set a separate random local gateway key. Do not reuse or place the ChatGPT OAuth bearer in this variable:

```bash
export LOCAL_CODEX_GATEWAY_KEY='<local-random-value>'
export HEADROOM_BASE_URL='http://127.0.0.1:8787/v1'
uv run litellm-codex-gateway
```

The gateway binds to `127.0.0.1:4000` by default. `CODEX_GATEWAY_HOST` accepts only `127.0.0.1`, `::1`, or
`localhost`. `HEADROOM_BASE_URL` must also resolve syntactically to a loopback address.

Add this opt-in provider to `~/.codex/config.toml` without changing the active default:

```toml
[model_providers.litellm_headroom]
name = "LiteLLM via Headroom"
base_url = "http://127.0.0.1:4000/v1"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
env_http_headers = { "x-litellm-api-key" = "LOCAL_CODEX_GATEWAY_KEY" }
```

Codex 0.134.0 and later loads named profiles from separate files. Create
`~/.codex/litellm_headroom.config.toml` with:

```toml
model_provider = "litellm_headroom"
```

Run only experimental sessions through the chain:

```bash
codex --profile litellm_headroom
```

Run `codex` without that profile to use the existing direct Headroom path.

## Operational endpoints

- `GET /healthz` returns a content-free local process health response
- `GET /readyz` probes Headroom's `/livez` endpoint without forwarding credentials
- `GET /metrics` requires `x-litellm-api-key`
- `GET /debug/traces/{trace_id}/export` requires `x-litellm-api-key`
- `/v1/responses` and its subpaths support HTTP and SSE passthrough

WebSocket passthrough is intentionally disabled. Keep `supports_websockets = false` until the Codex handshake has been
captured safely and a separate transport-completeness milestone is implemented.

## Capture

Capture is disabled by default. Enable it only for a bounded local experiment:

```bash
export CODEX_GATEWAY_CAPTURE=true
```

Optional controls are `CODEX_GATEWAY_TRACE_DIR`, `CODEX_GATEWAY_MAX_TRACE_BYTES`,
`CODEX_GATEWAY_MAX_TRACE_STORAGE_BYTES`, and `CODEX_GATEWAY_TRACE_RETENTION_SECONDS`. Defaults are 10 MiB per trace,
100 MiB total, and seven days. Trace directories and files are restricted to owner access. Requests and responses that
exceed their capture budget omit content at the truncation boundary rather than storing partial secrets.

Capture and storage failures do not change the request sent to Headroom. Redaction is defense in depth, not permission
to retain sensitive production traffic. Keep capture off unless the resulting artifacts are required, inspect exports
before sharing them, and delete them according to the experiment's data-handling policy.

## Environment reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_CODEX_GATEWAY_KEY` | required | Separate local LiteLLM authentication key |
| `HEADROOM_BASE_URL` | `http://127.0.0.1:8787/v1` | Exclusive downstream API base |
| `HEADROOM_READINESS_PATH` | `/livez` | Credential-free readiness path |
| `CODEX_GATEWAY_HOST` | `127.0.0.1` | Loopback gateway bind address |
| `CODEX_GATEWAY_PORT` | `4000` | Gateway port |
| `CODEX_GATEWAY_CAPTURE` | off | Enables bounded redacted trace capture |

Do not set an OpenAI API key for this flow. The ChatGPT OAuth bearer supplied by Codex is preserved for Headroom, while
the local LiteLLM key, cookies, host, caller content length, and hop-by-hop headers are removed before forwarding.
