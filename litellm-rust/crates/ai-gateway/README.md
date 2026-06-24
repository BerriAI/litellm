# LiteLLM Rust AI Gateway

A minimal Axum service that fronts OpenAI's realtime API. Clients open a
WebSocket to `GET /v1/realtime`; the gateway authenticates, selects a deployment,
dials OpenAI upstream, and splices the two sockets frame-by-frame.

## Crates

`litellm-rust` is exactly three crates (a crate is a **layer**, not a route):

| Crate | Role | Pure / I/O |
|-------|------|------------|
| litellm-core | Translation layer — types, route contracts (traits), provider transforms (modules under `providers/`), and the router. Builds requests/responses; no network. | Pure |
| litellm-ai-gateway | Routes + host — the only crate that touches the network. HTTP/WebSocket I/O (modules under `io/`) plus the Axum server binary (behind the `server` feature). | I/O |
| litellm-python-bridge | PyO3 cdylib exposing Rust to the litellm Python SDK — a thin adapter over litellm-ai-gateway's I/O. | Binding |

Dependency direction (acyclic): litellm-core ← litellm-ai-gateway ← litellm-python-bridge.

- **Client endpoint:** `wss://<host>/v1/realtime?model=<model>` (WebSocket)
- **Auth:** `Authorization: Bearer <key>` — the master key (admin) or a virtual key validated by the LiteLLM proxy (see below). Fails closed.
- **Health:** `GET /health/readiness`, `GET /health/liveness`, `GET /health/gil`

> **Realtime serving is pure Rust.** Python is used at **load time only** — to
> read the config once at boot. The realtime hot path never touches Python.

## Authentication — delegated to the LiteLLM proxy

The gateway does **not** re-implement key validation. It delegates virtual-key auth
to the LiteLLM proxy's FastAPI **control plane**, so keys, budgets, rate limits, and
route/model permissions are enforced in exactly one place. A client presents
`Authorization: Bearer <key>` on the `/v1/realtime` upgrade:

- **Master key** (`LITELLM_MASTER_KEY`) → treated as proxy admin, checked locally
  (constant-time compare). No network call.
- **Virtual key** (`sk-…`) → the gateway POSTs `{api_key, route, model}` to the proxy's
  **`POST /internal/v1/auth/verify`**, which runs the proxy's real `user_api_key_auth`
  (key lookup, expiry, budget, rate-limit, and route **+ model** permissions via
  `can_key_call_model`) and returns the resolved identity. The answer is cached
  in-memory (bounded ≤200 entries, short TTL), so it is not a round-trip per request.

Data plane → control plane is itself authenticated with a **dedicated data-plane key**
(NOT the master key — least privilege): the gateway sends
`X-LiteLLM-Data-Plane-Key: $LITELLM_DATA_PLANE_KEY` and the proxy rejects any call
without it. Set the **same** secret on both sides.

```text
client ──Bearer sk-…──▶ ai-gateway ──POST /internal/v1/auth/verify──▶ LiteLLM proxy
                         (X-LiteLLM-Data-Plane-Key)                    user_api_key_auth()
                         cache ≤200 / TTL                              → UserAPIKeyAuth | 401
```

### Wiring it up

On the **LiteLLM proxy** (control plane) — expose the verify endpoint:

```bash
export LITELLM_DATA_PLANE_KEY=<dedicated-secret>   # a NEW secret, not the master key
litellm --config proxy_config.yaml                 # serves POST /internal/v1/auth/verify
```

On the **gateway** (data plane) — point it at the proxy:

```bash
export LITELLM_DATA_PLANE_KEY=<same-secret>
export LITELLM_AUTH_VERIFY_URL=https://<proxy-host>/internal/v1/auth/verify
./litellm-ai-gateway
```

Fails closed: a missing/wrong data-plane key, an unreachable proxy, or a rejected key
all yield `401`. Revocation and budget changes take effect within the cache TTL
(`LITELLM_AUTH_CACHE_TTL_SECS`, default 60s). Keep the proxy on a private network —
the verify endpoint is internal-only and excluded from the public OpenAPI spec.

## Configuration (config.yaml)

The gateway loads its `model_list` from a **config.yaml**, the same as the
LiteLLM proxy. Point `LITELLM_CONFIG_PATH` at the file:

```yaml
# config.yaml
model_list:
  - model_name: gpt-realtime
    litellm_params:
      model: openai/gpt-realtime
      api_key: os.environ/OPENAI_API_KEY
```

```bash
LITELLM_CONFIG_PATH=./config.yaml ./litellm-ai-gateway
```

At boot the gateway calls into `litellm.proxy.read_model_list`, which reuses the
**real proxy config reader** (`ProxyConfig.get_config`). That means everything
the proxy supports in config.yaml works here too:

- `include:` to merge in other config files,
- `os.environ/VAR` secret references (resolved via the secret manager, never
  inlined),
- DB-stored models (when a database is configured).

Secrets stay out of the config — reference them with `os.environ/...` and set
the env var at deploy time. The shipped Docker image is built with the
`python-config` feature and **bundles litellm**, so config loading works out of
the box; the default baked config lives at `/app/config.yaml` and can be
overridden at deploy time (e.g. a Render secret file mounted at the same path).

### Environment variables

| Var | Required | Default | Purpose |
|---|---|---|---|
| `LITELLM_CONFIG_PATH` | yes (config mode) | — | Path to the config.yaml the gateway loads its `model_list` from. The Docker image defaults this to `/app/config.yaml`. |
| `LITELLM_MASTER_KEY` | yes | — | Admin bearer token (checked locally, no proxy call). Unset ⇒ the master-key path is disabled. |
| `LITELLM_DATA_PLANE_KEY` | for virtual keys | — | Dedicated secret the gateway sends as `X-LiteLLM-Data-Plane-Key` to authenticate itself to the proxy's verify endpoint. **Must match the proxy's `LITELLM_DATA_PLANE_KEY`.** Not the master key. |
| `LITELLM_AUTH_VERIFY_URL` | for virtual keys | `http://localhost:4000/internal/v1/auth/verify` | The proxy's verify endpoint the gateway delegates virtual-key auth to. |
| `LITELLM_AUTH_CACHE_TTL_SECS` | no | `60` | TTL for the gateway's in-memory verified-key cache. Lower = faster revocation, more proxy calls. |
| `OPENAI_API_KEY` | yes | — | Upstream OpenAI key. Referenced by config.yaml as `os.environ/OPENAI_API_KEY` for the gateway→OpenAI dial. |
| `HOST` | no | `127.0.0.1` | **Set to `0.0.0.0` in any container/deploy** or external traffic is refused. |
| `PORT` | no | `4001` | Listen port. Render and most PaaS inject this automatically. |

> Secrets (`LITELLM_MASTER_KEY`, `LITELLM_DATA_PLANE_KEY`, `OPENAI_API_KEY`) are never
> baked into the image or `render.yaml` — inject them at deploy time only.

### Lean env stand-in (fallback)

If the binary is built **without** `python-config` (default features), or
`LITELLM_CONFIG_PATH` is unset, the gateway falls back to a single-deployment
stand-in built from the environment:

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_REALTIME_MODEL` | `gpt-realtime` | The single deployment's model name (also the `?model=` clients pass). |

This mode links no libpython and needs no config file, but it only supports one
hard-coded OpenAI deployment. **config.yaml is the recommended path** — use the
stand-in only for the leanest possible build.

## Build & run with Docker

The image is built `--features python-config` and installs litellm **from this
repo's source** (the config reader is newer than any PyPI release), so the build
**context is the repo root**:

```bash
# from the repo root
docker build -f litellm-rust/crates/ai-gateway/Dockerfile -t litellm-ai-gateway .

docker run --rm -p 4001:4001 \
  -e HOST=0.0.0.0 -e PORT=4001 \
  -e LITELLM_MASTER_KEY=sk-local \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  litellm-ai-gateway          # LITELLM_CONFIG_PATH defaults to /app/config.yaml

# smoke test
curl -s -o /dev/null -w '%{http_code}\n' localhost:4001/health/readiness   # -> 200
curl -s -o /dev/null -w '%{http_code}\n' localhost:4001/v1/realtime         # -> 401 (auth fails closed)
```

On boot you should see `loaded model_list from /app/config.yaml via python
config reader` — that confirms the config path (not the env stand-in fallback).
To use your own config, mount it over the default:

```bash
docker run --rm -p 4001:4001 \
  -e HOST=0.0.0.0 -e LITELLM_MASTER_KEY=sk-local -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/my-config.yaml:/app/config.yaml:ro \
  litellm-ai-gateway
```

### Cargo-only (no Docker)

```bash
# config.yaml mode — needs litellm importable in the active python env
LITELLM_CONFIG_PATH=./crates/ai-gateway/config.yaml \
  cargo run --release -p litellm-ai-gateway --features python-config

# env stand-in mode — no python, no config
cargo run --release -p litellm-ai-gateway
```

## Deploy on Render

The service is a Docker **web service**; Render terminates TLS and supports
WebSockets, so the public endpoint is `wss://<service>.onrender.com/v1/realtime`.

### Option A — Blueprint (`render.yaml`)

`crates/ai-gateway/render.yaml` describes the service (Docker runtime,
`healthCheckPath: /health/readiness`, repo-root `dockerContext: .`,
`dockerfilePath: ./litellm-rust/crates/ai-gateway/Dockerfile`,
`LITELLM_CONFIG_PATH: /app/config.yaml`). `LITELLM_MASTER_KEY`,
`LITELLM_DATA_PLANE_KEY`, `LITELLM_AUTH_VERIFY_URL`, and `OPENAI_API_KEY` are
`sync: false` — set them in the dashboard after the first deploy (the data-plane
key + verify URL wire virtual-key auth to the proxy; see **Authentication**
above). To use a non-default model_list, mount a **Render Secret File** at
`/app/config.yaml`. Point a Render Blueprint at this repo/branch and apply.

### Option B — Render API

```bash
# create a Docker web service from this repo+branch, then set env vars:
curl -X POST https://api.render.com/v1/services \
  -H "Authorization: Bearer $RENDER_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "type": "web_service", "name": "litellm-rust-ai-gateway",
    "ownerId": "<owner-id>", "repo": "https://github.com/BerriAI/litellm",
    "branch": "<branch-with-this-dockerfile>",
    "serviceDetails": {
      "env": "docker",
      "envSpecificDetails": {
        "dockerfilePath": "./litellm-rust/crates/ai-gateway/Dockerfile",
        "dockerContext": "."
      },
      "healthCheckPath": "/health/readiness"
    }
  }'
# then set env vars LITELLM_MASTER_KEY, OPENAI_API_KEY, HOST=0.0.0.0,
# LITELLM_CONFIG_PATH=/app/config.yaml, and (for virtual-key auth)
# LITELLM_DATA_PLANE_KEY + LITELLM_AUTH_VERIFY_URL
```

Health check path **must** be `/health/readiness`. `autoDeploy` is off by default
in the blueprint — trigger deploys manually (or flip it on) to pick up new commits.

## Scaling

Concurrency is what matters, not total connections: each in-flight session holds
one client socket + one upstream socket. To scale, raise the instance count /
enable autoscaling on the Render service (e.g. baseline 10, max 100). Each
instance needs file descriptors for `2 × peak_concurrent_sessions` — raise
`ulimit -n` if you push very high concurrency.

## Latency note

The gateway adds the cost of one extra hop: client→gateway, then a fresh
gateway→OpenAI realtime handshake (TLS + WS upgrade + `session.created`). In
benchmarks this is ~100–150 ms of added session-establishment time; first-audio
and steady-state streaming add no measurable overhead. To minimize it, deploy the
gateway in the Render region with the lowest RTT to OpenAI's realtime endpoint.
