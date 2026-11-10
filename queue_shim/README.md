# queue_shim

A small FastAPI service that sits in front of litellm and makes agents (codex, pi,
opencode, ...) transparently tolerant of GPU downtime.

Point your agent's `OPENAI_BASE_URL` at the shim instead of litellm. While a target
deployment is healthy the shim forwards the request to litellm immediately and streams
the response back. While litellm reports unhealthy, the request is enqueued in Redis
and a worker drains the queue once a target recovers.

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI-compatible. Enqueues the request, forwards it to litellm now when healthy, or holds the queue when unhealthy. |
| `GET` | `/queue` | Returns `{"pending": int, "items": [...]}` of jobs currently queued. |

The worker runs on its own process (`python -m queue_shim.worker`), polls
`litellm_base/health` every `SCAN_INTERVAL` seconds and claims jobs from the queue when
a target deployment is healthy.

## Config

All configuration is via environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LITELLM_BASE_URL` | `http://localhost:4000` | Base URL of the litellm router. |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string. |
| `CLAIM_TTL` | `300` | Seconds before a claimed job's claim expires and it can be re-claimed. |
| `POLL_INTERVAL` | `1.0` | Seconds between worker scan iterations. |
| `SCAN_INTERVAL` | `0.5` | Seconds to sleep after each worker scan. |

## Run

```bash
# app
REDIS_URL=redis://localhost:6379/0 LITELLM_BASE_URL=http://localhost:4000 \
  python -m queue_shim.app

# worker
REDIS_URL=redis://localhost:6379/0 LITELLM_BASE_URL=http://localhost:4000 \
  python -m queue_shim.worker
```

## Run the tests

```bash
python -m pytest queue_shim/tests/
```

The tests run against fakeredis and a mocked `httpx.AsyncClient`, no live Redis or
litellm required.

## Sample client

`sample_client.py` points at the shim and shows two requests, the first while litellm is
unhealthy (held, then completed by the worker) and the second immediately after.

```bash
python sample_client.py
```

## Idempotency

Each request is keyed on `sha256(model + body)`. A retry of an identical request while
the original job is in-flight or done replays the existing job instead of re-running
litellm, so duplicate agent requests are safe.
