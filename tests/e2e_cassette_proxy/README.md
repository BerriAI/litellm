# e2e cassette proxy

A sidecar HTTP/HTTPS proxy that records and replays upstream responses
in Redis. Designed for CircleCI e2e jobs whose system-under-test runs
inside Docker — the in-process VCR persister at
`tests/_vcr_redis_persister.py` can't see those requests, this can.

## What it caches

Every HTTPS egress that flows through the proxy and:

- uses `GET` / `POST` / `PUT` / `PATCH` / `DELETE`
- is not destined for `localhost`, `127.0.0.1`, `host.docker.internal`,
  or any host listed in `LITELLM_E2E_CASS_PASSTHROUGH_HOSTS`
- received a `2xx` response from the upstream

is keyed on a canonical hash of `(method, scheme, host, path, sorted
query, allowlisted headers, canonical body)` and stored in Redis. On
subsequent runs, matching requests are served straight from Redis
without ever hitting the upstream.

What's intentionally *not* keyed on (so cache hits survive normal
churn):

- `Authorization`, `x-api-key`, `anthropic-api-key`, `openai-api-key`,
  `azure-api-key`, `cookie`, AWS sigv4 headers, `x-goog-api-key`,
  `x-goog-user-project` — auth rotates every run
- `User-Agent`, `x-stainless-*`, `traceparent`, `tracestate`,
  `x-request-id`, `request-id` — tracing / SDK metadata
- JSON key order or whitespace inside the request body — bodies are
  re-serialized in canonical form before hashing

## How to opt a CI job in

There are two patterns depending on where the upstream HTTP traffic
originates.

### Pattern A — System-under-test runs in a Docker container

Used by `e2e_openai_endpoints`, `build_and_test`,
`proxy_e2e_anthropic_messages_tests`, `proxy_pass_through_endpoint_tests`,
`proxy_logging_guardrails_model_info_tests`,
`proxy_spend_accuracy_tests`, `proxy_build_from_pip_tests`,
`proxy_store_model_in_db_tests`.

Two-step opt-in. Add the two reusable commands after your other
sidecars (postgres, redis, etc.) and *before* you start the SUT
container, then splice `$CASSETTE_PROXY_DOCKER_ARGS` into the
`docker run`:

```yaml
- start_postgres
- start_cassette_proxy
- export_cassette_proxy_docker_args
- run:
    name: Run Docker container
    command: |
      docker run -d \
        ...your existing env...
        $CASSETTE_PROXY_DOCKER_ARGS \
        --name my-app \
        ...your image and command...
```

`$CASSETTE_PROXY_DOCKER_ARGS` is a single string composed by
`export_cassette_proxy_docker_args` that sets every Python / curl /
boto3 / node trust-store env var, the proxy URL, and `AIOHTTP_TRUST_ENV`
in one shot.

### Pattern B — pytest runs in-process (no Docker container hosting the SUT)

Used by `llm_translation_testing`, `realtime_translation_testing`,
`agent_testing`, `guardrails_testing`,
`google_generate_content_endpoint_testing`,
`llm_responses_api_testing`, `ocr_testing`, `search_testing`,
`litellm_mapped_enterprise_tests`, `batches_testing`,
`litellm_utils_testing`, `pass_through_unit_testing`,
`image_gen_testing`, `logging_testing`, `audio_testing`,
`local_testing_part1`, `local_testing_part2`,
`langfuse_logging_unit_tests`.

Two-step opt-in. After install, before the test step:

```yaml
- run:
    name: Install Dependencies
    command: |
      uv sync --frozen --all-groups --all-extras --python 3.12
- start_cassette_proxy
- enable_cassette_proxy_for_pytest
- run:
    name: Run tests
    command: |
      uv run --no-sync python -m pytest ...
```

`enable_cassette_proxy_for_pytest` patches certifi's bundled
`cacert.pem` in every venv on the runner, exports
`HTTPS_PROXY` / `NO_PROXY` / `SSL_CERT_FILE` for every subsequent step,
and crucially flips `AIOHTTP_TRUST_ENV=true` — without that flag
litellm's aiohttp transport ignores the proxy variables (see
`litellm/llms/custom_httpx/http_handler.py`).

### Why `NO_PROXY` includes the Redis host

`mitmproxy` only proxies HTTP/HTTPS. The Redis client's TCP+TLS
connection to the project's managed Redis would be broken if it were
sent through the proxy. Both opt-in commands automatically append
`$REDIS_HOST` to `NO_PROXY` when it's set in the env.

## Knobs

| Env var | Where set | Effect |
|---|---|---|
| `LITELLM_E2E_CASS_REDIS_URL` | sidecar container | Override the Redis URL the sidecar uses to store cassettes. Falls back to `REDIS_URL` / `REDIS_SSL_URL` / `REDIS_HOST + REDIS_PORT + REDIS_PASSWORD`. |
| `LITELLM_E2E_CASS_PASSTHROUGH_HOSTS` | sidecar container | Extra hosts (comma-separated) to never cache. |
| `LITELLM_E2E_CASS_RECORD_ONLY` | sidecar container | When `1`, never serve from cache; always forward + persist. Use during cassette refresh runs. |
| `LITELLM_E2E_CASS_REPLAY_ONLY` | sidecar container | When `1`, never forward to upstream; serve `599` on miss. Use to *prove* a job is fully cached. |

## Why mitmproxy and not vcrpy

vcrpy is a Python in-process monkey-patch on `httpx`/`aiohttp`/`httpcore`.
It can only intercept HTTP traffic in the *same process* where it was
installed. Every CI job that runs the LiteLLM proxy in a Docker container
issues its upstream traffic from that container's process — not the
pytest process — so vcrpy literally has no hook to attach to.

A network-level recording proxy is language-agnostic, in-process-agnostic,
and transport-agnostic. It works for the LiteLLM proxy (Python aiohttp),
for the websocket realtime tests (a separate server), for the OpenAI SDK
(node fetch in some paths), and for any future containerized e2e job
without any changes to the SUT.

## Why one Redis key per request, not vcrpy-style cassettes

vcrpy stores an ordered list of `(request, response)` episodes per test
file. That model is the source of every footgun the
`tests/llm_translation/` recorder hit (unbounded per-key growth in
`new_episodes` mode, ordering brittleness, OOM under `noeviction`). Here
each Redis key holds exactly one `(request_summary, response)` pair,
so:

- Per-key size is bounded by one response (with an explicit
  `max_payload_bytes` ceiling on top).
- Two tests issuing the same request share the cache entry for free.
- "Order" is no longer a thing.
- "Record mode" is no longer a thing — if the entry is present, replay;
  else record.

## Refreshing cassettes

Add the `LITELLM_E2E_CASS_RECORD_ONLY=1` env var to the
`start_cassette_proxy` `docker run` flags for one CI run; every cassette
the job exercises will be re-recorded. Or wipe specific keys with
`redis-cli del litellm:e2ecass:<sha256>`.
