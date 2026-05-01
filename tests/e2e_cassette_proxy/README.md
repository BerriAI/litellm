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

Two changes to the job in `.circleci/config.yml`:

1. Add the `start_cassette_proxy` reusable command after your other
   sidecars (postgres, redis, etc.) and *before* you start the
   container under test:

   ```yaml
   - start_postgres
   - start_cassette_proxy
   ```

2. When you `docker run` the container under test, route its egress
   through the sidecar and trust its CA:

   ```yaml
   docker run -d \
     ...your existing env...
     -e HTTP_PROXY="$CASSETTE_PROXY_URL" \
     -e HTTPS_PROXY="$CASSETTE_PROXY_URL" \
     -e NO_PROXY="localhost,127.0.0.1,host.docker.internal" \
     -e SSL_CERT_FILE=/etc/litellm-cassette-proxy-ca.crt \
     -e REQUESTS_CA_BUNDLE=/etc/litellm-cassette-proxy-ca.crt \
     -e CURL_CA_BUNDLE=/etc/litellm-cassette-proxy-ca.crt \
     -e AWS_CA_BUNDLE=/etc/litellm-cassette-proxy-ca.crt \
     -e NODE_EXTRA_CA_CERTS=/etc/litellm-cassette-proxy-ca.crt \
     -v "$CASSETTE_PROXY_CA":/etc/litellm-cassette-proxy-ca.crt:ro \
     ...your image and command...
   ```

`e2e_openai_endpoints` is the canonical example in this PR. To opt the
others (`proxy_e2e_anthropic_messages_tests`,
`proxy_pass_through_endpoint_tests`, `e2e_ui_testing`,
`google_generate_content_endpoint_testing`,
`proxy_logging_guardrails_model_info_tests`,
`proxy_multi_instance_tests`, `proxy_spend_accuracy_tests`,
`proxy_store_model_in_db_tests`) in, copy the same two changes.

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
