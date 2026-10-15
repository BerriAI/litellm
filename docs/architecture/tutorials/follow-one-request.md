# Tutorial: follow one request through the proxy

Quadrant: tutorial. Tested on `main` at `13118948` (2026-10-15) with a mock model, so it costs nothing and needs no provider key

By the end you will have sent one chat completion through a local proxy and matched each stage of [03-data-plane-request-pipeline.md](../explanation/03-data-plane-request-pipeline.md) to a line in the debug log. That makes the rest of the series much easier to read, because every diagram stops being abstract

## 1. Install the proxy

From a checkout of `main`

```bash
uv venv .venv-tutorial
VIRTUAL_ENV=.venv-tutorial uv pip install -e '.[proxy]'
```

## 2. Write a config with a mock model

`mock_response` makes the SDK return a canned answer instead of calling the provider, while every other stage still runs

```bash
cat > /tmp/tutorial.yaml <<'EOF'
model_list:
  - model_name: demo
    litellm_params:
      model: openai/gpt-demo
      api_key: fake
      mock_response: "hello from the mock"
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
EOF
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)"
```

Don't put a literal like `sk-1234` in the config. The proxy refuses to start with a publicly known master key and prints how to fix it (`proxy/auth/master_key_boot_check.py`)

## 3. Start the proxy with debug logging

```bash
.venv-tutorial/bin/litellm --config /tmp/tutorial.yaml --port 4000 --detailed_debug 2>&1 | tee /tmp/tutorial.log
```

Watch the startup lines. You'll see the config being applied (`proxy_server.py:7116` background health check settings), the Router being built (`router.py:1679 - Routing strategy: simple-shuffle`), and proxy hooks that need a DB being skipped (`utils.py:1418 - Skipping proxy hook managed_files: it requires a database`). That last one is flow G in [user-flows-control-plane.md](../reference/user-flows-control-plane.md) running without Postgres

## 4. Send one request

In another terminal

```bash
curl -s localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"hi"}]}' -D -
```

You should get `200 OK`, the body `hello from the mock`, and a block of `x-litellm-*` headers including `x-litellm-model-group: demo`, `x-litellm-response-cost-original: 0.0` and `x-litellm-attempted-retries: 0`

## 5. Read the log against the stages

Open `/tmp/tutorial.log` and find these lines in order. The `file.py:line` prefix on each log line is the fastest way into the code

| Stage | Log line to look for | What just happened |
|---|---|---|
| 2 Auth | `user_api_key_auth.py:2588 - api key not found in cache` | Your master key was checked. A virtual key would show the cache lookup and DB fallback here |
| 2 Auth | `auth_checks.py:576 - Model demo has undefined cost` | `common_checks` looked at model access and budget |
| first request only | `__init__.py:158 - Discovered 41 guardrail translation mappings` | The unified guardrail layer scanned `litellm/llms/*/guardrail_translation` on first use |
| 3 Pre-call | `litellm_pre_call_utils.py:2664 - [PROXY] returned data from litellm_pre_call_utils` | `add_litellm_data_to_request` enriched the body with metadata |
| 3 Pre-call | `utils.py:2470 - Inside Proxy Logging Pre-call hook!` then `parallel_request_limiter_v3.py` and `cache_control_check.py` | `pre_call_hook` ran the built-in `PROXY_HOOKS` |
| 4 Route | `route_llm_request.py:445 - SESSION REUSE` | `route_request` attached the shared HTTP session and called the Router |
| 4 Route | `router.py:8147 - Inside async function with retries` then `router.py:13506 - initial list of deployments` and `cooldown_handlers.py:575` | Retries wrapper, deployment list, cooldown filter, strategy |
| 5 SDK | `utils.py:550 - litellm.acompletion(...)` and `main.py:589` | The `@client` wrapper logged the call and `main.acompletion` started |
| 6 Cost | `cost_calculator.py:1553 - selected model name for cost calculation` then `litellm_logging.py:1968 - response_cost: 0.0` | `response_cost_calculator` ran and the cost went into `_hidden_params` |
| 4 Route | `router.py:4059 - litellm.acompletion(model=openai/gpt-demo) 200 OK` | Control is back in the Router |
| 7 Post-call | `common_request_processing.py:1461 - response model mismatch ... Overriding` | The proxy put the requested model name `demo` back on the response |
| 6 Logging | `utils.py:550 - Async Wrapper: Completed Call, calling async_success_handler` followed by one `callback_controls.py:37 - Checking if <...> is disabled` per callback | Success logging fanned out to every object in the callback list |

## 6. Notice what's missing

There is no `_ProxyDBLogger` in the callback list, because there is no database, so spend was computed (`response_cost: 0.0`) but never queued. Add `database_url` under `general_settings` and run it again to see `proxy_track_cost_callback.py` and `db_spend_update_writer.py` appear after the success handler. That is flow J

Notice also that a `router.py:2022 - routing_group=default` line appears before the pre-call hook as well as during routing. The proxy asks the Router about the model group early, for pre-call fallbacks, and again when it routes. It's a small example of the routing ladder described in gap G-P4 of [gap-register.md](../reference/gap-register.md)

## Where to go next

Send the same mock model a Messages request

```bash
curl -s localhost:4000/v1/messages \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"demo","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'
```

The log now opens with `_lazy_features.py:510 - Lazy-loaded optional feature 'anthropic_passthrough'`, because `/v1/messages` didn't exist until this request. Then the Router takes its generic path (`router.py:5524 - Inside ageneric_api_call_with_fallbacks()`) instead of the chat path you saw before

What you won't see is the Messages -> Responses bridge from flow B. The Messages handler short-circuits `mock_response` (`llms/anthropic/pass_through/messages/handler.py:558`) before it picks a provider branch, and returns a fixed Claude-shaped body with made-up usage, so the cost step then looks up a Claude model instead of `openai/gpt-demo`. Mocks behave differently on each API surface. To watch the real bridge, swap the mock for a real `openai/` model and key and compare the log with flow B in [user-flows-data-plane.md](../reference/user-flows-data-plane.md)
