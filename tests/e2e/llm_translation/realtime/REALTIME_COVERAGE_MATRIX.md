# Realtime e2e coverage

Live tests for the proxy realtime websocket endpoint (`/v1/realtime`). One
GA-speaking websocket client drives every provider; the proxy normalizes each
provider's stream into the OpenAI GA event schema, so the same assertions hold
across providers and only the model alias changes.

## What is asserted

For each configured provider, `test_text_conversation` checks the session
lifecycle (`session.created`, then `session.update` echoed by `session.updated`),
the canonical response sequence (`response.created`, `response.output_item.added`,
through `response.done`), that the streamed deltas reconstruct a non-empty
transcript, and that `response.done` carries normalized usage.

`test_tool_call_round_trip` checks the full tool path: the model emits a
normalized `response.function_call_arguments.done` with valid JSON arguments and
a matching `function_call` output item, the test sends a `function_call_output`
back, and the follow-up response incorporates the result (the temperature 72
appears). That raw-websocket tool path is the source of truth for tool calling.

`test_pipecat_tool_smoke` is a realism layer through pipecat for openai, azure,
and gemini only (not vertex_ai: native-audio live is flaky under pipecat tool
calling while raw-ws tools pass; see pipecat-ai/pipecat#2544). Assertions are
coarse; raw-ws remains authoritative. Requires `pipecat-ai`.

Pipecat audio coverage lives in `test_realtime_pipecat_audio_e2e.py` (VAD / audio
I/O).

## Provisioning

The suite registers every provider's realtime deployment through `/model/new` at
session start (the `realtime_models` fixture) and deletes them on teardown, so it
never depends on a static or misconfigured gateway `model_list`. Each deployment
is created with `model_info.mode: realtime` and marker-unique names, and its
`litellm_params` point the credentials at `os.environ/*` refs the gateway resolves
at call time. The provider table below is the source of truth; edit `PROVIDERS` in
`realtime_client.py` to change a model or add one.

| provider | model alias | upstream model |
|----------|-------------|----------------|
| openai | `openai-realtime` | `openai/gpt-realtime-2` |
| azure | `azure-realtime` | `azure/gpt-realtime-2` (GA protocol) |
| gemini | `gemini-realtime` | `gemini/gemini-3.1-flash-live-preview` |
| vertex_ai | `vertex-realtime` | `vertex_ai/gemini-live-2.5-flash-preview-native-audio-09-2025` |

Bedrock and xai (`xai/grok-4-1-fast-non-reasoning`) are supported by the proxy but
kept commented out in `PROVIDERS` until they pass end-to-end here; re-enable them by
uncommenting their entry.

Every provider is provisioned and asserted; the suite never skips a provider. Per
`tests/e2e/CLAUDE.md` there is no sanctioned skip: the whole-suite proxy-liveness
probe hard-fails when no proxy answers, and a provider whose credentials or upstream
realtime model are missing on the gateway is likewise a hard failure, not a skip.
Give the gateway each provider's credentials to turn its tests green.

## Running

Start a proxy with the provider keys set in its environment (the suite registers
the deployments itself), then

```
uv run pytest tests/e2e/llm_translation/realtime/ -v
```

The whole suite hard-fails at setup when no proxy answers `GET /health/liveliness` at
`LITELLM_PROXY_URL` (default `http://localhost:4000`).
