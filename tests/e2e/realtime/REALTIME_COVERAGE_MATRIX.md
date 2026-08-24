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
appears).

`test_realtime_pipecat_e2e` is a realism layer that drives the same providers
through pipecat's GA `OpenAIRealtimeLLMService` (base_url pointed at the proxy)
rather than speaking the protocol by hand. Its assertions are coarse (the tool
callback fired, assistant text was produced); the raw-websocket suite is the
source of truth. It skips unless `pipecat-ai` is installed
(`uv pip install "pipecat-ai[openai]"`).

## Provider status

| provider | model alias | status |
|----------|-------------|--------|
| openai | `openai-realtime` | covered (in gateway config) |
| gemini | `gemini-realtime` | covered (in gateway config; needs Gemini Live API access) |
| azure | `azure-realtime` | gap: add to gateway config + AZURE creds |
| vertex_ai | `vertex-realtime` | gap: add to gateway config + Vertex creds |
| bedrock | `bedrock-realtime` | gap: add to gateway config + AWS creds |
| xai | `xai-realtime` | gap: add to gateway config + XAI_API_KEY |

A provider whose alias is not present in the proxy's `/model/info` skips (skip on
environment). To enable one, add a `model_info.mode: realtime` entry under that
alias to `tests/e2e/gateway/litellm-config.yml` and give the proxy the
provider's credentials; the test then runs with no code change.

## Running

Start a proxy with the gateway config and the provider keys set in its
environment, then

```
uv run pytest tests/e2e/realtime/ -v
```

Tests skip when no proxy answers `GET /health/liveliness` at `LITELLM_PROXY_URL`
(default `http://localhost:4000`).
