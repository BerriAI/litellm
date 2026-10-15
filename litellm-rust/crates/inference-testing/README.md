# Live provider route tests

Each route crate owns `tests/live.rs`. Use named `rstest` provider cases for the behaviors that route supports. Shared setup lives in `litellm_inference_testing::live`, with production HTTP settings, authentication services, environment secrets and a deadline

These tests execute the Rust route against a real provider. They do not prove Python parity, proxy authorization, billing persistence or agent compatibility. Those belong in the corresponding boundary and gateway suites

All live cases use `#[ignore = "calls a real provider, requires credentials and a live model"]`. Normal Cargo runs report them as ignored. An explicitly selected case fails on missing credentials, unavailable models, provider errors or timeouts. Never return early and count missing coverage as a pass

Set `LITELLM_LIVE_<ROUTE>_<PROVIDER>_MODEL` to the latest model available to the test account. The model has no compiled default because vendor availability changes independently of LiteLLM. Credentials use the provider's existing configuration rather than a second test-only key convention. Export credentials in the shell before running tests. The harness does not load `.env` or modify process environment while tests run

## Messages

Completion checks text and usage. Streaming consumes SSE to completion and checks text, usage and the terminal event. Tool round trips request a tool call and send its returned ID back with a result. Each case checks provider execution facts and rejects cache results

Messages setup and assertions live in `inference-messages/tests/live/support.rs`. Provider modules under `tests/live/` select their payloads and tool policy. The shared helpers consume streams completely and reconstruct text, thinking signatures and tool arguments for the follow-up request

The initial provider cases are `vertex_ai`, `edenai` and `github_copilot`. A registered ignored case is available coverage, not evidence that it passed against that provider

Run one provider's cases explicitly

```bash
LITELLM_LIVE_MESSAGES_VERTEX_AI_MODEL=<claude-model-on-vertex> \
  cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-inference-messages --test live vertex_ai \
  -- --ignored --nocapture --test-threads=1
```

For Vertex AI, set `LITELLM_LIVE_MESSAGES_VERTEX_AI_MODEL` to a Claude model published in the project's location. Credentials, project and location resolve through the production Rust GCP authentication, from `GOOGLE_APPLICATION_CREDENTIALS` or `VERTEXAI_CREDENTIALS`, with `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION`

Vertex AI cases live in `inference-messages/tests/live/vertex_ai.rs`. They cover plain and system messages, text streaming, and complete and streaming tool round trips. Tool cases use automatic selection because current Claude models reject forced `tool_choice`. Each case still requires a tool call, checks its arguments, and sends the tool result back

For EdenAI, set `LITELLM_LIVE_MESSAGES_EDENAI_MODEL` and `EDENAI_API_KEY`, and select `edenai`. EdenAI requires a production token for live coverage, because sandbox tokens return dummy responses

EdenAI cases live in `inference-messages/tests/live/edenai.rs`. They cover system messages, portable cache hints, provider-reported cost handoff, text streaming, and complete and streaming tool round trips. The cost assertion compares the public response with the execution facts instead of pinning a vendor price
EdenAI tool cases use automatic selection because current Claude models reject forced `tool_choice`. The test still requires a tool call and verifies its arguments and follow-up response

For GitHub Copilot, set `LITELLM_LIVE_MESSAGES_GITHUB_COPILOT_MODEL` to `github_copilot/<enabled-claude-model>`, a current model enabled on the account. Rust never acquires Copilot credentials, Python does, so set `LITELLM_LIVE_GITHUB_COPILOT_TOKEN` and `LITELLM_LIVE_GITHUB_COPILOT_API_BASE` to the `token` and `endpoints.api` of an unexpired session, e.g. from `~/.config/litellm/github_copilot/api-key.json` after any Python Copilot call refreshed it. Merely appearing in Copilot's `/models` response does not mean a model's policy is enabled

Copilot cases live in `inference-messages/tests/live/github_copilot.rs`. They cover system messages, caller credential and base isolation, text streaming, and complete and streaming tool round trips. Copilot uses automatic tool selection and the stream helper accepts its final `[DONE]` frame

Other tool cases require a model that supports forced `tool_choice`, except where a provider's section says otherwise. A provider rejection fails the selected case instead of skipping it

Select one behavior by its test name, for example `completion::case_1_basic`. To list every available cell without contacting any provider

```bash
cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-inference-messages --test live -- --list
```

For another route, reuse `LiveResources`, `model` and `within_deadline`. Keep payload construction and response assertions in that route's test file. Add only supported provider behaviors, and document any gap instead of substituting mocks or skipping failures. Run cases serially to keep spending and rate limits predictable
