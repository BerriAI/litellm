# Live provider route tests

Each route crate owns `tests/live.rs`. Use named `rstest` provider cases for the behaviors that route supports. Shared setup lives in `litellm_inference_testing::live`, with production HTTP settings, authentication services, environment secrets and a deadline

These tests execute the Rust route against a real provider. They do not prove Python parity, proxy authorization, billing persistence or agent compatibility. Those belong in the corresponding boundary and gateway suites

All live cases use `#[ignore = "calls a real provider, requires credentials and a live model"]`. Normal Cargo runs report them as ignored. An explicitly selected case fails on missing credentials, unavailable models, provider errors or timeouts. Never return early and count missing coverage as a pass

Set `LITELLM_LIVE_<ROUTE>_<PROVIDER>_MODEL` to the latest model available to the test account. The model has no compiled default because vendor availability changes independently of LiteLLM. Credentials use the provider's existing configuration rather than a second test-only key convention. Export credentials in the shell before running tests. The harness does not load `.env` or modify process environment while tests run

## Messages

Completion checks text and usage. Streaming consumes SSE to completion and checks text, usage and the terminal event. Tool round trips request a tool call and send its returned ID back with a result. Each case checks provider execution facts and rejects cache results

Messages setup and assertions live in `inference-messages/tests/live/support.rs`. Provider modules under `tests/live/` select their payloads and tool policy. The shared helpers consume streams completely and reconstruct text, thinking signatures and tool arguments for the follow-up request

The initial provider cases are `vertex_ai`, `edenai`, `openrouter` and `github_copilot`. A registered ignored case is available coverage, not evidence that it passed against that provider

Run one provider's cases explicitly

```bash
LITELLM_LIVE_MESSAGES_VERTEX_AI_MODEL=<claude-model-on-vertex> \
  cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-inference-messages --test live vertex_ai \
  -- --ignored --nocapture --test-threads=1
```

For Vertex AI, set `LITELLM_LIVE_MESSAGES_VERTEX_AI_MODEL` to a Claude model published in the project's location. Credentials, project and location resolve through the production Rust GCP authentication, from `GOOGLE_APPLICATION_CREDENTIALS` or `VERTEXAI_CREDENTIALS`, with `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION`

Vertex AI cases live in `inference-messages/tests/live/vertex_ai.rs`. They cover plain and system messages, text streaming, and complete and streaming tool round trips. Tool cases use automatic selection because current Claude models reject forced `tool_choice`. Each case still requires a tool call, checks its arguments, and sends the tool result back

For OpenRouter, set `LITELLM_LIVE_MESSAGES_OPENROUTER_MODEL` and `OPENROUTER_API_KEY`, and select `compatible_hosts` with the provider's case name. EdenAI is selected the same way

Anthropic-compatible host cases live in `inference-messages/tests/live/compatible_hosts.rs`, with EdenAI and OpenRouter as `#[values]` of every case. They cover system messages, cache hints, provider-reported cost handoff, text streaming, and complete and streaming tool round trips. The cost assertion reads the amount from the public response through the provider's own config and compares it with the settled execution facts instead of pinning a vendor price
EdenAI and OpenRouter tool cases use automatic selection because current Claude models reject forced `tool_choice`. Each test requires a tool call and verifies its arguments and follow-up response


For GitHub Copilot, set `LITELLM_LIVE_MESSAGES_GITHUB_COPILOT_MODEL` to `github_copilot/<enabled-claude-model>`, a current model enabled on the account. Copilot uses the existing device login under `~/.config/litellm/github_copilot`, or `GITHUB_COPILOT_TOKEN_DIR`. It acquires and refreshes the inference session through the production Rust authentication service. Merely appearing in Copilot's `/models` response does not mean a model's policy is enabled

Copilot cases live in `inference-messages/tests/live/github_copilot.rs`. They cover system messages, caller credential and base isolation, text streaming, and complete and streaming tool round trips. Copilot uses automatic tool selection and the stream helper accepts its final `[DONE]` frame

Other tool cases require a model that supports forced `tool_choice`, except where a provider's section says otherwise. A provider rejection fails the selected case instead of skipping it

Select one behavior by its test name, for example `completion::case_1_basic`. To list every available cell without contacting any provider

```bash
cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-inference-messages --test live -- --list
```

For another route, reuse `LiveResources`, `model` and `within_deadline`. Keep payload construction and response assertions in that route's test file. Add only supported provider behaviors, and document any gap instead of substituting mocks or skipping failures. Run cases serially to keep spending and rate limits predictable

### Bedrock Mantle Messages

Set `AWS_BEARER_TOKEN_BEDROCK` or `BEDROCK_MANTLE_API_KEY`, or use AWS credentials for SigV4. Set `LITELLM_LIVE_MESSAGES_BEDROCK_MANTLE_MODEL` to a Claude model advertised by `/v1/models`, with the `bedrock_mantle/` prefix.

```sh
cargo test -p litellm-inference-messages --test live bedrock_mantle -- --ignored --nocapture
```

This covers completion, system prompts, streaming, and tool round trips on `/anthropic/v1/messages`.
