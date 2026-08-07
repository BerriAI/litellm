# Claude Code LiteLLM E2E Test

This test validates the customer-critical Claude Code -> LiteLLM path in CircleCI for Anthropic and Bedrock upstream providers.

## What It Covers

- Builds/runs the LiteLLM proxy container from the current checkout.
- Starts an isolated Postgres container for proxy database state.
- Builds a separate Claude Code client container.
- Runs Claude Code as a non-root user with only the LiteLLM proxy key.
- Points Claude Code at LiteLLM via `ANTHROPIC_BASE_URL`.
- Verifies the configured model alias is visible from `/v1/models`.
- Runs a comprehensive test suite including:
  - **Basic Request**: Verifies successful text generation.
  - **Tool Use**: Verifies Claude Code can use its filesystem tools (via LiteLLM) to create and read files.
  - **Error Handling**: Verifies correct failure behavior for invalid models.
- Sends a direct Anthropic Messages API request through LiteLLM and checks proxy response headers:
  - `x-litellm-call-id`
  - `x-litellm-response-cost`

The direct `/v1/messages` header check gives a clear proxy-side signal that LiteLLM handled and accounted for the request, instead of only proving the client printed text.
For Bedrock, LiteLLM still exposes an Anthropic-compatible interface to Claude Code while routing upstream via Bedrock credentials.

## Claude Code Version Policy

CircleCI runs a pinned Claude Code version set instead of rolling `latest`:

- `2.1.100`
- `2.1.90`
- `2.1.80`

This keeps CI deterministic while still exercising multiple Claude Code versions.

Set `CLAUDE_CODE_VERSION=<version>` to test a specific version locally.

## Running Locally

Set provider credentials in the environment or in `tests/proxy_e2e_anthropic_messages_tests/claude_code/.env`, then run:

```bash
tests/proxy_e2e_anthropic_messages_tests/claude_code/run_claude_code_docker_test.sh
```

Useful overrides:

```bash
MODEL_NAME=claude-sonnet-4-6
LITELLM_UPSTREAM_MODEL=anthropic/claude-sonnet-4-6
UPSTREAM_PROVIDER=anthropic
CLAUDE_CODE_IMAGE=litellm-claude-code-client:local
CLAUDE_CODE_VERSION=2.1.100
KEEP_CONTAINERS=1
```

Bedrock example:

```bash
UPSTREAM_PROVIDER=bedrock
MODEL_NAME=claude-bedrock-sonnet
LITELLM_UPSTREAM_MODEL=bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-east-1
tests/proxy_e2e_anthropic_messages_tests/claude_code/run_claude_code_docker_test.sh
```

In CircleCI, the job sets `LITELLM_IMAGE=litellm-docker-database:ci` and `LITELLM_SKIP_BUILD=true` so the test uses the LiteLLM image already built from the checked-out commit.
The CircleCI job now has explicit component stages before test execution:

- verify LiteLLM image exists (`litellm-docker-database:ci`)
- use prebuilt Claude Code client image (`claude-code-client:ci`)
- run end-to-end test with both image builds skipped (`LITELLM_SKIP_BUILD=true` and `CLAUDE_CODE_SKIP_BUILD=true`)

## Intentionally Left Out

- Vertex and Azure provider coverage. Those belong in a follow-up matrix after Anthropic + Bedrock.
- Full streaming validation (though Claude Code uses streaming by default, we don't currently assert on chunk timing/presence).
- Wider Claude Code version expansion beyond the pinned CI set (`2.1.100`, `2.1.90`, `2.1.80`).

## Failure Output

The runner prints each phase, dumps LiteLLM logs if the proxy fails readiness, and emits a JUnit result in CircleCI. Header-check failures print the response headers and body so regressions are visible without SSHing into the job.
