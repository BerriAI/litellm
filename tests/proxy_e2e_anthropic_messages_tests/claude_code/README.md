# Claude Code LiteLLM E2E Test

This V0 test validates the customer-critical Claude Code -> LiteLLM -> Anthropic path in CircleCI.

## What It Covers

- Builds/runs the LiteLLM proxy container from the current checkout.
- Starts an isolated Postgres container for proxy database state.
- Builds a separate Claude Code client container.
- Runs Claude Code as a non-root user with only the LiteLLM proxy key.
- Points Claude Code at LiteLLM via `ANTHROPIC_BASE_URL`.
- Verifies the configured Anthropic model is visible from `/v1/models`.
- Sends back-to-back Claude Code prompts through LiteLLM and validates both responses.
- Sends a direct Anthropic Messages API request through LiteLLM and checks proxy response headers:
  - `x-litellm-call-id`
  - `x-litellm-response-cost`

The direct `/v1/messages` header check gives a clear proxy-side signal that LiteLLM handled and accounted for the request, instead of only proving the client printed text.

## Claude Code Version Policy

The Claude Code image resolves `@anthropic-ai/claude-code@latest` by querying npm publish times, selecting the newest version older than three days, then installing that exact version.

This intentionally follows the newest Claude Code release that has aged at least three days, which catches customer-facing compatibility issues while avoiding just-published releases that are more likely to be pulled or patched.

Set `CLAUDE_CODE_VERSION=<version>` to test a specific version locally or in a follow-up CI job.

## Running Locally

Set `ANTHROPIC_API_KEY` in the environment or in `tests/proxy_e2e_anthropic_messages_tests/claude_code/.env`, then run:

```bash
tests/proxy_e2e_anthropic_messages_tests/claude_code/run_claude_code_docker_test.sh
```

Useful overrides:

```bash
MODEL_NAME=claude-sonnet-4-6
LITELLM_UPSTREAM_MODEL=anthropic/claude-sonnet-4-6
CLAUDE_CODE_IMAGE=litellm-claude-code-client:local
CLAUDE_CODE_VERSION=latest
KEEP_CONTAINERS=1
```

In CircleCI, the job sets `LITELLM_IMAGE=litellm-docker-database:ci` and `LITELLM_SKIP_BUILD=true` so the test uses the LiteLLM image already built from the checked-out commit.
The CircleCI job now has explicit component stages before test execution:

- verify LiteLLM image exists (`litellm-docker-database:ci`)
- use prebuilt Claude Code client image (`claude-code-client:ci`)
- run end-to-end test with both image builds skipped (`LITELLM_SKIP_BUILD=true` and `CLAUDE_CODE_SKIP_BUILD=true`)

## Intentionally Left Out Of V0

- Bedrock and other providers. Those belong in V1 after the Anthropic path is stable.
- Back-to-back session behavior. V0 proves the request path works; V1 can add repeated requests and long-running session checks.
- Tool-use assertions. Claude Code can invoke tools in richer scenarios, but V0 keeps the signal focused on proxy compatibility and real Anthropic request success.
- A Claude Code version matrix. The Dockerfile supports `CLAUDE_CODE_VERSION`, but the default CI path tests one recent version to keep cost and flake surface low.

## Failure Output

The runner prints each phase, dumps LiteLLM logs if the proxy fails readiness, and emits a JUnit result in CircleCI. Header-check failures print the response headers and body so regressions are visible without SSHing into the job.
