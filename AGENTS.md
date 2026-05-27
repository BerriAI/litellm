# INSTRUCTIONS FOR LITELLM

This document provides comprehensive instructions for AI agents working in the LiteLLM repository.

## OVERVIEW

LiteLLM is a unified interface for 100+ LLMs that:
- Translates inputs to provider-specific completion, embedding, and image generation endpoints
- Provides consistent OpenAI-format output across all providers
- Includes retry/fallback logic across multiple deployments (Router)
- Offers a proxy server (LLM Gateway) with budgets, rate limits, and authentication
- Supports advanced features like function calling, streaming, caching, and observability

## REPOSITORY STRUCTURE

### Core Components
- `litellm/` - Main library code
  - `llms/` - Provider-specific implementations (OpenAI, Anthropic, Azure, etc.)
  - `proxy/` - Proxy server implementation (LLM Gateway)
  - `router_utils/` - Load balancing and fallback logic
  - `types/` - Type definitions and schemas
  - `integrations/` - Third-party integrations (observability, caching, etc.)

### Key Directories
- `tests/` - Comprehensive test suites
- `ui/litellm-dashboard/` - Admin dashboard UI
- `enterprise/` - Enterprise-specific features

Documentation lives in the separate [BerriAI/litellm-docs](https://github.com/BerriAI/litellm-docs) repository and is served at [docs.litellm.ai](https://docs.litellm.ai).

## DEVELOPMENT GUIDELINES

### MAKING CODE CHANGES

1. **Provider Implementations**: When adding/modifying LLM providers:
   - Follow existing patterns in `litellm/llms/{provider}/`
   - Implement proper transformation classes that inherit from `BaseConfig`
   - Support both sync and async operations
   - Handle streaming responses appropriately
   - Include proper error handling with provider-specific exceptions

2. **Type Safety**: 
   - Use proper type hints throughout
   - Update type definitions in `litellm/types/`
   - Ensure compatibility with both Pydantic v1 and v2

3. **Testing**:
   - Add tests in appropriate `tests/` subdirectories
   - Include both unit tests and integration tests
   - Test provider-specific functionality thoroughly
   - Consider adding load tests for performance-critical changes

### MAKING CODE CHANGES FOR THE UI (IGNORE FOR BACKEND)

1. **Always use `antd` for new UI components — Tremor is DEPRECATED**
   - We are migrating off of `@tremor/react`. Do not introduce new `Badge`, `Text`, `Card`, `Grid`, `Title`, or other imports from `@tremor/react` in any new or modified file.
   - Use `antd` equivalents: `Tag` for labels, plain `<span>`/`<div>` with Tailwind classes (or `Typography.Text`) for text, `Card` from `antd`, etc. Note that `antd` has no `"yellow"` Tag color — use `"gold"` for amber/yellow.
   - The only exception is the Tremor Table component and its required Tremor Table sub components.

2. **Use Common Components as much as possible**:
   - These are usually defined in the `common_components` directory
   - Use these components as much as possible and avoid building new components unless needed

3. **Testing**:
   - The codebase uses **Vitest** and **React Testing Library**
   - **Query Priority Order**: Use query methods in this order: `getByRole`, `getByLabelText`, `getByPlaceholderText`, `getByText`, `getByTestId`
   - **Always use `screen`** instead of destructuring from `render()` (e.g., use `screen.getByText()` not `getByText`)
   - **Wrap user interactions in `act()`**: Always wrap `fireEvent` calls with `act()` to ensure React state updates are properly handled
   - **Use `query` methods for absence checks**: Use `queryBy*` methods (not `getBy*`) when expecting an element to NOT be present
   - **Test names must start with "should"**: All test names should follow the pattern `it("should ...")`
   - **Mock external dependencies**: Check `setupTests.ts` for global mocks and mock child components/networking calls as needed
   - **Structure tests properly**:
     - First test should verify the component renders successfully
     - Subsequent tests should focus on functionality and user interactions
     - Use `waitFor` for async operations that aren't already awaited
   - **Avoid using `querySelector`**: Prefer React Testing Library queries over direct DOM manipulation

### IMPORTANT PATTERNS

1. **Function/Tool Calling**:
   - LiteLLM standardizes tool calling across providers
   - OpenAI format is the standard, with transformations for other providers
   - See `litellm/llms/anthropic/chat/transformation.py` for complex tool handling

2. **Streaming**:
   - All providers should support streaming where possible
   - Use consistent chunk formatting across providers
   - Handle both sync and async streaming

3. **Error Handling**:
   - Use provider-specific exception classes
   - Maintain consistent error formats across providers
   - Include proper retry logic and fallback mechanisms
   - Keep `error.message` low-cardinality. Stable template in `message=`,
     variable detail (provider response body, request id, model id) belongs
     on span attributes or the `detail=` kwarg — never inlined into the
     message. Never embed secrets, prompts, or customer identifiers.
     Full rules (S1–S4), provider-attributed error pattern, and phased
     rollout plan are in [Error-handling standards (observability)](#error-handling-standards-observability)
     below.

4. **Configuration**:
   - Support both environment variables and programmatic configuration
   - Use `BaseConfig` classes for provider configurations
   - Allow dynamic parameter passing

## Error-handling standards (observability)

> Status: **PROPOSED** (Ref: LIT-3196). This document captures the standards.
> Adoption is phased — see [Rollout timeline](#rollout-timeline) for what is
> required _today_ versus aspirational. Reviewers should treat the rules as
> guidance until the corresponding phase milestone is announced.

#### Why this exists

Operators wire LiteLLM error spans into observability backends (OpenTelemetry,
Datadog, etc.) on the `error.message` / `labels.error_message` attribute.
Today that attribute is sourced from `str(exception)` and is effectively
unconstrained: provider response bodies, deployment IDs, model IDs, request
IDs, file paths, IP literals, and one-shot strings all leak into it. That
breaks observability in three concrete ways:

1. **Unbounded cardinality.** Backends that index/tag on `error.message`
   (Datadog Tags, Honeycomb derived columns, Loki labels) explode in cost
   or get silently dropped when distinct values cross provider limits.
2. **Alert flake.** "Top error" dashboards fragment one logical failure
   (`InvalidRequestError: messages.0.content[0].text: must be non-empty`)
   across thousands of distinct messages that differ only by IDs.
3. **PII / token leakage.** Provider exceptions occasionally echo the
   inbound request — including bearer tokens, end-user prompts, or
   customer email addresses — verbatim into spans.

The fix is to separate the **stable, normalized** error identity (kept in
`error.message`) from the **high-cardinality** detail (kept in event
attributes / `error.detail` / structured logs).

#### Standards

These are the four hard rules:

##### S1 — `error.message` MUST be a low-cardinality, normalized template

Acceptable:

```
InvalidRequestError: messages.<index>.content.<index>.text: must be non-empty
RateLimitError: rate limit exceeded for deployment
ContextWindowExceededError: requested tokens exceed model context window
AuthenticationError: missing or invalid API key for provider
```

Not acceptable (current state — to be migrated):

```
litellm.BadRequestError: AnthropicException - {"type":"error","error":{"type":"invalid_request_error","message":"messages: text content blocks must be non-empty"},"request_id":"req_vrtx_011CYZCM5xwFNPwMuJndurak"}. Received Model Group=claude-sonnet-4-5
litellm.AuthenticationError: AzureException AuthenticationError - The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable
```

Heuristic: if two failures of the same _kind_ produce different `error.message`
strings, the message is too detailed. Move the variable bits to attributes.

##### S2 — Variable detail goes on attributes, not in the message

Required attributes when present (already part of
[`StandardLoggingPayloadErrorInformation`](litellm/types/utils.py)):

| Attribute        | Source                                          | Notes                                              |
| ---------------- | ----------------------------------------------- | -------------------------------------------------- |
| `error.code`     | `exception.code` or `exception.status_code`    | HTTP status as string; numeric mirror on `http.response.status_code` |
| `error.type`     | `exception.__class__.__name__`                  | e.g. `ContextWindowExceededError`                  |
| `llm.provider`   | `exception.llm_provider`                        | Normalized provider id                             |
| `error.detail`   | _new_ — see S3                                  | Free-form, high-cardinality payload                |

Request id, deployment id, model id, customer id, IP, and similar volatile
fields belong as **siblings** on the span (e.g. `gen_ai.request.id`,
`litellm.deployment_id`), not inlined into `error.message`.

##### S3 — Provider response bodies live on `error.detail`, never in the message

When LiteLLM wraps a provider exception, the upstream response body and any
provider request id MUST be preserved (operators rely on them), but on a
separate attribute that observability backends can choose to drop, sample,
or index differently:

```python
raise BadRequestError(
    message="invalid request to upstream provider",   # template (S1)
    llm_provider="anthropic",
    model=model,
    # New, optional kwarg. Backends route this to error.detail / status.description / log only.
    detail=upstream_response_body,
)
```

`detail` is **not** logged into `labels.error_message`. The
OpenTelemetry integration sets it on a new `error.detail` attribute that
defaults to **off** in low-resolution exporters.

##### S4 — Never embed secrets, end-user content, or customer identifiers

Provider exceptions that echo the inbound request (Bearer headers,
prompts, file uploads, customer email) MUST be redacted before being
attached to either `error.message` or `error.detail`. Use the existing
`redact_message_input_output_from_logging` helper as the redaction
boundary; do not invent ad-hoc regexes inside provider transformations.

#### Provider-attributed error pattern

The pattern below is the target shape for all `llms/<provider>/**`
exception construction. Three things matter: stable template, structured
`detail`, and `llm_provider`.

```python
# litellm/llms/<provider>/common_utils.py
from litellm.exceptions import BadRequestError

def _raise_for_invalid_messages(response_body: dict, model: str) -> None:
    raise BadRequestError(
        message="messages: text content blocks must be non-empty",  # S1
        llm_provider="<provider>",                                  # S2
        model=model,
        # `detail` is consumed by the OTEL integration and stored under
        # `error.detail`, NOT inlined into error.message.
        detail=response_body,                                       # S3
    )
```

The corresponding OTEL span attributes will be:

```
error.type    = "BadRequestError"
error.message = "messages: text content blocks must be non-empty"   # ← low cardinality
error.detail  = "{\"type\":\"error\",\"error\":{...},\"request_id\":\"...\"}"
llm.provider  = "<provider>"
gen_ai.request.model = "<model>"
http.response.status_code = 400
```

#### Enforcement (Greptile + Semgrep)

Two enforcement points. Both are advisory until phase M3 (see timeline).

##### Greptile review rule

Greptile already reviews every PR. Add the following to the repo's
Greptile prompt (`.greptile/style.md` or equivalent — to be created in
phase M1):

```
LiteLLM error-handling standards (LIT-3196):

When reviewing changes under `litellm/llms/**`, `litellm/proxy/**`,
or `litellm/exceptions.py`:

1. Flag any `raise <SomeError>(message=f"...{variable}...")` where the
   interpolated value looks like a request id, model id, deployment id,
   customer id, user id, file path, URL, or raw provider response body.
   The variable belongs on a span attribute (or the new `detail` kwarg),
   not inside `message`.
2. Flag any `raise <SomeError>(message=str(response.text))` or
   `raise <SomeError>(message=response.json())` — those are unbounded
   provider bodies. They must be passed via `detail=` and accompanied by
   a stable templated `message`.
3. Flag any `raise` that does not set `llm_provider=` when raised from
   `litellm/llms/<provider>/**`.
4. Flag exception classes added to `litellm/exceptions.py` that omit
   the `litellm_debug_info` parameter or do not inherit from an
   `openai.<Error>` (or existing LiteLLM) base — both are required by
   downstream code paths.

Each finding should cite the specific rule (S1–S4) from
ERROR_HANDLING_STANDARDS.md so authors can self-correct without a
maintainer round-trip.
```

##### Semgrep rule (proposed, not yet active)

The repo already uses Semgrep (see `.semgrep/rules/`). The rule below
catches the most common anti-pattern; ship it advisory (`severity:
INFO`) in phase M1, promote to `WARNING` in M2, `ERROR` in M3.

```yaml
# .semgrep/rules/python/reliability/error-message-cardinality.yml
rules:
  - id: error-message-from-provider-body
    message: |
      Do not put a raw provider response body or stringified exception into
      the `message=` kwarg of a LiteLLM exception. That field is exported
      to OpenTelemetry as `error.message` and high-cardinality values break
      observability backends. Use a stable template for `message=` and pass
      the full body via `detail=`. See ERROR_HANDLING_STANDARDS.md (S1, S3).
    severity: INFO
    languages: [python]
    paths:
      include:
        - litellm/llms/
        - litellm/proxy/
    pattern-either:
      - patterns:
          - pattern-inside: |
              raise $E(message=$MSG, ...)
          - pattern: $MSG
          - metavariable-pattern:
              metavariable: $MSG
              patterns:
                - pattern-either:
                    - pattern: str($RESPONSE.text)
                    - pattern: $RESPONSE.text
                    - pattern: $RESPONSE.json()
                    - pattern: str($EXC)
      - patterns:
          - pattern-inside: |
              raise $E(message=$MSG, ...)
          - pattern: $MSG
          - metavariable-pattern:
              metavariable: $MSG
              patterns:
                - pattern: f"...{$X}..."
                - metavariable-regex:
                    metavariable: $X
                    regex: '.*(request_id|model_id|deployment_id|customer_id|user_id|api_key|file_path|url).*'
    metadata:
      category: observability
      standard: #standards-s1s4
```

#### Rollout timeline

| Milestone | Target date | Scope                                                                                                  |
| --------- | ----------- | ------------------------------------------------------------------------------------------------------ |
| **M0**    | Done        | Ship this document. No code changes required.                                                          |
| **M1**    | + 2 weeks   | Greptile prompt block live. Semgrep rule landed at `severity: INFO`. New PRs reviewed against S1–S4.  |
| **M2**    | + 6 weeks   | Add `detail=` kwarg to the seven exception classes documented in [Affected classes](#affected-classes). Migrate the top 10 highest-traffic provider error sites (Anthropic, Vertex AI Anthropic, Bedrock, Azure OpenAI, OpenAI, Cohere, Mistral) to the new pattern. Semgrep promoted to `WARNING`. |
| **M3**    | + 10 weeks  | Audit pass over `litellm/llms/**` for S1–S4 violations; track remaining sites in a checklist issue. Semgrep promoted to `ERROR` for files in the migrated set. |
| **M4**    | + 14 weeks  | Whole-tree enforcement (Semgrep `ERROR` everywhere); CI fails new violations. Document the new
                attribute schema in the OTEL integration page.                                                                       |

The dates are anchored to the day this document merges, not to a fixed
calendar date — if M1 slips, M2/M3/M4 slip with it.

#### Affected classes

The following classes in [`litellm/exceptions.py`](litellm/exceptions.py)
are the seven that need a `detail` kwarg in phase M2. They cover ~95%
of provider-wrapping call sites today:

- `AuthenticationError`
- `BadRequestError`
- `NotFoundError`
- `PermissionDeniedError`
- `RateLimitError`
- `ServiceUnavailableError`
- `InternalServerError`

`detail` is a new optional kwarg; existing call sites continue to work
unchanged because none of them pass it.

#### Open questions for human review

1. **Greptile location.** The team currently has no repo-level
   `.greptile/` config; the prompt block above assumes we will create
   `.greptile/style.md` in phase M1. If Greptile prompts live elsewhere
   (Greptile UI, separate repo), point me at the canonical location.
2. **Detail attribute mapping.** Should `error.detail` be a
   first-class OTEL attribute, or only attached to
   `Status.set_status(description=...)`? The former is queryable, the
   latter is per-OTEL-spec.
3. **Migration sequencing.** M2 lists provider order alphabetically;
   the right order is by **production error volume**. Need a dashboard
   reference (e.g. Datadog metric over last 30d) before M2 starts.

## PROXY SERVER (LLM GATEWAY)

The proxy server is a critical component that provides:
- Authentication and authorization
- Rate limiting and budget management
- Load balancing across multiple models/deployments
- Observability and logging
- Admin dashboard UI
- Enterprise features

Key files:
- `litellm/proxy/proxy_server.py` - Main server implementation
- `litellm/proxy/auth/` - Authentication logic
- `litellm/proxy/management_endpoints/` - Admin API endpoints

**Database (proxy)**: Use Prisma model methods (`prisma_client.db.<model>.upsert`, `.find_many`, `.find_unique`, etc.), not raw SQL (`execute_raw`/`query_raw`). See COMMON PITFALLS for details.

## MCP (MODEL CONTEXT PROTOCOL) SUPPORT

LiteLLM supports MCP for agent workflows:
- MCP server integration for tool calling
- Transformation between OpenAI and MCP tool formats
- Support for external MCP servers (Zapier, Jira, Linear, etc.)
- See `litellm/experimental_mcp_client/` and `litellm/proxy/_experimental/mcp_server/`

## RUNNING SCRIPTS

Use `uv run python script.py` to run Python scripts in the project environment (for non-test files).

## GITHUB TEMPLATES

When opening issues or pull requests, follow these templates:

### Bug Reports (`.github/ISSUE_TEMPLATE/bug_report.yml`)
- Describe what happened vs. expected behavior
- Include relevant log output
- Specify LiteLLM version
- Indicate if you're part of an ML Ops team (helps with prioritization)

### Feature Requests (`.github/ISSUE_TEMPLATE/feature_request.yml`)
- Clearly describe the feature
- Explain motivation and use case with concrete examples

### Pull Requests (`.github/pull_request_template.md`)
- Add at least 1 test in `tests/litellm/`
- Ensure `make test-unit` passes


## TESTING CONSIDERATIONS

1. **Provider Tests**: Test against real provider APIs when possible
2. **Proxy Tests**: Include authentication, rate limiting, and routing tests
3. **Performance Tests**: Load testing for high-throughput scenarios
4. **Integration Tests**: End-to-end workflows including tool calling

## DOCUMENTATION

- Keep documentation in sync with code changes
- Update provider documentation when adding new providers
- Include code examples for new features
- Update changelog and release notes

## SECURITY CONSIDERATIONS

- Handle API keys securely
- Validate all inputs, especially for proxy endpoints
- Consider rate limiting and abuse prevention
- Follow security best practices for authentication

## ENTERPRISE FEATURES

- Some features are enterprise-only
- Check `enterprise/` directory for enterprise-specific code
- Maintain compatibility between open-source and enterprise versions

## COMMON PITFALLS TO AVOID

1. **Breaking Changes**: LiteLLM has many users - avoid breaking existing APIs
2. **Provider Specifics**: Each provider has unique quirks - handle them properly
3. **Rate Limits**: Respect provider rate limits in tests
4. **Memory Usage**: Be mindful of memory usage in streaming scenarios
5. **Dependencies**: Keep dependencies minimal and well-justified
6. **UI/Backend Contract Mismatch**: When adding a new entity type to the UI, always check whether the backend endpoint accepts a single value or an array. Match the UI control accordingly (single-select vs. multi-select) to avoid silently dropping user selections
7. **Missing Tests for New Entity Types**: When adding a new entity type (e.g., in `EntityUsage`, `UsageViewSelect`), always add corresponding tests in the existing test files and update any icon/component mocks
8. **Raw SQL in proxy DB code**: Do not use `execute_raw` or `query_raw` for proxy database access. Use Prisma model methods (e.g. `prisma_client.db.litellm_tooltable.upsert()`, `.find_many()`, `.find_unique()`) so behavior stays consistent with the schema, the client stays mockable in tests, and you avoid the pitfalls of hand-written SQL (parameter ordering, type casting, schema drift)

8. **Do not hardcode model-specific flags**: Put model-specific capability flags in `model_prices_and_context_window.json` and read them via `get_model_info` (or existing helpers like `supports_reasoning`). This prevents users from needing to upgrade LiteLLM each time a new model supports a feature.

   **Example of BAD** (hardcoded model checks):

   ```python
   @staticmethod
   def _is_effort_supported_model(model: str) -> bool:
       """Check if the model supports the output_config.effort parameter..."""
       model_lower = model.lower()
       if AnthropicConfig._is_claude_4_6_model(model):
           return True
       return any(
           v in model_lower for v in ("opus-4-5", "opus_4_5", "opus-4.5", "opus_4.5")
       )
   ```

   **Example of GOOD** (config-driven or helper that reads from config):

   ```python
   if (
       "claude-3-7-sonnet" in model
       or AnthropicConfig._is_claude_4_6_model(model)
       or supports_reasoning(
           model=model,
           custom_llm_provider=self.custom_llm_provider,
       )
   ):
       ...
   ```

   Using helpers like `supports_reasoning` (which read from `model_prices_and_context_window.json` / `get_model_info`) allows future model updates to "just work" without code changes.

9. **Never close HTTP/SDK clients on cache eviction**: Do not add `close()`, `aclose()`, or `create_task(close_fn())` inside `LLMClientCache._remove_key()` or any cache eviction path. Evicted clients may still be held by in-flight requests; closing them causes `RuntimeError: Cannot send a request, as the client has been closed.` in production after the cache TTL (1 hour) expires. Connection cleanup is handled at shutdown by `close_litellm_async_clients()`. See PR #22247 for the full incident history.

## HELPFUL RESOURCES

- Main documentation: https://docs.litellm.ai/ (source: [BerriAI/litellm-docs](https://github.com/BerriAI/litellm-docs))
- Provider-specific docs: https://docs.litellm.ai/docs/providers/
- Admin UI for testing proxy features

## WHEN IN DOUBT

- Follow existing patterns in the codebase
- Check similar provider implementations
- Ensure comprehensive test coverage
- Update documentation appropriately
- Consider backward compatibility impact

## Cursor Cloud specific instructions

### Environment

- uv is installed in `~/.local/bin`; the update script ensures it is on `PATH`.
- Python 3.12, Node 22 are pre-installed.
- The project virtual environment lives under `.venv/`.

### Running the proxy server

Create a minimal config file and start the proxy:

```yaml
# config.yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake-model
      api_key: fake-key
      api_base: https://fake-api.example.com

general_settings:
  master_key: sk-1234

litellm_settings:
  drop_params: True
  telemetry: False
```

```bash
uv run litellm --config config.yaml --port 4000
```

The proxy takes ~15-20 seconds to fully start (it runs Prisma migrations on boot). Wait for `/health` to return before sending requests. Without a PostgreSQL `DATABASE_URL`, the proxy connects to a default Neon dev database embedded in the `litellm-proxy-extras` package.

### Running tests

See `CLAUDE.md` and the `Makefile` for standard commands. Key notes:

- `uv sync --group proxy-dev --extra proxy` installs the Prisma and proxy-side test dependencies used by the standard local workflow.
- The `--timeout` pytest flag is NOT available; don't pass it.
- Unit tests: `uv run pytest tests/test_litellm/ -x -vv -n 4`
- **Before committing, always run `uv run black .` to format your code.** Black formatting is enforced in CI.
- If `uv sync` fails because the lockfile is outdated, run `uv lock` and retry.

### Lint

```bash
cd litellm && uv run ruff check .
```

Ruff is the primary fast linter. For the full lint suite (including mypy, black, circular imports), run `make lint` per `CLAUDE.md`.

### UI Dashboard development

- The UI is at `ui/litellm-dashboard/`. Run `npm run dev` from that directory for the Next.js dev server on port 3000.
- The proxy at port 4000 serves a **pre-built** static UI from `litellm/proxy/_experimental/out/`. After making UI code changes, you must run `npm run build` in the dashboard directory and copy the output: `cp -r ui/litellm-dashboard/out/* litellm/proxy/_experimental/out/` for the proxy to serve the updated UI.
- SVGs used as provider logos (loaded via `<img>` tags) must NOT use `fill="currentColor"` — replace with an explicit color like `#000000` or use the `-color` variant from lobehub icons, since CSS color inheritance does not work inside `<img>` elements.
- Provider logos live in `ui/litellm-dashboard/public/assets/logos/` (source) and `litellm/proxy/_experimental/out/assets/logos/` (pre-built). Both locations must have the file for it to work in dev and proxy-served modes.
- UI Vitest tests: `cd ui/litellm-dashboard && npx vitest run`
