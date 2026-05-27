# Error-Handling Standards (Observability)

> Status: **PROPOSED** (Ref: LIT-3196). This document captures the standards.
> Adoption is phased — see [Rollout timeline](#rollout-timeline) for what is
> required _today_ versus aspirational. Reviewers should treat the rules as
> guidance until the corresponding phase milestone is announced.

## Why this exists

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

## Standards

These are the four hard rules:

### S1 — `error.message` MUST be a low-cardinality, normalized template

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

### S2 — Variable detail goes on attributes, not in the message

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

### S3 — Provider response bodies live on `error.detail`, never in the message

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

### S4 — Never embed secrets, end-user content, or customer identifiers

Provider exceptions that echo the inbound request (Bearer headers,
prompts, file uploads, customer email) MUST be redacted before being
attached to either `error.message` or `error.detail`. Use the existing
`redact_message_input_output_from_logging` helper as the redaction
boundary; do not invent ad-hoc regexes inside provider transformations.

## Provider-attributed error pattern

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

## Enforcement (Greptile + Semgrep)

Two enforcement points. Both are advisory until phase M3 (see timeline).

### Greptile review rule

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

### Semgrep rule (proposed, not yet active)

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
      standard: ERROR_HANDLING_STANDARDS.md#standards
```

## Rollout timeline

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

## Affected classes

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

## Open questions for human review

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
