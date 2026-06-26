# Call lifecycle

`litellm-core::call_lifecycle` is the shared execution wrapper for LiteLLM call
types that are migrated to Rust.

The runner owns ordering and timing only. Route-specific modules still own
provider selection, transforms, request payloads, callback payloads, and error
mapping.

## Order

Every wrapped call runs in this order:

1. `pre_call`
2. `during_call`
3. `provider_call`
4. `success_callback` or `failure_callback`

`pre_call` receives the initial LiteLLM request shape. `during_call` returns the
provider-ready request shape. This lets OCR expose the raw document to pre-call
guardrails, then expose the transformed provider body to during-call guardrails.
Chat, messages, responses, and future call types can use the same shape even
when their initial and provider request types are identical.

## Failure behavior

If `pre_call`, `during_call`, or `provider_call` fails, the lifecycle runner
still attempts `async_log_failure_event`. Callback logging must not replace the
original provider or guardrail error.

## Timing

The runner records wall-clock start and end times for the full call and per-phase
timings for:

- `pre_call`
- `during_call`
- `provider_call`
- `success_callback`
- `failure_callback`

`CallLifecycleObserver` receives phase start and end events. The default observer
is a no-op; future OTEL support should implement this observer instead of
changing route-level call code.

## Adding a call type

Add a small route adapter that implements
`CallLifecycleHooks<InitialReq, ProviderReq, Resp>`. Keep call-type-specific
logic in that adapter:

- build the guardrail payload for `pre_call`
- build the provider-ready request for `during_call`
- convert the response/error into the call type's callback payload

Then call `CallLifecycle::run(...)` around the provider I/O function.
