# Call lifecycle

`litellm_core::call_lifecycle` is the shared execution wrapper for LiteLLM call
types migrated to Rust. It owns lifecycle ordering, phase timing, and trace
observer calls. It must not know about OCR, chat, messages, responses,
completions, provider auth, request transforms, or response normalization.

Call-type modules own their domain behavior. For example, OCR owns document
payloads, OCR provider transforms, safe document fetch, guardrail payload shape,
callback payload shape, and provider HTTP execution.

## Runtime order

Every wrapped call runs in this order:

1. `async_pre_call_hook`
2. `async_during_call_hook`
3. provider call
4. `async_log_success_event` or `async_log_failure_event`

`async_pre_call_hook` receives the initial LiteLLM request shape. It is where
pre-call custom guardrails run.

`async_during_call_hook` converts the initial request into the provider-ready
request. It is where provider config selection, parameter mapping, auth/header
resolution, request transforms, and during-call guardrails belong.

The provider call receives only the provider-ready request. It should execute
I/O and call the provider response transform.

Success and failure callbacks receive `CallLifecycleTiming`. Callback failures
must not replace the original provider or guardrail result.

## Trace contract

The lifecycle runner records:

- full call start and end time
- `pre_call` phase timing
- `during_call` phase timing
- `provider_call` phase timing
- `success_callback` phase timing
- `failure_callback` phase timing

`CallLifecycleObserver` receives phase start and end events. The default
observer is a no-op. Future OTEL support should implement this observer instead
of editing OCR, chat, messages, responses, completions, or provider modules.

## Required shape

Each migrated call type should use this folder shape:

```text
litellm-rust/crates/ai-gateway/src/<call_type>/
  mod.rs          # thin public entrypoint
  types.rs        # public request, prepared request, provider request, response types
  prepare.rs      # model/provider/callback/guardrail setup
  hooks.rs        # CallLifecycleHooks implementation
  handler.rs      # provider I/O and response normalization
  tests.rs        # call-type lifecycle and handler tests
```

Provider transforms can live in `litellm-rust/crates/core/src/providers/...`.
Shared call-type helpers can live beside the call type, but generic lifecycle
code stays in this folder.

## Core API

The prepared request implements `CallLifecycleRequest`:

```rust
impl CallLifecycleRequest for PreparedMessagesRequest {
    fn lifecycle_context(&self) -> CallLifecycleContext {
        CallLifecycleContext::new(
            "messages",
            self.model.clone(),
            self.custom_llm_provider.clone(),
            self.litellm_call_id.clone(),
        )
    }
}
```

The call-type hooks implement `CallLifecycleHooks`:

```rust
impl CallLifecycleHooks<
    PreparedMessagesRequest,
    ProviderMessagesRequest,
    MessagesResponse,
> for MessagesLifecycleHooks {
    fn async_pre_call_hook(...) {
        // run pre-call custom guardrails against the LiteLLM request shape
    }

    fn async_during_call_hook(...) {
        // map params, validate env, transform request, run during-call guardrails
    }

    fn async_log_success_event(...) {
        // call async_log_success_event on configured custom loggers
    }

    fn async_log_failure_event(...) {
        // call async_log_failure_event without swallowing the original error
    }
}
```

The public entrypoint stays thin:

```rust
pub async fn messages(request: MessagesRequest<'_>) -> CoreResult<MessagesResponse> {
    let PreparedMessagesCall { request, hooks } = prepare_messages_call(request)?;

    CallLifecycle::default()
        .run_request(request, &hooks, execute_messages_provider_call)
        .await
}
```

Use `run_request` for new call types. Keep `run` available only for specialized
tests or existing code that already has a `CallLifecycleContext`.

## Adding a new call type

1. Add `<call_type>/types.rs`

Define the public request accepted by the bridge, the prepared request used by
the lifecycle runner, and the provider request consumed by the handler.

2. Implement `CallLifecycleRequest`

Return `call_type`, `model`, `custom_llm_provider`, and `litellm_call_id`.
Do not put provider-specific logic here.

3. Add `<call_type>/prepare.rs`

Resolve model/provider once, generate or preserve `litellm_call_id`, construct
callback and guardrail runners, and return `Prepared<CallType>Call`.

4. Add `<call_type>/hooks.rs`

Implement `CallLifecycleHooks`. Put pre-call guardrail payload construction,
provider config selection, param mapping, request transform, during-call
guardrail payload construction, and callback payload construction here.

5. Add `<call_type>/handler.rs`

Execute the provider request and normalize the provider response. Do not repeat
provider-specific transforms here; call the provider config.

6. Add tests

Cover hook order, success callback payload, failure callback payload, pre-call
guardrail blocking before provider I/O, during-call body mutation, and provider
error mapping.

## Review checklist

- Core lifecycle has no call-type or provider-specific branches
- Public call-type entrypoint only prepares and calls `run_request`
- Provider behavior lives behind provider config/transformation code
- Hook method names map to the Python custom logger and guardrail concepts
- Phase timing is recorded once in lifecycle, not separately per call type
- Callback failures never hide the original provider or guardrail error
- Tests prove the provider socket is not touched when pre-call guardrails block
