# Call lifecycle

`litellm_core::call_lifecycle` executes typed SDK calls. It owns lifecycle
ordering, phase timing, exactly-once completion, and observer notification. It
does not know about providers, route payloads, callbacks, guardrails, or
transport.

## Runtime order

Every call runs in this order:

1. `CallInterceptor::before_call`
2. Route-owned provider preparation
3. `CallInterceptor::before_send`
4. Provider execution
5. `CallInterceptor::complete`

`complete` runs once for failures from any preceding phase and for successful
provider responses. It returns `()`, so instrumentation failures cannot replace
the original call result.

## Typed extension point

Each route defines a zero-sized `CallSpec` marker with three associated types:

```rust
pub enum OcrCall {}

impl CallSpec for OcrCall {
    const NAME: &'static str = "ocr";
    type BeforeCall = PreparedOcrRequest;
    type BeforeSend = ProviderOcrRequest;
    type Response = serde_json::Value;
}
```

Phase types are capability boundaries. Fields that an interceptor may change
are public. Credentials, headers, provider configuration, and endpoint identity
stay private. Read-only facts can be exposed through getters.

Native Rust extensions implement `CallInterceptor<C>`. The lifecycle boundary
uses static dispatch and native `impl Future + Send`; it does not require
`async_trait` or Tower. Dynamic dispatch remains inside callback registries,
where runtime-selected integrations require it.

## Route entrypoints

The normal route entrypoint uses `NoopCallInterceptor`:

```rust
pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    ocr_with_interceptor(request, &NoopCallInterceptor).await
}
```

Routes expose a generic variant for native interceptors and may expose a
callback convenience variant backed by `CallbackOptions`. Callback, guardrail,
authentication metadata, and tracing configuration must not be fields on the
route request.

## SDK callbacks

`litellm_core::callbacks` owns SDK callback and guardrail contracts. Route-local
callback interceptors adapt typed lifecycle phases to those dynamic contracts.
Host crates may provide concrete logger implementations and callback transport,
but the generic SDK contracts remain in core.

The Python bridge uses the normal no-op entrypoint because Python already owns
its callback lifecycle. Forwarding the same callbacks into Rust would dispatch
them twice.

## Streaming sessions

A provider future may represent an entire streaming or WebSocket session. Its
response should contain the neutral session summary needed by completion
interceptors. Event collectors accumulate state only; they do not own callback
registries or decide when terminal callbacks run.

## Review checklist

- Route request types contain only route inputs
- `CallSpec` phase types expose only intentional mutation capabilities
- Provider preparation is separate from interception
- Request signing happens after `before_send`
- `complete` runs exactly once and cannot replace the call result
- Python bridge calls remain callback-free
- Callback transport and customer I/O stay outside core
