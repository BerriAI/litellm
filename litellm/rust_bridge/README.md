# Native bridge catalog

Every SDK API has one `NativeComponent` in the immutable `COMPONENTS` catalog. A component declares its native exports and one `CapabilitySpec`, which resolves implementation availability and rollout from `CapabilityContext(provider, model, delivery)`

`DeliveryMode` contains `COMPLETED`, `STREAMING`, and `WEBSOCKET`. Lifecycle is an implementation detail, so lifecycle and value entrypoints for the same API share the same completed-delivery policy

`RustImplementationState` records whether Rust is unimplemented, experimental, or ready. `RolloutPolicy` independently selects unsupported, Python-only, Rust opt-in, Rust opt-out, or Rust-required execution. Optional Rust execution can fall back to Python. Rust-required execution cannot

OCR completed delivery is ready and default-on. Messages, chat completions, raw-request input token counting, and Responses WebSocket transport are experimental and opt-in. The public `litellm.token_counter()` and other completed APIs remain Python-only. Bedrock transcription requires Rust because it has no Python implementation; Python-backed transcription providers remain on Python

```python
execution = COMPONENT.resolve(
    CapabilityContext(
        provider=provider,
        model=model,
        delivery=DeliveryMode.COMPLETED,
    )
)
```

Optional capabilities use the `litellm.rust(bool)` process override first, `LITELLM_RUST=1` or `LITELLM_RUST=0` second, then their catalog default. Python-only, Rust-required, and unsupported capabilities ignore overrides

## Fallback contract

Each API calls its native entrypoint at most once. Rust performs request admission inside that entrypoint before provider calls or host callbacks. An unavailable binding or `RustBridgeDeclined` selects the supplied Python fallback only when the policy allows it

Provider failures, host callback failures, cancellation, conversion failures, and response adaptation failures propagate without replay. Adaptation runs outside the decline-catching boundary

`invoke` and `ainvoke` return the native result or execute the supplied fallback directly. There is no public admission, prepare, accepts, or can-handle API

Token counting has two catalog entries. `UtilityName.TOKEN_COUNTER` declares the public `litellm.token_counter()` as Python-only, with no native exports. The public function continues to execute Python directly regardless of `litellm.rust(bool)` or `LITELLM_RUST`

`UtilityName.REQUEST_INPUT_TOKEN_COUNTER` owns the experimental raw-request optimization. Budget reservation keeps its direct `litellm.rust_bridge.token_counter.count_input_tokens` import. That adapter uses `REQUEST_COMPONENT`; `COMPONENT` describes the public API

The request optimization follows its component policy. Its one native counting entrypoint validates the tokenizer configuration and request body, obtains and caches the required tokenizer resource, then counts. Unsupported inputs decline, known resource loading failures report native unavailability, and unexpected counting failures propagate

## Package layout

Python component packages keep their descriptor in `definition.py`, dynamic call protocols in `types.py`, and entrypoint adapters in `value.py`, `lifecycle.py`, or transport modules. Rust mirrors those APIs below `crates/python-bridge/src/routes/`
