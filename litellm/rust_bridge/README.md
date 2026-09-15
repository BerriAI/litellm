# Native bridge catalog

Every SDK API has one `NativeComponent` in the immutable `COMPONENTS` catalog. A component declares its native exports and one `CapabilitySpec`, which resolves implementation availability and rollout from `CapabilityContext(provider, model, delivery)`

`DeliveryMode` contains `COMPLETED`, `STREAMING`, and `WEBSOCKET`. Lifecycle is an implementation detail, so lifecycle and value entrypoints for the same API share the same completed-delivery policy

`RustImplementationState` records whether Rust is unimplemented, experimental, or ready. `RolloutPolicy` independently selects unsupported, Python-only, Rust opt-in, Rust opt-out, or Rust-required execution. Optional Rust execution can fall back to Python. Rust-required execution cannot

OCR completed delivery is ready and default-on. Messages and chat completions are experimental and opt-in. Responses WebSocket transport remains a separate experimental surface. The public `litellm.token_counter()` and other completed APIs remain Python-only. Bedrock transcription requires Rust because it has no Python implementation; Python-backed transcription providers remain on Python

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

Each API calls its native entrypoint at most once. Rust performs request admission inside that entrypoint before provider calls or host callbacks. An unavailable binding or `RustBridgeDeclined` permits fallback only when the resolved capability declares Python availability and the caller supplies a real Python callback. Python-capable decisions without a callback, and Rust-required decisions with one, are contract errors

Provider failures, host callback failures, cancellation, conversion failures, and response adaptation failures propagate without replay. Adaptation runs outside the decline-catching boundary

Lifecycle boundaries use `invoke_lifecycle` and `ainvoke_lifecycle`. They catch a decline only while entering the native operation; once execution starts, reserved decline or unavailable errors are terminal. The generic `invoke` helpers retain the contracts required by token counting and WebSocket transport. There is no public admission, prepare, accepts, or can-handle API

`ComponentName` identifies every API in the catalog, including token counting. The public `litellm.token_counter()` resolves `ComponentName.TOKEN_COUNTER` and executes through `invoke`. Its policy is `PYTHON_ONLY`, so environment and process overrides keep public calls on the Python implementation

The direct `litellm.rust_bridge.token_counter.count_input_tokens` adapter bypasses public rollout policy and attempts its native binding even with `LITELLM_RUST=0` or `litellm.rust(False)`. Budget reservation keeps that direct import. Missing native bindings or request bodies, unsupported inputs, and unavailable tokenizer resources use the supplied Python fallback. Unexpected failures propagate without replay

The raw-body binding remains experimental. It does not implement the synchronous public signature, so public dispatch has no native callable yet. The catalog declares the existing native export and the public Python-only policy in one component

## Package layout

Python component packages keep their descriptor in `definition.py`, dynamic call protocols in `types.py`, public selectors in `lifecycle.py`, and core-requested Python work in `host.py`. Implemented completed operations expose exactly one sync/async pair: `chat_completions`/`achat_completions`, `messages`/`amessages`, `ocr`/`aocr`, and `transcription`/`atranscription`. Their selection occurs at the public Python boundary, before the captured legacy lifecycle

Python-only completed operations expose uniform placeholder pairs that decline without inspecting arguments. Responses WebSocket and token counting keep their delivery-specific bindings. Rust mirrors these APIs below `crates/python-bridge/src/routes/`
