# Cache boundary

This folder owns Python cache API compatibility: argument projection, facade identity, public result construction, Python embedding calls and per-operation composition of native backends. Cache algorithms, storage protocols and response-cache semantics belong to their cache crates

`mod.rs` owns global cache selection and route admission. `python/` delegates operations to the selected Python cache without discovering configuration. `native/` owns native backend construction, configuration projection, facade validation, embedding and storage bindings, including experimental V2 handles. The shared `binding.rs` retains the runtime that can wrap either native storage or a Python callback

`SemanticExecution` belongs here because its steps select cache operations and invoke the Python embedder. Use the shared `Execution` handle and inline lifecycle driver; do not duplicate coroutine state validation, runtime waiting or GIL machinery. Python embedding awaits stay in the caller's task, and cancellation must prevent later backend or batch operations from starting

Resolved asyncio Future construction is generic host machinery. Use `litellm-host-python::ready_future` with an already constructed Python value. Keep cache-specific conversion and disabled-cache return values here. Preserve the Future-returning API and running-loop requirement

Tests for Future mechanics belong in `host-python`; tests for disabled-cache values, embedding failure policy, batch sequencing and cancellation belong with this cache adapter. Assert public behavior rather than the location or name of a helper
