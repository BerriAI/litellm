# Cache boundary

This folder owns how Rust inference reaches the selected cache: global cache selection, route admission, delegation to a Python cache and the experimental V2 native handles. Cache algorithms, storage protocols and response-cache semantics belong to their cache crates

`mod.rs` exposes the cache boundary to routes and module registration; adapter directories remain private. `selection.rs` owns global cache selection, route admission and inference protocol composition for both adapters

`python/` delegates operations to the selected Python cache without discovering configuration. `native/` owns native backend construction, configuration projection, facade validation, embedding and storage bindings, including experimental V2 handles. Neither adapter depends on shared selection or the other adapter. Shared composition depends on the adapters, and routes use only the parent module's exports

Keys follow the selected storage, as `litellm-cache-response/AGENTS.md` describes. `python/` asks the Python cache for its key and does not project one in Rust. `native/` builds keys through `litellm-cache-response` with the same participation rules Python uses

`SemanticExecution` belongs here because its steps select cache operations and invoke the Python embedder. Use the shared `Execution` handle and inline lifecycle driver; do not duplicate coroutine state validation, runtime waiting or GIL machinery. Python embedding awaits stay in the caller's task, and cancellation must prevent later backend or batch operations from starting

Resolved asyncio Future construction is generic host machinery. Use `litellm-host-python::ready_future` with an already constructed Python value. Keep cache-specific conversion and disabled-cache return values here. Preserve the Future-returning API and running-loop requirement

Tests for Future mechanics belong in `host-python`; tests for disabled-cache values, embedding failure policy, batch sequencing and cancellation belong with this cache adapter. Assert public behavior rather than the location or name of a helper
