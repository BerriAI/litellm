# Cache boundary

This folder owns how Rust inference reaches the configured cache. Rust holds no cache rules: `litellm.rust_bridge.host.cache` decides whether a call may read or write, builds the key and stores the entry, using the same Python handler functions the Python route runs. Cache algorithms, storage protocols and response-cache semantics belong to their cache crates

`mod.rs` exposes the cache boundary to routes and module registration. Adapter directories remain private. `python/service.rs` owns the `Cached<P>` protocol and typed cache operations. `python/binding.rs` owns `CachedBinding<H>`, which wraps a route's Python binding, asks the Python host module for access and serves cache operations. Route bindings never hold cache state

`python/` delegates operations to the Python cache the host module granted. `native/` owns the experimental V2 native cache handles exposed to Python. Inference does not use them until the Rust cache is ready
