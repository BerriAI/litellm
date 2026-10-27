# Cache boundary

This folder owns how a Python-hosted native call reaches the configured cache. Rust holds no cache rules: `litellm.rust_bridge.host.cache` decides whether a call may read or write, builds the key, picks the entry format and stores the entry, using the same Python handler functions the Python route runs. Cache algorithms, storage protocols and response-cache semantics belong to their cache crates

`mod.rs` exposes the cache boundary to routes and module registration. Adapter directories remain private. `python/binding.rs` owns `CachedBinding<H>`, which wraps a route's Python binding and answers the driver's lookup and store seams. Route bindings never hold cache state

`python/` serves the Python cache around the native machine, never from inside it. `native/` owns the experimental V2 native cache handles exposed to Python. Python-hosted inference does not attach them until the Rust cache is ready
