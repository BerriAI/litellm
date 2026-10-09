# Cache boundary

This folder owns how Rust inference reaches the selected cache: global cache selection, route admission, delegation to a Python cache and the experimental V2 native handles. Cache algorithms, storage protocols and response-cache semantics belong to their cache crates

`mod.rs` exposes the cache boundary to routes and module registration; adapter directories remain private. `selection.rs` asks `litellm.rust_bridge.host.cache` for the live configured cache, and owns route admission and inference protocol composition for both adapters

`python/` delegates operations to the selected Python cache without discovering configuration. `native/` owns the experimental V2 native cache handles. Neither adapter depends on shared selection or the other adapter. Shared composition depends on the adapters, and routes use only the parent module's exports
