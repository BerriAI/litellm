# Native cache bindings

This directory exposes the experimental V2 native cache handles (`NativeCacheHandle`) to Python. Cache algorithms and storage protocols remain in their cache crates

Inference does not select these handles. A Python `Cache` backed by one is reached through the Python cache delegation like any other backend

Keep Python-facing cache classes and method signatures stable when reorganizing modules. Native internals stay private to this directory

Verify changes with the V2 cache tests using a freshly built extension. Test cache behavior, not module paths or file structure
