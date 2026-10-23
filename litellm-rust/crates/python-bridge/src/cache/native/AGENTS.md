# Native cache bindings

This directory exposes the experimental V2 native cache handles (`NativeCacheHandle`) to Python. Cache algorithms and storage protocols remain in their cache crates

Accept the cache object selected by the parent module. Do not read global `litellm.cache`, decide route admission, or select the Python cache adapter here

Keep Python-facing cache classes and method signatures stable when reorganizing modules. Native internals stay private to this directory unless the shared cache boundary or Python module registration needs them

Verify changes with the V2 cache tests using a freshly built extension. Test cache behavior, not module paths or file structure
