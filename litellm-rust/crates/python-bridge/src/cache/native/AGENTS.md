# Native cache bindings

This directory constructs and exposes Rust cache backends to Python. It owns backend configuration projection, facade validation, native request conversion, semantic embedding integration and experimental V2 handles. Cache algorithms and storage protocols remain in their cache crates

Accept the cache object or projected configuration selected by the parent module. Do not read global `litellm.cache`, decide route admission, or select the Python cache adapter here

Keep Python-facing cache classes and method signatures stable when reorganizing modules. Native internals stay private to this directory unless the shared cache boundary or Python module registration needs them. Python embedding awaits use the existing inline lifecycle driver, preserving caller task identity and cancellation

Verify changes with the existing backend and facade tests using a freshly built extension. Test cache behavior, not module paths or file structure
