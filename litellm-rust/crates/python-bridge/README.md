Native OCR uses `litellm_secrets::source::SecretSource`. Registered built-in secret managers resolve to retained Rust backends. Custom Python managers and overrides keep the callback path. Readable managers still require the Rust secret-manager binding to be enabled

Secret-manager loaders capture native configuration without loading the extension or doing native I/O. `_SecretManagerRuntime.from_client` constructs a backend on first use and keeps its handle on the Python client. Python reads call that handle; Rust routes extract the backend directly. Configuration changes replace the handle, while calls already bound to the previous backend keep using it. Handles cannot be reused after fork. Manually supplied built-in SDK clients need explicit native configuration rather than inferred credentials

Retention describes ownership and lifetime. `callbacks-legacy-python::PublicCall` owns Python references for one call to preserve identity. A native cache or secret-manager handle owns shared Rust state across calls to preserve connection pools and caches. Both use existing `Py<T>` and shared Rust ownership, with execution and GIL transitions handled by `litellm-host-python`


Cache and secret-manager catalog entries remain Python-only, including when `LITELLM_RUST=1`. This wiring does not change rollout policy

OCR provider requests use the shared `litellm-http` pool. AWS and Google secret-manager SDK clients keep their SDK transports, which do not yet inherit the pool's proxy, TLS, certificate, timeout, or observability configuration. Preserve those SDK transports and configure them equivalently instead of forcing them through reqwest
