Native OCR uses `litellm_secrets::source::SecretSource`. Built-in secret managers resolve to retained Rust backends. Custom Python managers and overrides keep the callback path. Readable managers still require the Rust secret-manager binding to be enabled

The shared proxy initializer captures native configuration without loading the extension or doing native I/O. `_SecretManagerRuntime.from_client` constructs a backend on first use and keeps its handle on the Python client. The secret-manager dispatcher selects Python or Rust through `catalog.py`. Native reads call that handle; Rust routes extract the backend directly. Configuration changes replace the handle, while calls already bound to the previous backend keep using it. Handles cannot be reused after fork. Directly constructed LiteLLM managers are adapted on first native use. Manually supplied SDK clients keep their Python behavior because their credentials cannot be inferred safely. Provider implementations contain no bridge registration

Retention describes ownership and lifetime. `callbacks-legacy-python::PublicCall` owns Python references for one call to preserve identity. A native cache or secret-manager handle owns shared Rust state across calls to preserve connection pools and caches. Both use existing `Py<T>` and shared Rust ownership, with execution and GIL transitions handled by `litellm-host-python`


Cache and secret-manager catalog entries remain Python-only, including when `LITELLM_RUST=1`. This wiring does not change rollout policy

OCR provider requests use the shared `litellm-http` pool. AWS and Google secret-manager SDK clients keep their SDK transports, which do not yet inherit the pool's proxy, TLS, certificate, timeout, or observability configuration. Preserve those SDK transports and configure them equivalently instead of forcing them through reqwest
