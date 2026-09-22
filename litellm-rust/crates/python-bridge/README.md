Native OCR uses the shared `litellm_secrets::source::SecretSource`. Unreadable or unconfigured managers use `EnvironmentSecrets`. Readable managers decline to Python unless the Rust secret-manager binding is enabled. When enabled, `ResolvedSecrets` delegates existing Python clients through the Python handler and shared Rust resolver. Synchronous transformations receive explicit snapshots without implicit environment reads for undeclared names

Cache and secret-manager catalog entries remain Python-only, including when `LITELLM_RUST=1`. The new cache runtime is not connected to SDK or gateway caching

OCR provider requests use the shared `litellm-http` pool. AWS and Google secret-manager SDK clients keep their SDK transports, which do not yet inherit the pool's proxy, TLS, certificate, timeout, or observability configuration. Preserve those SDK transports and configure them equivalently instead of forcing them through reqwest
