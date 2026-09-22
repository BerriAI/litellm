Native OCR uses `SecretSource` with `EnvironmentSecrets`, preserving process-environment reads. Readable Python secret managers still make OCR decline to the existing Python implementation. `ResolvedSecrets` and the separate `secret_manager_binding()` snapshot are inactive foundations for a later rollout

Cache and secret-manager catalog entries remain Python-only, including when `LITELLM_RUST=1`. The new cache runtime is not connected to SDK or gateway caching

OCR provider requests use the shared `litellm-http` pool. AWS and Google secret-manager SDK clients keep their SDK transports, which do not yet inherit the pool's proxy, TLS, certificate, timeout, or observability configuration. Preserve those SDK transports and configure them equivalently instead of forcing them through reqwest
