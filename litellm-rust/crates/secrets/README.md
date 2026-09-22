# Secret resolution

Construct `SecretManagerState::new(backend, settings)` for a configured manager or use `SecretManagerState::default()` for environment lookups. The configured backend determines its provider identity. Write-only settings and names excluded by `hosted_keys` use the environment directly. `secret_manager_would_be_consulted` follows the same routing decision as resolution

`get_secret` distinguishes a found value, confirmed absence, and a failed read. A missing value returns `Ok(None)` without consulting another source or the caller's default. Empty strings are found values. Backend failures propagate by default, unless the caller supplied a default. `.with_failure_policy(FailurePolicy::EnvironmentFallback)` instead answers failed reads from the environment, returning `None` if it is absent. This matches Python's inner error fallback. Cancellation and other Python `BaseException`s always propagate unchanged. Explicit OIDC references keep their own errors and never use these fallbacks

The getters follow Python's conversion policy. Environment values use case-insensitive, whitespace-trimmed boolean parsing. Manager strings become booleans only when Python literal evaluation yields a boolean; other strings retain their exact contents. Non-string manager results produce `None`. `get_secret_str` returns only strings, and `get_secret_bool` accepts booleans or strings containing `true` or `false`. A type mismatch returns `None` and does not activate fallback

## Route integration

Inject `Arc<dyn SecretSource>` from `litellm_secrets::source` into route preparation. `SecretResolver` implements this interface and supports arbitrary names through its asynchronous `get_secret_str`. It applies the same manager selection, conversion, and fallback policy to every lookup

For synchronous provider transformations, call `source.resolve(names).await` during preparation and inject the returned `Secrets` snapshot. A snapshot contains only those names and never reads the process environment implicitly. Resolve runtime names through the source before invoking a synchronous transformation. OCR uses this pattern; other routes can adopt it as they are implemented

The Python bridge uses this shared source and resolver. Existing Python manager clients remain explicit external callbacks through Python's handler, preserving their configured credentials and settings. This does not establish native backend execution in Python routes

## Backend contracts

AWS Secrets Manager, Azure Key Vault, Google Secret Manager, Vault, and CyberArk implement `BaseSecretManager` for reads with an operation context. Foreign provider contexts are rejected before cache access or I/O. Writes and deletes use separate `SecretWriter` and `SecretDeleter` capabilities. CyberArk supports writes but not deletion, so it cannot use the shared rotation operation that deletes an old secret

Shared rotation verifies that the replacement has the requested value before deleting the old secret. Same-name rotation keeps the replacement. Provider-specific update APIs, such as AWS version updates, remain provider-specific

Backend reads preserve payload strings. Conversion belongs to the resolver. Google caches only successfully decoded payloads, so values agree before and after caching. Confirmed absence and failed reads are not cached. Resource-not-found responses indicate absence; authentication, permission, transport, and malformed successful responses remain errors

The HashiCorp Vault backend is enabled with the `hashicorp` feature and reads KV v2 values from `HCP_VAULT_*` environment variables. It supports static tokens, AppRole authentication, and TLS certificate authentication

## Intentional differences from Python

Native backends consistently distinguish absence from failure instead of swallowing provider errors. The resolver applies the chosen failure policy

`hosted_keys` excludes a name for every backend. Python's handler recognizes Azure `SecretClient` and Google `KeyManagementServiceClient` instances before the `local` branch, allowing excluded names to reach those providers. Rust treats that as a routing bug. `test_rust_hosted_keys_exclude_azure_sdk_clients_too` in `tests/test_litellm/rust_bridge/ocr/test_secrets.py` pins this behavior
