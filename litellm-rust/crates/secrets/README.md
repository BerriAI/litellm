# Secret resolution

Construct `SecretManagerState::new(backend, settings)` for a configured manager or use `SecretManagerState::default()` for environment lookups. The configured backend determines its provider identity. Write-only settings and names excluded by `hosted_keys` use the environment directly. `secret_manager_would_be_consulted` follows the same routing decision as resolution

Native resolution distinguishes a found value, confirmed absence, and a failed read. Missing values use the caller's default, while provider errors propagate. Empty strings are found values

`new_python_compatible` uses Python's environment fallback and conversion rules. Manager exceptions fall back to the environment, including custom-manager exceptions. AWS missing secrets, failed HTTP reads, missing string payloads, and missing or empty primary secrets return `None` without fallback, matching Python. The standard Python HTTP handler wraps network timeouts in `litellm.Timeout`, which AWS reads also swallow. Invalid primary JSON still raises. An absent AWS primary JSON field and a successful Azure response without a value also remain `None`. Defaults do not replace these results. `.with_failure_policy(FailurePolicy::Propagate)` exposes manager failures explicitly instead. Cancellation and other Python `BaseException`s always propagate unchanged. Explicit OIDC references keep their own errors and never use these fallbacks

The getters follow Python's conversion policy. Environment values use case-insensitive, whitespace-trimmed boolean parsing. Manager strings become booleans only when Python literal evaluation yields a boolean; other strings retain their exact contents. Non-string manager results produce `None`. `get_secret_str` returns only strings, and `get_secret_bool` accepts booleans or strings containing `true` or `false`. A type mismatch returns `None` and does not activate fallback

## Route integration

Inject `Arc<dyn SecretSource>` from `litellm_secrets::source` into route preparation. `SecretResolver` implements this interface and supports arbitrary names through its asynchronous `get_secret_str`. It applies the same manager selection, conversion, and fallback policy to every lookup

For synchronous provider transformations, call `source.resolve(names).await` during preparation and inject the returned `Secrets` snapshot. A snapshot contains only those names and never reads the process environment implicitly. Resolve runtime names through the source before invoking a synchronous transformation. OCR uses this pattern; other routes can adopt it as they are implemented

The Python bridge uses this shared source and resolver. The shared proxy initializer captures effective configuration, and directly constructed LiteLLM managers are adapted at the dispatch boundary, and the bridge retains a native backend per configured client. Python reads and Rust routes share that backend. Custom Python implementations remain external callbacks. Rollout policy controls whether the native binding is selected. Public provider reads and Vault/CyberArk mutations use this selection; AWS mutation bindings remain unfinished. Mutations update the same native cache used by public reads. Vault retains complete write response bodies and Python-compatible error dictionaries while verifying rotation with fresh reads. The Python boundary retains primary JSON until return conversion so Python JSON numbers, values, and exception details survive unchanged

## Backend contracts

AWS Secrets Manager, Azure Key Vault, Google Secret Manager, Vault, and CyberArk implement `BaseSecretManager` for reads with an operation context. Foreign provider contexts are rejected before cache access or I/O. Writes and deletes use separate `SecretWriter` and `SecretDeleter` capabilities. CyberArk rotation writes and verifies the replacement while retaining the old alias because Conjur does not support deletion through this API

Shared rotation verifies that the replacement has the requested value before deleting the old secret. Same-name rotation keeps the replacement. AWS same-name rotation uses its version update API directly, matching Python

Backend reads preserve payload strings. Conversion belongs to the resolver. Google caches only successfully decoded payloads, so values agree before and after caching. Native reads do not cache absence or failures. Python-compatible Google reads preserve Python's negative cache and its always-read override. Resource-not-found responses indicate absence; authentication, permission, transport, and malformed successful responses remain errors

The HashiCorp Vault backend is enabled with the `hashicorp` feature and reads KV v2 values from `HCP_VAULT_*` environment variables. It supports static tokens, AppRole authentication, and TLS certificate authentication

## Intentional differences from Python

Native backends consistently distinguish absence from failure instead of swallowing provider errors. Python-compatible resolution maps these results back to the Python handler contract before applying fallback

`hosted_keys` excludes a name for every backend. Python's handler recognizes Azure `SecretClient` and Google `KeyManagementServiceClient` instances before the `local` branch, allowing excluded names to reach those providers. Rust treats that as a routing bug. `test_rust_hosted_keys_exclude_azure_sdk_clients_too` in `tests/test_litellm/rust_bridge/ocr/test_secrets.py` pins this behavior

Google rejects malformed base64 and mismatched CRC32C values instead of accepting corrupted payloads. Python currently ignores the checksum and uses permissive base64 decoding. Rust follows [RFC 4648](https://www.rfc-editor.org/rfc/rfc4648#section-3.3) and [Google's integrity guidance](https://docs.cloud.google.com/secret-manager/docs/data-integrity); `failed_or_missing_reads_are_not_cached` covers rejection and recovery

CyberArk cached reads preserve the original secret text. Python's shared cache attempts JSON decoding, so a secret such as `"password"` changes to `password` after the first read, and `true` changes to a Boolean. This corrupts the stored credential representation. `test_public_cyberark_reads_reuse_authentication_and_cached_values` demonstrates the Python defect and verifies stable native results

## Test parity

[The Python test inventory](PARITY.md) maps each secret-manager test to Rust coverage or its owning boundary

AWS, Vault, and CyberArk keep client/authentication, reads, and writes/rotation in private provider modules. Their existing integration-test targets group configuration, read/cache, and write/rotation cases, with shared fixtures local to each target. Python-compatible dispatch and string coercion live separately from native dispatch
