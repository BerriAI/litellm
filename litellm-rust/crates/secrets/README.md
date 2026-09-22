# Secret resolution

Construct `SecretManagerState::new(backend, settings)` for a configured manager or use `SecretManagerState::default()` for environment lookups. The configured backend determines its provider identity. Write-only settings and names excluded by `hosted_keys` use the environment directly. `secret_manager_would_be_consulted` follows the same routing decision as resolution

`get_secret` returns `Ok(Some(value))` for a found value, `Ok(None)` when no source contains the value, and `Err(error)` when lookup fails. For managed names, resolution checks the manager, then the environment, then the caller's default. An empty string, `false`, or an explicitly stored JSON null is a found value

Backend failures propagate by default. To allow fallback during a backend failure, construct the resolver with `.with_failure_policy(FailurePolicy::EnvironmentFallback)`. It then tries the environment and default, in that order. If neither exists, the original error is returned. This policy applies to manager lookups. Explicit OIDC references retain their own authentication errors and never fall back to environment secrets under the reference name

`get_secret` preserves value types. `get_secret_str` accepts a string default and rejects boolean or JSON values with `Error::TypeMismatch`. `get_secret_bool` accepts a boolean default and converts strings containing `true` or `false`, ignoring surrounding whitespace and ASCII case. Other strings and JSON values produce `Error::TypeMismatch`. Conversion failures never activate fallback or replace a found value with the default

Provider payloads remain strings unless explicitly selecting a field from an AWS primary JSON secret. Google caches only successfully decoded string payloads, so reads have identical values and types before and after caching. Confirmed absence and failed reads are not cached. AWS resource-not-found responses and Google HTTP 404 responses indicate absence. Other provider errors remain errors, and successful responses without the required payload are malformed responses rather than missing secrets

The HashiCorp Vault backend is enabled with the `hashicorp` feature and reads KV v2 values from `HCP_VAULT_*` environment variables. It supports static tokens, AppRole authentication, and TLS certificate authentication
