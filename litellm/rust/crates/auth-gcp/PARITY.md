# Vertex AI Messages login parity

Vertex AI logins go through `src/vertex/`, built on the Google login in `src/google/` and Google's Rust SDK (`google_cloud_auth` 1.16). Claude on Vertex AI `/v1/messages` is the route that uses it today. This file lists the login paths Python's google-auth handled that the SDK alone does not. A path is Legacy when it works only because of code we added on top of the SDK, and Unsupported when it worked in Python and fails here. Paths that work, or that fail the same way they did in Python, are not listed

A login that fails raises the google-auth exception class Python raised, with litellm's own message text, before anything is sent to Vertex. The proxy answers HTTP 500, and failure callbacks, deployment cooldown and router retries run as before. An unreachable token endpoint raises `TransportError` on every path

The "How common" column is an estimate, since there is no telemetry on how deployments log in. Rows 7 and 8 are the regressions most likely to reach a real deployment, because they fail every call and the setups behind them are ordinary ones

| # | Status | Auth method or setup | How common | Rust | Caller gets |
| --- | --- | --- | --- | --- | --- |
| 1 | Legacy | X.509 certificate workload identity | Rare | `google/legacy/certificate_source.rs` | Works. On failure 500 `MalformedError` or `RefreshError` (E1) |
| 2 | Legacy | Credential helper executable | Rare | `google/legacy/executable_source.rs` | Works. On failure 500 `MalformedError`, `DefaultCredentialsError`, `InvalidValue` or `RefreshError` (E2) |
| 3 | Legacy | Workforce identity user from gcloud ADC | Uncommon | `google/legacy/external_account_authorized_user.rs` | Works. On failure 500 `RefreshError` or `OAuthError` (E3) |
| 4 | Legacy | Impersonated ADC with a workforce identity user source | Rare | rows 3 and 5 | Works. Fails as rows 3 and 5 do |
| 5 | Legacy | Impersonation through a non-default host (Private Service Connect, other universes) | Rare | `google/legacy/impersonation.rs` | Works. On failure 500 `RefreshError` (E4) |
| 6 | Legacy | Service account key in PKCS#1 (`BEGIN RSA PRIVATE KEY`) | Uncommon | Re-encoded as PKCS#8 in `google/builders.rs` before the SDK reads it | Works |
| 7 | Unsupported | Service account key behind a gateway that only accepts OAuth access tokens | Rare | Sends the SDK's self-signed JWT and never calls the key's `token_uri`, so the gateway rejects it | 401 `AuthenticationError` with the gateway's body |
| 8 | Unsupported | Corporate CA only in `REQUESTS_CA_BUNDLE` or `CURL_CA_BUNDLE`, on SDK paths (authorized user, workload identity, SDK impersonation) | Moderate in enterprises | The SDK trusts only `SSL_CERT_FILE`, `SSL_CERT_DIR` and the system store, so TLS to Google's token endpoint fails. Legacy paths honour both variables | 500 `TransportError` (E6) |
| 9 | Unsupported | Service account key outside 2048 to 8192 bits | Rare | Refused by the SDK | 500 `ValueError` (E5) |
| 10 | Unsupported | Credential helper command with `#` comments, `\r` or `\$` inside double quotes | Very rare | The `shlex` crate splits it differently from Python's `shlex.split` | 500, likely `RefreshError` |
| 11 | Unsupported | Non-Google STS reply without `issued_token_type` or `token_type` | Very rare | Refused | 500 `RefreshError` (E7) |
| 12 | Unsupported | Metadata server reached only through `GCE_METADATA_ROOT` | Rare | The SDK reads `GCE_METADATA_HOST` only | 500 `DefaultCredentialsError` |
| 13 | Unsupported | Metadata server in mTLS-only mode | Very rare | The SDK has no metadata mTLS | 500, likely `TransportError` |
| 14 | Unsupported | `gdch_service_account` in an ADC file | Essentially never for Claude | Refused at login | 500 `DefaultCredentialsError` (E8) |
| 15 | Unsupported | External account whose `service_account_impersonation_url` is plain `http` to a host other than this machine's loopback | Very rare | Refused before the source token is sent, so it never crosses the network unencrypted. google-auth sent it | 500 `RefreshError` (E9) |
| 16 | Unsupported | AWS workload identity (a `credential_source` with `environment_id`) without litellm's `aws_*` keys, FIPS build only | Rare | Refused in `google/builders.rs`, since Google's Rust SDK signs the AWS request outside the FIPS module. Litellm's `aws_*` login still works | 500 `MalformedError` for `vertex_credentials`, `DefaultCredentialsError` for an ADC file (E10) |

## Error texts

E1. "Cannot read the certificate configuration (path): (reason)" for an unusable configuration, and `RefreshError` when the exchange is refused

E2. "Executables need to be explicitly allowed (set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES to '1') to run." raised as `MalformedError` for `vertex_credentials` and `DefaultCredentialsError` for an ADC file. Timeouts out of range raise `InvalidValue`, and a malformed helper answer raises `MalformedError`, `ValueError` or `RefreshError`

E3. "The external account authorized user credentials need refresh_token, token_url, client_id and client_secret to refresh." A refused grant raises `OAuthError` with "The token endpoint refused the grant with HTTP (status): (body)", which carries the endpoint's response body

E4. "Vertex: the service account impersonation endpoint refused the request." or "Vertex: the service account impersonation endpoint answered no token."

E5. "Vertex: the service account private key cannot sign a token."

E6. "Vertex: the Google token endpoint could not be reached." When `REQUESTS_CA_BUNDLE` or `CURL_CA_BUNDLE` is set without `SSL_CERT_FILE`, it adds a hint to set `SSL_CERT_FILE` to the same CA bundle

E7. "Vertex: Google refused or could not complete the token refresh."

E8. "The file (path) holds gdch_service_account credentials, which the Rust Vertex AI login does not support."

E9. "Vertex: the service account impersonation URL must use https unless it names this machine."

E10. "The FIPS build cannot use an AWS credential_source through Google's Rust SDK, which signs the AWS request outside the FIPS module. Put the credentials in vertex_credentials with LiteLLM's aws_region_name and other aws_* keys so LiteLLM signs the request."
