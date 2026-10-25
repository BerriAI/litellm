## Scope

- This crate is the vocabulary for authenticating LiteLLM to a provider: how a credential is named before it is known (`CredentialRef`, `CredentialPlan`, `ParamSpec`), how it is obtained (`CredentialResolver`, `TokenProvider`, `ResolvedCredential`), how it is carried (`SecretValue`, `Sourced`), how it lands on a request (`CredentialPlacement`, `ProviderAuthPolicy`) and how that fails (`Error`)
- Outbound only. Who is calling the gateway is `gateway-auth`. The one exception is `UiCredentials`, which is here because `gateway-auth` and `gateway-ui` both read it and neither may depend on the other
- No I/O and no clock. Resolving a reference, fetching a token or reading a file is a `CredentialResolver` or `TokenProvider` implementation in `auth-aws`, `auth-azure`, `auth-gcp` or a host. This crate only defines the traits and runs the plan over an injected handle
- Depends on `serde`, `serde_json`, `subtle`, `thiserror` and `veil` only. Nothing here may pull in an HTTP client, a cloud SDK or a provider crate

## Layering

- `auth-types` (contracts) is below `auth-aws`, `auth-azure` and `auth-gcp` (one cloud's acquisition each), which are below `auth` (the `AuthServices` facade that bundles them behind features). `llms`, `inference-*`, `secrets-*` and `cache-*` take the types from here and the services from `auth`
- Import every type from `litellm_auth_types`. `auth` still glob re-exports this crate and `secrets-types` re-exports `SecretValue`, both against the workspace rule in `../inference/AGENTS.md`. Do not add a third path, and move consumers off those two when touching them
- A provider's own credential shape (`AwsParams`, `VertexParams`) lives in `src/params/` only because routing, config and the provider all read it. A field or env name used by one provider stays in that provider's crate

## Contracts

- `SecretValue` is the only way a secret moves between crates. `Debug` is redacted through `veil`, equality is constant time, and `expose()` is the single read. Never add `Display`, `Serialize` or an `AsRef<str>` to it
- `Sourced<T>` records whether a value came from the request, the deployment or the environment. Precedence between those is decided by whoever builds the `Sourced`, never by this crate
- `CredentialPlan::resolve` has three outcomes and they are not interchangeable. `Resolved` carries a credential, `Unavailable` means the caller may fall back to the next plan, and `Err` is terminal. A resolver that cannot answer returns `Declined` or `Missing`, it never errors to mean "not mine"
- A `CredentialRef::Explicit` resolves without consulting the resolver. Everything else, including `Host`, is resolved at acquisition time so rotated secrets are read fresh
- `apply_credential` prepends the credential header and refuses an empty credential or a header already present. Whether an existing header is preserved or rejected is `ProviderAuthPolicy::existing_header_behavior`, chosen per provider, never a default here
- `ParamSpec` lists every place Python reads one connection param, in order: wire names, the `litellm.<name>` module global, then environment names. `resolve` walks that order and treats blank as absent. The static tables in `src/params/` are the one copy of those names

## Errors

- `Error` is the single credential error every provider nests. `auth-aws` maps its SDK failures into it with `From`, `auth-azure` and `auth-gcp` build the variants directly. Either way a consumer never sees a cloud SDK error type
- `MissingApiKey`, `MissingApiBase` and `MissingParam` are what the end user reads when a setting is absent. The message names what to pass or set, and `ParamSpec::guidance` derives that from the spec so the two never drift
- `ErrorDetail::Failed` wraps a source error by `Arc`, so equality is pointer identity. Tests compare the `Message`, `Empty`, `InvalidType` and `DuplicateHeader` variants and only match on `Failed`

## Known gaps

- `ProviderAuthPolicy` and `CredentialPlan` have no production consumer yet, `auth/tests/facade.rs` is the only user. Wire them through `llms/src/base_llm/auth.rs` or delete them, do not grow them unused

## Tests

- Test what crosses the boundary: redaction in `Debug` output, which header a placement writes, which plan outcome a resolver answer produces. Never test that a type derives something
