# Gateway authentication

`Auth` turns a presented secret into a `Caller`. Today the only caller is `Caller::Master`, verified by a constant-time comparison against the configured master key, which may be an `os.environ/` reference resolved through the injected `SecretSource`. New credential kinds (virtual keys, JWT) become new `Caller` variants and new branches in `Auth::authenticate`, not pluggable verifier traits. Ports exist only at real infrastructure boundaries such as secrets and, later, key storage

Credential extractors are plain Axum `FromRequestParts` types that convert into `SecretValue`, built on typed `headers::Header` decoders, never hand-scanned strings. `Bearer` takes exactly one RFC 6750 Authorization bearer token. `ApiKey` walks a priority table of typed key headers in the Python proxy's order, and only the first present source must be well formed. `ApiKeySources` adds a configured custom header ahead of the table and enables the `?key=` query for routes that need it. A route profile picks its extractor with `from_fn_with_state(state, authenticate::<E>)`, and extractor configuration comes from that state through `FromRef`

The `authenticate` middleware inserts the `Caller` as a request extension and replaces any preexisting one. Handlers extract `Caller`, which fails closed when the middleware did not run, and call `Caller::authorize` with the typed `AccessRequest` before calling a provider or MCP server. `Caller::owner` scopes MCP sessions and response caches

Local UI login uses axum-login and tower-sessions. `UiSession` validates the session and its CSRF bearer and never produces a `Caller`, so a UI session cannot authorize inference

`general_settings.custom_auth` is served by the Python proxy, and the gateway refuses to start with it. `CustomAuth` is the reserved seam for running it natively. Its outcome maps Python's modes, where `Caller` and `Reject` are the hook's decision, `Fallthrough` is `auto` mode moving on to built-in auth, and `Rekey` is the enterprise hook returning a replacement key for built-in auth. It runs first in `Auth::authenticate` and receives request parts plus the optional extracted key, so a profile with a hook uses `Option<E>` as its extractor. The runtime behind it (Python extension host, Lua, or other) is undecided

Errors map to HTTP status codes in `Error::status`. Inference keeps its API-specific error envelopes and MCP translates authorization failures to its own protocol
