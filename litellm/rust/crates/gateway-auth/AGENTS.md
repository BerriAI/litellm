# Gateway authentication

Axum wiring for `gateway-auth-core`, and nothing else: it reads a credential off a request, hands it to an authenticator, puts the caller that comes back on the request and turns a failure into a response

## Boundary with the auth core

- No domain type, key column or rule lives here. Who the caller is, what role it has and what it may reach are `gateway-auth-core`'s, and the rows are the host's. A type that describes a caller, a key or a permission does not belong in this crate
- There is one authenticator trait, `Authenticator<R>`, with one result type, the core's `Caller<R>`. `Auth<R>` holds one behind an extractor, and `Authenticated<R>` is that caller as a handler extracts it. Never add a second trait, a second result type or a wrapper that carries facts the core did not decide
- `CoreAuthenticator<H>` is the one implementation. It asks the core's `plan` which path the request takes, hands the credential to `authenticate_key` or `authenticate_jwt` over the `AuthHost` it was built with, and answers `NotNative` for a plan this server does not serve. The master key and a JWT are verified by the core, never by a comparison here. Every route this crate guards is an inference route, which is what the request it hands the core says
- `From<gateway_auth_core::Error>` and `From<NotACaller>` in `error.rs` are the only place a core answer becomes this server's `Error`. The core's own text is the proxy's, which answers instead wherever one stands behind this server

## Credentials

`CredentialExtractor` selects a token or transport credential from HTTP request parts. The default `Bearer` extractor is strict. `BearerOrXApiKey` follows the Python proxy's order when the request has no non-empty Authorization header. It reads one non-empty `api-key` header, the key header older OpenAI SDKs send, and otherwise one `x-api-key` header, the Anthropic SDK's key header, both taken as sent with no `Bearer ` stripping. A malformed Authorization header never falls back to either, and a malformed `api-key` never falls back to `x-api-key`. A malformed or rejected `api-key` or `x-api-key` answers 401 `missing or invalid API key`, and every other 401, a request with no credential included, keeps `missing or invalid bearer token`. `CredentialExtractor::reject` is where a profile renames an authenticator's rejection for the header it read. A host can supply another extractor with `Auth::with_extractor` for a separate route profile. A transport credential is only a selection marker, never proof of identity, and presents no token to the key path

## Requests

The Axum `authenticate` middleware runs the route profile's extractor. Missing, duplicate, empty, or malformed credentials fail. Authentication replaces any preexisting caller extension. A `Preauthenticated` extension, which a server layer in front puts on a request it already authenticated with the same `Auth`, is taken in place of reading the credential again. Handlers extract `Authenticated<R>`, and missing authenticated context fails closed. There is no fallback chain after an authenticator fails

`RequireMasterKey` admits the master key alone. A request with no usable bearer is still put to the authenticator, with no token, so a server with no master key answers 500 before it calls a token missing

## Failures

Failures distinguish invalid or expired credentials, forbidden operations, unavailable authentication services, and missing server configuration/context. Every credential fails with the one `Error`, classified once by `status()`, and `is_denied()` tells a rejected caller from an auth failure. Inference keeps its API-specific error envelopes

The plain-text rejection body is the top-level message only, never the `Report` chain, because the caller is unauthenticated and the sources carry secret-manager and backend detail. That stays reachable through `source()` for logs

A host declines a key that holds what the server does not serve (`HostDeclined`), which is answered `NotNative`, and so is an identity the core authenticated that this server serves no request for, so no such caller ever reaches a route. Where the keys are stored, and which of the core's host steps are answered, belong to the mounting server
