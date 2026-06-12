# auth_v2

The proxy's authentication and authorization layer. A request arrives with some
credential (API key, JWT, basic auth, mTLS cert, session cookie), and this module
turns it into a `Principal` (a normalized caller identity) and decides whether that
principal is allowed to reach the route.

Everything is consumed through FastAPI `Security()` dependencies, so routes opt in
declaratively and never call into this module by hand.

## The two core types

`Credential` is what an authenticator produces: a verified but un-resolved fact about
the caller ("this bearer token is valid and its subject is `alice@corp`"). It carries
the scheme, the subject, scopes/claims from the token, and, for exchangeable bearer
tokens, the raw token for downstream RFC 8693 token exchange.

`Principal` is what the route handler receives: a normalized identity with the user,
organization, teams, roles and scopes filled in. It holds identity only, no budget or
policy state. Scope checking lives here as `Principal.has_required_scopes`.

The split matters: authentication proves the credential, resolution turns the proven
credential into a known identity, and only then can authorization run.

## How a request flows

A route declares one of three dependencies, all rooted at the same `AuthSecurity`
instance. The intended integration is to build that instance at startup and expose it as
`request.app.state.auth_v2` so routes reach it at request time:

```
Security(auth.principal)                  -> authenticated caller, scopes enforced
Security(auth.require_roles(...))         -> the above, plus a role gate
Security(auth.require_permission(obj,act))-> the above, plus a Casbin permission gate
```

`require_roles` and `require_permission` both depend on `principal`, so the steps below
always run first.

1. Authenticate. `AuthSecurity.principal` walks the authenticator chain in
   `config.scheme_order` and takes the first one that returns a `Credential` (scheme OR,
   not AND). The chain always ends with the session-cookie authenticator. If every
   authenticator declines, it raises `401` with a combined `WWW-Authenticate` challenge
   built from each scheme.

2. Resolve identity. The winning `Credential` is handed to the configured
   `IdentityResolver`. The DB resolver looks the subject up in the proxy's Prisma tables
   (key object, user, teams, org) and builds the `Principal`. A blocked key or unknown
   subject raises `401`/`403` here, before any route logic runs.

3. Attach network context. The client IP and host are resolved (trusted-proxy aware, see
   `network.py`) and copied onto the principal.

4. Enforce scopes. The scopes declared on the `Security()` dependency must be a subset of
   the principal's scopes (`principal.has_required_scopes`). A miss raises `403`
   insufficient_scope. An empty scope requirement always passes.

5. Authorize (only for the role/permission variants).
   - `require_roles(*roles)` calls `authorizer.has_any_role`, which is hierarchy-aware
     (a `platform_admin` satisfies an `org_admin` gate via the Casbin role graph). A miss
     raises `403` forbidden_role.
   - `require_permission(obj, act)` calls `authorizer.enforce`, matching the principal's
     roles against the Casbin policy (`keyMatch` on the path, anchored `regexMatch` on the
     method). A miss raises `403` forbidden_permission.

The resolved `Principal` is then injected into the route handler.

```
request
  -> authenticators (scheme_order, first match wins)   [401 if none]
  -> resolver.resolve(credential) -> Principal          [401/403 on bad/blocked identity]
  -> resolve_network_context
  -> principal.has_required_scopes(scopes)              [403 insufficient_scope]
  -> require_roles / require_permission (optional)      [403 forbidden_*]
  -> route handler(principal)
```

## Where each piece lives

`security.py` is the orchestrator described above; it is the only stateful object and is
built once at the composition root.

`authenticators/` holds one authenticator per scheme behind the `Authenticator` protocol
(`authenticate -> Optional[Credential]`, plus a `challenge`). `build_authenticators`
constructs and orders them from `AuthConfig`. JWT verification for OIDC/OAuth2 is shared
via `JWTVerifier`.

`resolvers/` holds the `IdentityResolver` / `IdentityStore` protocols and their
implementations (`DbIdentityStore` against Prisma, an in-memory store for tests). The
store also handles SCIM user/group provisioning so a provisioned user is immediately
resolvable.

`authorization/` holds the `Authorizer` protocol and its implementations: `RBACEngine`
(Casbin role hierarchy and policy) and `ABACEngine`. `Role` and the JWT/OIDC/SAML role
allowlist gate `filter_claim_roles` live here too. Scope checking does not, it is a
method on `Principal`.

`sessions/` holds the session and OAuth-transaction stores (in-memory or Redis, chosen
from the environment), used by the cookie authenticator and the OIDC/SAML login flows.

`config.py` defines `AuthConfig` and the per-scheme config models. `models.py` defines
`Credential`, `Principal` and the identity sub-models. `network.py` resolves the client
IP with trusted-proxy handling. `errors.py` maps every failure to an `HTTPException` with
the right status and challenge header.

## Adding things

A new credential scheme is a new `Authenticator` plus a branch in `build_authenticators`.
A new identity backend is a new `IdentityStore`. A new authorization method (ReBAC, an
external PDP) is a new `Authorizer` passed as `AuthSecurity(..., authorizer=...)`.
Dependencies are injected at construction, so each of these is unit-testable with a fake
in place of the real backend.
