# litellm/proxy/auth/core

This package decides who a request comes from and whether it may proceed. It holds the decisions only. The proxy does every cache, DB and Redis read, sends every alert, and turns each result into its own types and errors. A core written in another language and exposed through pyo3 implements the same interface, so callers never change when the core does.

## The core is pure

Modules here import only the standard library and each other. A core does no IO itself, reads no globals and keeps no state between calls, so the same inputs and the same host answers always give the same result.

## Facets

`core.py` splits the decision into facets, each a protocol with its own host. A core hands each facet out through an accessor, the way `RouterCore` hands out `registry()`. Nothing inherits from another facet.

`AuthCore` has one accessor per facet. `plan()` picks the credential path a request takes, in the proxy's order: custom auth, public routes, OAuth2 introspection, OAuth2 proxy headers, JWT, unauthenticated pass-through routes, then keys. A path that can hand the request on (custom auth issuing a key, a JWT mapped to a virtual key) resumes planning through `after_custom_auth()` or `after_jwt()`, whose narrower return types exclude the paths already taken. `anonymous()`, `oauth2()`, `custom()` and `jwt()` turn a credential into an identity. `keys()` authenticates a key credential (master key, virtual key, UI session, cached admin key) and verifies a key the proxy resolved itself. `authorize()` decides whether a key identity may make the request: master-key-only routes, model access and the key, team member and per-model budgets. `model_budgets()` checks per-model budgets for the key and user, the user alone, or a team member.

`litellm/proxy/auth/selected_core.py` picks the core the proxy runs, `PythonCore`. `RustCore` declares every accessor and implements none. `PythonCore` does not implement `jwt()` yet, so its signature is provisional and the proxy still validates JWTs itself. `authorize().authorize()` checks a key identity, and `authorize().check_request()` then runs the common checks every authenticated request passes, whatever authenticated it. `check_request()` decides which requests are exempt (public routes by the planner's own rule, pass-through routes without `auth: true`, no-auth dev mode, custom auth that opts out) and the order of every step, while `common_checks` itself and the per-model budget checks still run inside host methods that raise the proxy's errors. Moving those decisions into the core is the next step.

Authentication and authorization are separate steps with separate results. An authenticating facet returns an identity or the reason the credential is unusable (`keys()` returns a `KeyAuthentication`). `authorize()` takes that identity and returns an `Outcome`, `Allow` or `Deny` with a `Denial`. A check about the credential itself (missing, malformed, blocked, expired, owner deactivated) belongs in authentication, and a check about what the identity may do (routes, models, budgets) belongs in `authorize()`, so a credential that is unusable is always refused as such before any budget is read or alert sent. `ModelBudgetCore` returns `WithinModelBudgets` or `ModelBudgetExceeded`, and the planner returns a validator or `PremiumRequired`.

## Hosts carry the IO

`host.py` lists every read, write and alert a facet may ask for, as typed methods, one host per kind of IO a facet needs (OAuth2 introspection calls out over HTTP while proxy-header mode only reads the request, so each has its own host). A host performs IO and returns data. Anything that compares, chooses or denies belongs in the core. When a decision needs data the core doesn't have, add a host method that returns that data instead of letting the host decide. When one check runs another, its host hands out the other host through an accessor, as `AuthorizeHost.model_budgets()` does.

## Model the data, not flags

The values that cross the boundary live in one module per direction. `inputs.py` holds what a caller passes in (`RequestFacts`, `PlanFacts` and the settings each facet reads), `identity.py` holds what authentication returns (identities and why a credential was refused), `outcome.py` holds the authorization verdict, `validators.py` holds the planner's choices, and `answers.py` holds what host methods return. Each is a frozen dataclass, and each choice is a union of them. Express "absent" as `None` on a typed value (a key without a team membership has `team_member=None`) instead of a boolean next to optional fields.

## Order is behavior

The order in which a core calls host methods is part of its contract. It decides which denial wins when several apply and which reads and alerts happen before a request is refused. Reordering checks or host calls is a behavior change. Two cores are equivalent only when they reach the same result through the same sequence of host calls.

## Projection stays outside

Nothing here knows about `UserAPIKeyAuth`, HTTP status codes or the proxy's exception types. The proxy turns each identity into its token object and maps every `KeyDenial` and `Denial` to the public error contract in one place, `litellm/proxy/auth/auth_denial.py`. A new denial needs a case there.

## Credentials

`credential.py` picks the key from the presented headers, query and WebSocket values, owns the precedence between those sources and tells a JWT from a key. `key.py` hashes tokens and compares the master key in constant time. `network.py` decides whether a peer address falls inside the trusted proxy ranges. Code that needs either goes through these two modules instead of parsing headers or comparing keys itself.

## Tests

`tests/unit/proxy/auth/core` runs a core against a host that answers from a script and records each call. Tests assert both the result and the host calls in order, never core internals.
