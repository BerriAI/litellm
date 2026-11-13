# Gateway auth core

Every proxy auth decision for the planner, the master key, virtual keys and JWTs, behind one `AuthHost` trait. The Python proxy is the only host, and runs it through `python-bridge/src/auth/`. `common_checks`, the authorization stage that runs after either path, is still Python. Design and scope live in the Notion page "Auth" under "Rust Migration Depth First Plan"

These are target invariants. Where the code still lags them, the Notion page lists the gap, and new code follows the invariant, not the gap

## One implementation per component

- Each rule, route table and keyspace codec lives in Rust or in Python, never both. JWT is all Rust, `handle_jwt.py` stays deleted, and every Python caller (SSO, MCP, pass-through, agents) goes through `litellm/proxy/auth/jwt.py`
- A rule that a core path and a Python-only path both need (route groups, OAuth2 routing) lives here and Python calls it through `_native`
- The only allowed duplicates are the Redis client (redis-py and `redis-client`), the DB client (Prisma and SQLx) and request extraction (FastAPI dependency and axum extractor). Keyspace names and value encoding still live in one place

## The core decides, hosts fetch

- No IO, no pyo3, no axum. `AuthHost` in `host.rs` is the only way out, one trait for every path so a host implements it once. The operator's own code sits apart in `OperatorHooks`, which `AuthHost::operator_hooks` hands out only on a host that can run it. Methods return `impl Future + Send`, and the core takes `&impl AuthHost`, never `dyn`
- A method fetches or stores one named thing, or runs one named host-owned step. Comparing, choosing, ordering and denying happen here. A method that still answers a decision goes in the last sections of `AuthHost`, with a TODO naming what has to move to Rust first, and takes the rows it decides on as arguments instead of reading host state
- A host call exists because of what it touches, and that says how it leaves. `host.rs` is grouped the same way
  - A store port reads or writes one row, cache entry or remote document. It is permanent, and only its answerer changes when the bridge answers it natively
  - Operator code is `custom_validate`, behind `OperatorHooks`. It is permanent and only the Python host offers it
  - State only the Python process holds (router aliases, access groups, request pricing, spend counters, alerts) makes a temporary method. It narrows to a read of that state, then goes once the core decides
- Pure logic is never a host call. The test is a second host. If it could only answer a method by re-implementing a rule, that rule belongs in the core
- No generic methods such as `cached_json(key)` or `run_step()`. One method per question, typed in and out
- The order of host calls is behavior, since it decides which reads and writes happen before a denial. Tests share the scripted host in `testing.rs` and assert the result and the calls in order

## No Python semantics

- No `litellm-python-compat`, no `repr`, `str()`, truthiness or `int()` copies. Untyped claim values follow the JSON rules in `value.rs`: set means not null, false, zero or empty; an id is a string or an integer; a NumericDate is a JSON number
- Messages format values the core's own way: RFC 3339 times, JSON lists and objects, `unset` for a missing value. Keep each message's opening sentence stable, since clients match on it
- A host turns Python-shaped input into typed values before calling the core, such as a body field that arrives as a JSON string or a Python number's `str()`. That code lives in `python-bridge`, which may use `python-compat`
- An operator's glob (`jwt_role`, a routing override selector) is matched by `litellm-fnmatch`, the dialect existing configs are written in. Never hand-roll a glob here or pull in another glob crate
- The router's own matching is not a Python semantic. Route templates match a `%0A`-terminated path like the route without it, because the router dispatches it that way

## Typed rows in, one context out

- Methods return the row they read (key, user, team, membership, end user, org, project), next to the facts the core decides on. No method returns `()` to leave data behind in the host, and no host keeps per-request state between calls. The request is the one exception. It is an argument the whole call takes in and hands back, so a host step that still resolves the model (`normalize_model`, the budget fallback) rewrites it in place, where the budget steps, the handlers after auth and the Python-only path read it. Returning the model for the core to pass along would only echo what is already on the request. Once the core resolves the model itself, the rewrite comes back in the outcome and the host applies it once
- One reader per row, with one set of facts beside it. The read policy (cache only, cache then database, database only, upsert) is a typed argument, never a second method
- Pass-through grants are decided in `routes/passthrough.rs` from the team's facts and the request's forwarded routes. `RouteChecks` calls the same functions through `_native`
- Success is one `AuthContext` holding those rows. The host only converts it, to `UserAPIKeyAuth` in `litellm/proxy/auth/host/identity.py`

## Failures are values

- Every failure is one `crate::Error` in `src/error.rs`, with JWT denials in `src/credential/jwt/error.rs` and every message in `#[error(...)]`. A host method returns the same `Error`: `HostFailed` means the host keeps its own failure, an unreachable database included, and a host-owned step's verdict is its typed denial. `is_host_failure()` separates the two at every boundary
- `public.rs` is the one mapping from `Error` to `PublicDenial`. It picks the status, type and param, and takes the message from `Display`
- The core renders every denial to `PublicDenial`. Python only constructs the exception, in `core.raise_denial`. A host failure is the one Python exception that passes through, so `allow_requests_on_db_unavailable` stays in Python's `handle_authentication_error` and the core never reads it

## Host-owned steps

- `custom_validate` is the only permanent host step. It runs the operator's Python callable, so there is no rule to port. The core decides when it runs and what a `False` means
  - `plan` returns `Plan::JwtWithOperatorCode` when one is configured, so a host without `OperatorHooks` hands the request to Python before authenticating it. The owner is still picked before authentication
  - A host without hooks that runs the JWT path anyway is refused with `JwtDenial::CustomValidateUnavailable`. The check is never skipped
- Everything else the host runs is temporary, one named method with a typed result and a TODO naming what has to reach Rust first
  - The key path's budgets (key, team member, user per-model, end user, proxy budget alert) are decisions the host still makes, because the spend counters, alerts and budget fallback rewrite are Python-owned. Each moves onto `spend` and a comparison in the core
  - Key minting for JWT mappings is a write the host performs with its race handling. The core has already decided whether to mint and for whom
  - Project lookup is a plain read, not a host step
  - The row rebuilds (`with_team`, `with_temp_budget`, the end-user limits `db_key` stamps) are pure rules the host still runs, because the core cannot write a row. Nothing Python-owned blocks them. Each becomes facts the core returns beside the row, which the host applies once in `host/identity.py`
  - The UI and `lite login` session keys are Python's whole. `ui_key` decodes one and refreshes a `lite login` key's grants from its user's and team's current rows, and a denial there is the proxy's own exception, which passes through like a host failure. They move to Rust as their own credential path, not as part of `key/`
- When a host-owned step's TODO lands and the core decides it, its trait method goes too, together with everything listed below
- When the bridge answers a read or write natively, the trait method stays, since every host still has to answer it. Only the Python side goes: its `RustAuthStep`, its bridge `Ask`, and its `answer.py` case and `PythonAuthHost` method, all in `litellm/proxy/auth/host/`
- So the trait shrinks only by decisions, while the Python host shrinks by both. It reaches `custom_validate` alone only once every Prisma query has ported and `user_api_key_cache` has moved to Rust. Until then the reads that are left stay Python answers, and nothing schedules those ports

## IO

- The core awaits only `AuthHost`. A host may answer any method natively, so the bridge driver awaits Rust futures as well as Python answers
- Under the Python host, DB reads and writes are Python answers on Prisma until a query ports on demand. JWKS, OIDC discovery and userinfo are answered in Rust with a client from `HttpClientPool`. The JWT keyspaces are answered in Rust from `redis-store`
- Python answers are awaited on the caller's asyncio task, never from a task Rust spawned

## Keyspaces

- JWKS, OIDC discovery, OIDC userinfo and JWT key mapping entries live in Rust, memory and Redis together, with the same keys, values and TTLs as main
- `user_api_key_cache` (key, team, user, end user, membership, org, project, object permission, access groups) stays on the Python stack behind named methods. It moves whole later, together with every Python writer (management endpoints, eviction, pub/sub)

## Settings

- `settings.rs` builds `Settings` from `litellm_config::GeneralSettings` plus `HostFacts`, what the host knows that the config does not say. Both hosts hand in the same typed config, resolved once at config load
- Secrets stay `SecretValue` all the way through. A presented credential is a `Token`, the master key is a `MasterKey`, and both wrap `SecretValue`. A key's SHA-256 is a `KeyHash`, which is not a secret. None of the three travels as a bare `&str`
- The key path takes a `Credential`, either the token the request presented or a key the caller already read, which is how a JWT mapped to a key enters it. A resolved key is named by the `KeyHash` it stores, so nothing hashes it again or compares it with the master key
- `MasterKey::matches` is the only comparison of a token with the master key. A host calls `expose()` only where it hands the value to Python or to a third-party client

## Layout

- `lib.rs` holds the entry points. `plan` picks the path, and each host matches on it exhaustively. `authenticate_key` and `authenticate_jwt` run the two paths the core owns and return an `Outcome`, `Authenticated` or `Denied`. Custom auth runs in the host before `plan`, and a key it hands back sets `Request::from_custom_auth`. OAuth2 introspection and the OAuth2 proxy run in the Python host
- On the Python side, `litellm/proxy/auth/core.py` drives a call and converts its outcome, `litellm/proxy/auth/host/python.py::PythonAuthHost` answers the trait's methods under the same names, and `tests/unit/proxy/auth/host/test_answer.py` fails when a `RustAuthStep` has no `answer.py` case, the way `router_rust/core.py` and `router_python_host.py` pair for the router. Rust never imports either, so neither is a `litellm/rust_bridge/host/` module
- `plan/mod.rs` picks the credential path in the proxy's order. `credential/` holds the two paths the core owns, `key/` and `jwt/`. The key path runs as stages that `mod.rs::authenticate` sequences: `gate.rs` decides what the request and the looked-up key alone settle, `lookup.rs` names the end user (the key's own id before the one the host's driver read from the request into `Request::end_user_id`), has the host validate and read it, and reads the key with its team values, and `validate.rs` checks a virtual key, ordering the budget steps the host still decides around the expiry check
- `routes/` decides whether a route may be called. `template.rs` matches starlette route templates, `groups.rs` the `LiteLLMRoutes` group lookups and an operator's `allowed_routes`, `managed.rs` the managed-agent routes, and `passthrough.rs` the pass-through grants, which Python also calls through `_native`. The `LiteLLMRoutes` groups stay Python's, and the host hands them in as `HostFacts::route_groups`, read through `litellm/rust_bridge/host/route_groups.py`
- `routes/table.rs` is the only file in the core that names a path or a `LiteLLMRoutes` group. The core reads a group through `RouteGroup`, never by its string name, and `RouteGroups::new` refuses a table that lacks one. A fixed set of templates is a `RouteSet` there. Only an operator's `allowed_routes` entries reach `RouteGroups` as strings
