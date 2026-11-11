# Gateway auth core

Every proxy auth decision for the planner, the master key, virtual keys and JWTs, shared by two hosts. The Python proxy (the only host in prod) runs it through `python-bridge/src/auth/`. The Rust gateway (proof of concept) runs it through `gateway-auth`'s `CoreAuthenticator`, with its `AuthHost` in `crates/gateway/src/auth/host.rs`. Design and scope live in the Notion page "Auth" under "Rust Migration Depth First Plan"

These are target invariants. Where the code still lags them, the Notion page lists the gap, and new code follows the invariant, not the gap

## One implementation per component

- Each rule, route table and keyspace codec lives in Rust or in Python, never both. JWT is all Rust, `handle_jwt.py` stays deleted, and every Python caller (SSO, MCP, pass-through, agents) goes through `litellm/proxy/auth/rust_jwt.py`
- A rule that a core path and a Python-only path both need (end-user id, route groups, OAuth2 routing, fail-open) lives here and Python calls it through `_native`
- The only allowed duplicates are the Redis client (redis-py and `redis-client`), the DB client (Prisma and SQLx) and request extraction (FastAPI dependency and axum extractor). Keyspace names and value encoding still live in one place

## The core decides, hosts fetch

- No IO, no pyo3, no axum. `AuthHost` in `host.rs` is the only way out, one trait for every path so a host implements it once. Methods return `impl Future + Send`, and the core takes `&impl AuthHost`, never `dyn`
- A method fetches or stores one named thing, or runs one named host-owned step. Comparing, choosing, ordering and denying happen here. A method that still answers a decision goes in the last section of `AuthHost`, with a TODO naming what has to move to Rust first, and takes the rows it decides on as arguments instead of reading host state
- No generic methods such as `cached_json(key)` or `run_step()`. One method per question, typed in and out
- The order of host calls is behavior, since it decides which reads and writes happen before a denial. Tests share the scripted host in `testing.rs` and assert the result and the calls in order

## No Python semantics

- No `litellm-python-compat`, no `repr`, `str()`, truthiness or `int()` copies. Untyped claim values follow the JSON rules in `value.rs`: set means not null, false, zero or empty; an id is a string or an integer; a NumericDate is a JSON number
- Messages format values the core's own way: RFC 3339 times, JSON lists and objects, `unset` for a missing value. Keep each message's opening sentence stable, since clients match on it
- A host turns Python-shaped input into typed values before calling the core, such as a body field that arrives as a JSON string or a Python number's `str()`. That code lives in `python-bridge`, which may use `python-compat`
- The router's own matching is not a Python semantic. Route templates match a `%0A`-terminated path like the route without it, because the router dispatches it that way

## Typed rows in, one context out

- Methods return the row they read (key, user, team, membership, end user, org, project), next to the facts the core decides on. No method returns `()` to leave data behind in the host, and no host keeps per-request state between calls
- Pass-through grants are decided in `passthrough.rs` from the team's facts and the request's forwarded routes. `RouteChecks` calls the same rules through `_native`
- Success is one `AuthContext` holding those rows. Each host only converts it, to `UserAPIKeyAuth` in `litellm/proxy/auth/core.py::project` or to the gateway's caller extension

## Failures are values

- Every failure is one `crate::Error` in `src/error.rs`, with JWT denials in `src/jwt/error.rs` and every message in `#[error(...)]`. A host method returns the same `Error`: `Unavailable` names the dependency (DB, Redis, identity provider), `HostFailed` means the host keeps its own failure, and a host-owned step's verdict is its typed denial. `is_host_failure()` separates the two at every boundary
- `public.rs` is the one mapping from `Error` to `PublicDenial`. It picks the status, type and param, and takes the message from `Display`
- The core decides fail-open (`allow_requests_on_db_unavailable`) and renders every denial to `PublicDenial`. Python only constructs the exception, in `core.raise_denial`. A Python exception never passes through as the public error of a core path

## Host-owned steps

- `custom_validate` is the only permanent host step. It runs the operator's Python callable, so there is no rule to port. The core decides when it runs and what a `False` means
- Everything else the host runs is temporary, one named method with a typed result and a TODO naming what has to reach Rust first
  - The authorization stage's budgets are decided here in `budgets.rs`. The host only reads spend from its counters (`spend`), reads budget rows, and sends alerts (`budget_alert`)
  - The key path's budgets (key, team member, user per-model, end user, proxy budget alert) are decisions the host still makes, because the spend counters, alerts and budget fallback rewrite are Python-owned. Each moves the same way, onto `spend`, `budget_alert` and the comparisons in `budgets.rs`
  - Key minting for JWT mappings is a write the host performs with its race handling. The core has already decided whether to mint and for whom
  - Project lookup is a plain read, not a host step
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
- Secrets stay `SecretValue` all the way through, the master key included

## Layout

- `lib.rs` holds the entry points. `plan` picks the path, and each host matches on it exhaustively. `authenticate_key` and `authenticate_jwt` run the two paths the core owns and return an `Outcome`, `Authenticated` or `Denied`. Custom auth runs in the host before `plan`, and a key it hands back sets `Request::from_custom_auth`. OAuth2 introspection and the OAuth2 proxy run in the Python host, and the gateway forwards them to Python
- On the Python side, `litellm/proxy/auth/core.py` drives a call and converts its outcome, `litellm/proxy/auth/host/python.py::PythonAuthHost` answers the trait's methods under the same names, and `tests/unit/proxy/auth/host/test_answer.py` fails when a `RustAuthStep` has no `answer.py` case, the way `router_rust/core.py` and `router_python_host.py` pair for the router. Rust never imports either, so neither is a `litellm/rust_bridge/host/` module
- `plan.rs` picks the credential path in the proxy's order. `key.rs` is the key path. `jwt/` is the JWT path. `authorize.rs` is `common_checks`, the authorization stage that runs after either path, `budgets.rs` its budget checks, and `route_access.rs` its organization and route checks, which Python-only callers reach through `_native`. `routes.rs` and `route_groups.json` own the route tables, and Python reads them from here
