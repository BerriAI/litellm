# Gateway auth core

Every proxy auth decision for the planner, the master key, virtual keys and JWTs, behind one `AuthHost` trait. The Python proxy runs it through `python-bridge/src/auth/`, and the gateway's native host through `gateway/src/host/auth`. `common_checks`, the authorization stage that runs after either path, is still Python. Design and scope live in the Notion page "Auth" under "Rust Migration Depth First Plan"

These are target invariants. Where the code still lags them, the Notion page lists the gap, and new code follows the invariant, not the gap

## One implementation per component

- Each rule, route table and keyspace codec lives in Rust or in Python, never both. JWT is all Rust, `handle_jwt.py` stays deleted, and every Python caller (SSO, MCP, pass-through, agents) goes through `litellm/proxy/auth/jwt.py`
- A rule that a core path and a Python-only path both need (route groups, OAuth2 routing) lives here and Python calls it through `_native`
- The only allowed duplicates are the Redis client (redis-py and `redis-client`), the DB client (Prisma and SQLx) and request extraction (FastAPI dependency and axum extractor). Keyspace names and value encoding still live in one place

## The core decides, hosts fetch

- No IO, no pyo3, no axum. `AuthHost` in `host.rs` is the only way out, one trait for every path so a host implements it once. The operator's own code sits apart in `OperatorHooks`, which `AuthHost::operator_hooks` hands out only on a host that can run it. Methods return `impl Future + Send`, and the core takes `&impl AuthHost`, never `dyn`
- A method fetches or stores one named thing, or runs one named host-owned step. Comparing, choosing, ordering and denying happen here. A method that still answers a decision goes in the last sections of `AuthHost`, with a TODO naming what has to move to Rust first, and takes the rows it decides on as arguments instead of reading host state
- A host call exists because of what it touches, and that says how it leaves
  - A store port reads or writes one row, cache entry or remote document. It is permanent, and only its answerer changes when the bridge answers it natively
  - Operator code is `custom_validate`, behind `OperatorHooks`. It is permanent and only the Python host offers it
  - State only the Python process holds (router aliases, access groups, request pricing, spend counters, alerts) makes a temporary method, with a TODO. It narrows to a read of that state, then goes once the core decides
- `host.rs` groups the methods by the stage that calls them: credential, identity reads, provisioning, admission, grants and finish. Route policy calls none
- Pure logic is never a host call. The test is a second host. If it could only answer a method by re-implementing a rule, that rule belongs in the core
- No generic methods such as `cached_json(key)` or `run_step()`. One method per question, typed in and out
- The order of host calls is behavior, since it decides which reads and writes happen before a denial. Tests share the scripted host in `testing.rs` and assert the result and the calls in order
- Where one path hands off to another, such as a mapped JWT continuing on the key path, a test at the entry point in `lib.rs` covers it, with inputs built by the stage that produces them. A test that seeds a record by hand proves only its own stage

## The caller

- Every answer about who the caller is and what it may reach is decided here, with no IO. A host asks, it never works one out from a row
- `Caller` in `caller.rs` is the only thing a host carries past authentication. An identity that is not a caller a host serves, as no master key, the user auth route, a cached admin key and a JWT are, is `NotACaller` and never becomes one, so a host cannot serve it by accident
- The role is derived here, by `Caller::role`. Admin is a role and never a kind of credential, so a host never reads it off how the caller authenticated or off a permission it holds
- A key's id is the hash its row is stored under, `Caller::hash`, read from the `KeyRecord` the host's key row keeps through `StoredKey`. The key path refuses a key whose record has none, so a caller that is a key always has one and only the master key has none. It is never the hash of the token the request presented, which a key a JWT maps to does not have
- For access, the host supplies the layers it read (the key's, the team's and so on) and the core decides. A host that did not read a layer a key has must have declined that key at authentication, so no decision is ever made on a layer that is missing
  - `authz::authorize` in `authz/models.rs` is the one model rule. It takes each list as a `Layer` with the `ModelAsk` the host's router resolved the name to (the name asked for, its model group and that group's access groups), and a request has to be reached by every layer. No layer, as the master key has, reaches every model
  - A list reaches a model when it is empty, holds `*` or `all-proxy-models`, or names the model, its group or one of its access groups
  - An entry the rule does not read, a provider pattern such as `anthropic/*`, `all-team-models` or `no-default-models`, is the host's to decline at authentication. Reading one means porting it here, never matching it in a host
  - A team's or a user's list is a new `Layer` variant with its own `Denied`, fed by the host method that yields the layers. A route never names a row type and never compares a list itself

## No Python semantics

- No `litellm-python-compat`, no `repr`, `str()`, truthiness or `int()` copies. Untyped claim values follow the JSON rules in `value.rs`: set means not null, false, zero or empty; an id is a string or an integer; a NumericDate is a JSON number
- Messages format values the core's own way: RFC 3339 times, JSON lists and objects, `unset` for a missing value. Keep each message's opening sentence stable, since clients match on it
- A host turns Python-shaped input into typed values before calling the core, such as a body field that arrives as a JSON string or a Python number's `str()`. That code lives in `python-bridge`, which may use `python-compat`
- An operator's glob (`jwt_role`, a routing override selector) is matched by `litellm-fnmatch`, the dialect existing configs are written in. Never hand-roll a glob here or pull in another glob crate
- The router's own matching is not a Python semantic. Route templates match a `%0A`-terminated path like the route without it, because the router dispatches it that way

## Typed rows in, one context out

- Methods return the row they read (key, user, team, membership, end user, org, project), next to the facts the core decides on. No method returns `()` to leave data behind in the host, and no host keeps per-request state between calls. The request is the one exception. It is an argument the whole call takes in and hands back, so a host step that still resolves the model (`normalize_model`, the budget fallback) rewrites it in place, where the budget steps, the handlers after auth and the Python-only path read it. Returning the model for the core to pass along would only echo what is already on the request. Once the core resolves the model itself, the rewrite comes back in the outcome and the host applies it once
- A record such as `KeyRecord` holds only what the stored row says. A value one stage derives from the request or a claim reaches the next stage as an argument, never through a field a host fills from a row. A JWT's end-user claim reaches the key path as `Credential::Resolved::end_user_id`
- One reader per row, with one set of facts beside it. The read policy (cache only, cache then database, database only, upsert) is a typed argument, never a second method
- Pass-through grants are decided in `routes/passthrough.rs` from the team's facts and the request's forwarded routes. `RouteChecks` calls the same functions through `_native`
- Success is one `AuthContext` holding those rows. The host only converts it, to `UserAPIKeyAuth` in `litellm/proxy/auth/host/identity.py`

## Failures are values

- Every failure is one `crate::Error` in `src/error.rs`, with key and JWT denials in `src/error/key.rs` and `src/error/jwt.rs` and every message in `#[error(...)]`. A host method returns the same `Error`: `HostFailed` means the host keeps its own failure, an unreachable database included, and a host-owned step's verdict is its typed denial. `is_host_failure()` separates the two at every boundary. `HostDeclined` is a host that holds what a step asked for and does not serve it, as a native host does for a row whose columns it cannot honor yet. A read that treats a failure as nothing found goes through `Error::absent_unless_declined`, so a declined read never authenticates the request
- `public.rs` is the one mapping from `Error` to `PublicDenial`. It picks the status, type and param, and takes the message from `Display`
- The exception class is part of a denial, since the failure spend row records its name and the auth failure log reads its fields. A denial the proxy raised as its own class is its own `PublicDenial` variant, and `tests/unit/proxy/auth/test_denial.py` fails when a variant has no class
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

## Request body check

- `body::check` decides whether a caller's request body may reach a handler, a port of Python's `is_request_body_safe`. It is pure. The caller hands in everything it decides on, the way `authz::authorize` takes a `ModelAsk`
- A caller runs it in three steps
  - `body::needs` names what the body can reach: the model names (its model, the route's, every fallback target the body names) and every `litellm_credential_name` it carries. It walks the body as `check` does with every override granted, so it lists everything a check on the real data can read before it stops
  - The caller resolves the model names to a `Reachable`, the deployments of each group they reach through aliases, the router's fallbacks (`Router::fallback_model_names` in router-core) and the groups each deployment found routes under, and reads the credentials with `body::credentials`, which asks `AuthHost::credential`, a store port
  - `body::check` takes the body, the `Reachable`, the `Credentials`, `Names` and `Policy`
- The rules a second host would otherwise write again live here: decoding a mapping sent as JSON text, the form's bracket-spelled metadata (`body/form.rs`), the URL destination allowlist (`body/destination.rs`) and an `api_base` opt-in pattern, read in Python's regex dialect through `fancy-regex`
- A model call (`Surface::model_call`) is read everywhere outside its content (`CONTENT_KEYS` in `names.rs`), since providers resolve secret references in more parameters than any list names. A reference anywhere a parameter can sit is refused, and so is a value nested below the depth a host read it to (`CallerValue::TooDeep`)
- No rule reads a raw body. The host turns it into `Params` and `CallerValue`, which hold only what a rule may read and have no method that resolves a value as a secret
- `Names` is built once from `HostTables`, the parameter lists Python still owns. `Policy` holds the `general_settings` and `litellm_settings` facts
- The gateway runs it on `/v1/messages` and `/ocr` through `gateway-inference`'s `Checked` extractor. The Python proxy still runs its own copy until it switches over
- Gaps against the invariants above
  - A refusal is a `body::Rejection` in `error/body.rs`, with the caller's text in `Display`. It is not a `crate::Error` and has no `PublicDenial`, because every Python caller still raises it as a `ValueError`
  - `body/surface.rs` names the routes whose bodies carry their own fields, outside `routes/table.rs`

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
- Two rules run before `plan` and read no caller, so neither is authn or authz: `client_address.rs` decides which addresses reach the proxy (`allowed_ips`) and `routes/exposed.rs` which routes it serves at all (`allowed_routes`). Each has its own settings, predicate and denial, so a host can adopt one without the other. `Allowlists` holds both, and `check_request` in `lib.rs` runs them in the proxy's order, the address first, so a host that runs both calls that. `ClientAddresses::reads_address` tells a host whether to read the peer at all
- `plan/mod.rs` picks the credential path in the proxy's order. The two paths the core owns, key and JWT, run as stages, each its own module with typed inputs and outputs:
  - `authn/` decides whether the credential is genuine and accepted: `key/` prechecks the token, reads the key (`lookup.rs` names the end user, the key's own id before the request's, then reads the key with its team values) and refuses a blocked, expired or tokenless one. `jwt/` verifies the signature or reads UserInfo, runs `custom_validate`, and resolves the virtual key a JWT maps to
  - `identity/` reads who is acting: a key's owner, team and project, or a JWT's caller (`caller.rs`), team, user and organization. It may provision, and collects the writes a request earns as `commit::Grants`
  - `authz/` decides whether that caller may call the route or model. Every check is pure and takes no host. `TeamEligibility` is the one place identity's team selection reads the route
  - `admission/` runs the checks that need reads after identity: the key's temporary budget, the model it resolves to and its budgets
  - `commit.rs` applies the grants (`record_agent_authentication`, `add_team_member`, `create_key_mapping`) once nothing refuses the request, and is the only caller of those methods
- `pipeline/` holds the orchestrators the entry points call. Each runs every check at the earliest point its input exists, so nothing is read or written for a request a check that could have run earlier refuses
- `routes/` decides whether a route may be called. `template.rs` matches starlette route templates, `groups.rs` the `LiteLLMRoutes` group lookups and an operator's `allowed_routes`, `managed.rs` the managed-agent routes, and `passthrough.rs` the pass-through grants, which Python also calls through `_native`. The `LiteLLMRoutes` groups stay Python's, and the host hands them in as `HostFacts::route_groups`, read through `litellm/rust_bridge/host/route_groups.py`
- `routes/table.rs` is the only file in the core that names a path or a `LiteLLMRoutes` group. The core reads a group through `RouteGroup`, never by its string name, and `RouteGroups::new` refuses a table that lacks one. A fixed set of templates is a `RouteSet` there. Only an operator's `allowed_routes` entries reach `RouteGroups` as strings
