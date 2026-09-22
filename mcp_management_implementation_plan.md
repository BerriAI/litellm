# Built-in LiteLLM management MCP: implementation plan

Status: independently reviewed and agreed with Devin on 2026-09-22; ready for user review, not implementation approval or runtime verification

Code baseline: `origin/main` at `7177d3b6d11dc208e2531f05df4df7a1f33b228a`, inspected on 2026-09-22

## Objective

Let an authenticated agent manage LiteLLM through a built-in MCP endpoint, beginning with virtual keys and access groups. The MCP interface must preserve the existing management API's authorization, validation, persistence, cache invalidation, and audit behavior

The long-term goal is broad management API coverage. The first release is a deliberately bounded foundation, not an automatic export of every HTTP route

## Verified code context

All paths below refer to the baseline commit, rather than the unrelated working branch in the local checkout

| Area | Existing code and implication |
| --- | --- |
| SDK and images | `pyproject.toml` requires `mcp>=2.2.0,<3`; `uv.lock` resolves 2.2.0; `Dockerfile` and `gateway/Dockerfile` install frozen dependencies with the proxy extra |
| Current MCP routes | `litellm/proxy/proxy_server.py`: `aggregate_mcp_route`, `proxy_mcp_route`, toolset routes, and dynamic `/{mcp_server_name}/mcp` routes already exist |
| MCP transport | `litellm/proxy/_experimental/mcp_server/server.py` owns SDK transport/session wiring and delegates tool operations to `operations.py` |
| Existing policy | `operations.py`, `contracts.py`, and `auth/user_api_key_auth_mcp.py` operate on upstream MCP server grants. `_raise_if_initialize_grants_no_mcp_servers` can deny callers without upstream grants. Management access must not depend on granting an unrelated upstream MCP server |
| OpenAPI conversion | `openapi_to_mcp_generator.py::build_input_schema` copies body properties but does not fully resolve body schema references or preserve all nested constraints. `_register_openapi_tools` in `mcp_server_manager.py` uses this converter |
| Local tool registry | `tool_registry.py` supports configured handlers. It is not a ready-made management authorization or audit boundary |
| Worker admission | `middleware/admission_control_middleware.py::AdmissionControlMiddleware` acquires a worker permit for each non-exempt HTTP request. Re-entering the full application can self-starve and repeat middleware accounting |
| Existing internal client | `litellm/llms/custom_httpx/asgi_handler.py::get_async_asgi_client` already pools ASGI clients with context-local client/root-path state. It does not solve full-application re-entry and is not needed by the selected design |
| Inner management hooks | `management_helpers/utils.py::management_endpoint_wrapper` executes alerts, OTEL, and cache updates before returning. `integrations/SlackAlerting/slack_alerting.py::send_virtual_key_event_slack` renders kwargs, so outer MCP redaction alone cannot prevent secret exposure |
| Virtual keys | `management_endpoints/key_management_endpoints.py` has authenticated generate/list/info/update/delete routes. `generate_key_fn` uses `GenerateKeyRequest`, `management_endpoint_wrapper`, hooks, and caller permission checks |
| Access groups | `management_endpoints/access_group_endpoints.py` exposes `/v1/access_group` CRUD. `_require_proxy_admin` controls writes; `_require_admin_view` controls reads. Team/key cache synchronization is already implemented there |
| Access-group models | `litellm/types/access_group.py` defines create/update/response models, including assigned keys, teams, models, agents, and MCP servers |
| Split deployment | `gateway/main.py` and `backend/main.py` filter the shared application routes through their component allowlists. `backend/routes/allowlist.py` retains `/mcp`; `gateway/routes/allowlist.py` does not retain the aggregate `/mcp` route or mount |

Correction to the initial feasibility summary: having MCP SDK dependencies in the standalone gateway image does not itself make the aggregate MCP endpoint reachable there. The full proxy and backend are the appropriate starting hosts

## Scope and product contract

### First release

- A dedicated built-in endpoint, proposed as `/litellm-management/mcp`, enabled explicitly by deployment configuration
- Stateless Streamable HTTP with JSON responses, using the official MCP SDK already installed in the repository; no long-lived SSE stream for these finite operations
- Full-proxy and backend-component support; operators can route the endpoint to the backend behind the same public hostname as the gateway
- Virtual-key and access-group tools listed below
- Initial callers: existing proxy-admin credentials only
- Explicitly narrower access than REST is acceptable for the first release. Admin-viewer, team-admin, organization-admin, internal-user, and broader delegated OAuth support are subsequent work. MCP uses POST for reads too, so viewer support needs a narrow admission-policy design. Per-tool REST verb checks already distinguish viewer reads and writes on the synthetic requests
- Works with no external MCP servers configured or granted
- Disabled means the endpoint is unavailable and no management tools appear in existing catalogs

### Outside the first release

- Exporting every OpenAPI operation or changing the generic OpenAPI converter
- Automatically adding administration tools to the existing aggregate `/mcp` catalog
- Gateway-to-backend credential delegation, token exchange, or a new authorization server
- Dashboard configuration UI, new database tables, or new migrations
- Inference streaming, uploads, realtime sessions, provider credentials, SSO settings, arbitrary HTTP execution, or arbitrary SQL
- New durable approval, idempotency, or job systems

### Initial tools

Names are proposed stable public contracts. Descriptions must identify effects and required privileges

| Tool | Existing API | Release behavior |
| --- | --- | --- |
| `list_virtual_keys` | `GET /key/list` | Reuse existing pagination and filters; redact secrets |
| `get_virtual_key` | `GET /key/info` | Prefer the key hash, as REST recommends; accept existing identifier semantics; remove echoed raw keys from the safe metadata result |
| `create_virtual_key` | `POST /key/generate` | Proxy-admin only; return the newly generated key once in the authorized tool response |
| `update_virtual_key` | `POST /key/update` | Proxy-admin only; preserve omitted versus explicitly supplied fields |
| `delete_virtual_keys` | `POST /key/delete` | Proxy-admin only; preserve existing batch and `key_aliases` semantics; destructive annotation; redact echoed raw keys |
| `list_access_groups` | `GET /v1/access_group` | Preserve current response semantics; do not invent server-side pagination |
| `get_access_group` | `GET /v1/access_group/{access_group_id}` | Proxy-admin for this release; retain endpoint checks |
| `create_access_group` | `POST /v1/access_group` | Proxy-admin only; existing relation validation and cache updates |
| `update_access_group` | `PUT /v1/access_group/{access_group_id}` | Proxy-admin only; preserve null, empty-array, and omitted-field semantics |
| `delete_access_group` | `DELETE /v1/access_group/{access_group_id}` | Proxy-admin only; preserve cleanup semantics; destructive annotation |

Every tool requires proxy-admin admission in this release. Any later viewer support must check each actual REST contract and keep direct calls to hidden write tools denied

## Architecture decisions

### 1. A dedicated management surface

Use a small management MCP application under `litellm/proxy/_experimental/mcp_server/management/`, with explicit catalog and execution owners. Reuse the official SDK and appropriate existing HTTP/auth helpers, while keeping management policy distinct from upstream MCP server entitlements

Register literal handlers for `/litellm-management/mcp` and `/litellm-management/mcp/` before the dynamic MCP alias route, with OpenAPI inclusion disabled. The former does match `/{mcp_server_name}/mcp`, so order matters. Keep the literal handlers as 404 responders when disabled to prevent alias fallthrough, reserve `litellm-management` at config/database MCP registration, and report existing conflicts explicitly. Add both exact paths to `BACKEND_EXACT_PATHS`; make no gateway allowlist changes

Own a separate SDK server and stateless session manager, configured for JSON responses. Create and enter the manager once per application lifespan in `proxy_startup_event`, close it on shutdown, and create a fresh manager for a later lifespan. Do not reuse the aggregate server's manager or auth/session ContextVars, and do not copy its lazy startup pattern

Do not change the existing aggregate server's no-upstream-grants check to accommodate management tools. That would broaden unrelated access paths

### 2. Explicit tools with schemas derived from existing models

Use an explicit immutable allowlisted mapping of tool name, HTTP method, fixed path, request model, query/path model, response projection, and read/write metadata. Inject auth and handler dependencies for tests. Use typed per-tool call adapters rather than generic untyped kwargs or callable reflection

Derive JSON Schema from the existing Pydantic request types, with typed wrappers for query/path parameters. Preserve references through self-contained definitions, required fields, enums, nested lists, unions, defaults, and nullable values. Validate arguments using the same model contract before dispatch

Keep tool schemas deliberate and agent-readable. Do not bind arbitrary functions or blindly register every OpenAPI operation. New tools require a reviewed mapping and behavioral tests

### 3. Explicit dispatch preserving REST policy and handler behavior

The reviewed direction replaces full-app ASGI re-entry with an explicit ten-tool dispatcher. For each catalog entry, construct a fresh per-tool Request with the mapped REST method, route, validated arguments, verified client context, and caller credential. Authenticate and authorize that request through the existing `user_api_key_auth`, require the returned caller to remain `PROXY_ADMIN`, then call the existing decorated endpoint coroutine with every required argument explicitly supplied

Build the synthetic request from trusted `client`, `scheme`, `server`, and `root_path` values with fresh request state. Replace stale outer route/endpoint/path parameters with the correct REST values; construct GET query strings from the query model and mutation bodies from the request model using `exclude_unset=True`. Preserve null versus missing fields. Supply a receive function that yields the body once, and retain that Request for auth and any handler that accepts it

Pass `request` to `update_key_fn` and `list_keys`; pass every query parameter explicitly to `list_keys` and `info_key_fn`. Preserve the existing `litellm-changed-by` header where the key handlers accept it, but record verified caller identity separately from this caller-declared attribution. Pass no additional MCP kwargs to decorated functions. Import lazy access-group handlers explicitly rather than assuming their router has already loaded

The adapter must preserve the caller credential for the same LiteLLM service, request correlation, and the verified client context required by IP restrictions. It must not accept arbitrary forwarded identity headers, tool-supplied credentials, target URLs, or elevated server credentials

Check authentication and the admin role restriction at MCP admission, then re-run existing endpoint authorization on every tool execution. Classify the new endpoint explicitly as a management route, governed by `DISABLE_ADMIN_ENDPOINTS`, not as MCP inference. It must not inherit generic MCP admission fallbacks or become public through deployment overrides. The existing REST policy remains authoritative for object ownership, route permissions, and custom authorization. Never call a route function with unresolved FastAPI `Depends`, `Header`, or `Query` defaults

Define a new `management_mcp_routes` group containing only the two endpoint spellings and include it in `management_routes`. Do not include it in `mcp_inference_routes`, `mcp_management_routes`, or their backwards-compatible `mcp_routes` union. The latter also grants existing MCP server CRUD and appears in JWT defaults, so reusing it would broaden an existing key's route permission. Require both the admin role and appropriate management route permission; per-tool authorization additionally checks the concrete REST route

Full-app ASGI re-entry is excluded: `AdmissionControlMiddleware` holds the outer request's permit while an inner REST request acquires another, causing self-starvation at one permit or saturation. It also repeats request accounting and middleware. The existing `get_async_asgi_client` helper is useful elsewhere but does not remove these effects

Direct dispatch does not automatically provide FastAPI request parsing, dependency injection, or response-model filtering. Before adding each tool, audit its application/router/route dependencies, request/header/query arguments, decorators, response type, and exceptions. At this baseline, the selected routes use `user_api_key_auth` with no additional application/router dependencies. Implement explicit typed argument adapters and safe response projections with existing Pydantic models; do not use private FastAPI dependency-solving APIs or a generic reflection dispatcher. Add REST-versus-MCP behavioral comparison tests for these contracts

For declared response types, validate with the relevant model or `TypeAdapter.validate_python` before JSON serialization. Serialization alone is not validation. For endpoints without a response model, use explicit typed safe-result projections and the public JSON encoder. Do not expose arbitrary metadata or raw handler dictionaries blindly. Treat access-group deletion's 204/None as a successful empty result, not invalid JSON

Normalize `ValidationError` into the corresponding REST validation structure with body/query/path locations, removing credential-bearing input details. Preserve useful status classes for `ProxyException`, `ManagementProblem`, and `HTTPException` through existing public helpers. Recognize database-unavailable chains using the existing classifier; otherwise return a generic safe server error. Promise semantic parity and deliberate secret redaction, not byte-identical output where the MCP transport or projections differ

The outer HTTP request still traverses normal middleware once. Preserve client IP/root path/correlation from trusted request context, and re-check route policy against the actual per-tool operation. Do not add a user-controllable middleware bypass marker or an admission exemption for management traffic

### 4. Credential and session boundaries

Start with header-configured clients using credentials already accepted by LiteLLM's management APIs. Build dedicated management admission around the existing credential parser and `user_api_key_auth`, using the literal management path and an isolated empty body for transport admission before the SDK consumes the original request. Preserve the configured custom-auth behavior and test it at both admission and per-tool authorization. Do not call `MCPRequestHandler.process_mcp_request`, which includes upstream passthrough/DCR flows irrelevant to administration

An upstream MCP OAuth credential, MCP session bearer, browser cookie, or external server grant is not automatically a management credential. Fail closed if management authentication is unavailable, even in a development deployment with no master key

Do not forward a token issued for the MCP resource to a different backend audience. Split deployments initially route the public management MCP URL directly to the backend; any later internal forwarding needs an explicit audience/delegation design

Authenticate each request and check the role again after per-tool auth. Stateless mode stores no caller/session authority, and arbitrary incoming session IDs must not import aggregate-server authority. Revocation behavior must be no weaker than the current REST auth cache behavior

### 5. Results, failures, and operational behavior

- Return typed structured results where supported and a compatible textual representation; preserve useful backend errors without echoing credentials or tracebacks
- Map tool execution failures to MCP tool error results; keep transport/authentication failures at their appropriate protocol boundary
- Never retry mutations automatically. A timeout or cancellation after dispatch may have an unknown outcome; tell the caller to inspect state before repeating a create operation
- Mark read operations read-only and deletions destructive. Client confirmation is advisory and not a substitute for authorization
- Newly created virtual keys are intentionally sensitive tool output. Exclude raw arguments/results containing secrets from MCP traces, callbacks, analytics, audit payloads, and error messages. Metadata-only reads must not expose usable keys. Cover caller-supplied keys and get/delete echoes as well as newly generated keys
- Reuse and extend field-aware redaction at the inner alert/OTEL owners before they stringify inputs. A focused alert-redaction change is in scope because preserving the existing hook unchanged would expose raw key inputs through this feature. Keep audit events and safe metadata; do not disable auditing. Capture hook payloads in tests without actually sending messages
- Do not reuse aggregate MCP tool I/O logging for management tools. Ensure SDK telemetry excludes arguments/results. Correlate the synthetic request span with the outer request and explicitly close spans on success, failure, and cancellation using existing tracing helpers
- Keep the existing management audit events and attribute them to the verified caller. Add only missing management-MCP correlation metadata; do not assume every endpoint already emits identical audit events
- Proposed initial bounds: one tool execution per stateless request with no batch fan-out, a 30-second execution deadline, and a 1 MiB encoded MCP result limit. Keep existing worker admission as the concurrency control; do not add a per-session semaphore to a stateless protocol. Preserve REST `list_virtual_keys` page size of 1 to 100. Oversized responses are explicit tool errors without silent truncation; if a mutation already completed, report that the result could not be returned and warn against blind retry
- Attach the SDK server's lifecycle to the existing application lifespan; do not start an application lifespan per tool call or duplicate database startup

## Implementation sequence and code ownership

1. **Prove the boundary**: audit the ten handlers' full REST contracts and current auth routing; write dispatcher-versus-REST behavioral tests, including one available admission permit; verify official SDK lifecycle against the locked version
2. **Add endpoint and typed config**: implement the disabled-by-default flag, dedicated management transport, exact backend routing, collision detection, and shutdown cleanup. Extend `proxy_server.py` only for wiring, not management business logic
3. **Add typed catalog and dispatcher**: explicit ten-tool mappings, model-derived schemas, credential-safe REST execution, deterministic role-filtered discovery, and safe result/error conversion
4. **Validate behavior and document operation**: full-proxy/backend tests, real client flows, configuration and ingress examples, limitations, and rollback instructions
5. **Expand separately after acceptance**: teams/users/organizations, budgets/models, spend/analytics, then remaining management families. Review each family's permission and secret contracts before adding it. Aggregate catalog integration and delegated identities are separate follow-ups

Expected new capability owners: `management/server.py`, `management/catalog.py`, and `management/dispatcher.py`, adjusted only if repository conventions favor fewer files. Existing transport/auth helpers should be factored only where this feature requires it

Existing owners potentially touched: `proxy_server.py`, `_types.py::ConfigGeneralSettings` and route classifications, `backend/routes/allowlist.py`, `auth/user_api_key_auth.py`, `auth/route_checks.py`, MCP alias registration/config owners, `management_helpers/utils.py`, `integrations/SlackAlerting/slack_alerting.py`, and existing tracing/redaction helpers. No endpoint business-logic rewrite is planned

## Validation and acceptance criteria

### Automated behavior

- Disabled endpoint, enabled endpoint, bare/trailing-slash paths, alias collision, and backend/full-proxy route availability
- Management route classification: `DISABLE_ADMIN_ENDPOINTS` blocks it, `DISABLE_LLM_API_ENDPOINTS` does not, and generic MCP inference permission does not grant administration
- A proxy-admin key restricted to `allowed_routes=["mcp_routes"]` is denied at management MCP admission; the equivalent key with `allowed_routes=["management_routes"]` is admitted. Explicit MCP transport permission never bypasses concrete per-tool REST route restrictions
- Startup/shutdown/repeated test lifespans own fresh SDK managers; cancellation closes tracing and releases worker admission capacity
- Initialize/list/call using real SDK types with zero configured upstream servers
- Admin read/write success; denied admin-viewer and other non-admin callers; denied anonymous/expired/revoked credentials; direct calls to hidden tools remain denied
- REST authorization parity for custom auth, route restrictions, verified client IP, and object restrictions; no service-account substitution
- One available worker admission permit must support a successful call; check saturated workers and request accounting. Admission plus per-tool auth may perform two cached lookups. At this baseline these management requests must cause no LLM budget reservation or LLM parallel-request counter movement; assert that invariant in tests
- Two concurrent callers and reused session identifiers cannot leak credentials, tool visibility, results, or authority
- Schema fidelity for references, nested access-group assignments, enums, null, empty lists, false, zero, omitted fields, and unknown fields under the existing model policy
- Create/update/delete effects match direct REST calls, including team/key cache invalidation and existing hooks
- Error propagation for invalid arguments, missing resources, conflicts, database failures, oversized results, timeout, cancellation, and unknown mutation outcome
- No raw generated or supplied key in captured logs, alerts, traces, callbacks, validation errors, or get/delete metadata output; key creation returns its secret only in the authorized result
- Existing aggregate MCP discovery/calls, OpenAPI-backed tools, and upstream authentication remain unchanged

New tests mirror the new capability under `tests/test_litellm/proxy/_experimental/mcp_server/management/`. Extend existing mapped tests when changing `test_operations.py`, `test_dynamic_mcp_route.py`, auth behavior, key endpoints, or access-group endpoints. Follow `tests/e2e/AGENTS.md` for real end-to-end coverage

Place management MCP end-to-end scenarios under `tests/e2e/management/`, exercising the actual LiteLLM server and database. The existing `tests/e2e/mcp/` suite specifically requires real external MCP servers and is not the appropriate home for this built-in management surface

### Real before/after evidence

After implementation is authorized, present local setup steps and obtain setup/service approval before installation or startup. Use source builds tied to the merge base and implementation tip

With identical client steps, show the absent management feature at the base and successful initialize/list/create/read/update/delete flows at the tip. Cover keys and access groups, non-admin denial, full proxy, and split ingress-to-backend routing. Record commit hashes, commands, sanitized results, and cleanup. No paid LLM calls are needed for protocol/API verification

### Submission requirements

Before a future implementation PR, re-read repository guidelines, the PR template, and current CI. Use the verified default branch and a `litellm_` branch name. Run applicable full local CI lint/type/budget gates against the merge base and affected backend tests; do not increase budgets or add blanket suppressions

Measure changed executable-line coverage separately from touched-function branch coverage. Aim for at least 95% changed-line coverage for this feature and cover all meaningful authorization and secret-handling branches. Require completed current-tip Codecov evidence, required CI, contributor agreement, and current-tip Veria, Greptile confidence at least 4/5, and Bugbot reviews with zero unresolved actionable findings before declaring the implementation ready

## Rollout and rollback

Enable first in a test deployment, then an administrator-only pilot. Configuration removal disables the endpoint without schema rollback. Keep the management endpoint separate from the existing aggregate catalog during the pilot

Success means an agent can complete the ten supported operations with the same observable authorization and state changes as REST, and operators can identify the caller without recording secrets

## Independent review record

Devin session: [Independent plan cross-check](https://app.devin.ai/sessions/c94fda7266fe4e3bb40e3ac3adf08dea)

The user explicitly requested a fresh session. The complete draft was shared with Devin on 2026-09-22 for independent code inspection and iterative review, with no implementation authorized

Agreement status: agreed on the scoped design and acceptance criteria after applying Devin's final route-permission correction. No unresolved design blockers remain from this review; implementation tests and runtime proof remain future work

| Review exchange | Resolution |
| --- | --- |
| Initial independent review and local admission finding | Removed recursive ASGI dispatch. Explicit per-tool auth and decorated-handler adapters preserve the relevant REST behavior without consuming worker capacity twice |
| Viewer and transport scope | Admin-only first release; stateless JSON over Streamable HTTP; keep existing key-alias deletion semantics |
| Challenge to direct-handler parity | Added explicit query/header/body adaptation, response validation before serialization, safe result projections, public error normalization, and parity tests |
| Corrections to review details | The dynamic alias pattern does match the proposed literal, so literal ordering and disabled 404 handling remain required. Serialization alone does not validate. Raw keys occur in supplied arguments and echoed results too. Recheck the admin role after per-tool auth |
| Stateless resource bounds | Existing worker admission, one execution per request, 30-second deadline, 1 MiB result limit; no new per-session semaphore |
| Final independent cross-check | Created a separate `management_mcp_routes` permission group rather than joining the legacy `mcp_routes` union. Added explicit restricted-admin-key tests and corrected the quota/viewer wording |

Devin's final response at 2026-09-22 23:21:51 UTC, event `event-01a0cb6cc3ab7640bf1508df6681edff`, explicitly authorizes recording agreement after that route-list correction. All three final amendments are applied above

The original findings and subsequent correction record are retained in `mcp_management_devin_review.md`. That transcript contains superseded review assertions; this plan is the authoritative proposal

## References

- [Baseline repository](https://github.com/BerriAI/litellm/tree/7177d3b6d11dc208e2531f05df4df7a1f33b228a)
- [MCP tools specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP security guidance](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices)
