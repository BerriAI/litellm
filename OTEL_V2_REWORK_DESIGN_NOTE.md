# OTEL v2 admin-owned destinations: rework design note

Working note for the LIT-3850 rework (branch `otel_v2_rework_lit3850`, off PR head `190190cce6`). Not intended for the PR diff. Every claim is cited to a verified `file:line` in this checkout.

## 1. What the current PR actually does (verified source of truth)

Destinations, access grants, assignments, resolution, and export as they exist today:

- Destination = a logging credential. Stored in `LiteLLM_CredentialsTable` (`litellm/proxy/schema.prisma:37-46`), synced into the in-memory global `litellm.credential_list`. A credential is a logging destination when `credential_info.credential_type == "logging"` (`litellm/proxy/management_endpoints/logging_exporter_validation.py:74-79`). The backend name rides in `credential_info.description`; the per-backend secrets ride in `credential_values` (`litellm/proxy/litellm_pre_call_utils.py:642-647`).
- Access grant = `credential_info.access` `{global: bool, teams: [str], orgs: [str]}`, an UNTYPED dict. Shape validated on write (`logging_exporter_validation.py:39-71`). Read via `access_grants(access, team_id, org_id)` (`litellm/proxy/management_endpoints/logging_exporter_access.py:16-31`). Enablement (`auto_enable`) via `is_auto_enable` (`logging_exporter_access.py:34-41`).
- Assignment = `metadata.logging_exporters`, a list of credential-name strings on the key / team / org row, stored UNTYPED inside the generic `metadata Json` column (`schema.prisma:126` team, `:88` org, `:406` key). Written through `validate_logging_exporter_assignment` (`logging_exporter_validation.py:173-227`), called from key/team/org endpoints (`key_management_endpoints.py:1504,1708,2509,4601`; `team_endpoints.py:1014,1733`; `organization_endpoints.py:205,465`).
- Resolution = `_resolve_logging_exporters` (`litellm_pre_call_utils.py:604-676`): unions the assigned names across key+team+org, re-checks `access_grants` defensively, builds each survivor into an `OtelDestination` via `build_destination` (`litellm/integrations/otel/presets/destinations.py:116-130`), dedupes. Also run early at the auth boundary by `_hoist_request_destinations` (`litellm/proxy/auth/user_api_key_auth.py:914-951`).
- Carry-through = TWO carriers for the same resolved list:
  1. Server-only `ContextVar` `_request_destinations` (`litellm/integrations/otel/plumbing/context.py:44-58`), set at the auth boundary and re-set in `_apply_admin_logging_exporters` (`litellm_pre_call_utils.py:729`). Read by the proxy-internal-span fan-out (`fan_out.py:65`). Inherited by the `asyncio.create_task` logging children (documented `context.py:29-46`).
  2. Request-carried `data["litellm_metadata"]["otel_destinations"]` (`litellm_pre_call_utils.py:730-734`), which becomes `StandardCallbackDynamicParams.otel_destinations` (`litellm/litellm_core_utils/initialize_dynamic_callback_params.py:113-118`) and drives the gen-AI-span tracer (`metadata.py:53-75,248`; `logger.py:237,412`). The client's value is wiped first (`litellm_pre_call_utils.py:1888-1890`).
- Export = `TenantFanOutSpanProcessor` for proxy-internal spans (`litellm/integrations/otel/plumbing/fan_out.py:47-152`, skips gen-AI spans) and `TenantTracerCache` for gen-AI spans (`litellm/integrations/otel/plumbing/routing.py:56-227`).
- `OtelDestination` = a frozen Pydantic value object at `litellm/integrations/otel/model/destination.py:14-27`. It is NOT persisted; it is the resolved runtime projection (endpoint + headers + resource attrs) built from `credential_values`.

## 2. The trust-boundary problem Yassin is pointing at

The provider-body scrub is real: `litellm_metadata` is in `all_litellm_params` (`litellm/types/utils.py:3087-3091`) and `get_non_default_params` strips those before the provider body (`litellm/utils.py:3497`). So `litellm_metadata.otel_destinations` does not reach the wire today. But the resolved destination (endpoint + auth headers = infrastructure credentials) is still written INTO the request `data` object (`litellm_pre_call_utils.py:733`) and reconstituted from a request-shaped field (`otel_destinations` on dynamic params). That is the "internal control data living in provider-facing request data" smell (Y5/Y6), and it is a request-shaped surface whose safety depends entirely on the wipe at `:1888-1890` running before every write path (Y3 spoofing). The server-only `ContextVar` carrier has none of these properties and already carries the same data for the other span class.

## 3. Confirmed bug (must fix)

`GET /credentials` (`litellm/proxy/credential_endpoints/endpoints.py:262-308`): for a non-proxy-admin it computes the caller's grantable scope (`_caller_grantable_team_ids`, `:283`) only to gate a 403, then returns EVERY logging-typed destination filtered solely by `is_admin_gated_credential_info` (`:295-299`) with values masked. So any team-admin or org-admin sees the name, host, and `access` scope of destinations belonging to other tenants. The frontend picker compounds it: `LoggingExportersSelect.tsx:34` sets `seesEveryDestination = isAdminRole(userRole)` and `isAdminRole` (`utils/roles.ts:20-22`) includes `org_admin`, so an org-admin's picker shows every destination client-side. The assignment gate (`_reject_unassignable_destinations`, `logging_exporter_validation.py:86-118`) and the resolver (`_selected`, `litellm_pre_call_utils.py:629-637`) already scope correctly; the list endpoint and the picker are the two places that disagree.

## 4. Verified repo conventions

- ORM: Prisma + prisma-client-py (`schema.prisma:1-9`). A new table means editing `schema.prisma` (three byte-identical copies), adding a timestamped migration under `litellm-proxy-extras/litellm_proxy_extras/migrations/`, and hand-writing a Pydantic mirror in `litellm/models/`. Heavy, and it touches the migration package.
- Typed named-resource assignment already exists as a column, not a blob: `LiteLLM_TeamTable.policies String[]` and `access_group_ids String[]` (`schema.prisma:143`). This is the closest analog to `logging_exporters` (a list of named resources assigned to a team).
- Metadata Json is also a documented convention for per-identity config (guardrails, tags, disabled callbacks all live in `metadata`).
- Repository layer: `litellm/repositories/` with `BaseRepository(ABC, Generic[T])` (`base_repository.py:23`); ctor takes a prisma client only, NO in-memory cache. `CredentialsRepository` is standalone, no cache (`credentials_repository.py:14-55`). The in-memory cache is the global `litellm.credential_list`, mutated on CRUD by `CredentialAccessor.upsert_credentials` (`litellm/litellm_core_utils/credential_accessor.py:22-35`) and by the endpoints (`credential_endpoints/endpoints.py:249,423,608-653`). Repositories are instantiated per-call inline; no singleton.
- Domain model layer = `litellm/models/`. `CredentialItem(CredentialBase)` mirrors the credentials table (`litellm/models/credentials.py:18-20`). Teams use a `@model_validator` to coerce Json string columns into typed fields (`litellm/models/team.py:103-134`) — the idiom for typing a Json column.
- Roles: `LitellmUserRoles` (`_types.py:99-135`). Proxy-admin idiom `user_role == LitellmUserRoles.PROXY_ADMIN`. Team-admin = `role == "admin"` in `members_with_roles` (`common_utils.py:106-111`). Org-admin = an `ORG_ADMIN` org membership (`common_utils.py:114-143`). Visibility-list idiom (`/team/list` `_authorize_and_filter_teams`, `team_endpoints.py:4316-4391`): admin sees all, org-admin sees their orgs, member sees own.

## 5. Comment-by-comment plan (option -> choice -> evidence)

### Y1 (destination.py): "move OtelDestination to the DB model layer; why pydantic here?"
The persisted destination state is the logging credential, and it is ALREADY at the model layer (`CredentialItem`, `litellm/models/credentials.py:18-20`). `OtelDestination` is a derived runtime projection, not persisted state, so moving it into `litellm/models/` would misfile a value object as a table mirror. What is genuinely untyped and belongs at the model layer is `credential_info` (currently a bare dict read with `.get()`).
- Option A (chosen): add typed `CredentialAccess` + `LoggingCredentialInfo` Pydantic models next to `CredentialItem` in `litellm/models/credentials.py`; keep `OtelDestination` as the runtime projection and document why it is not persisted. Uses the team.py `@model_validator` idiom for typing a Json column.
- Option B (rejected): physically move `OtelDestination` into `litellm/models/`. Rejected: it is not a table mirror, and the OTEL export code (`fan_out.py`, `routing.py`, `logger.py`) is its only consumer, so it belongs with them.

### Y2 (destination.py, metadata.py): docstrings should describe the abstraction, not narrate the anti-pattern
Rewrite the flagged docstrings (`destination.py:1-9`, `metadata.py:53-61`, `context.py`) to state what the type/function IS. Pure edit, no behavior change.

### Y3 (metadata.py): is dynamic_params guarded against spoofing?
Remove the request-carried carrier entirely rather than guard it. See Y5-resolution / Y6.

### Y4 (metadata.py): query DB + cache in-memory; repository-like class owning lifecycle
- Option A (chosen): a single read surface (`LoggingDestinationRegistry` or a module) that reads the existing DB-synced `litellm.credential_list`, applies the ONE shared predicate, and exposes `visible_to(team_id, org_id)`, `assignable_to(...)`, `resolve_for(identity)`. Reuses the existing cache (already invalidated on CRUD) instead of adding a second, staleness-prone cache.
- Option B (rejected): a new `CredentialsRepository` method that hits the DB directly and caches inside the repo. Rejected: duplicates the `credential_list` cache and would need its own invalidation wired into every CRUD path.

### Y5 (litellm_pre_call_utils.py:674): destinations/assignments are untyped objects
Two halves:
- Destinations + access grant -> typed via Y1 Option A.
- Assignment (`metadata.logging_exporters`) -> FORK, see section 6.
Resolution relocation (the part of Y5 about resolving in the OTEL module, not request metadata) is handled with Y6.

### Y6 (litellm_pre_call_utils.py:693): don't write control data into provider-facing request data
Drop `data["litellm_metadata"]["otel_destinations"]` (`:730-734`) and its read in `initialize_dynamic_callback_params.py:113-118`. Make the gen-AI-span path read the server-only `ContextVar` (`request_destinations()`, `context.py:56`) that the fan-out processor already uses and that is inherited by the logging-callback tasks. Net: one server-only carrier, nothing request-shaped, the `:1888-1890` wipe and the spoofing surface both become moot.

### Y7 (fan_out.py): fold into routing.py?
`fan_out.py` (`TenantFanOutSpanProcessor`, proxy-internal spans, attached to the global provider) and `routing.py` (`TenantTracerCache`, gen-AI spans, per-tenant providers) are two different span classes with two different mechanisms. FORK, see section 6. Default: keep separate under `plumbing/`, co-locate only the shared destination read, and document why (merging makes a god module, violating the repo's own no-god-object rule).

### Y8: extensibility / boilerplate documented
After the above, document how to add a backend (one adapter in `presets/destinations.py`) and the single source of truth for destinations vs assignments, in the PR body and a module docstring.

## 6. Resolved forks (decided 2026-07-01)

FORK A - assignment storage: **A2 chosen**. Typed `logging_exporters String[]` columns on `LiteLLM_TeamTable`, `LiteLLM_OrganizationTable`, `LiteLLM_VerificationToken` (+ the Deleted* mirrors, per the `policies` migration precedent), migrating the resolver / endpoints / UI off `metadata.logging_exporters`. Matches `LiteLLM_TeamTable.policies String[]` (`schema.prisma:143`). Migration template: `litellm-proxy-extras/litellm_proxy_extras/migrations/20260123131407_add_policy_tables_and_policies_field/migration.sql` (`ADD COLUMN IF NOT EXISTS "policies" TEXT[] DEFAULT ARRAY[]::TEXT[]`).

FORK B - Y1 OtelDestination: **B1 chosen**. Type `credential_info` (`CredentialAccess` + `LoggingCredentialInfo` at `litellm/models/credentials.py`); keep `OtelDestination` as the runtime projection with its OTEL consumers.

FORK C - Y7 fan_out/routing: **C2 chosen**. Merge `TenantFanOutSpanProcessor` into `routing.py`; delete `fan_out.py`. Both processors read the one server-only `ContextVar`.

## 6a. Environment constraints (verified)

- Zero `prisma.models.*` imports in `litellm/` (grep: 0). All DB access is dynamic `prisma_client.db.<table>.<op>(data=...)` plus hand-written Pydantic mirrors in `litellm/models/`. So a new column needs: schema.prisma x3 + a proxy-extras migration + updates to the hand mirrors (`team.py`, `organization.py`, `verification_token.py`), and no local `prisma generate` is required for the code to be correct (deploy regenerates).
- `import prisma` works, `import litellm` works, `pytest` runs (existing `test_logging_exporter_validation.py`: 38 passed) against the repo's mocked-prisma convention (`MockPrismaClient`/`MockTable`). So unit tests are the validation surface here.
- Live proxy against real Postgres + real provider APIs is not runnable in this environment (needs `prisma generate` + DB); live proof-of-fix is delivered as a curl/UI runbook for the maintainer to run, per CLAUDE.md.

## 7. Migration / compatibility risk

- The feature is not merged, so there is no production `logging_exporters` data to migrate; A1 and A2 are both green-field on data. A2's risk is purely schema-change blast radius on core tables.
- Removing the `litellm_metadata.otel_destinations` carrier (Y6) changes the gen-AI-span routing source. Risk: a code path where the logging callback runs OUTSIDE the request task's copied context would lose destinations. Mitigation: a test that asserts the gen-AI span still routes via the ContextVar in the success AND failure callback, plus the streaming path.
- Fixing the list leak narrows what non-admins receive from `GET /credentials`; the picker already intends to be a UX filter (`LoggingExportersSelect.tsx:22-26`), so tightening the backend cannot break the picker, only make it authoritative.

## 7a. Implementation findings (as built)

- Leak fix + predicate centralization (committed): `GET /credentials` now filters non-admin visibility through the same `is_destination_visible` predicate as the resolver and the assignment validator, scoped by `_caller_admin_scope` (teams AND orgs). The predicate operates on typed `CredentialInfo` / `CredentialAccess`. `_caller_grantable_team_ids` is now a thin wrapper over `_caller_admin_scope` so the PATCH decider is unchanged. Tests: `test_logging_exporter_access.py` (new, 15), leak regressions in `test_endpoints.py`.

- Y3 spoofing guard: found and fixed a real gap. The wipe at `litellm_pre_call_utils.py:1890` cleared only the top-level and `litellm_metadata` carriers; a client `otel_destinations` still landed in `data[_metadata_variable_name]` ("metadata") during metadata construction. It was inert (the dynamic-params reader consults only `litellm_metadata`, and `metadata` is scrubbed from the provider body), but it left attacker-controlled control data in request data. The wipe now covers every metadata carrier. Regression: `test_client_cannot_control_otel_destinations` drives the full `add_litellm_data_to_request` so the wipe-then-resolve order is under test, and asserts the dynamic-params reader yields nothing.

- Y6 full carrier removal (request-data -> ContextVar for the gen-AI span): NOT done blind. Evidence against a naive removal: the gen-AI logger binds the tenant tracer from `standard_callback_dynamic_params.otel_destinations` (`logger.py:237`, and the deferred close path `:344/:364`). The tracer is bound at `log_pre_api_call` (request task, ContextVar present) for the mainline, but `streaming_handler.py:1668` closes via `asyncio.run(...)`, a fresh event-loop context where the server-only ContextVar may not survive, whereas the kwargs-borne `dynamic_params` always does. So moving the gen-AI path to the ContextVar risks a streaming tenant-routing regression that cannot be validated without a live streaming proxy. Recommendation: keep `dynamic_params` as the robust carrier, rely on the now-hardened wipe for the trust boundary (Y3 proven), and treat ContextVar unification as a follow-up gated on live streaming validation. Open for the maintainer's call.

## 7b. A2 progress (as built, uncommitted WIP)

Environment self-unblocked: installed + generated prisma-client-py 0.11.0 (was an ungenerated namespace stub), stood up a throwaway Postgres (docker `litellm-pg`, port 5599), `prisma db push` applied the schema.

Done and DB-validated:
- `logging_exporters String[] @default([])` added to `LiteLLM_TeamTable`, `LiteLLM_OrganizationTable`, `LiteLLM_VerificationToken`, and the `Deleted*` mirrors, in all three synced `schema.prisma` copies. Round-trip proven against real Postgres: `team.logging_exporters` and `key.logging_exporters` persist and read back through the generated client.
- Pydantic mirrors carry the field (`litellm/models/team.py`, `organization.py`, `verification_token.py`); it propagates to `LiteLLM_VerificationTokenView` / `UserAPIKeyAuth`. `get_key_object`/`get_org_object`/`get_team_object` hydrate via `**model_dump()`, so the column flows in automatically.
- Resolver read path (`_union_logging_exporter_names`) migrated from `metadata.logging_exporters` to the `.logging_exporters` column, reading each level from its own object (key/team/org), DB-required.

Remaining for a complete, consistent A2 (this is why the tree currently has 5 red resolver tests, since a half-migration is intentionally not committed):
- Write path across 6 endpoint sites (key generate/update/regenerate, team new/update, org new/update), each with bespoke persistence (team `/new` auto-persists via `data.json()` -> mirror; keys map fields explicitly). Recommended low-churn approach: add `logging_exporters: Optional[list[str]]` to the request models, feed the existing `validate_logging_exporter_assignment` a synthesized `{"logging_exporters": data.logging_exporters}` dict (leaving the validator and its 38 tests unchanged), and persist the field to the column.
- Update the 5 resolver tests to the column model (mock `get_team_object`; set `prisma_client`).
- Add the proxy-extras `migration.sql` (`ADD COLUMN IF NOT EXISTS "logging_exporters" TEXT[] DEFAULT ARRAY[]::TEXT[]` on the four tables, per the `policies` precedent).
- UI: send `logging_exporters` as a top-level field, read from the column.

## 8. Test plan (maps to the 6 required areas)

1. Visibility: proxy-admin sees per policy; team/org-admin cannot see out-of-scope destinations from `GET /credentials`; global/scoped/auto_enable follow policy.
2. Assignment authz: key generate/update/regenerate, team update, org update, credential PATCH access path.
3. Spoofing: client `otel_destinations` (top-level and under `litellm_metadata`) is ignored; dynamic params cannot control export; provider-facing body contains no destination control data.
4. Runtime export: authorized destination gets spans, attacker destination gets zero, multiple destinations fan out, bad endpoint does not crash the request.
5. Lifecycle/cache: create/update/delete reflects in what resolves; deleted/inaccessible destinations stop resolving.
6. UI: picker only shows backend-visible destinations; no client-side "admin sees all" beyond real proxy-admin semantics; displayed fields accurate.
