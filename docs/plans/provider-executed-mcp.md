# Provider-executed MCP registration

Status: database, isolation, and admin registration API slices approved. Scoped retrieval and provider config loading are intentionally deferred

## Implementation plan

Work through these layers in order on `litellm_provider_mcp_registration`. Update this table as each layer is reviewed and implemented. Open changes in VS Code for review before committing. Keep each review slice uncommitted until the user has reviewed it

| Layer | Scope | Status |
| --- | --- | --- |
| 1 | Settle registration fields, values, storage, and validation | Approved |
| 2 | Implement persistence and execution-mode isolation before enabling provider registration | Database and isolation committed; API persistence validation approved; YAML provider inventory pending |
| 3 | Management UI/API, stored tool options, and existing key/team/toolset selectors | Admin CRUD and inventory API approved; UI and scoped selectors pending |
| 4 | Authorize effective OpenAI Responses and Anthropic Messages MCP attachments through existing MCP permissions | Pending |
| 5 | Live provider tests, streaming admission tests, UI tests, and gateway regressions | Pending |

V1 supports URL-based OpenAI Responses MCP and both Anthropic Messages connector formats. Request authorization applies to streaming and non-streaming calls before provider dispatch. Server restrictions, team permissions, and key permissions intersect. Existing applicable permission ceilings and explicit denies continue to apply

Response/SSE inspection, per-invocation argument policy inside a provider's tool loop, automatic MCP discovery for provider registrations, local MCP references, provider connector IDs, and additional API bridges are outside v1

## Current review acceptance criteria

This batch covers admin CRUD, input validation, cached management inventory, and execution isolation. Key/team-scoped retrieval, selectors, toolsets, and native-request admission remain later layers. Their absence is intentional for this batch

Core provider fields remain `execution_mode`, `server_name`, `url`, `allowed_tools`, and optional `tool_name_to_description`, alongside the existing shared management fields. Neutral connection values remain accepted where the shared schema permits them, but never enable MCP execution. Active credentials, transports, OAuth settings, commands, and other connection configuration are rejected. No strict five-field schema rewrite is part of this revision

| Acceptance area | Required proof in this batch |
| --- | --- |
| Admin CRUD | Create, list, detail, partial update, delete, missing/deleted IDs, duplicate URL, immutable mode |
| Input validation | Active connection settings reject; supported neutral settings succeed; rejected updates leave the row unchanged; omitted tools preserve scope and an explicit empty list stays enforced |
| Cached inventory | Provider registrations appear in admin inventory and refresh after updates and deletion without becoming executable |
| Execution isolation | No provider initialization, discovery, health probe, or tool call from management or gateway endpoints |
| Compatibility | Existing LiteLLM credential merging and update transaction rules remain in place |

Provider updates keep read-merge-validation atomic through the existing writer. The gateway credential path is not moved into a new blanket transaction. One initial row lookup selects the immutable execution mode and is reused when the gateway credential flow needs stored values

The later permissions layer will test matching list/detail visibility under key/team intersection, explicit denial, disjoint scopes, revocation after caching, and separation of read grants from mutation privileges. The admin-only live run is not evidence for that layer

The next config slice should accept provider entries under the existing `mcp_servers` key, reuse registration validation, and publish read-only management records in the shared catalog without constructing gateway clients. Verify startup and reload, removal, stable IDs, conflicting config/database identities or URLs, and zero upstream activity. Existing LiteLLM config loading must remain unchanged

## Current invocation routing

On the Responses path, `type: "mcp"` alone does not tell LiteLLM to connect to the server. An external `server_url` such as `https://custom.example/mcp` stays in the provider-bound tools. On a provider/API supporting native MCP, the provider discovers and calls that endpoint, subject to its own connectivity, authentication, and approval requirements. An unsupported provider/API may reject the attachment. LiteLLM does not automatically become the MCP executor

The explicit LiteLLM gateway reference is `litellm_proxy/mcp/<name>`, or `litellm_proxy` for the whole gateway. This selects registered gateway resources. LiteLLM discovers their tools and supplies function definitions to the model. Automatic execution additionally requires every gateway reference in that request to set `require_approval: "never"`. That approval value by itself does not convert an external native attachment into LiteLLM execution

Current main also recognizes HTTP(S) URLs ending in `/mcp/<name>` when the name resolves to a registered server, access group, or toolset. It rewrites those references to `litellm_proxy/mcp/<name>`. This shortcut matches the path name without validating the hostname. A URL on an unrelated host can therefore select a local registered resource if its final name matches. The gateway then uses the registered resource's connection configuration, rather than connecting to that unrelated host

These branches are implemented in [tool_precheck.py](../../litellm/responses/mcp/tool_precheck.py), [split_mcp_tools and gateway name resolution](../../litellm/responses/mcp/litellm_proxy_mcp_handler.py), and [Responses orchestration](../../litellm/responses/main.py). The prior live reproduction demonstrated provider execution of custom URLs under metadata and tool-permission guardrail restrictions. It did not configure registered-server object permissions

The proposed `execution_mode` constrains which registered resource may be used in each path. Provider registrations must never enter gateway discovery or execution, including through the HTTPS path shortcut or toolset membership. Layer 2 must isolate them before enabling registration. Layer 4 must resolve native attachments by the approved canonical endpoint and prevent path-name collisions from silently selecting a different executor. The existing hostname-blind shortcut needs an explicit compatibility decision during that work

## Layer 1 schema proposal

### Entity and terminology

Store both execution modes in `LiteLLM_MCPServerTable`, with the same stable `server_id`. Existing `LiteLLM_ObjectPermissionTable` and `LiteLLM_MCPToolsetTable` shapes remain unchanged

| Surface | Proposed wording |
| --- | --- |
| API, YAML, and database field | `execution_mode` |
| Value `litellm` | LiteLLM-executed |
| Value `provider` | Provider-executed |
| UI field label | Execution mode |
| Provider-mode help | The LLM provider discovers and calls this MCP server. LiteLLM authorizes the server and tool scope without connecting to it |
| Provider-mode tool editor label | Allowed tools |
| Provider-mode tool editor help | Enter the upstream tool names permitted on this server. Keys, teams, and toolsets can select from this list. LiteLLM does not discover or verify these tools |

Execution mode identifies who may initiate the upstream MCP connection. A caller using LiteLLM's gateway still uses a `litellm` registration. A provider registration is usable as an authorized native attachment on the LLM API

No `native_mcp_policy`, `declared_tools`, `native_url`, provider-specific server table, or additional toolset kind is introduced. The registration does not need an OpenAI/Anthropic discriminator because the same endpoint can be used by either provider

### Storage

Add one enum column and make the existing transport column nullable. The first implementation slice adds these fields to all three synchronized Prisma schemas and a schema-only migration. The migration has not been applied to a running LiteLLM deployment

```prisma
enum MCPExecutionMode {
  litellm
  provider
}

model LiteLLM_MCPServerTable {
  execution_mode MCPExecutionMode @default(litellm)
  transport      String?          @default("sse")
}
```

The model excerpt shows only the changed fields. All other columns remain in place

| Field | Provider-executed storage and meaning |
| --- | --- |
| `execution_mode` | Required enum value `provider`. Omitted on existing/legacy registrations means `litellm` |
| `server_id` | Existing stable identifier used by object permissions and toolsets |
| `server_name`, `alias`, `description` | Existing management identity and display fields |
| `url` | Required approved HTTPS MCP endpoint, stored in canonical form |
| `allowed_tools` | Existing `String[]`. Explicit finite server-level allowlist and source of management tool choices |
| `mcp_info.tool_allowlist_enforced` | Existing marker, persisted as `true` for provider mode so `[]` means deny all |
| `tool_name_to_description` | Existing optional descriptions for names in `allowed_tools` |
| `transport` | SQL `NULL`. LiteLLM neither chooses nor opens the provider's MCP transport |
| `auth_type`, `credentials` | No upstream authentication settings or credentials. Existing empty storage representations are acceptable |
| Other connection-only settings | Neutral existing defaults, with no active connection configuration accepted |
| Object permissions and toolsets | Existing tables and existing server IDs/tool names |

The provider registration's tool list retains the meaning of an authorization ceiling. Reusing it for management choices does not turn it into a verified discovery cache. Provider discovery supplies real descriptions and input schemas to the model; LiteLLM does not need to store those schemas

Add a partial unique index on `url` for rows whose `execution_mode` is `provider`. One canonical endpoint maps to one provider registration in v1. The same URL may also exist on a separate LiteLLM-executed registration, and existing duplicate gateway URLs remain valid

```sql
CREATE UNIQUE INDEX "LiteLLM_MCPServerTable_provider_url_key"
ON "LiteLLM_MCPServerTable" ("url")
WHERE "execution_mode" = 'provider';
```

Use the same canonicalization and collision checks for YAML-defined registrations. Reject ambiguous provider registrations across database/config sources rather than selecting whichever record grants more access

### API and YAML contract

The existing create/update/read schemas gain the enum field. Use the same vocabulary in API payloads, YAML, and the UI. Validate the merged stored record on partial updates

```json
{
  "server_name": "docs",
  "execution_mode": "provider",
  "url": "https://docs.example/mcp",
  "allowed_tools": ["search_docs", "fetch_doc"],
  "tool_name_to_description": {
    "search_docs": "Search documentation",
    "fetch_doc": "Read a documentation page"
  }
}
```

Management reads return the stored field values, including `execution_mode`, `transport: null`, and the enforced-allowlist marker. They do not probe the endpoint. Existing response redaction conventions still apply

| Input | Proposed behavior |
| --- | --- |
| Omitted `execution_mode` on create/config | Preserve existing LiteLLM execution and transport defaults |
| Unknown or null `execution_mode` | Reject |
| `execution_mode` omitted on update | Preserve stored mode |
| `execution_mode` supplied on update | Reject in v1; this field is immutable and must be omitted from updates |
| Missing or null provider `url` | Reject |
| Provider URL | Require HTTPS; reject URL userinfo and fragments; lowercase scheme/hostname and omit default port 443; preserve path, trailing slash, and query identity |
| Provider `transport` omitted or null | Persist null explicitly, rather than applying the database's legacy SSE default |
| Provider `transport` set to SSE, HTTP, or stdio | Reject; transport is not a configuration responsibility of this registration |
| `allowed_tools` omitted on provider create/config | Reject; require an explicit administrative choice |
| `allowed_tools` omitted on update | Preserve the existing list |
| `allowed_tools: null` | Reject for provider mode |
| `allowed_tools: []` | Register a server with zero permitted tools; never interpret as unrestricted |
| Tool names | Exact upstream names, case-sensitive; reject blank names and wildcard entries, deduplicate identical names |
| Description-map keys | Must reference entries in `allowed_tools` |
| Allowlist marker omitted | Persist `mcp_info.tool_allowlist_enforced=true`, preserving unrelated metadata |
| Allowlist marker explicitly false or null | Reject for provider mode |

Endpoint matching uses this same canonicalization later in authorization. It does not follow redirects, perform discovery, collapse tenant paths, or equate URLs solely by hostname. Changing a provider URL changes the resource behind existing grants and requires the existing privileged registration-update authorization

Reject active provider-mode connection settings, including stored tokens, OAuth/BYOK configuration, static/forwarded credential headers, stdio commands and environments, connection probes, and schema-pinning configuration. Accept neutral null/empty/default values needed to round-trip the existing shared response shape. Do not store `auth_type="none"` to imply the upstream server itself is unauthenticated

The provider may receive credentials in a native LLM request according to its API. That does not authorize LiteLLM to store them in the registration or use them to connect to MCP

### Permissions and toolsets

The existing shapes remain sufficient

```json
{
  "object_permission": {
    "mcp_servers": ["server_docs"],
    "mcp_tool_permissions": {
      "server_docs": ["search_docs"]
    },
    "mcp_toolsets": ["toolset_research"]
  }
}
```

```json
{
  "toolset_name": "research",
  "tools": [
    {"server_id": "server_docs", "tool_name": "search_docs"},
    {"server_id": "server_gateway", "tool_name": "read_record"}
  ]
}
```

A toolset may include both execution modes. Membership contributes grants through the existing resolver and never changes execution mode. Removing a name from server-level `allowed_tools` immediately restricts its effective authorization even if an object permission or toolset still references it

Because provider registrations in v1 have a finite allowlist, an unbounded native attachment is rejected. Granting all tools of this registration means all of its approved names, not every tool the upstream server might expose now or later. Clients must send an explicitly bounded scope that passes authorization; v1 does not silently rewrite requests

### Compatibility and implementation touchpoints

Migration work adds the enum column, relaxes transport nullability, and adds the provider-only unique index. Existing rows retain LiteLLM behavior through the constant default. No data-rewriting migration is needed. Layer 2 must install the no-connection boundary before provider registrations can be created or loaded into executable gateway paths

| Existing location | Later implementation change |
| --- | --- |
| `litellm/proxy/schema.prisma` and migrations | Enum, column, nullable transport, partial unique index |
| `litellm/types/mcp.py` | Shared execution-mode vocabulary |
| `NewMCPServerRequest` / `UpdateMCPServerRequest` in `litellm/proxy/_types.py` | Mode and mode-dependent validation |
| `litellm/models/mcp_server.py` | Management response/storage representation |
| `litellm/types/mcp_server/mcp_server_manager.py` and registry loading | Preserve mode; exclude provider registrations from connection-capable use |
| Existing server repository and config loader | Canonical storage, partial updates, config collision checks |
| Management UI/API and tool selectors | Stored allowed-tool choices for provider mode, with no `/mcp-rest/tools/list` call |
| Existing MCP auth resolver | Reuse server/tool grants, groups, toolsets, denies, and applicable ceilings |
| LLM request dispatch | Resolve and authorize effective native attachments after config/translation and before dispatch |

Existing `metadata.allowed_tools` remains an independent compatibility restriction. It does not grant access that object permissions or server restrictions deny. The exact rollout setting for strict native-attachment enforcement belongs to layer 4, without changing this storage contract

## Acceptance checks for the implementation layers

| Area | Required proof |
| --- | --- |
| Schema and compatibility | Old registrations load as `litellm`; provider records round-trip with null transport, no upstream credentials, and exact allowed tools |
| Field validation | Mode/URL/list validation; partial updates; enforced empty list; duplicate endpoint rejection |
| Management | API and UI create/edit/select the same server and tool IDs without live MCP discovery |
| Toolsets | Mixed-mode toolsets remain assignable; grants cannot change execution mode or exceed server/team ceilings |
| Isolation | Zero outbound MCP initialize/list/call/probe activity for provider registrations, including startup, refresh, and malicious gateway references |
| Authorization | Registered allowed native calls succeed; unknown URL, wrong mode, disallowed tools, unbounded scope, and narrower key/team grants reject before provider dispatch |
| API coverage | OpenAI Responses and both Anthropic Messages formats; request-supplied and model-configured attachments; streaming and non-streaming |
| Live evidence | Real providers execute authorized read-only MCP tools; denied requests produce zero provider dispatches; existing gateway execution still works |

The approved database slice was committed as `fd0ac3ca76` and the approved isolation slice as `42ea534932`. Stored records can represent both modes, while the connection-capable `MCPServer` type permits only `litellm`. The redundant named gateway-mode alias is replaced with a literal annotation, leaving one named execution-mode enum

Provider records remain excluded during incremental add/update and database catalog reload, before cached gateway entries can be reused. Direct table-to-runtime conversion and both shared client-creation boundaries reject provider mode before OAuth discovery, credential resolution, or MCP connection. YAML loading still rejects provider mode until its inventory path is implemented

The current review slice enables provider registration on the existing create API, canonicalizes HTTPS endpoint identity, requires an explicit finite tool list, and rejects active gateway connection settings. Create persists null transport/auth fields and an enforced allowlist. Update validates the merged record under a row lock, preserves omitted fields, rejects null tool lists, and rejects changes to execution mode. The existing provider-only unique index rejects duplicate endpoints with HTTP 409

Admin management inventory includes active provider records. Detail reads return stored allowed tools without running health probes. The UI, scoped key/team selectors, toolset integration, YAML provider inventory, and native LLM admission authorization remain pending. Registration management alone does not yet enforce native-request permissions

Management inventory and executable servers share one revision-based catalog snapshot. A refresh reads active rows once, retains provider registrations for management, and excludes them from runtime construction. Management listing does not issue a second database inventory query

Create normalizes provider input in a dedicated request-model validator, then serializes once. Update locks and reads the row, merges the raw patch, validates through the same request model, then serializes and writes. Gateway credential merging retains its existing flow and transaction rules. No extra credential-cleanup SQL or null-preservation flag is needed

For LiteLLM execution, an empty `allowed_tools` list keeps its existing unrestricted meaning unless `mcp_info.tool_allowlist_enforced=true` is set. Provider registration always persists that marker as true, making its empty list deny all

Database-slice validation passed with Prisma schema validation, identical schema copies, the repository's no-data-rewriting migration check, and six checks on a disposable PostgreSQL 16 database built from the pre-change schema. Those checks cover existing rows and duplicate gateway URLs, new legacy defaults, provider records with null connection fields and exact tool lists, duplicate provider URL rejection, distinct provider URLs with empty tool lists, and invalid/null mode rejection. Application-level URL canonicalization, field validation, and authorization are not covered by this database-only proof

Runtime isolation was exercised against a real local MCP server. The existing gateway successfully initialized, listed tools, and called an echo tool, producing 11 HTTP requests. Provider-mode table conversion, config loading, manager client creation, and direct upstream client creation were rejected with zero additional HTTP requests. Provider rows produced zero executable catalog entries. This proves the local execution boundary, not native-provider request authorization

The runtime slice passes 369 focused model, request, catalog, and MCP utility tests, along with Ruff and the test lint rules. A type-check comparison of the six changed runtime files found no new diagnostics against the approved commit, and strict Ruff rule counts did not increase. These are focused slice checks, not a full repository CI run

Registration API validation uses a real local proxy and disposable PostgreSQL database. The live checks cover create/read/admin inventory, partial updates, explicit empty lists, invalid/null tool scopes, immutable mode, duplicate endpoints, rejected connection settings, gateway-listing denial, and concurrent tool-list/description edits. The prior run verified null connection fields and zero outbound connections. Revised validation checks absence of actual credentials without requiring a particular JSON-null representation. Evidence is local to the review workspace and contains no customer credentials

The registration slice passes 899 focused tests, including the CI directory-coverage check, and 16 live API/database checks. The new API tests retain legacy gateway update coverage alongside provider-specific validation


Revision review follows the code in this order

1. `NewMCPServerRequest.normalize_registration` owns provider normalization, separately from transport validation
2. `update_mcp_server` reads the stored execution mode. Provider updates use the existing writer with a preparation callback under the row lock, validating the merged registration through `NewMCPServerRequest`. Only supplied fields plus normalized `mcp_info` are serialized for the patch. Explicit null metadata clears the value, while omitted tools remain untouched
3. `CatalogSnapshots._reload` reads active records once. The published snapshot contains both management inventory and gateway-only executable servers. `_resolve_accessible_mcp_servers` applies visibility to that cached inventory
4. Provider credentials use the existing empty representation. No additional SQL update enforces a distinction between SQL NULL and JSON null

The revision passed 966 focused tests plus 132 catalog/manager tests. The live proxy and PostgreSQL run passed 20 checks including explicit metadata clearing, inventory refresh after updates and deletion, and concurrent edits. The provider endpoint received zero connections. Changed-file Ruff checks passed, strict rule counts did not increase, and a five-file type-check comparison introduced no diagnostics against the approved base. These revisions were reviewed and approved


The latest revision restores the gateway credential-update flow and its existing transaction rules. The shared writer accepts a preparation callback only for provider updates, keeping their stored-row validation under the lock. Neutral connection fields remain accepted without enabling upstream activity. No strict five-field rewrite or scoped-retrieval implementation was added

Validation after this trim passed 967 focused tests, followed by 606 affected tests after the final adjustment. All 24 live proxy/PostgreSQL checks passed, including neutral fields on create/update, unchanged records after rejection, deleted-record errors, and concurrent edits. Provider connections remained zero. Type comparison found no new diagnostics and strict Ruff rule counts did not increase. The revision was reviewed in VS Code and approved for commit and push
