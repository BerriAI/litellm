# LiteLLM MCP gateway feature map

Inventory of MCP features in the codebase, mapped to the e2e coverage registry
(`mcp.yaml`, plus related `mgmt` / `guardrail` cells). Cross-checked against
customer open-MCP issues (anon export, 2026-08).

Every customer-noticeable behavior should become a registry cell, then a
`@pytest.mark.covers(...)` test.

**Primary code:** `litellm/proxy/_experimental/mcp_server/`  
**Admin API:** `litellm/proxy/management_endpoints/mcp_management_endpoints.py`  
**Types:** `litellm/types/mcp.py`, `litellm/types/mcp_server/`  
**LLM bridges:**  
- Chat completions: `litellm/responses/mcp/chat_completions_handler.py` (`acompletion_with_mcp`)  
- Responses API: `litellm/responses/main.py` (`aresponses_api_with_mcp`), `mcp_streaming_iterator.py`  
- **Messages API:** `litellm/llms/anthropic/experimental_pass_through/messages/mcp_handler.py` (`anthropic_messages_with_mcp`) wired from `messages/handler.py`  
- Shared expansion: `litellm/responses/mcp/litellm_proxy_mcp_handler.py`  
**Guardrails:** `guardrail_hooks/mcp_*`, `guardrail_translation/`

Registry grammar (MCP module):

```
mcp.<operation>.<auth_family>.<assertion>
  auth_family : none | api_key | bearer | oauth   # how the *client* authenticates to LiteLLM
```

`auth_family` is **not** the upstream MCP server auth type. Upstream auth
(oauth2, sigv4, basic, …) is expressed in the assertion or operation variant.

---

## 1. Protocol operations (MCP server surface)

Exposed via MCP protocol handlers in `server.py` and REST mirrors in
`rest_endpoints.py` (`/tools/list`, `/tools/call`, test helpers). Aggregate
route is `/mcp/` (trailing slash matters for some clients); per-server and
toolset routes exist via `dynamic_mcp_route` / `toolset_mcp_route`.

| Feature | Code | Registry cell(s) | E2E today |
| --- | --- | --- | --- |
| list_tools | `server.py` handle_list_tools; REST `/tools/list` | `mcp.list_tools.{api_key,bearer,oauth,none}.succeeds` | api_key often skipped LIT-5052; oauth bridge e2e TBD on a non-Linear real OAuth MCP |
| call_tool | `server.py` mcp_server_tool_call; REST `/tools/call` | `mcp.call_tool.{api_key,bearer,oauth,none}.succeeds` | same |
| list_tools denied without key scope | permission path | `mcp.list_tools.api_key.denied_without_permission` | yes |
| call_tool denied without permission | rest/manager | `mcp.call_tool.api_key.denied_without_permission` | skipped LIT-5052 |
| list_prompts / get_prompt | `server.py` | `mcp.list_prompts` / `mcp.get_prompt` | missing |
| list_resources / read_resource | `server.py` | `mcp.list_resources` / `mcp.read_resource` | missing (**customer pain:** Atlassian / Claude Code ListResources) |
| list_resource_templates | `server.py` | `mcp.list_resource_templates.api_key.succeeds` | missing |
| Progress on call_tool | `server.py` forward_progress | `mcp.call_tool.api_key.progress_forwarded` | missing |
| Multi-server tool namespace / prefix | short_prefix / `MCP_TOOL_PREFIX_SEPARATOR` | `mcp.list_tools.api_key.namespaced_multi_server` | missing |
| Toolset tool FQ names + separator | toolset UI/DB + prefix | `mcp.toolset.api_key.prefix_separator_honored` | missing (customer PR #34559) |
| Virtual tools: mcp_tool_search / mcp_tool_call | `tool_search.py` | `mcp.tool_search` / `mcp.tool_call_virtual` | missing |
| Partial list on upstream failure | `faults/` | `mcp.list_tools.api_key.partial_on_upstream_fault` | missing |
| Per-server MCP path (`/mcp/{alias}/…`) | dynamic route | `mcp.list_tools.api_key.per_server_route` | missing (customer open) |
| Aggregate `/mcp/` picks up new servers without restart | registry reload | `mcp.list_tools.api_key.sees_newly_added_server` | missing (customer open) |
| Trailing slash parity `/mcp` vs `/mcp/` | server routing | `mcp.list_tools.api_key.trailing_slash_parity` | partial historical fix |
| REST test connection / test list tools | `rest_endpoints.py` | `mcp.test_connection.api_key.succeeds` | missing |
| BYOK health without user token | health + is_byok | `mcp.health.api_key.byok_not_false_unhealthy` | missing (LIT-4896/5136) |

---

## 2. Client auth to the gateway (`auth_family`)

| Family | Meaning | Registry |
| --- | --- | --- |
| `api_key` | Virtual key / master key | P0 list/call + deny |
| `bearer` | Bearer token (incl. OAuth access token as Bearer) | P1 list/call |
| `oauth` | Interactive OAuth2; gateway-managed per-user tokens | P1 + chat/messages bridges |
| `none` | Anonymous / public / `delegate_auth_to_upstream` | P1 list/call |

| Related behavior | Code | Registry | Customer issue |
| --- | --- | --- | --- |
| Auth fail → 401 + WWW-Authenticate (not 500) | `extract_mcp_auth_context`, ProxyException mapping | `mcp.auth.api_key.returns_401_not_500` | #2 (PR #31011 claimed) |
| Budget exceeded on /mcp → 429 | same path | `mcp.auth.api_key.returns_429_on_budget` | #2 |
| Pre-emptive 401 for unauthenticated OAuth servers | `server.py` | oauth cells | #15 |
| Stateful session auth contexts + cap | `server.py` session managers | P2 | — |

---

## 3. Upstream server auth (`MCPAuth`)

| `auth_type` / flag | Behavior | Cell | E2E / issues |
| --- | --- | --- | --- |
| `none` | No upstream auth | generic succeeds | partial |
| static (`api_key`, `bearer_token`, `basic`, `authorization`, `token`) | Inject static headers | `upstream_static_auth` | missing dedicated |
| `oauth2` + `authorization_code` | Per-user vault | oauth cells | e2e TBD on a real OAuth MCP; UI flaky (#15) |
| `oauth2` + `client_credentials` | M2M | `upstream_oauth2_client_credentials` | missing |
| `delegate_auth_to_upstream` | Client PKCE with upstream | `delegate_auth_upstream` | missing |
| `oauth_passthrough` | Proxy metadata + 401 challenges | `oauth_passthrough` | unit |
| `true_passthrough` | Forward client Authorization | `upstream_true_passthrough` | missing |
| `oauth2_token_exchange` / OBO | RFC 8693 / entra_obo | `upstream_token_exchange` | missing |
| `oauth2_id_jag` | ID-JAG | `upstream_id_jag` | missing |
| `oauth_delegate` | Bridge / SSO assertion | `upstream_oauth_delegate` | missing |
| `aws_sigv4` | SigV4 (AgentCore etc.) | `upstream_sigv4` | unit; UI onboard flaky (#18) |
| Session handshake (Databricks-style mint) | **not implemented** as full handshake | product gap | #5 open |
| Resource metadata `https` behind TLS terminator | `oauth_utils` X-Forwarded-Proto / public base | `mcp.oauth.api_key.resource_metadata_public_https` | #16 open |

---

## 4. Transports and MCP spec versions

| Feature | Values | Notes / issues |
| --- | --- | --- |
| Transport | `sse`, `http`, `stdio` | stdio poorly e2e'd; customer #20 |
| Spec version | 2024-11-05, 2025-03-26, 2025-06-18 | |
| OpenAPI → tools | `openapi_to_mcp_generator.py` | backend exists; **UI missing** (#20) |
| gRPC transport | **not supported** | FR #23 |
| Admin transport allowlist | **not supported** | FR #22 |

---

## 5. Permission and multi-tenant safety

| Feature | Code | Registry | Customer / e2e |
| --- | --- | --- | --- |
| Key `object_permission.mcp_servers` | manager + auth_mcp | deny cells | partial |
| Empty key∩team (or similar) intersection = **deny-all** | `user_api_key_auth_mcp.py` hierarchy | `mcp.list_tools.api_key.empty_intersection_denies` | **#3 open (A2A same class; MCP too)** |
| MCP access groups | manager + e2e | `access_group_scoped` | list yes; call missing |
| Key access group grants beyond team list | product gap / disputed | `mcp.list_tools.api_key.key_access_group_beyond_team` | #4 open |
| Model-level MCP scoping | **not supported** | FR cell if added | #4 FR |
| `allowed_tools` / `disallowed_tools` | MCPServer | allowed/disallowed cells | missing e2e |
| Toolsets → permission expansion | toolset_db | toolset_scoped | missing; prefix bug #6 |
| `allowed_params` | MCPServer | params_filtered | missing |
| `allow_all_keys` | MCPServer | document + test | missing |
| User-only / exclude service accounts | **FR** | `mcp.call_tool.api_key.user_scoped_only` when built | #8 open |
| `require_key_mcp_access_defined` | auth hierarchy | cell | missing e2e |
| End-user MCP permission guardrail | mcp_end_user_permission | guardrail | missing e2e |
| Team-scoped server list | management | mgmt | missing |
| Gateway allowlist: only registered MCPs reachable | product intent | `mcp.call_tool.api_key.unregistered_server_blocked` | #17 FR |
| Invalid server name / tool prefix validation | utils / UI | `mcp.server.api_key.rejects_invalid_prefix` | #17 |

### Stable identity (config.yaml servers)

| Feature | Code | Issue |
| --- | --- | --- |
| `server_id` = hash(name, url, transport, auth_type, alias) | `_generate_stable_server_id` | **#1 OPEN:** rename/repoint URL changes ID; grants silently dangle |
| Stable id OR migrate grants + loud dangling-grant error | needed | cell: `mcp.permission.api_key.dangling_grant_errors` + identity migration |

---

## 6. Admin / management API (`/v1/mcp/...`)

Registry module often `mgmt`, not `mcp`.

| Feature | Registry | Customer |
| --- | --- | --- |
| Create / update / delete / list / health | mgmt cells | UI onboard #18; health BYOK #12 |
| Non-admin register + approve/reject | mgmt approve/reject | |
| Temporary / session MCP | gap | |
| User OAuth credential CRUD | mgmt + docs gap | **#9:** deposit API exists, docs + acting-user resolution unclear |
| User env vars CRUD | mgmt | |
| Toolsets CRUD | mgmt | prefix #6 |
| Make public / discover / registry | gap | local registry FR #19 |
| Multi-pod: UI save visible on all processes | reliability / pub-sub | **#7** (v1.96 Redis push claimed) |
| Key update must not wipe MCP toolsets/servers | key management | **#10 OPEN** |
| Remove stale deleted MCP IDs from key (UI) | key UI | **#11 LIT-3278** |
| OpenAPI converter in UI | UI | #20 |
| First-party LiteLLM admin MCP server | **FR product** | #21 |

---

## 7. LLM bridges (MCP tools inside model APIs)

All three share `LiteLLM_Proxy_MCP_Handler` for `litellm_proxy` / `litellm_proxy/mcp/...` tool references: expand tools under the caller's credentials, optional auto-execute loop.

| Surface | Entry | Registry | E2E |
| --- | --- | --- | --- |
| **Chat completions** `/v1/chat/completions` | `acompletion_with_mcp` via `main.py` | `mcp.chat_completion.{api_key,oauth}.auto_executes_tools` | e2e TBD on a shared real MCP for api_key and oauth |
| **Responses** `/v1/responses` | `aresponses_api_with_mcp` | `mcp.responses.api_key.auto_executes_tools` | missing |
| **Messages** `/v1/messages` | `anthropic_messages_with_mcp` via experimental pass-through handler | `mcp.messages.{api_key,oauth}.auto_executes_tools` | **was missing from map; no e2e** |
| Semantic tool filter on chat tools | `semantic_tool_filter` + hook | `mcp.chat_completion.api_key.semantic_filter_narrows` | missing |
| Provider-native Anthropic `mcp_servers` tool | `AnthropicMcpServerTool` / beta header | **not gateway**; provider-side MCP | separate from litellm_proxy bridge |
| Playground auto-execution | UI → same bridges | exercises chat/messages paths | customer #12 |

Claude Code / Desktop often use **messages** + OAuth + resources; that is why resources + messages bridge + OAuth metadata are P0 for gateway maturity.

---

## 8. Sampling and elicitation

| Feature | Flag / code | Registry |
| --- | --- | --- |
| Sampling createMessage → completion | `allow_sampling`, `sampling_handler.py` | `mcp.sampling.api_key.succeeds` |
| Sampling model access + budget | same | `mcp.sampling.api_key.enforces_model_access` |
| Elicitation relay | `allow_elicitation` | `mcp.elicitation.api_key.succeeds` |

---

## 9. Cost, spend, concurrency, headers

| Feature | Registry | Customer |
| --- | --- | --- |
| Per-server / per-tool cost on call_tool | `mcp.call_tool.api_key.cost_logged` | |
| list_tools spend log | `mcp.list_tools.api_key.cost_logged` | |
| UI MCP Server Activity: tool invocations not tokens | product/UI | **#13 LIT-4897** (tokens always 0) |
| max_concurrent_requests | `concurrent_limit` | |
| extra_headers / static_headers / user env vars | forward / resolve cells | |
| Timeout per server | reliability-adjacent | |

---

## 10. Guardrails on MCP

| Feature | Registry | E2E |
| --- | --- | --- |
| Content filter `pre_mcp_call` | `guardrail.litellm_content_filter.pre_mcp_call.blocks` | `test_mcp_guardrail_e2e.py` |
| MCP security hook | `guardrail.mcp_security.pre_call.blocks` | missing |
| MCP JWT signer | gap | unit |
| End-user permission | gap | missing |

---

## 11. Config load / multi-pod / lifecycle

| Feature | Notes | Issue |
| --- | --- | --- |
| YAML `mcp_servers` load | manager | #1 identity |
| DB store + reload / Redis pub-sub config sync | proxy config sync | #7 multi-process |
| Temporary server Redis cache | management | |
| Discovery well-known OAuth | discoverable + byok | #15/#16 |
| DCR | gateway_dcr_flow | |
| Operator open servers / allow_all union | manager | |

---

## 12. Customer open-issue → product/registry map

Source: open MCP issues export (26 rows, anon). Status abbreviated.

| # | Theme | Product status | Registry / e2e action |
| --- | --- | --- | --- |
| 1 | Stable server_id vs rename/URL; dangling grants silent | OPEN | `mcp.permission.api_key.stable_id_survives_url_change` or migration + `dangling_grant_errors` |
| 2 | /mcp auth 500 vs 401/429 | Claimed fixed #31011 | `returns_401_not_500`, `returns_429_on_budget` — regression e2e |
| 3 | Empty permission intersection allow-all (A2A; same class MCP) | OPEN blocking | `empty_intersection_denies` (MCP + a2a suite) |
| 4 | Model-level MCP scope; key AG beyond team | FR / open | FR cells when designed |
| 5 | Central MCP inherit without per-key; Databricks handshake | OPEN | product; handshake not in codebase |
| 6 | Toolset prefix separator | Claimed #34559 | `toolset.prefix_separator_honored` |
| 7 | Multi-process UI save not sticky | Claimed v1.96 Redis | multi-pod e2e after release |
| 8 | User-only MCP (exclude service accounts) | FR | when built |
| 9 | OBO deposit API + acting user for agents | OPEN docs/behavior | mgmt credential + `call_tool` with acting user |
| 10 | Key budget edit wipes MCP toolset | OPEN | mgmt key update regression |
| 11 | Stale MCP IDs on key UI | LIT-3278 | UI/mgmt e2e |
| 12 | BYOK health false unhealthy | LIT-4896/5136 | `byok_not_false_unhealthy` |
| 13 | MCP activity Total Tokens = 0 | LIT-4897 | UI metric |
| 14 | Per-server routes; aggregate tool list stale | OPEN | `per_server_route`, `sees_newly_added_server` |
| 15 | OAuth E2E UI + token forward + clients | OPEN / churn | oauth + messages/chat bridges |
| 16 | Resource metadata http behind TLS | OPEN | `resource_metadata_public_https` |
| 17 | Gateway allowlist unregistered MCPs; invalid names | FR | allowlist + validation cells |
| 18 | UI register / playground tools empty | OPEN/stale | mgmt + playground |
| 19 | Local/dev MCP registry governance | FR | product |
| 20 | OpenAPI UI; stdio/oauth maturity | OPEN / lost deal | openapi + transports |
| 21 | First-party LiteLLM admin MCP | FR | product |
| 22 | Transport allowlist setting | FR | product |
| 23 | gRPC transport | FR low | product |
| 24 | Resources + Atlassian OAuth via Claude Code | OPEN recurring | **resources + messages + oauth** cells |
| 25–26 | Competitive losses (MCP immaturity) | LOST | treat as quality bar, not single cells |

---

## Coverage snapshot

| Bucket | In product | In registry | Live e2e |
| --- | --- | --- | --- |
| Core list/call api_key | yes | yes | blocked LIT-5052 |
| Deny / access groups | yes | yes | partial |
| bearer / oauth / none | yes | yes | oauth skipif |
| prompts / **resources** | yes | yes | **none** (Atlassian #24) |
| LLM bridges chat | yes | yes | oauth only |
| LLM bridges **responses** | yes | yes | none |
| LLM bridges **messages** | yes | **added** | **none** |
| Auth status codes | claimed | **added** | need regression |
| Stable id / dangling grants | bug | **added** | need |
| Empty intersection deny | bug | **added** | need |
| Toolsets / multi-pod / BYOK health | partial | partial | thin |
| Model-level scope, user-only mode, gRPC, admin MCP | FR / missing | FR notes only | — |

Collector after registry expansion: **MCPs ~2/N live** until tests land; denominator is intentionally honest.

---

## Priority for new e2e (reliability / fewer regressions)

1. **P0** Unskip LIT-5052 (Datadog list/call/deny); guardrail pre_mcp_call is covered by test_mcp_guardrail_e2e.py.  
2. **P0** `/mcp` auth status codes 401/429 regression (#2).  
3. **P0** Empty permission intersection denies (#3).  
4. **P0** **messages** bridge auto-execute tools (Claude Code path) + **list_resources**.  
5. **P0** OAuth resource metadata public https (#16) + token used on tool call.  
6. **P1** Stable server_id / dangling grant loud failure (#1).  
7. **P1** allowed/disallowed tools; call_tool access groups.  
8. **P1** chat + responses auto-exec with api_key.  
9. **P1** Per-server route + sees newly added server (#14).  
10. **P1** Multi-pod MCP edit propagates (#7).  
11. **P1** Key update does not wipe MCP grants (#10).  
12. **P2** Toolset prefix, BYOK health UX, cost/invocation metrics, sampling.

---

## Related docs

- E2E MCP suite: `tests/e2e/CLAUDE.md` (real Datadog for api_key; separate real OAuth MCP for vault/oauth bridges)  
- Registry: `tests/e2e/coverage_registry/mcp.yaml`  
- MCP internal note: `litellm/proxy/_experimental/mcp_server/CLAUDE.md`  
