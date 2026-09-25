# 09 Agent and MCP gateway

Agentic surfaces of the gateway: MCP server registration and the `/mcp` protocol endpoint, MCP auth and access control, toolsets and tool search, A2A agents, managed agents, workflow runs, memory, the skills gateway, the Claude Code gateway, and search tools. Guardrails applied to MCP calls are described in [06-guardrails-policies.md](06-guardrails-policies.md); the search endpoint itself lives in [01-llm-endpoints.md](01-llm-endpoints.md) (`llm.search`)

## MCP

### mcp.servers: MCP servers (/v1/mcp/server CRUD, UI, config mcp_servers)
surfaces: api, ui, config | flags: db
docs: https://docs.litellm.ai/docs/mcp, https://docs.litellm.ai/docs/mcp_deployment, https://docs.litellm.ai/docs/mcp_config_reference
code: `litellm/proxy/management_endpoints/mcp_management_endpoints.py` (POST/GET/DELETE `/server`, `/server/import`, `/server/register`, `/server/health`), `litellm/proxy/_experimental/mcp_server/` (server manager), `mcp_servers.json`
tests: `tests/test_litellm/proxy/_experimental/mcp_server/`, `tests/e2e/mcp/`
registry: mcp.yaml mcp.*, mgmt.yaml mgmt.mcp_server.*
verify: POST /v1/mcp/server with a remote MCP url, then open http://localhost:4000/ui/?page=mcp-servers and confirm it lists

### mcp.protocol_endpoint: /mcp endpoint (list tools, call tool, streamable http, sse)
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/mcp, https://docs.litellm.ai/docs/mcp_usage
code: `litellm/proxy/proxy_server.py` (`/{mcp_server_name}/mcp`, `/mcp/proxy`, `/toolset/{toolset_name}/mcp`), `litellm/proxy/_experimental/mcp_server/server.py`
tests: `tests/e2e/mcp/`
registry: mcp.yaml mcp.list_tools.*, mcp.call_tool.*
verify: point an MCP client at http://localhost:4000/{server_name}/mcp with a virtual key and run tools/list then tools/call

### mcp.rest_api: MCP REST API and OpenAPI tools
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/mcp_rest_api, https://docs.litellm.ai/docs/mcp_openapi
code: `litellm/proxy/_experimental/mcp_server/rest_endpoints.py` (GET `/tools/list`, POST `/tools/call`, `/test/connection`), `litellm/proxy/management_endpoints/mcp_management_endpoints.py` (GET `/openapi-registry`)
tests: `tests/test_litellm/proxy/_experimental/mcp_server/`
registry: mcp.yaml mcp.* (rest surface cases)
verify: GET http://localhost:4000/mcp-rest/tools/list -H "Authorization: Bearer sk-1234" and POST /tools/call with a tool name and arguments

### mcp.in_chat_tools: MCP tools inside chat completions / responses
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/mcp
code: `litellm/proxy/proxy_server.py` (`chat_completion` MCP tool handling), `litellm/proxy/_experimental/mcp_server/`
tests: `tests/e2e/mcp/` (chat completion MCP coverage)
registry: mcp.yaml mcp.call_tool.*
verify: send /chat/completions with an MCP tool exposed by a registered server and confirm the tool executes and results return inline

### mcp.auth: MCP auth (api_key, per-user auth, OAuth, OAuth passthrough, OBO, ID-JAG, SigV4)
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/mcp_authentication, https://docs.litellm.ai/docs/mcp_oauth, https://docs.litellm.ai/docs/mcp_oauth_passthrough, https://docs.litellm.ai/docs/mcp_obo_auth, https://docs.litellm.ai/docs/mcp_id_jag, https://docs.litellm.ai/docs/mcp_aws_sigv4, https://docs.litellm.ai/docs/mcp_per_user_auth
code: `litellm/proxy/_experimental/mcp_server/byok_oauth_endpoints.py` (`/v1/mcp/oauth/*`), `litellm/proxy/_experimental/mcp_server/discoverable_endpoints.py` (OAuth discovery, `/authorize`, `/token`, `/register`), `litellm/proxy/management_endpoints/mcp_management_endpoints.py` (`/server/oauth/*`, `/user-credentials`)
tests: `tests/test_litellm/proxy/_experimental/mcp_server/`, `tests/e2e/mcp/` (OAuth happy path tests)
registry: mcp.yaml mcp.call_tool.oauth.*, mcp.call_tool.api_key.*
verify: register an OAuth-protected MCP server, run the authorize/token dance through /v1/mcp/oauth, then tools/call with the issued token

### mcp.access_control: MCP access control (key/team/org grants, client allowlist, zero trust, internal ranges)
surfaces: api, ui, config | flags: none
docs: https://docs.litellm.ai/docs/mcp_control, https://docs.litellm.ai/docs/mcp_grant_access, https://docs.litellm.ai/docs/mcp_client_allowlist, https://docs.litellm.ai/docs/mcp_zero_trust, https://docs.litellm.ai/docs/mcp_public_internet
code: `litellm/proxy/management_endpoints/mcp_management_endpoints.py` (`/access_groups`, `/tools`, `/network/client-ip`), `litellm/proxy/_experimental/mcp_server/`
tests: `tests/test_litellm/proxy/_experimental/mcp_server/`, `tests/e2e/mcp/` (permission denial tests)
registry: mcp.yaml mcp.list_tools.api_key.*
verify: scope an MCP server to one team, call /mcp tools/list with a foreign key, and confirm tools are hidden or denied

### mcp.toolsets: Toolsets and tool search / semantic filter
surfaces: api, ui | flags: none
docs: https://docs.litellm.ai/docs/mcp_toolsets, https://docs.litellm.ai/docs/mcp_tool_search, https://docs.litellm.ai/docs/mcp_semantic_filter
code: `litellm/proxy/management_endpoints/mcp_management_endpoints.py` (`/toolset` CRUD), `litellm/proxy/proxy_server.py` (`/toolset/{toolset_name}/mcp`), `litellm/proxy/hooks/mcp_semantic_filter/`, `litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py` (`/get/mcp_semantic_filter_settings`, `/get/mcp_tool_search_settings`)
tests: `tests/test_litellm/proxy/_experimental/mcp_server/`
registry: mgmt.yaml mgmt.mcp_toolset.*
verify: POST /toolset with a subset of tools, then hit /toolset/{name}/mcp and confirm only the subset is listed

### mcp.cost: MCP cost tracking
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/mcp_cost
code: `litellm/proxy/_experimental/mcp_server/` (cost per call), `litellm/proxy/spend_tracking/spend_management_endpoints.py` (spend rows)
tests: `tests/e2e/mcp/`
registry: mcp.yaml mcp.call_tool.*
verify: set a cost on a tool call per the doc, run tools/call, and check /spend/logs for the MCP spend row

### mcp.guardrails: Guardrails on MCP calls
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/mcp_guardrail
code: `litellm/proxy/guardrails/guardrail_hooks/mcp_security/`, `litellm/proxy/guardrails/guardrail_registry.py`
tests: `tests/e2e/guardrails/`, `tests/e2e/mcp/`
registry: guardrail.yaml guardrail.mcp_security.*
verify: attach a guardrail to an MCP server, call a tool with disallowed input, and confirm the call is blocked

### mcp.server_submissions: MCP server submissions and registry
surfaces: api, ui | flags: none
docs: https://docs.litellm.ai/docs/mcp_server_submissions
code: `litellm/proxy/management_endpoints/mcp_management_endpoints.py` (`/server/submissions`, `/server/{server_id}/approve`, `/server/{server_id}/reject`, `/registry.json`, `/discover`)
tests: `tests/test_litellm/proxy/_experimental/mcp_server/`
registry: mgmt.yaml mgmt.mcp_server.*
verify: submit a server through the submission flow, approve it via PUT /server/{id}/approve, and see it in /server list

### mcp.user_managed: User-managed MCP servers
surfaces: config, ui | flags: none
docs: https://docs.litellm.ai/docs/mcp
code: `litellm/proxy/management_endpoints/mcp_management_endpoints.py` (`/server/{server_id}/user-credential`, `/user-env-vars`), `litellm/proxy/_experimental/mcp_server/`
tests: `tests/test_litellm/proxy/_experimental/mcp_server/`
registry: none
verify: enable user-managed servers, add a server as a non-admin user in the UI, and confirm only that user sees it

## Agents

### a2a.agents: A2A agents (/v1/agents CRUD, agent card, invoke)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/a2a, https://docs.litellm.ai/docs/a2a_agent_card, https://docs.litellm.ai/docs/a2a_invoking_agents
code: `litellm/proxy/agent_endpoints/endpoints.py` (`/v1/agents` CRUD), `litellm/proxy/agent_endpoints/a2a_endpoints.py` (`/.well-known/agent-card.json`, `/{agent_id}/message/send`), `litellm/proxy/a2a/endpoints.py` (`/v1/a2a/discover`)
tests: `tests/e2e/a2a/`
registry: other.yaml other.a2a.*
verify: POST /v1/agents to register an agent, GET /a2a/{id}/.well-known/agent-card.json, then POST /a2a/{id}/message/send and check the JSON-RPC reply

### a2a.permissions_budgets: Agent permissions, iteration budgets, kill switch, cost tracking
surfaces: api, ui | flags: none
docs: https://docs.litellm.ai/docs/a2a_agent_permissions, https://docs.litellm.ai/docs/a2a_iteration_budgets, https://docs.litellm.ai/docs/a2a_kill_switch, https://docs.litellm.ai/docs/a2a_cost_tracking, https://docs.litellm.ai/docs/a2a_agent_headers
code: `litellm/proxy/agent_endpoints/endpoints.py` (POST `/v1/agents/{agent_id}/kill_switch`, permissions fields), `litellm/proxy/hooks/max_iterations_limiter.py`
tests: `tests/e2e/a2a/`
registry: other.yaml other.a2a.*
verify: set an iteration budget on an agent, invoke until the cap, then POST /v1/agents/{id}/kill_switch and confirm further calls are denied

### agents.managed_agents: Managed agents / agentic loop hook
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/managed_agents, https://docs.litellm.ai/docs/proxy/agentic_loop_hook
code: `litellm/litellm_core_utils/` (agentic loop in completions), `litellm/proxy/agent_endpoints/endpoints.py`
tests: `tests/test_litellm/litellm_core_utils/test_chat_completion_agentic_loop.py`
registry: none
verify: enable the agentic loop on a model with tools, send a multi-step prompt, and confirm the loop iterates until done

### agents.workflows: Workflow runs
surfaces: api, ui | flags: db
docs: none found
code: `litellm/proxy/management_endpoints/workflow_management_endpoints.py` (`/v1/workflows/runs` CRUD, `/events`, `/messages`), `ui/litellm-dashboard/src/app/(dashboard)/workflows/page.tsx`
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: mgmt.yaml mgmt.workflow.*
verify: POST /v1/workflows/runs, then open http://localhost:4000/ui/?page=workflows and inspect the run detail

### agents.memory: Memory (/v1/memory)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/memory, https://docs.litellm.ai/docs/memory_management
code: `litellm/proxy/memory/memory_endpoints.py` (POST/GET/PUT/DELETE `/v1/memory`), `ui/litellm-dashboard/src/app/(dashboard)/memory/page.tsx`
tests: `tests/test_litellm/proxy/memory/`
registry: none
verify: POST /v1/memory {"key":"k","value":"v"}, GET it back, and see it on http://localhost:4000/ui/?page=memory

### agents.skills_gateway: Skills gateway (/v1/skills)
surfaces: api, ui | flags: none
docs: https://docs.litellm.ai/docs/skills, https://docs.litellm.ai/docs/skills_gateway
code: `litellm/proxy/anthropic_endpoints/skills_endpoints.py` (POST/GET/DELETE `/v1/skills`), `litellm/proxy/discovery_endpoints/agent_skills_endpoints.py`, `ui/litellm-dashboard/src/app/(dashboard)/skills/page.tsx`
tests: `tests/test_litellm/proxy/` (skills files)
registry: none
verify: POST /v1/skills to register a skill, then open http://localhost:4000/ui/?page=skills and confirm it lists

### agents.claude_code_gateway: Claude Code gateway managed settings and plugins
surfaces: api, config, ui | flags: none
docs: https://docs.litellm.ai/docs/tutorials/claude_code_gateway, https://docs.litellm.ai/docs/tutorials/claude_code_plugin_marketplace
code: `litellm/proxy/anthropic_endpoints/claude_code_endpoints/claude_code_marketplace.py` (`/claude-code/marketplace.json`, `/claude-code/plugins/*`), `litellm/proxy/anthropic_endpoints/gateway_endpoints.py` (GET `/managed/settings`)
tests: `tests/test_litellm/proxy/` (claude code gateway files)
registry: llm_claude_code_compat.yaml (compat cells for the gateway surface)
verify: set `general_settings.enable_claude_code_gateway: true`, GET /managed/settings, and confirm a `claude` CLI pointed at the proxy picks up managed settings

### agents.search_tools: Search tools management
surfaces: api, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/ui_search_tools, https://docs.litellm.ai/docs/search/index
code: `litellm/proxy/search_endpoints/search_tool_management.py` (`/search_tools` CRUD, `/search_tools/test_connection`), `ui/litellm-dashboard/src/app/(dashboard)/search-tools/page.tsx`
tests: `tests/test_litellm/proxy/` (search tools files)
registry: none
verify: POST /search_tools with a provider config, open http://localhost:4000/ui/?page=search-tools, and confirm it lists; call /v1/search with the tool

### agents.vector_stores_ui: Vector stores management UI
surfaces: ui | flags: none
docs: https://docs.litellm.ai/docs/vector_stores/managed_vector_stores
code: `ui/litellm-dashboard/src/app/(dashboard)/vector-stores/page.tsx`, `litellm/proxy/vector_store_endpoints/management_endpoints.py` (`/vector_store/*`)
tests: `tests/test_litellm/vector_stores/`
registry: llm_nonconversational.yaml llm.vector_stores.*
verify: open http://localhost:4000/ui/?page=vector-stores, create a store, and confirm it appears via GET /vector_store/list
