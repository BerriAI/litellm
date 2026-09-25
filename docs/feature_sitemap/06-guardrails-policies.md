# 06 Guardrails and policies

Content controls that run inside or alongside LLM calls: the guardrail framework and lifecycle hooks, the 50+ provider hooks under `litellm/proxy/guardrails/guardrail_hooks/`, guardrail management and scoping, the policy engine, tool policies, and MCP security. The `/guardrails/apply_guardrail` standalone endpoint is covered in [01-llm-endpoints.md](01-llm-endpoints.md); MCP server management lives in [09-agent-mcp-gateway.md](09-agent-mcp-gateway.md); provider-by-provider hook list is in [appendix-integrations.md](appendix-integrations.md)

## Guardrail framework

### guardrails.framework: Guardrail framework (pre_call, during_call, post_call, logging_only; per-request guardrails param)
surfaces: config, api, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/quick_start
code: `litellm/proxy/guardrails/guardrail_registry.py`, `litellm/proxy/guardrails/init_guardrails.py`, `litellm/integrations/custom_guardrail.py` (`CustomGuardrail`), `litellm/proxy/guardrails/guardrail_helpers.py`
tests: `tests/test_litellm/proxy/guardrails/`, `tests/e2e/guardrails/`
registry: guardrail.yaml guardrail.*.pre_call.*, guardrail.*.post_call.*, guardrail.dispatch.*
verify: add a `guardrails:` block to config.yaml with a litellm_content_filter hook on `pre_call`, send a blocked word in a chat completion, and confirm a 400

### guardrails.management_api: Guardrail management (/guardrails/*) and UI
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/guardrails/quick_start, https://docs.litellm.ai/docs/apply_guardrail
code: `litellm/proxy/guardrails/guardrail_endpoints.py` (`/guardrails` CRUD, `/guardrails/list`, `/guardrails/register`, `/guardrails/submissions`), `litellm/proxy/guardrails/usage_endpoints.py` (`/guardrails/usage/*`)
tests: `tests/test_litellm/proxy/guardrails/`, `tests/e2e/guardrails/`
registry: guardrail.yaml guardrail.dispatch.*
verify: POST /guardrails with a litellm_content_filter definition, GET /guardrails/list to see it, and use `guardrails:["name"]` in a chat completion

### guardrails.providers: Guardrail providers (presidio, bedrock, lakera, aim, aporia, azure content safety, model armor, ... 50+)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/quick_start, https://docs.litellm.ai/docs/proxy/guardrails/bedrock, https://docs.litellm.ai/docs/proxy/guardrails/lakera_ai, https://docs.litellm.ai/docs/proxy/guardrails/pii_masking_v2
code: `litellm/proxy/guardrails/guardrail_hooks/` (one module per provider), `litellm/proxy/guardrails/guardrail_registry.py`
tests: `tests/test_litellm/proxy/guardrails/guardrail_hooks/`, `tests/e2e/guardrails/`
registry: guardrail.yaml guardrail.<provider>.*
verify: add a `bedrock` guardrail with AWS creds, mark it pre_call, and confirm a disallowed prompt is blocked; full matrix in [appendix-integrations.md](appendix-integrations.md)

### guardrails.litellm_content_filter: LiteLLM content filter (built-in)
surfaces: config, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/litellm_content_filter
code: `litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/`
tests: `tests/test_litellm/proxy/guardrails/guardrail_hooks/content_filter/`, `tests/e2e/guardrails/`
registry: guardrail.yaml guardrail.litellm_content_filter.*
verify: register litellm_content_filter with blocked_words, send a blocked word in a completion, and confirm the block response names the guardrail

### guardrails.llm_as_judge: LLM-as-a-judge guardrail
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/llm_as_a_judge
code: `litellm/proxy/guardrails/guardrail_hooks/llm_as_a_judge/`
tests: `tests/test_litellm/proxy/guardrails/`, `tests/e2e/guardrails/`
registry: guardrail.yaml guardrail.llm_as_a_judge.*
verify: configure llm_as_a_judge with a judge prompt, send borderline content, and confirm the judge model decides allow/block

### guardrails.custom_code: Custom code and custom guardrail class
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail, https://docs.litellm.ai/docs/proxy/guardrails/custom_code_guardrail, https://docs.litellm.ai/docs/adding_provider/simple_guardrail_tutorial
code: `litellm/integrations/custom_guardrail.py` (`CustomGuardrail`), `litellm/proxy/guardrails/guardrail_hooks/custom_code/`
tests: `tests/test_litellm/proxy/guardrails/`
registry: none
verify: subclass `CustomGuardrail` with an async_pre_call_hook, point `guardrails` config at your file, and confirm it runs per call

### guardrails.generic_api: Generic guardrail API
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api
code: `litellm/proxy/guardrails/guardrail_hooks/generic_guardrail_api/`
tests: `tests/test_litellm/proxy/guardrails/`
registry: guardrail.yaml guardrail.generic_guardrail_api.*
verify: run a tiny HTTP service implementing the generic guardrail contract, register it, and confirm requests are screened through it

## Scoping and delivery

### guardrails.team_based: Team / key scoped guardrails and BYO guardrails
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/guardrails/team_based_guardrails
code: `litellm/proxy/guardrails/guardrail_endpoints.py` (team-scoped fields), `litellm/proxy/management_endpoints/team_endpoints.py` (`guardrails` on teams)
tests: `tests/test_litellm/proxy/guardrails/`
registry: guardrail.yaml guardrail.* (team-scoped cases)
verify: attach a guardrail to a team, call with a team key and an unrelated key, and confirm only the team traffic is screened

### guardrails.load_balancing: Guardrail load balancing
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/guardrail_load_balancing
code: `litellm/proxy/guardrails/guardrail_registry.py`, `litellm/proxy/guardrails/init_guardrails.py`
tests: `tests/test_litellm/proxy/guardrails/`
registry: none
verify: register two deployments of the same guardrail, run calls, and confirm screening traffic spreads across them

### guardrails.realtime_batch: Guardrails on realtime and batch endpoints
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/realtime_guardrails, https://docs.litellm.ai/docs/proxy/guardrails/batch_guardrails
code: `litellm/proxy/guardrails/guardrail_registry.py`, `litellm/proxy/realtime_endpoints/endpoints.py`
tests: `tests/test_litellm/proxy/guardrails/`
registry: guardrail.yaml guardrail.* (realtime/batch cases)
verify: enable a realtime guardrail, open a /v1/realtime session with a banned utterance, and confirm the session is cut

### guardrails.test_playground: Guardrail test playground
surfaces: ui, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/test_playground
code: `litellm/proxy/guardrails/guardrail_endpoints.py` (POST `/guardrails/apply_guardrail`, UI endpoints), `ui/litellm-dashboard/src/app/(dashboard)/guardrails/page.tsx`
tests: `tests/test_litellm/proxy/guardrails/`
registry: guardrail.yaml guardrail.litellm_content_filter.apply_endpoint.*
verify: open http://localhost:4000/ui/?page=guardrails, open the test playground, submit blocked text, and confirm the verdict renders

### guardrails.monitor: Guardrails monitor page
surfaces: ui, api | flags: db
docs: none found
code: `litellm/proxy/guardrails/usage_endpoints.py` (GET `/guardrails/usage/overview`, `/guardrails/usage/detail/{id}`, `/guardrails/usage/logs`), `ui/litellm-dashboard/src/app/(dashboard)/guardrails-monitor/page.tsx`
tests: `tests/test_litellm/proxy/guardrails/`
registry: none
verify: run a few guarded calls, then open http://localhost:4000/ui/?page=guardrails-monitor and confirm block/allow counts appear

## Policies

### policies.engine: Policy engine (policies, policy tags, templates)
surfaces: api, ui, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies, https://docs.litellm.ai/docs/proxy/guardrails/policy_tags, https://docs.litellm.ai/docs/proxy/guardrails/policy_templates
code: `litellm/proxy/policy_engine/policy_endpoints.py` (`/policies` CRUD, `/policies/attachments`, `/policies/resolve`), `litellm/proxy/policy_engine/policy_resolve_endpoints.py`, `policy_templates.json`, `litellm/proxy/management_endpoints/policy_endpoints/endpoints.py` (`/policy/*`)
tests: `tests/test_litellm/proxy/policy_engine/`, `tests/test_litellm/proxy/management_endpoints/policy_endpoints/`, `tests/test_litellm/types/proxy/policy_engine/`
registry: guardrail.yaml guardrail.tool_policy.*
verify: POST /policies with a policy per the doc, attach it to a key or team via /policies/attachments, and confirm matching calls enforce it

### policies.flow_builder: Policy flow builder
surfaces: ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/policy_flow_builder
code: `ui/litellm-dashboard/src/app/(dashboard)/policies/page.tsx`, `litellm/proxy/policy_engine/policy_endpoints.py` (POST `/policies/test-pipeline`)
tests: `ui/litellm-dashboard/` (policies component tests)
registry: none
verify: open http://localhost:4000/ui/?page=policies, build a two-stage flow in the builder, and run the test pipeline on a sample call

### policies.tool_policies: Tool policies and tool permission guardrail
surfaces: api, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/tool_policies, https://docs.litellm.ai/docs/proxy/guardrails/tool_permission
code: `litellm/proxy/management_endpoints/tool_management_endpoints.py` (POST `/v1/tool/policy`, GET `/v1/tool/*`), `litellm/proxy/guardrails/guardrail_hooks/tool_permission.py`, `litellm/proxy/guardrails/guardrail_hooks/tool_policy/`
tests: `tests/test_litellm/proxy/guardrails/`, `tests/e2e/guardrails/`
registry: guardrail.yaml guardrail.tool_permission.*, guardrail.tool_policy.*, mgmt.yaml mgmt.tool_management.*
verify: register a tool_permission guardrail denying a tool name, send a completion calling that tool, and confirm the call is blocked

### policies.mcp_security: MCP security guardrails (mcp_security, mcp_jwt_signer, end user permission)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/mcp_guardrail
code: `litellm/proxy/guardrails/guardrail_hooks/mcp_security/`, `litellm/proxy/guardrails/guardrail_hooks/mcp_jwt_signer/`, `litellm/proxy/guardrails/guardrail_hooks/mcp_end_user_permission/`
tests: `tests/test_litellm/proxy/guardrails/`, `tests/e2e/mcp/`
registry: guardrail.yaml guardrail.mcp_security.*, guardrail.litellm_content_filter.pre_mcp_call.*
verify: enable mcp_security on a registered MCP server, invoke a flagged tool through /mcp, and confirm it is blocked

### policies.secret_detection: Secret detection / redaction
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/secret_detection
code: `litellm/litellm_core_utils/secret_redaction.py`, `litellm/proxy/guardrails/guardrail_hooks/` (secret detection capable hooks)
tests: `tests/test_litellm/litellm_core_utils/` (redaction files)
registry: none
verify: enable secret detection, send a prompt containing a fake API key, and confirm the key is masked in logs or blocked

### policies.prompt_injection: Prompt injection detection
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/guardrails/prompt_injection
code: `litellm/proxy/guardrails/guardrail_hooks/prompt_security/`, `litellm/proxy/guardrails/guardrail_hooks/lakera_ai_v2.py`
tests: `tests/test_litellm/proxy/guardrails/`
registry: none
verify: enable a prompt-injection-capable hook, send a jailbreak prompt, and confirm it is flagged or blocked

### policies.block_code_execution: Block code execution
surfaces: config | flags: none
docs: none found
code: `litellm/proxy/guardrails/guardrail_hooks/block_code_execution/`
tests: `tests/test_litellm/proxy/guardrails/`
registry: guardrail.yaml guardrail.block_code_execution.*
verify: enable block_code_execution, send a prompt asking the model to emit runnable code, and confirm the response is blocked
