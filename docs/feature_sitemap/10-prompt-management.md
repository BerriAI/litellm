# 10 Prompt management

Versioned, reusable prompts: the native LiteLLM prompt store, external prompt management providers (Langfuse, Humanloop, Arize Phoenix, custom, generic API), and the Prompts UI page. Everything is flagged beta in the skeleton, so expect the surface to move. Related: [09-agent-mcp-gateway.md](09-agent-mcp-gateway.md) for the skills gateway, [08-observability.md](08-observability.md) for the integrations these providers also belong to

## Prompt management

### prompts.native: Native LiteLLM prompts (/prompts CRUD, dotprompt)
surfaces: api, ui, config | flags: db, beta
docs: https://docs.litellm.ai/docs/proxy/native_litellm_prompt
code: `litellm/proxy/prompts/prompt_endpoints.py` (POST/GET/PUT/DELETE `/prompts`, `/prompts/{id}/versions`, `/prompts/test`, `/utils/dotprompt_json_converter`), `litellm/integrations/dotprompt/`
tests: `tests/test_litellm/proxy/prompts/`
registry: none
verify: POST /prompts with a dotprompt template, render it via /prompts/test with variables, and call /chat/completions referencing the prompt id

### prompts.providers: Prompt management providers (Langfuse, Humanloop, Arize Phoenix, custom, generic API)
surfaces: config | flags: beta
docs: https://docs.litellm.ai/docs/proxy/prompt_management, https://docs.litellm.ai/docs/observability/langfuse_integration, https://docs.litellm.ai/docs/observability/humanloop, https://docs.litellm.ai/docs/proxy/arize_phoenix_prompts, https://docs.litellm.ai/docs/adding_provider/generic_prompt_management_api
code: `litellm/integrations/langfuse/langfuse_prompt_management.py`, `litellm/integrations/humanloop.py`, `litellm/integrations/custom_prompt_management.py`, `litellm/integrations/generic_prompt_management/`, `litellm/integrations/prompt_management_base.py`
tests: `tests/test_litellm/integrations/` (prompt management files)
registry: none
verify: set `litellm_settings.global_prompt_directory` pointing at a Langfuse prompt, call a completion with `prompt_id`, and confirm the managed prompt renders

### prompts.custom: Custom prompt management class
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/custom_prompt_management, https://docs.litellm.ai/docs/adding_provider/generic_prompt_management_api
code: `litellm/integrations/custom_prompt_management.py`, `litellm/integrations/prompt_management_base.py`
tests: `tests/test_litellm/integrations/`
registry: none
verify: implement a custom prompt manager class per the doc, wire it via config, and confirm prompt_id resolution goes through your class

### prompts.versioning_ui: Prompts page
surfaces: ui | flags: beta
docs: https://docs.litellm.ai/docs/proxy/native_litellm_prompt
code: `ui/litellm-dashboard/src/app/(dashboard)/prompts/page.tsx`, `litellm/proxy/prompts/prompt_endpoints.py`
tests: `ui/litellm-dashboard/` (prompts component tests)
registry: none
verify: open http://localhost:4000/ui/?page=prompts, create a prompt, publish a new version, and confirm GET /prompts/{id}/versions lists both
