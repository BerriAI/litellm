# 02 Models and providers

What deployments exist on the gateway and how they are configured: `model_list` in config.yaml, the `/model/*` management API, credentials, provider registration, the cost map, model info and health checks, and the Admin UI surfaces for models. Routing behavior between deployments lives in [03-routing-reliability.md](03-routing-reliability.md); the inference endpoints themselves live in [01-llm-endpoints.md](01-llm-endpoints.md); the provider x endpoint capability matrix is in [appendix-providers.md](appendix-providers.md)

## Model configuration and management

### models.model_list: model_list config (deployments, litellm_params, model_info)
surfaces: config, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/configs, https://docs.litellm.ai/docs/proxy/model_management
code: `litellm/proxy/proxy_server.py` (config load), `litellm/router.py` (`Router` model_list handling), `litellm/proxy/_types.py` (`ConfigYAML`)
tests: `tests/test_litellm/proxy/`, `tests/test_litellm/test_router.py`
registry: mgmt.yaml mgmt.model.*
verify: add a `model_list` entry to config.yaml, boot the proxy, and GET /model/info to see the deployment

### models.management_api: /model/new, /model/update, /model/delete, /model/info
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/model_management, https://docs.litellm.ai/docs/proxy/team_model_add
code: `litellm/proxy/management_endpoints/model_management_endpoints.py` (POST `/model/new`, `/model/update`, `/model/delete`, `/model/block`), `litellm/proxy/proxy_server.py` (GET `/model/info`)
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/management/`
registry: mgmt.yaml mgmt.model.*
verify: curl -X POST http://localhost:4000/model/new -H "Authorization: Bearer sk-1234" -d '{"model_name":"t","litellm_params":{"model":"gpt-5.5","api_key":"os.environ/OPENAI_API_KEY"}}' then GET /model/info

### models.wildcard: Wildcard models (provider/*)
surfaces: config, api | flags: none
docs: https://docs.litellm.ai/docs/wildcard_routing
code: `litellm/router.py` (wildcard resolution), `litellm/litellm_core_utils/` (model name parsing)
tests: `tests/test_litellm/test_router.py`
registry: none
verify: add `model_name: openai/*` to model_list, then POST /chat/completions with an unconfigured openai model name and confirm it routes

### models.aliases: Model aliases and model_group_alias
surfaces: config, api | flags: none
docs: https://docs.litellm.ai/docs/completion/model_alias, https://docs.litellm.ai/docs/proxy/model_access_guide
code: `litellm/router.py`, `litellm/proxy/proxy_server.py`
tests: `tests/test_litellm/test_router.py`
registry: none
verify: set `model_group_alias: {"fast":"gpt-5.5"}` in config, call /chat/completions with model `fast`, confirm it lands on gpt-5.5

### models.credentials: Reusable credentials (/credentials, credential_name)
surfaces: api, config, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/model_management
code: `litellm/proxy/credential_endpoints/endpoints.py` (POST `/credentials`, GET `/credentials`, DELETE `/credentials/{name}`)
tests: `tests/test_litellm/proxy/credential_endpoints/`
registry: mgmt.yaml mgmt.credential.*
verify: curl -X POST http://localhost:4000/credentials -H "Authorization: Bearer sk-1234" -d '{"credential_name":"oai","credential_values":{"api_key":"sk-..."},"credential_info":{}}' then use `credential_name` in /model/new

### models.credential_routing: Credential routing (per-key provider credentials)
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/credential_routing
code: `litellm/proxy/credential_endpoints/endpoints.py`, `litellm/proxy/litellm_pre_call_utils.py`
tests: `tests/test_litellm/proxy/credential_endpoints/`
registry: none
verify: attach credentials to a virtual key, call /chat/completions with that key, and confirm the request uses the key-scoped credential

### models.provider_registry: Provider registry (LlmProviders, custom_llm_provider, provider_endpoints_support)
surfaces: sdk, config | flags: none
docs: https://docs.litellm.ai/docs/provider_registration/index, https://docs.litellm.ai/docs/providers/openai
code: `litellm/types/utils.py` (`LlmProviders`), `provider_endpoints_support.json`, `litellm/litellm_core_utils/` (provider resolution in `litellm/main.py`)
tests: `tests/test_litellm/test_main.py`
registry: none
verify: python -c 'from litellm import LlmProviders; print(len(LlmProviders))' and cross-check a provider dir under `litellm/llms/`

### models.openai_compatible_custom: Add OpenAI-compatible provider via JSON / custom api_base
surfaces: sdk, config | flags: none
docs: https://docs.litellm.ai/docs/providers/openai_compatible, https://docs.litellm.ai/docs/contributing/adding_openai_compatible_providers
code: `litellm/llms/openai_like/`, `litellm/main.py` (`openai_like` dispatch)
tests: `tests/test_litellm/llms/openai_like/`
registry: none
verify: add a model_list entry with `model: openai/<name>` and `api_base` pointing at an OpenAI-compatible server, then call /chat/completions

### models.custom_llm: Custom LLM handler class (CustomLLM)
surfaces: sdk, config | flags: none
docs: https://docs.litellm.ai/docs/providers/custom_llm_server
code: `litellm/llms/custom_llm.py` (`CustomLLM`)
tests: `tests/test_litellm/llms/` (custom llm related files)
registry: none
verify: subclass `litellm.CustomLLM`, register it under `custom_provider_map`, and call `litellm.completion` with `custom_llm_provider` set

### models.cost_map: Model cost map and context windows (model_prices_and_context_window.json)
surfaces: sdk, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/custom_model_cost_map, https://docs.litellm.ai/docs/provider_registration/add_model_pricing
code: `model_prices_and_context_window.json`, `litellm/litellm_core_utils/` (cost map load in `litellm/utils.py`), `litellm/proxy/proxy_server.py` (`/model/cost_map/source`, `/reload/model_cost_map`)
tests: `tests/test_litellm/` (cost map related files)
registry: none
verify: set `LITELLM_LOCAL_MODEL_COST_MAP=true`, boot, then GET /model/cost_map/source and POST /reload/model_cost_map

### models.custom_pricing: Custom pricing per deployment (input/output cost, PTU flat cost)
surfaces: config, ui, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/custom_pricing, https://docs.litellm.ai/docs/proxy/ptu_flat_cost, https://docs.litellm.ai/docs/sdk_custom_pricing
code: `litellm/proxy/management_endpoints/model_management_endpoints.py` (`litellm_params` cost fields), `litellm/litellm_core_utils/llm_cost_calc/`
tests: `tests/e2e/llm_translation/test_custom_pricing_e2e.py`
registry: quota_management.yaml quota_management.spend_tracking.*
verify: create a deployment with `input_cost_per_token`/`output_cost_per_token`, run one call, and check /spend/logs reports the configured rate

### models.provider_margins_discounts: Provider margins, discounts, off-peak pricing
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/provider_margins, https://docs.litellm.ai/docs/proxy/provider_discounts, https://docs.litellm.ai/docs/proxy/off_peak_pricing
code: `litellm/proxy/management_endpoints/cost_tracking_settings.py` (`/config/cost_margin_config`, `/config/cost_discount_config`)
tests: `tests/test_litellm/proxy/` (cost tracking settings related files)
registry: none
verify: PATCH /config/cost_margin_config with a percent margin, run a call, and confirm /spend/logs cost reflects the margin

## Discovery, health, and UI

### models.model_info: Model capabilities lookup (supports_vision, supports_function_calling, get_model_info)
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/model_discovery
code: `litellm/utils.py` (`get_model_info`, `supports_vision`, `supports_function_calling`), `litellm/proxy/proxy_server.py` (GET `/utils/model_info`)
tests: `tests/test_litellm/` (model info related files)
registry: none
verify: python -c 'import litellm; print(litellm.get_model_info("gpt-5.5"))' or GET /utils/model_info?litellm_model_id=gpt-5.5

### models.ai_hub: AI Hub / public model hub
surfaces: ui, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/ai_hub
code: `litellm/proxy/public_endpoints/public_endpoints.py` (GET `/public/model_hub`, `/public/model_hub/info`), `litellm/proxy/public_endpoints/public_v1/model_hub.py`
tests: `tests/test_litellm/proxy/` (model hub related files)
registry: none
verify: set `general_settings.enable_public_model_hub: true`, then GET http://localhost:4000/public/model_hub and open http://localhost:4000/ui/?page=model-hub-table

### models.model_compare: Model compare UI
surfaces: ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/model_compare_ui, https://docs.litellm.ai/docs/tutorials/compare_llms
code: `ui/litellm-dashboard/` (model compare components)
tests: `ui/litellm-dashboard/` (component tests)
registry: none
verify: open http://localhost:4000/ui/?page=models-and-endpoints, pick two models, and run the same prompt in the compare view

### models.health_checks: Model health checks (/health, background health checks, health_check_routing)
surfaces: api, config, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/health, https://docs.litellm.ai/docs/proxy/health_check_routing
code: `litellm/proxy/health_endpoints/_health_endpoints.py` (GET `/health`, `/health/services`, `/health/test_connection`), `litellm/main.py` (`health_check`)
tests: `tests/test_litellm/litellm_core_utils/test_health_check_helpers.py`
registry: none
verify: curl http://localhost:4000/health?model=gpt-5.5 -H "Authorization: Bearer sk-1234" and check healthy_endpoints

### models.sync_from_github: Sync models from GitHub
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/sync_models_github
code: `litellm/proxy/management_endpoints/model_management_endpoints.py` (model sync logic)
tests: `tests/test_litellm/proxy/management_endpoints/` (sync related files)
registry: none
verify: configure the GitHub sync settings per the doc, run one sync, and confirm new deployments appear in /model/info

### models.deprecation_dates: Model deprecation and retirement dates
surfaces: sdk, ui | flags: none
docs: none found
code: `litellm/proxy/proxy_server.py` (GET `/model/deprecations`), `model_prices_and_context_window.json` (`deprecation_date` fields)
tests: `tests/test_litellm/proxy/` (deprecations related files)
registry: none
verify: GET http://localhost:4000/model/deprecations -H "Authorization: Bearer sk-1234" and check models with upcoming dates are listed

### models.playground: Admin UI Playground (chat, images, etc.)
surfaces: ui | flags: none
docs: https://docs.litellm.ai/docs/tutorials/first_playground
code: `ui/litellm-dashboard/src/app/(dashboard)/playground/page.tsx`
tests: `ui/litellm-dashboard/` (playground component tests)
registry: none
verify: open http://localhost:4000/ui/?page=llm-playground, select a model, send a message, and confirm a completion renders

### models.api_playground: API Playground / transform request
surfaces: ui, api | flags: none
docs: none found
code: `litellm/proxy/proxy_server.py` (POST `/utils/transform_request`), `ui/litellm-dashboard/src/app/(dashboard)/transform-request/page.tsx`
tests: `tests/test_litellm/proxy/` (transform request related files)
registry: none
verify: curl -X POST http://localhost:4000/utils/transform_request -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","request_body":{"messages":[{"role":"user","content":"hi"}]}}' or open http://localhost:4000/ui/?page=transform-request
