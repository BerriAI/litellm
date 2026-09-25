# 12 SDK utilities

Features that exist only inside the Python SDK and never cross the proxy boundary: the client-side Router, token and cost helpers, exception mapping, BudgetManager, batch helpers, streaming utilities, the proxy-pointed client mode, global settings, and third-party library compatibility. When the same capability exists through the proxy it is filed under its own domain (for example budgets in [05-budgets-ratelimits-spend.md](05-budgets-ratelimits-spend.md))

## SDK utilities

### sdk.router_class: litellm.Router (client-side load balancing)
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/routing, https://docs.litellm.ai/docs/proxy/load_balancing
code: `litellm/router.py` (`Router`), `litellm/types/router.py` (`RouterConfig`, `RouterGeneralSettings`)
tests: `tests/test_litellm/test_router.py`, `tests/test_litellm/router_utils/`
registry: reliability.yaml reliability.routing.* (same logic proxied)
verify: python -c 'import litellm; r=litellm.Router(model_list=[{"model_name":"m","litellm_params":{"model":"gpt-5.5","api_key":"..."}}]); print(r.completion(model="m",messages=[{"role":"user","content":"hi"}]))'

### sdk.token_counter: token_counter, encode/decode, get_max_tokens
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/count_tokens
code: `litellm/litellm_core_utils/token_counter.py`, `litellm/litellm_core_utils/default_encoding.py`, `litellm/utils.py` (`get_max_tokens`)
tests: `tests/test_litellm/litellm_core_utils/test_token_counter.py`, `tests/test_litellm/litellm_core_utils/test_tokenizer.py`
registry: none
verify: python -c 'import litellm; print(litellm.token_counter(model="gpt-5.5", messages=[{"role":"user","content":"hi"}]))'

### sdk.completion_cost: completion_cost, cost_per_token
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/proxy/cost_tracking, https://docs.litellm.ai/docs/completion/usage
code: `litellm/litellm_core_utils/llm_cost_calc/` (`completion_cost`, `cost_per_token`)
tests: `tests/test_litellm/litellm_core_utils/llm_cost_calc/`
registry: none
verify: python -c 'import litellm; r=litellm.completion(model="gpt-5.5",messages=[{"role":"user","content":"hi"}],mock_response="ok"); print(litellm.completion_cost(completion_response=r))'

### sdk.exception_mapping: Exception mapping to OpenAI exception types
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/exception_mapping
code: `litellm/litellm_core_utils/exception_mapping_utils.py`, `litellm/proxy/auth/auth_exception_handler.py` (proxy-side error shape)
tests: `tests/test_litellm/litellm_core_utils/` (exception mapping files)
registry: none
verify: call litellm.completion with a bad key and confirm the raised exception is litellm.AuthenticationError / openai.AuthenticationError compatible

### sdk.budget_manager: BudgetManager
surfaces: sdk | flags: none
docs: none found
code: `litellm/budget_manager.py` (`BudgetManager`)
tests: `tests/test_litellm/` (budget manager related files)
registry: none
verify: python -c 'import litellm; b=litellm.BudgetManager(project_name="t"); b.create_budget(10, "user1"); print(b.is_valid_user("user1"))'

### sdk.batch_completion: batch_completion, batch_completion_models
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/completion/batching
code: `litellm/main.py` (`batch_completion`, `batch_completion_models`)
tests: `tests/test_litellm/test_batch_completion_models_all_responses.py`
registry: none
verify: python -c 'import litellm; print(litellm.batch_completion(model="gpt-5.5", messages=[[{"role":"user","content":"a"}],[{"role":"user","content":"b"}]], mock_response="ok"))'

### sdk.async_and_streaming_helpers: acompletion, stream_chunk_builder, CustomStreamWrapper
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/completion/stream
code: `litellm/main.py` (`acompletion`), `litellm/litellm_core_utils/streaming_handler.py` (`CustomStreamWrapper`), `litellm/main.py` (`stream_chunk_builder`)
tests: `tests/test_litellm/litellm_core_utils/test_streaming_handler.py`, `tests/test_litellm/litellm_core_utils/test_streaming_chunk_builder_utils.py`
registry: none
verify: python -c 'import asyncio, litellm; asyncio.run(litellm.acompletion(model="gpt-5.5", messages=[{"role":"user","content":"hi"}], mock_response="ok"))'

### sdk.proxy_client: SDK against proxy (api_base, proxy_auth, litellm_proxy provider)
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/providers/litellm_proxy, https://docs.litellm.ai/docs/proxy_auth, https://docs.litellm.ai/docs/proxy_server
code: `litellm/main.py` (`litellm_proxy` provider), `litellm/proxy/` (virtual key auth used by the client)
tests: `tests/test_litellm/` (litellm_proxy provider files)
registry: none
verify: point the OpenAI SDK at the proxy: `openai.OpenAI(base_url="http://localhost:4000", api_key="sk-1234").chat.completions.create(model="gpt-5.5", messages=[...])`

### sdk.set_keys: set_verbose, api keys, api_base, headers, global settings
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/set_keys
code: `litellm/__init__.py` (module-level settings), `litellm/main.py`
tests: `tests/test_litellm/test_main.py`
registry: none
verify: python -c 'import litellm; litellm.set_verbose=True; litellm.api_base="..."; litellm.completion(...)' and watch the verbose request log

### sdk.adapters: Adapters (creating_adapters)
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/extras/creating_adapters
code: `litellm/types/adapter.py` (`AdapterItem`), `litellm/__init__.py` (`adapters`), `litellm/google_genai/adapters/`
tests: `tests/test_litellm/` (adapter related files)
registry: none
verify: register an adapter on `litellm.adapters` translating one API shape into another, and call the adapted function

### sdk.supports_x: supports_function_calling, supports_vision, etc.
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/completion/function_call, https://docs.litellm.ai/docs/completion/vision
code: `litellm/utils.py` (`supports_function_calling`, `supports_vision`, `supports_prompt_caching`, `supports_audio_input`, ...)
tests: `tests/test_litellm/` (supports_* related files)
registry: none
verify: python -c 'import litellm; print(litellm.supports_function_calling("gpt-5.5"), litellm.supports_vision("gpt-5.5"))'

### sdk.model_alias_map: model_alias_map, register_model
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/completion/model_alias
code: `litellm/__init__.py` (`model_alias_map`), `litellm/utils.py` (`register_model`)
tests: `tests/test_litellm/` (alias related files)
registry: none
verify: python -c 'import litellm; litellm.register_model({"my-model":{"input_cost_per_token":0,"output_cost_per_token":0}})' then call completion(model="my-model")

### sdk.caching: SDK-level caching (litellm.cache)
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/proxy/caching (same semantics SDK-side)
code: `litellm/caching/caching.py` (`Cache`), `litellm/__init__.py` (`cache` global)
tests: `tests/test_litellm/caching/`
registry: none
verify: python -c 'import litellm; litellm.cache=litellm.Cache(); litellm.completion(model="gpt-5.5", messages=m, mock_response="ok"); litellm.completion(model="gpt-5.5", messages=m, mock_response="different")' and confirm identical output

### sdk.callbacks: SDK callbacks (success_callback, input_callback)
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/observability/callbacks, https://docs.litellm.ai/docs/observability/custom_callback
code: `litellm/__init__.py` (`success_callback`, `failure_callback`, `input_callback`), `litellm/integrations/custom_logger.py`
tests: `tests/test_litellm/integrations/`
registry: none
verify: python -c 'import litellm; litellm.success_callback=[lambda *a,**k: print("fired", k.get("response_cost"))]; litellm.completion(model="gpt-5.5",messages=[{"role":"user","content":"hi"}],mock_response="ok")'

### sdk.langchain_openai_compat: Using with LangChain, Instructor, OpenAI SDK drop-in
surfaces: sdk | flags: none
docs: https://docs.litellm.ai/docs/langchain/langchain, https://docs.litellm.ai/docs/tutorials/instructor, https://docs.litellm.ai/docs/proxy_server
code: `litellm/main.py` (OpenAI-shaped responses), `litellm/proxy/proxy_server.py` (proxy drop-in for the OpenAI SDK)
tests: `tests/test_litellm/` (compat related files)
registry: none
verify: instantiate ChatOpenAI(openai_api_base="http://localhost:4000", model="gpt-5.5") in LangChain and invoke; or point the OpenAI SDK at the proxy
