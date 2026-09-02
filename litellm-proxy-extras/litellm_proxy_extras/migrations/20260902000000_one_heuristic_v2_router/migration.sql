-- Atomically reserve the proxy-wide heuristic_v2 classifier slot
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ProxyModelTable_one_heuristic_v2_router"
    ON "LiteLLM_ProxyModelTable" ((litellm_params #>> '{complexity_router_config,classifier_type}'))
    WHERE (litellm_params #>> '{complexity_router_config,classifier_type}') = 'heuristic_v2';
