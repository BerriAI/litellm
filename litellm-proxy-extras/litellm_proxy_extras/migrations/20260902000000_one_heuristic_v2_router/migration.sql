ALTER TABLE "LiteLLM_ProxyModelTable"
    ADD COLUMN IF NOT EXISTS "heuristic_v2_unlimited" BOOLEAN NOT NULL DEFAULT FALSE;

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ProxyModelTable_one_heuristic_v2_router"
    ON "LiteLLM_ProxyModelTable" ((1))
    WHERE NOT "heuristic_v2_unlimited" AND CASE
        WHEN jsonb_typeof(litellm_params) = 'object'
            THEN (litellm_params #>> '{complexity_router_config,classifier_type}') = 'heuristic_v2'
        WHEN jsonb_typeof(litellm_params) = 'string'
            THEN (((litellm_params #>> '{}')::jsonb) #>> '{complexity_router_config,classifier_type}') = 'heuristic_v2'
        ELSE FALSE
    END;
