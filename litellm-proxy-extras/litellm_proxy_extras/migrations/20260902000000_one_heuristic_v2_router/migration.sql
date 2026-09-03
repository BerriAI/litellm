ALTER TABLE "LiteLLM_ProxyModelTable"
    ADD COLUMN IF NOT EXISTS "heuristic_v2_unlimited" BOOLEAN,
    ADD COLUMN IF NOT EXISTS "heuristic_v2_license_blocked" BOOLEAN;

-- Existing rows remain NULL so pre-gating duplicates cannot break this migration.
-- Proxy startup reconciles them atomically before loading database-backed routers.
ALTER TABLE "LiteLLM_ProxyModelTable"
    ALTER COLUMN "heuristic_v2_unlimited" SET DEFAULT FALSE,
    ALTER COLUMN "heuristic_v2_license_blocked" SET DEFAULT FALSE;

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ProxyModelTable_one_heuristic_v2_router"
    ON "LiteLLM_ProxyModelTable" ((1))
    WHERE "heuristic_v2_unlimited" IS FALSE
      AND "heuristic_v2_license_blocked" IS FALSE
      AND CASE
        WHEN jsonb_typeof(litellm_params) = 'object'
            THEN (litellm_params #>> '{complexity_router_config,classifier_type}') = 'heuristic_v2'
        WHEN jsonb_typeof(litellm_params) = 'string'
            THEN (((litellm_params #>> '{}')::jsonb) #>> '{complexity_router_config,classifier_type}') = 'heuristic_v2'
        ELSE FALSE
    END;
