SELECT
    EXISTS (
        SELECT 1
        FROM "LiteLLM_GuardrailsTable"
        WHERE status = 'active'
          AND COALESCE(lower(litellm_params ->> 'default_on'), 'false') NOT IN ('false', '0', 'no', 'off', 'f', 'n', '')
    ) AS "default_on_guardrails!",
    EXISTS (SELECT 1 FROM "LiteLLM_PolicyAttachmentTable") AS "policy_attachments!",
    EXISTS (
        SELECT 1
        FROM "LiteLLM_ToolTable"
        WHERE input_policy <> 'untrusted' OR output_policy <> 'untrusted'
    ) AS "tool_policies!",
    EXISTS (
        SELECT 1
        FROM "LiteLLM_CacheConfig"
        WHERE id = 'cache_config'
          AND cache_settings NOT IN ('{}'::jsonb, 'null'::jsonb, '""'::jsonb)
    ) AS "cache_settings!",
    EXISTS (
        SELECT 1
        FROM "LiteLLM_ConfigOverrides"
        WHERE config_type IN ('hashicorp_vault', 'cyberark')
    ) AS "config_overrides!",
    EXISTS (
        SELECT 1
        FROM "LiteLLM_Config"
        WHERE param_name = 'anthropic_beta_headers_reload_config'
    ) AS "anthropic_beta_headers_reload!",
    EXISTS (
        SELECT 1
        FROM "LiteLLM_Config"
        WHERE param_name = 'model_cost_map_reload_config'
    ) AS "model_cost_map_reload!"
