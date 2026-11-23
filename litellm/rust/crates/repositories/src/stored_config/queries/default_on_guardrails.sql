SELECT guardrail_name
FROM "LiteLLM_GuardrailsTable"
WHERE status = 'active'
  AND COALESCE(lower(litellm_params ->> 'default_on'), 'false') NOT IN ('false', '0', 'no', 'off', 'f', 'n', '')
ORDER BY guardrail_name
