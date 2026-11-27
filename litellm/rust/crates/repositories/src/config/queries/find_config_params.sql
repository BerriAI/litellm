SELECT param_name,
       param_value AS "param_value: RawJson",
       reload_revision
FROM "LiteLLM_Config"
WHERE param_name = ANY($1)
