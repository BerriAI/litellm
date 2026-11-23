SELECT param_name AS "param_name!",
       ARRAY(SELECT jsonb_object_keys(param_value) ORDER BY 1) AS "keys!",
       COALESCE(lower(param_value ->> 'store_model_in_db') = 'true', FALSE) AS "store_model_in_db!"
FROM "LiteLLM_Config"
WHERE param_name = ANY($1)
  AND jsonb_typeof(param_value) = 'object'
