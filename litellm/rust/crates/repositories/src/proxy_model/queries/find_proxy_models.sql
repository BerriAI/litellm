SELECT model_id,
       model_name,
       litellm_params AS "litellm_params: RawJson",
       model_info AS "model_info: RawJson",
       blocked,
       created_at,
       created_by,
       updated_at,
       updated_by
FROM "LiteLLM_ProxyModelTable"
