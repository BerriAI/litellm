SELECT t.tag_name AS "tag_name!",
       t.description,
       COALESCE(t.models, '{}') AS "models!",
       t.model_info AS "model_info: RawJson",
       t.spend AS "spend!",
       t.budget_id,
       t.created_at AS "created_at!",
       t.created_by,
       t.updated_at AS "updated_at!",
       to_jsonb(b) AS "litellm_budget_table?: Json<Budget>"
FROM "LiteLLM_TagTable" AS t
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = t.budget_id
WHERE t.tag_name = ANY($1)
