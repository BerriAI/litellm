SELECT to_jsonb(a) AS "data!: AuthData"
FROM "LiteLLM_RetiredAgent" a
WHERE a.original_agent_id = ANY($1::text[])
ORDER BY a.original_agent_id
