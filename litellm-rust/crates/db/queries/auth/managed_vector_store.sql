SELECT to_jsonb(v) AS "data!: AuthData"
FROM "LiteLLM_ManagedVectorStoresTable" v
WHERE v.vector_store_id = ANY($1::text[])
ORDER BY v.vector_store_id
