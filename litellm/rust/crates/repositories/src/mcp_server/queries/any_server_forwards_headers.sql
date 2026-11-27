SELECT EXISTS (
    SELECT 1
    FROM "LiteLLM_MCPServerTable" AS server, unnest(server.extra_headers) AS name
    WHERE lower(name) <> 'authorization'
) AS "forwards!"
