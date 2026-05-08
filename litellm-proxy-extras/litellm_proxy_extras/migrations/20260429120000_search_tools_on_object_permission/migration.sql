-- Search tool allowlists live on LiteLLM_ObjectPermissionTable (with agents, MCP, vector stores).
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "search_tools" TEXT[] DEFAULT ARRAY[]::TEXT[];
