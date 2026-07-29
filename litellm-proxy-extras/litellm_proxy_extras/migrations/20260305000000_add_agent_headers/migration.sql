-- Add static_headers and extra_headers to LiteLLM_AgentsTable

ALTER TABLE "LiteLLM_AgentsTable"
  ADD COLUMN IF NOT EXISTS "static_headers" JSONB DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS "extra_headers"  TEXT[] DEFAULT ARRAY[]::TEXT[];
