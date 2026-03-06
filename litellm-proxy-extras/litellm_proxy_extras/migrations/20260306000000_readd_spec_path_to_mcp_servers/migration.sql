-- Re-add spec_path column to LiteLLM_MCPServerTable
-- (was dropped in 20260224203854_add_agent_object_permissions_table, now re-added to schema)

ALTER TABLE "LiteLLM_MCPServerTable"
  ADD COLUMN IF NOT EXISTS "spec_path" TEXT;
