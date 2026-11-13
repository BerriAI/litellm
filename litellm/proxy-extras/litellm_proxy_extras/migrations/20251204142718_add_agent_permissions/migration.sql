-- Add agent permission fields to LiteLLM_ObjectPermissionTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "agents" TEXT[] DEFAULT ARRAY[]::TEXT[];
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "agent_access_groups" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- Add agent_access_groups field to LiteLLM_AgentsTable  
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN IF NOT EXISTS "agent_access_groups" TEXT[] DEFAULT ARRAY[]::TEXT[];

