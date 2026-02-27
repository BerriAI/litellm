-- AlterTable
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN     "object_permission_id" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" DROP COLUMN "spec_path";

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN     "agent_id" TEXT;

-- CreateTable
CREATE TABLE "LiteLLM_ToolTable" (
    "tool_id" TEXT NOT NULL,
    "tool_name" TEXT NOT NULL,
    "origin" TEXT,
    "call_policy" TEXT NOT NULL DEFAULT 'untrusted',
    "call_count" INTEGER NOT NULL DEFAULT 0,
    "assignments" JSONB DEFAULT '{}',
    "key_hash" TEXT,
    "team_id" TEXT,
    "key_alias" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ToolTable_pkey" PRIMARY KEY ("tool_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_ToolTable_tool_name_key" ON "LiteLLM_ToolTable"("tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_ToolTable_call_policy_idx" ON "LiteLLM_ToolTable"("call_policy");

-- CreateIndex
CREATE INDEX "LiteLLM_ToolTable_team_id_idx" ON "LiteLLM_ToolTable"("team_id");

-- AddForeignKey
ALTER TABLE "LiteLLM_AgentsTable" ADD CONSTRAINT "LiteLLM_AgentsTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

