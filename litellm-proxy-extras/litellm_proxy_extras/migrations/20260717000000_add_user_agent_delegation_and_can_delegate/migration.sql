-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN "mcp_can_delegate" BOOLEAN;

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_UserAgentDelegationTable" (
    "delegation_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "granted_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "granted_by" TEXT NOT NULL,
    "revoked_at" TIMESTAMP(3),
    "revoked_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_UserAgentDelegationTable_pkey" PRIMARY KEY ("delegation_id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_UserAgentDelegationTable_user_id_agent_id_key" ON "LiteLLM_UserAgentDelegationTable"("user_id", "agent_id");
