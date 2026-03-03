-- AlterTable
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN     "access_group_ids" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN     "access_group_ids" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN     "access_group_ids" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN     "access_group_ids" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- CreateTable
CREATE TABLE "LiteLLM_AccessGroupTable" (
    "access_group_id" TEXT NOT NULL,
    "access_group_name" TEXT NOT NULL,
    "description" TEXT,
    "access_model_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "access_mcp_server_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "access_agent_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "assigned_team_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "assigned_key_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_AccessGroupTable_pkey" PRIMARY KEY ("access_group_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AccessGroupTable_access_group_name_key" ON "LiteLLM_AccessGroupTable"("access_group_name");

