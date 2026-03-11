-- AlterTable
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN     "policies" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN     "policies" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN     "policies" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN     "policies" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN     "policies" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- CreateTable
CREATE TABLE "LiteLLM_PolicyTable" (
    "policy_id" TEXT NOT NULL,
    "policy_name" TEXT NOT NULL,
    "inherit" TEXT,
    "description" TEXT,
    "guardrails_add" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "guardrails_remove" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "condition" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_PolicyTable_pkey" PRIMARY KEY ("policy_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_PolicyAttachmentTable" (
    "attachment_id" TEXT NOT NULL,
    "policy_name" TEXT NOT NULL,
    "scope" TEXT,
    "teams" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "keys" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "models" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_PolicyAttachmentTable_pkey" PRIMARY KEY ("attachment_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_PolicyTable_policy_name_key" ON "LiteLLM_PolicyTable"("policy_name");

