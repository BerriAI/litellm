-- CreateTable
CREATE TABLE "LiteLLM_SkillTable" (
    "skill_name" TEXT NOT NULL,
    "description" TEXT,
    "content" TEXT NOT NULL,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_SkillTable_pkey" PRIMARY KEY ("skill_name")
);

-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN     "skills" TEXT[] DEFAULT ARRAY[]::TEXT[];
