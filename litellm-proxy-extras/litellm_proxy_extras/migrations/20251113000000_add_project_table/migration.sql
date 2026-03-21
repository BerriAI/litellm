-- CreateTable
CREATE TABLE "LiteLLM_ProjectTable" (
    "project_id" TEXT NOT NULL,
    "project_alias" TEXT,
    "team_id" TEXT,
    "budget_id" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "models" TEXT[],
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "blocked" BOOLEAN NOT NULL DEFAULT false,
    "object_permission_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_ProjectTable_pkey" PRIMARY KEY ("project_id")
);

-- AddForeignKey
ALTER TABLE "LiteLLM_ProjectTable" ADD CONSTRAINT "LiteLLM_ProjectTable_team_id_fkey" FOREIGN KEY ("team_id") REFERENCES "LiteLLM_TeamTable"("team_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_ProjectTable" ADD CONSTRAINT "LiteLLM_ProjectTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_ProjectTable" ADD CONSTRAINT "LiteLLM_ProjectTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AlterTable: Add project_id to LiteLLM_VerificationToken
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "project_id" TEXT;

-- AddForeignKey
ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "LiteLLM_ProjectTable"("project_id") ON DELETE SET NULL ON UPDATE CASCADE;

