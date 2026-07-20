-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_PTUReservation" (
    "id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "model" TEXT NOT NULL,
    "cost_source" TEXT NOT NULL DEFAULT 'manual',
    "ptu_count" INTEGER,
    "cost_per_ptu" DOUBLE PRECISION,
    "azure_resource_id" TEXT,
    "effective_from" TIMESTAMP(3) NOT NULL,
    "effective_to" TIMESTAMP(3),
    "created_by" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_PTUReservation_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_PTUReservation_team_id_model_effective_from_idx" ON "LiteLLM_PTUReservation"("team_id", "model", "effective_from");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_PTUReservation_cost_source_azure_resource_id_idx" ON "LiteLLM_PTUReservation"("cost_source", "azure_resource_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_PTUReservation_effective_from_effective_to_idx" ON "LiteLLM_PTUReservation"("effective_from", "effective_to");
