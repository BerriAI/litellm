-- CreateTable
CREATE TABLE "LiteLLM_AgentsTable" (
    "agent_id" TEXT NOT NULL,
    "agent_name" TEXT NOT NULL,
    "litellm_params" JSONB,
    "agent_card_params" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_AgentsTable_pkey" PRIMARY KEY ("agent_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AgentsTable_agent_name_key" ON "LiteLLM_AgentsTable"("agent_name");

