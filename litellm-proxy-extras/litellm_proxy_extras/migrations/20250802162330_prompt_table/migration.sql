-- CreateTable
CREATE TABLE "LiteLLM_PromptTable" (
    "id" TEXT NOT NULL,
    "prompt_id" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "prompt_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_PromptTable_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_PromptTable_prompt_id_key" ON "LiteLLM_PromptTable"("prompt_id");

