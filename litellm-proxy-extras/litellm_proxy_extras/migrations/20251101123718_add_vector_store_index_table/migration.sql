-- CreateTable
CREATE TABLE "LiteLLM_IndexTable" (
    "id" TEXT NOT NULL,
    "index_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "index_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_IndexTable_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_IndexTable_index_name_key" ON "LiteLLM_IndexTable"("index_name");

