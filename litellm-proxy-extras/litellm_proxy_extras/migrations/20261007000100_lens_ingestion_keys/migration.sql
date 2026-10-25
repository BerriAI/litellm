CREATE TABLE IF NOT EXISTS "LiteLLM_LensIngestionKey" (
    "id" TEXT NOT NULL,
    "data" JSONB NOT NULL,
    CONSTRAINT "LiteLLM_LensIngestionKey_pkey" PRIMARY KEY ("id")
);
