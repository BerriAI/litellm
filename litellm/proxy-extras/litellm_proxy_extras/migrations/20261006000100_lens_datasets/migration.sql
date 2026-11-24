CREATE TABLE IF NOT EXISTS "LiteLLM_LensDataset" (
    "id" TEXT NOT NULL,
    "revision" INTEGER NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL,
    "data" JSONB NOT NULL,
    CONSTRAINT "LiteLLM_LensDataset_pkey" PRIMARY KEY ("id", "revision")
);
