-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_BackgroundResponseSettlement" (
    "response_id" TEXT NOT NULL,
    "settled_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_BackgroundResponseSettlement_pkey" PRIMARY KEY ("response_id")
);
