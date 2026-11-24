-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogToolIndex" (
    "request_id" TEXT NOT NULL,
    "tool_name" TEXT NOT NULL,
    "start_time" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_SpendLogToolIndex_pkey" PRIMARY KEY ("request_id","tool_name")
);
