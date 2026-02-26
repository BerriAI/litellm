-- CreateTable
CREATE TABLE "LiteLLM_SpendLogToolIndex" (
    "request_id" TEXT NOT NULL,
    "tool_name" TEXT NOT NULL,
    "start_time" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_SpendLogToolIndex_pkey" PRIMARY KEY ("request_id","tool_name")
);

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogToolIndex_tool_name_start_time_idx" ON "LiteLLM_SpendLogToolIndex"("tool_name", "start_time");
