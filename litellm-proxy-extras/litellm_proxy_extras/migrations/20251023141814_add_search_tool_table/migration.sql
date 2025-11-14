-- CreateTable
CREATE TABLE "LiteLLM_SearchToolsTable" (
    "search_tool_id" TEXT NOT NULL,
    "search_tool_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "search_tool_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_SearchToolsTable_pkey" PRIMARY KEY ("search_tool_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_SearchToolsTable_search_tool_name_key" ON "LiteLLM_SearchToolsTable"("search_tool_name");

