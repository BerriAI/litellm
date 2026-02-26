-- CreateTable
CREATE TABLE "LiteLLM_ToolPolicyOverrideTable" (
    "override_id" TEXT NOT NULL,
    "tool_name" TEXT NOT NULL,
    "team_id" TEXT NOT NULL DEFAULT '',
    "key_hash" TEXT NOT NULL DEFAULT '',
    "call_policy" TEXT NOT NULL DEFAULT 'blocked',
    "key_alias" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ToolPolicyOverrideTable_pkey" PRIMARY KEY ("override_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_ToolPolicyOverrideTable_tool_name_idx" ON "LiteLLM_ToolPolicyOverrideTable"("tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_ToolPolicyOverrideTable_team_id_idx" ON "LiteLLM_ToolPolicyOverrideTable"("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ToolPolicyOverrideTable_key_hash_idx" ON "LiteLLM_ToolPolicyOverrideTable"("key_hash");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_ToolPolicyOverrideTable_tool_name_team_id_key_hash_key" ON "LiteLLM_ToolPolicyOverrideTable"("tool_name", "team_id", "key_hash");
