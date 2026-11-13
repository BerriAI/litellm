ALTER TABLE "LiteLLM_CustomerMemberTable" ADD COLUMN IF NOT EXISTS "email" TEXT;

CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_parent_team_id_email_idx"
ON "LiteLLM_CustomerMemberTable"("parent_team_id", "email");
