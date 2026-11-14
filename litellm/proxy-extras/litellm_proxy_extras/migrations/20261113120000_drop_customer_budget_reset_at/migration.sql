SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_CustomerOrganizationTable" DROP COLUMN IF EXISTS "budget_reset_at";
ALTER TABLE "LiteLLM_CustomerTeamTable" DROP COLUMN IF EXISTS "budget_reset_at";
ALTER TABLE "LiteLLM_CustomerMemberTable" DROP COLUMN IF EXISTS "budget_reset_at";
