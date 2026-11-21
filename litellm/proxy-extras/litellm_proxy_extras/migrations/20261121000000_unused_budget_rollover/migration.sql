ALTER TABLE "LiteLLM_UserTable" ADD COLUMN "rollover_max_budget" DOUBLE PRECISION,
    ADD COLUMN "rollover_credit" DOUBLE PRECISION;

ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN "rollover_max_budget" DOUBLE PRECISION;

ALTER TABLE "LiteLLM_TeamMembership" ADD COLUMN "rollover_credit" DOUBLE PRECISION;
