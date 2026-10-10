CREATE TABLE IF NOT EXISTS "LiteLLM_AutoRouterDailySpend" (
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "router_name" TEXT NOT NULL,
    "router_type" TEXT NOT NULL,
    "turns" INTEGER NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "saved_spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "savings_estimated_turns" INTEGER NOT NULL DEFAULT 0,
    "savings_estimated_actual_spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "savings_estimated_saved_spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "classifier_cost" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "classifier_cost_recorded_turns" INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT "LiteLLM_AutoRouterDailySpend_pkey" PRIMARY KEY ("date", "api_key", "user_id", "router_name", "router_type")
);
