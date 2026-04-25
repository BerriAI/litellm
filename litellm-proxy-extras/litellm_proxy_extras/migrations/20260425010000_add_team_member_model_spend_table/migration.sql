-- CreateTable: per-model spend rows with atomic increment support
CREATE TABLE IF NOT EXISTS "LiteLLM_TeamMemberModelSpend" (
    "user_id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "model"   TEXT NOT NULL,
    "spend"   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    CONSTRAINT "LiteLLM_TeamMemberModelSpend_pkey" PRIMARY KEY ("user_id", "team_id", "model")
);
