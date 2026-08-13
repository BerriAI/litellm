-- CreateTable
CREATE TABLE "LiteLLM_CLISessionTable" (
    "session_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "team_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "revoked_at" TIMESTAMP(3),
    "revoked_by" TEXT,

    CONSTRAINT "LiteLLM_CLISessionTable_pkey" PRIMARY KEY ("session_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_CLISessionTable_user_id_idx" ON "LiteLLM_CLISessionTable"("user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_CLISessionTable_expires_at_idx" ON "LiteLLM_CLISessionTable"("expires_at");
