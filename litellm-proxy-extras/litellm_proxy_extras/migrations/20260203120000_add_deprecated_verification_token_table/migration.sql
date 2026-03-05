-- CreateTable
CREATE TABLE "LiteLLM_DeprecatedVerificationToken" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "active_token_id" TEXT NOT NULL,
    "revoke_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_DeprecatedVerificationToken_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DeprecatedVerificationToken_token_key" ON "LiteLLM_DeprecatedVerificationToken"("token");

-- CreateIndex
CREATE INDEX "LiteLLM_DeprecatedVerificationToken_token_revoke_at_idx" ON "LiteLLM_DeprecatedVerificationToken"("token", "revoke_at");

-- CreateIndex
CREATE INDEX "LiteLLM_DeprecatedVerificationToken_revoke_at_idx" ON "LiteLLM_DeprecatedVerificationToken"("revoke_at");
