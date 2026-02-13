-- CreateIndex
CREATE INDEX "LiteLLM_VerificationToken_user_id_team_id_idx" ON "LiteLLM_VerificationToken"("user_id", "team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_VerificationToken_team_id_idx" ON "LiteLLM_VerificationToken"("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_VerificationToken_budget_reset_at_expires_idx" ON "LiteLLM_VerificationToken"("budget_reset_at", "expires");
