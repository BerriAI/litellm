-- AlterTable: add model_spend JSON column to LiteLLM_TeamMembership for per-model budget tracking
ALTER TABLE "LiteLLM_TeamMembership" ADD COLUMN IF NOT EXISTS "model_spend" JSONB NOT NULL DEFAULT '{}';
