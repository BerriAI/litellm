-- Timestamp sorts before some already-applied migrations; this is safe: the
-- runner is `prisma migrate deploy`, which applies every pending migration
-- regardless of name order (utils.py has an informational check for exactly
-- this), and IF NOT EXISTS keeps a re-apply idempotent.
-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "token_exchange_endpoint" TEXT;
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "audience" TEXT;
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "subject_token_type" TEXT;
