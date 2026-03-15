-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "authorization_url" TEXT,
ADD COLUMN IF NOT EXISTS "registration_url" TEXT,
ADD COLUMN IF NOT EXISTS "token_url" TEXT;

