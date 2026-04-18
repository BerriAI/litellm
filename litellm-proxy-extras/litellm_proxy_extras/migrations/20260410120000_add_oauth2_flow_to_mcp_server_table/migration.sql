-- AlterTable
-- OAuth2 flow discriminator: "client_credentials" (M2M) | "authorization_code" (interactive); nullable for legacy rows / inference
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "oauth2_flow" TEXT;
