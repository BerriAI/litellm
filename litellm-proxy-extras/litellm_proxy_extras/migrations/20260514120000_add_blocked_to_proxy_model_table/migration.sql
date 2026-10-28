-- AlterTable
-- Adds the admin-toggleable pause flag used by the router's blocked filter and the
-- credential lookup helpers; defaults to false so existing rows behave unchanged.
ALTER TABLE "LiteLLM_ProxyModelTable" ADD COLUMN IF NOT EXISTS "blocked" BOOLEAN NOT NULL DEFAULT false;
