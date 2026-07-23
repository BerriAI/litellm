-- Adds password audit and reset-password-required state to LiteLLM_UserTable.
-- Used by self-service password change and admin reset-password flows.

ALTER TABLE "LiteLLM_UserTable" ADD COLUMN IF NOT EXISTS "password_updated_at" TIMESTAMP(3);
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN IF NOT EXISTS "reset_password_required" BOOLEAN NOT NULL DEFAULT false;
