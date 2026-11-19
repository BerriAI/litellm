SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_AppTable"
    ADD COLUMN IF NOT EXISTS "resource_type" TEXT,
    ADD COLUMN IF NOT EXISTS "resource_id" TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_AppTable_resource_type_resource_id_key"
    ON "LiteLLM_AppTable"("resource_type", "resource_id");

ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "key_id" TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_VerificationToken_key_id_key"
    ON "LiteLLM_VerificationToken"("key_id");

ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "key_id" TEXT;
