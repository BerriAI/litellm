-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_UserProviderCredentials" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "credential_name" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "credential_b64" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_UserProviderCredentials_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_UserProviderCredentials_user_id_credential_name_key" ON "LiteLLM_UserProviderCredentials"("user_id", "credential_name");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_UserProviderCredentials_credential_name_idx" ON "LiteLLM_UserProviderCredentials"("credential_name");
