SET LOCAL lock_timeout = '10s';

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_VaultSecrets" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "secret_name" TEXT NOT NULL,
    "description" TEXT,
    "value_b64" TEXT NOT NULL,
    "last_accessed_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_VaultSecrets_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_VaultSecrets_user_id_secret_name_key" ON "LiteLLM_VaultSecrets"("user_id", "secret_name");
