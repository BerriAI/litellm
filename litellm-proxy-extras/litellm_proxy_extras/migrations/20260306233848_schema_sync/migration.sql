-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "byok_api_key_help_url" TEXT,
ADD COLUMN IF NOT EXISTS "byok_description" TEXT[] DEFAULT ARRAY[]::TEXT[],
ADD COLUMN IF NOT EXISTS "is_byok" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS "tool_name_to_description" JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS "tool_name_to_display_name" JSONB DEFAULT '{}';

-- CreateTable
CREATE TABLE "LiteLLM_MCPUserCredentials" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "server_id" TEXT NOT NULL,
    "credential_b64" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_MCPUserCredentials_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_JWTKeyMapping" (
    "id" TEXT NOT NULL,
    "jwt_claim_name" TEXT NOT NULL,
    "jwt_claim_value" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "description" TEXT,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_JWTKeyMapping_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ConfigOverrides" (
    "config_type" TEXT NOT NULL,
    "config_value" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_ConfigOverrides_pkey" PRIMARY KEY ("config_type")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_MCPUserCredentials_user_id_server_id_key" ON "LiteLLM_MCPUserCredentials"("user_id", "server_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_JWTKeyMapping_jwt_claim_name_jwt_claim_value_is_act_idx" ON "LiteLLM_JWTKeyMapping"("jwt_claim_name", "jwt_claim_value", "is_active");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_JWTKeyMapping_jwt_claim_name_jwt_claim_value_key" ON "LiteLLM_JWTKeyMapping"("jwt_claim_name", "jwt_claim_value");

-- AddForeignKey
ALTER TABLE "LiteLLM_JWTKeyMapping" ADD CONSTRAINT "LiteLLM_JWTKeyMapping_token_fkey" FOREIGN KEY ("token") REFERENCES "LiteLLM_VerificationToken"("token") ON DELETE RESTRICT ON UPDATE CASCADE;

