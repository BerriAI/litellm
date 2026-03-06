-- Re-add spec_path (was added in 20260220, dropped in 20260224, re-added in schema)
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "spec_path" TEXT;

-- Add BYOK MCP fields
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "tool_name_to_display_name" JSONB DEFAULT '{}';
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "tool_name_to_description" JSONB DEFAULT '{}';
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "is_byok" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "byok_description" TEXT[] DEFAULT ARRAY[]::TEXT[];
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "byok_api_key_help_url" TEXT;

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

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_MCPUserCredentials_user_id_server_id_key" ON "LiteLLM_MCPUserCredentials"("user_id", "server_id");

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

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_JWTKeyMapping_jwt_claim_name_jwt_claim_value_key" ON "LiteLLM_JWTKeyMapping"("jwt_claim_name", "jwt_claim_value");

-- CreateIndex
CREATE INDEX "LiteLLM_JWTKeyMapping_jwt_claim_name_jwt_claim_value_is_act_idx" ON "LiteLLM_JWTKeyMapping"("jwt_claim_name", "jwt_claim_value", "is_active");

-- AddForeignKey
ALTER TABLE "LiteLLM_JWTKeyMapping" ADD CONSTRAINT "LiteLLM_JWTKeyMapping_token_fkey" FOREIGN KEY ("token") REFERENCES "LiteLLM_VerificationToken"("token") ON DELETE RESTRICT ON UPDATE CASCADE;
