CREATE TABLE "LiteLLM_MCPKeyedOAuthGrant" (
    "id" TEXT NOT NULL,
    "binding_b64" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "code_hash" TEXT,
    "refresh_hash" TEXT,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_MCPKeyedOAuthGrant_pkey" PRIMARY KEY ("id")
);
CREATE INDEX "LiteLLM_MCPKeyedOAuthGrant_expires_at_idx" ON "LiteLLM_MCPKeyedOAuthGrant"("expires_at");
