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
CREATE INDEX "LiteLLM_JWTKeyMapping_jwt_claim_name_jwt_claim_value_is_acti_idx" ON "LiteLLM_JWTKeyMapping"("jwt_claim_name", "jwt_claim_value", "is_active");

-- AddForeignKey
ALTER TABLE "LiteLLM_JWTKeyMapping" ADD CONSTRAINT "LiteLLM_JWTKeyMapping_token_fkey" FOREIGN KEY ("token") REFERENCES "LiteLLM_VerificationToken"("token") ON DELETE RESTRICT ON UPDATE CASCADE;
