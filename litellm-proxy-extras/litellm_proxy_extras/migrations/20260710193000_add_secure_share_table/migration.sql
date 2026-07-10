-- CreateTable
CREATE TABLE "LiteLLM_SecureShareTable" (
    "share_id" TEXT NOT NULL,
    "ciphertext" TEXT NOT NULL,
    "salt" TEXT NOT NULL,
    "iv" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_by" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_SecureShareTable_pkey" PRIMARY KEY ("share_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_SecureShareTable_expires_at_idx" ON "LiteLLM_SecureShareTable"("expires_at");
