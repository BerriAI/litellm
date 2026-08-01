-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_PasswordResetToken" (
    "token_hash" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "requested_ip" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "used_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_PasswordResetToken_pkey" PRIMARY KEY ("token_hash")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_PasswordResetToken_user_id_idx" ON "LiteLLM_PasswordResetToken"("user_id");

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_PasswordResetToken_user_id_fkey') THEN
        ALTER TABLE "LiteLLM_PasswordResetToken" ADD CONSTRAINT "LiteLLM_PasswordResetToken_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
