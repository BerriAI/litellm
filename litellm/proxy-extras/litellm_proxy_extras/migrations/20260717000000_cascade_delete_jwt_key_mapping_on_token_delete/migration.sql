-- DropForeignKey
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_JWTKeyMapping_token_fkey') THEN
        ALTER TABLE "LiteLLM_JWTKeyMapping" DROP CONSTRAINT "LiteLLM_JWTKeyMapping_token_fkey";
    END IF;
END $$;

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_JWTKeyMapping_token_fkey') THEN
        ALTER TABLE "LiteLLM_JWTKeyMapping" ADD CONSTRAINT "LiteLLM_JWTKeyMapping_token_fkey" FOREIGN KEY ("token") REFERENCES "LiteLLM_VerificationToken"("token") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
