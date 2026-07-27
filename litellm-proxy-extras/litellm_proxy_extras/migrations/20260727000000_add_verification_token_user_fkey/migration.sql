-- Null out user_ids that reference users that no longer exist; the foreign key
-- below cannot be added over them, and ON DELETE SET NULL would have produced
-- the same rows had the constraint existed when those users were deleted
UPDATE "LiteLLM_VerificationToken" vt
SET "user_id" = NULL
WHERE vt."user_id" IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM "LiteLLM_UserTable" u WHERE u."user_id" = vt."user_id");

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_VerificationToken_user_id_fkey') THEN
        ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;
