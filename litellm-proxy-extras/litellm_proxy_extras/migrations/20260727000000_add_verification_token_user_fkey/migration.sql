-- AddForeignKey
-- NOT VALID so no existing row is touched or judged: keys whose user_id points
-- at a user that no longer existed before this migration keep their value, and
-- the constraint only enforces new INSERTs and UPDATEs of user_id.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_VerificationToken_user_id_fkey') THEN
        ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE SET NULL ON UPDATE CASCADE NOT VALID;
    END IF;
END $$;

-- Best-effort validation: on databases with no orphaned user_ids (the normal
-- case; user deletion removes the user's keys) this marks the constraint fully
-- valid. Where orphans exist the constraint simply stays NOT VALID and keeps
-- enforcing go-forward writes; the migration never fails and never mutates data.
DO $$
BEGIN
    BEGIN
        ALTER TABLE "LiteLLM_VerificationToken" VALIDATE CONSTRAINT "LiteLLM_VerificationToken_user_id_fkey";
    EXCEPTION WHEN foreign_key_violation THEN
        RAISE NOTICE 'LiteLLM_VerificationToken has user_ids referencing missing users; constraint left NOT VALID';
    END;
END $$;
