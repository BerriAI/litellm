SET LOCAL lock_timeout = '10s';

-- DropForeignKey
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_InvitationLink_created_by_fkey' AND conrelid = '"LiteLLM_InvitationLink"'::regclass) THEN
        ALTER TABLE "LiteLLM_InvitationLink" DROP CONSTRAINT "LiteLLM_InvitationLink_created_by_fkey";
    END IF;
END $$;

-- DropForeignKey
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_InvitationLink_updated_by_fkey' AND conrelid = '"LiteLLM_InvitationLink"'::regclass) THEN
        ALTER TABLE "LiteLLM_InvitationLink" DROP CONSTRAINT "LiteLLM_InvitationLink_updated_by_fkey";
    END IF;
END $$;
