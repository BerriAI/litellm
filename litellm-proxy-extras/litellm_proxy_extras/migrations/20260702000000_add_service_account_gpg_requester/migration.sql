-- AlterTable: add GPG public key + requester to LiteLLM_ServiceAccountTable
-- public_key  — the requester's ASCII-armored OpenPGP public key, used at
--               creation-approve time to GPG-encrypt the issued service-account
--               key before relaying it (the plaintext key never leaves litellm).
-- requester   — the user_id of the user who filed the creation request, used to
--               resolve a Slack DM recipient at notification time.
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "public_key" TEXT;
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "requester" TEXT;
