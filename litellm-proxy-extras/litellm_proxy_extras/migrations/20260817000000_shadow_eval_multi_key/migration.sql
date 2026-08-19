ALTER TABLE "LiteLLM_ShadowEvalJob" ADD COLUMN IF NOT EXISTS "group_id" TEXT;

UPDATE "LiteLLM_ShadowEvalJob" SET "group_id" = "id" WHERE "group_id" IS NULL;

ALTER TABLE "LiteLLM_ShadowEvalJob" ALTER COLUMN "group_id" SET NOT NULL;

CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJob_group_id_idx" ON "LiteLLM_ShadowEvalJob"("group_id");
