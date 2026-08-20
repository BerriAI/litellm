-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalJob" ADD COLUMN IF NOT EXISTS "released_at" TIMESTAMP(3);

UPDATE "LiteLLM_ShadowEvalJob" SET released_at = stopped_at
WHERE stopped_at IS NOT NULL AND released_at IS NULL;

UPDATE "LiteLLM_ShadowEvalJob" SET stopped_at = NULL
WHERE stopped_at IS NOT NULL AND stopped_by IS NULL;

DROP INDEX IF EXISTS "LiteLLM_ShadowEvalJob_one_active_per_key_direction";

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJob_one_active_per_key_direction"
    ON "LiteLLM_ShadowEvalJob"("api_key_id", "direction") WHERE "released_at" IS NULL;
