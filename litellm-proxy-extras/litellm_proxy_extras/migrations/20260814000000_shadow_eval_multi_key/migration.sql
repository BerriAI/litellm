CREATE TABLE IF NOT EXISTS "LiteLLM_ShadowEvalJobKey" (
    "id" TEXT NOT NULL,
    "job_id" TEXT NOT NULL,
    "direction" TEXT NOT NULL,
    "api_key_id" TEXT NOT NULL,
    "max_turns" INTEGER NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "stopped_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_ShadowEvalJobKey_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJobKey_job_id_idx" ON "LiteLLM_ShadowEvalJobKey"("job_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJobKey_api_key_id_idx" ON "LiteLLM_ShadowEvalJobKey"("api_key_id");

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJob_id_direction_key" ON "LiteLLM_ShadowEvalJob"("id", "direction");

DO $$ BEGIN
    ALTER TABLE "LiteLLM_ShadowEvalJobKey" ADD CONSTRAINT "LiteLLM_ShadowEvalJobKey_job_id_direction_fkey"
        FOREIGN KEY ("job_id", "direction") REFERENCES "LiteLLM_ShadowEvalJob"("id", "direction")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- One active job per key per direction, enforced by the database rather than a read-then-create in
-- the start endpoint, which races against a concurrent start on another pod. Partial indexes are not
-- expressible in schema.prisma, so this lives here only. Active means not yet stopped; the start
-- endpoint stamps stopped_at on exhausted key rows before creating.
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJobKey_one_active_per_key_direction"
    ON "LiteLLM_ShadowEvalJobKey"("api_key_id", "direction") WHERE "stopped_at" IS NULL;

INSERT INTO "LiteLLM_ShadowEvalJobKey" ("id", "job_id", "direction", "api_key_id", "max_turns", "created_at", "stopped_at")
SELECT 'jobkey_' || "id", "id", "direction", "api_key_id", "max_turns", "created_at", "stopped_at"
FROM "LiteLLM_ShadowEvalJob"
ON CONFLICT DO NOTHING;

ALTER TABLE "LiteLLM_ShadowEvalAttempt" ADD COLUMN IF NOT EXISTS "api_key_id" TEXT;

UPDATE "LiteLLM_ShadowEvalAttempt" a SET "api_key_id" = j."api_key_id"
FROM "LiteLLM_ShadowEvalJob" j WHERE a."job_id" = j."id" AND a."api_key_id" IS NULL;

ALTER TABLE "LiteLLM_ShadowEvalAttempt" ALTER COLUMN "api_key_id" SET NOT NULL;

CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalAttempt_job_id_api_key_id_idx" ON "LiteLLM_ShadowEvalAttempt"("job_id", "api_key_id");

DROP INDEX IF EXISTS "LiteLLM_ShadowEvalAttempt_job_id_idx";

DROP INDEX IF EXISTS "LiteLLM_ShadowEvalJob_one_active_per_key_direction";

DROP INDEX IF EXISTS "LiteLLM_ShadowEvalJob_api_key_id_idx";

-- Key scope, per-key budget and stop state moved into LiteLLM_ShadowEvalJobKey above. Dropping them
-- in the same migration is safe because no tagged release ships shadow eval, so no pod is serving
-- the single key code and there is no mixed version window that needs these columns readable.
ALTER TABLE "LiteLLM_ShadowEvalJob" DROP COLUMN IF EXISTS "api_key_id",
    DROP COLUMN IF EXISTS "max_turns",
    DROP COLUMN IF EXISTS "stopped_at";
