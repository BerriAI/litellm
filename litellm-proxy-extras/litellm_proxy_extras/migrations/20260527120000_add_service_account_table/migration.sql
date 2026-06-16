-- CreateTable
CREATE TABLE "LiteLLM_ServiceAccountTable" (
    "user_id" TEXT NOT NULL,
    "owner_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc'),
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc'),

    CONSTRAINT "LiteLLM_ServiceAccountTable_pkey" PRIMARY KEY ("user_id")
);

-- Auto-update updated_at on row change
CREATE OR REPLACE FUNCTION "LiteLLM_ServiceAccountTable_set_updated_at"()
RETURNS TRIGGER AS $$
BEGIN
    NEW."updated_at" = NOW() AT TIME ZONE 'utc';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER "LiteLLM_ServiceAccountTable_updated_at_trigger"
    BEFORE UPDATE ON "LiteLLM_ServiceAccountTable"
    FOR EACH ROW
    EXECUTE FUNCTION "LiteLLM_ServiceAccountTable_set_updated_at"();
