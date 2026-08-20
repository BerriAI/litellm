UPDATE "LiteLLM_SpendLogs"
SET "created_at" = "endTime",
    "updated_at" = "endTime"
WHERE "created_at" > "endTime" + interval '1 hour';
