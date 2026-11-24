WITH failed AS (
    INSERT INTO "LiteLLM_DailyGatewayFailedRequests" (
        "date", "category", "route", "status_code", "failed_requests", "created_at", "updated_at"
    )
    SELECT
        row."date", row."category", row."route", row."status_code", row."failed_requests",
        NOW() AT TIME ZONE 'UTC', NOW() AT TIME ZONE 'UTC'
    FROM UNNEST($1::text[], $2::text[], $3::text[], $4::integer[], $5::bigint[])
        WITH ORDINALITY AS row ("date", "category", "route", "status_code", "failed_requests", "position")
    ORDER BY row."position"
    ON CONFLICT ("date", "category", "route", "status_code") DO UPDATE SET
        "failed_requests" = "LiteLLM_DailyGatewayFailedRequests"."failed_requests" + EXCLUDED."failed_requests",
        "updated_at" = NOW() AT TIME ZONE 'UTC'
    RETURNING "status_code"
)
INSERT INTO "LiteLLM_DailyGatewayRequests" (
    "date", "category", "route", "successful_requests", "failed_requests", "created_at", "updated_at"
)
SELECT
    row."date", row."category", row."route", row."successful_requests", row."failed_requests",
    NOW() AT TIME ZONE 'UTC', NOW() AT TIME ZONE 'UTC'
FROM UNNEST($6::text[], $7::text[], $8::text[], $9::bigint[], $10::bigint[])
    WITH ORDINALITY AS row ("date", "category", "route", "successful_requests", "failed_requests", "position")
ORDER BY row."position"
ON CONFLICT ("date", "category", "route") DO UPDATE SET
    "successful_requests" = "LiteLLM_DailyGatewayRequests"."successful_requests" + EXCLUDED."successful_requests",
    "failed_requests" = "LiteLLM_DailyGatewayRequests"."failed_requests" + EXCLUDED."failed_requests",
    "updated_at" = NOW() AT TIME ZONE 'UTC'
