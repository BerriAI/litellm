DELETE FROM "LiteLLM_AutoRouterBaselineObservation" WHERE request_id IN (
    SELECT event.request_id FROM "LiteLLM_AutoRouterBaselineObservation" AS event
    JOIN "LiteLLM_AutoRouterBaselineComparison" AS comparison USING (scope)
    WHERE comparison.retired AND comparison.updated_at < $1::timestamptz
    LIMIT $2::int
)
