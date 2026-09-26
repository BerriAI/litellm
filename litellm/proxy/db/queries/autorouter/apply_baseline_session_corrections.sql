WITH changes AS (
    SELECT * FROM jsonb_to_recordset($1::jsonb) AS x(
        user_id text, api_key text, session_id text, router_name text, baseline_model text,
        covered_delta int, actual_delta float8, savings_delta float8
    )
), totals AS (
    SELECT api_key, session_id, router_name, SUM(covered_delta)::int AS covered_delta,
        SUM(actual_delta) AS actual_delta, SUM(savings_delta) AS savings_delta
    FROM changes GROUP BY api_key, session_id, router_name
), models AS (
    SELECT api_key, session_id, router_name, jsonb_object_agg(baseline_model, delta) AS deltas
    FROM (
        SELECT api_key, session_id, router_name, baseline_model, SUM(covered_delta)::int AS delta
        FROM changes GROUP BY api_key, session_id, router_name, baseline_model
    ) grouped GROUP BY api_key, session_id, router_name
)
UPDATE "LiteLLM_AutoRouterSession" AS session
SET saved_spend = session.saved_spend + totals.savings_delta,
    savings_estimated_turns = session.savings_estimated_turns + totals.covered_delta,
    savings_estimated_actual_spend = session.savings_estimated_actual_spend + totals.actual_delta,
    savings_estimated_saved_spend = session.savings_estimated_saved_spend + totals.savings_delta,
    savings_estimated_baseline_models = (
        SELECT COALESCE(jsonb_object_agg(key, value), '{}'::jsonb) FROM (
            SELECT key, SUM(value::int)::int AS value FROM (
                SELECT * FROM jsonb_each_text(session.savings_estimated_baseline_models)
                UNION ALL SELECT * FROM jsonb_each_text(models.deltas)
            ) combined GROUP BY key HAVING SUM(value::int) > 0
        ) counts
    )
FROM totals JOIN models USING (api_key, session_id, router_name)
WHERE session.api_key = totals.api_key AND session.session_id = totals.session_id
    AND session.router_name = totals.router_name
