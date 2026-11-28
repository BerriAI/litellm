SELECT token AS id, budget_duration AS "duration!", budget_reset_at AS reset_at,
       COALESCE(spend, 0.0) AS "spend!", max_budget,
       NULL::double precision AS rollover_max_budget,
       NULL::double precision AS rollover_credit
FROM "LiteLLM_VerificationToken"
WHERE token > $1 AND budget_duration IS NOT NULL
  AND budget_reset_at < $2 AND (expires IS NULL OR expires > $2)
ORDER BY token
LIMIT $3
