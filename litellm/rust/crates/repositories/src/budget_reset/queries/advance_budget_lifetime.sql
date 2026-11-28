WITH tier AS (
    UPDATE "LiteLLM_BudgetTable"
    SET budget_reset_at = $2,
        budget_reset_count = budget_reset_count + 1,
        updated_at = NOW() AT TIME ZONE 'UTC'
    WHERE budget_id = $1 AND budget_reset_at IS NOT DISTINCT FROM $3 AND budget_duration = $4
      AND NOT EXISTS (SELECT 1 FROM "LiteLLM_TagTable" WHERE budget_id = $1)
      AND NOT EXISTS (SELECT 1 FROM "LiteLLM_ModelAccessGroupBudgetTable" WHERE budget_id = $1)
      AND NOT EXISTS (SELECT 1 FROM "LiteLLM_CustomerOrganizationTable" WHERE budget_id = $1)
      AND NOT EXISTS (SELECT 1 FROM "LiteLLM_CustomerTeamTable" WHERE budget_id = $1 OR budget_override_id = $1)
      AND NOT EXISTS (SELECT 1 FROM "LiteLLM_CustomerMemberTable" WHERE budget_id = $1 OR budget_override_id = $1)
    RETURNING budget_id, max_budget, created_at, budget_reset_count,
        CASE WHEN ABS(max_budget) < 'Infinity'::double precision AND $5::boolean THEN max_budget END AS cap,
        CASE WHEN max_budget > 0 AND rollover_max_budget > max_budget
                   AND rollover_max_budget < 'Infinity'::double precision
             THEN rollover_max_budget - max_budget END AS credit_room
), orgs AS (
    UPDATE "LiteLLM_OrganizationTable" AS o
    SET spend = CASE WHEN tier.cap IS NULL THEN 0 ELSE GREATEST(o.spend - tier.cap, 0) END,
        updated_at = NOW() AT TIME ZONE 'UTC'
    FROM tier WHERE o.budget_id = tier.budget_id AND o.spend > 0
    RETURNING o.organization_id AS id, o.spend
), members AS (
    UPDATE "LiteLLM_TeamMembership" AS m
    SET total_spend = GREATEST(m.total_spend, m.spend),
        rollover_credit = CASE WHEN tier.credit_room IS NOT NULL THEN
            LEAST(tier.credit_room, GREATEST(0, tier.max_budget
                + LEAST(tier.credit_room, GREATEST(0, COALESCE(m.rollover_credit, 0))) - m.spend))
            ELSE m.rollover_credit END,
        spend = CASE WHEN tier.cap IS NULL THEN 0 ELSE GREATEST(m.spend - tier.cap
            - LEAST(COALESCE(tier.credit_room, 0), GREATEST(0, COALESCE(m.rollover_credit, 0))), 0) END
    FROM tier
    WHERE m.budget_id = tier.budget_id OR (
        m.budget_id IS NULL AND EXISTS (
            SELECT 1 FROM "LiteLLM_TeamTable" t
            WHERE t.team_id = m.team_id AND t.metadata->>'team_member_budget_id' = tier.budget_id
        )
    )
    RETURNING m.user_id AS id, m.team_id, m.spend
), keys AS (
    UPDATE "LiteLLM_VerificationToken" AS k
    SET spend = CASE WHEN tier.cap IS NULL THEN 0 ELSE GREATEST(k.spend - tier.cap, 0) END,
        updated_at = NOW() AT TIME ZONE 'UTC'
    FROM tier WHERE k.budget_id = tier.budget_id AND k.budget_duration IS NULL AND k.spend > 0
    RETURNING k.token AS id, k.spend
), projects AS (
    UPDATE "LiteLLM_ProjectTable" AS p
    SET spend = CASE WHEN tier.cap IS NULL THEN 0 ELSE GREATEST(p.spend - tier.cap, 0) END,
        updated_at = NOW() AT TIME ZONE 'UTC'
    FROM tier WHERE p.budget_id = tier.budget_id AND p.spend > 0
    RETURNING p.project_id AS id, p.spend
), closing AS (
    SELECT tier.*,
        CONCAT(budget_id, ':', to_char(created_at, 'YYYY-MM-DD"T"HH24:MI:SS'),
            CASE WHEN date_trunc('second', created_at) <> created_at THEN to_char(created_at, '.US') END,
            '+00:00:', budget_reset_count - 1, ':',
            CASE WHEN $3::timestamp IS NULL THEN 'unset' ELSE CONCAT(
                to_char($3::timestamp, 'YYYY-MM-DD"T"HH24:MI:SS'),
                CASE WHEN date_trunc('second', $3::timestamp) <> $3::timestamp THEN to_char($3::timestamp, '.US') END,
                '+00:00') END) AS window
    FROM tier
), end_users AS (
    UPDATE "LiteLLM_EndUserTable" AS e
    SET spend = CASE WHEN closing.cap IS NULL THEN 0 ELSE GREATEST(e.spend - closing.cap, 0) END,
        spend_reset_window = closing.window
    FROM closing
    WHERE COALESCE(e.budget_id, $6::text) = closing.budget_id
      AND e.spend > 0
      AND e.spend_reset_window IS DISTINCT FROM closing.window
    RETURNING e.user_id AS id
)
SELECT 'org' AS "kind!", id AS "id!", NULL::text AS team_id FROM orgs
UNION ALL SELECT 'member', id, team_id FROM members
UNION ALL SELECT 'key', id, NULL FROM keys
UNION ALL SELECT 'project', id, NULL FROM projects
UNION ALL SELECT 'end_user', id, NULL FROM end_users
