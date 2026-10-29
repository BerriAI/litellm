INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, total_spend)
SELECT member.user_id, member.team_id, member.cost, member.cost
FROM jsonb_to_recordset($1::jsonb) AS member(user_id text, team_id text, cost float8)
WHERE EXISTS (
    SELECT 1 FROM "LiteLLM_TeamTable" t
    WHERE t.team_id = member.team_id
      AND t.members_with_roles @> jsonb_build_array(jsonb_build_object('user_id', member.user_id))
)
   OR EXISTS (
    SELECT 1 FROM "LiteLLM_TeamMembership" m
    WHERE m.user_id = member.user_id AND m.team_id = member.team_id
)
ON CONFLICT (user_id, team_id) DO UPDATE
SET spend = "LiteLLM_TeamMembership".spend + EXCLUDED.spend,
    total_spend = "LiteLLM_TeamMembership".total_spend + EXCLUDED.total_spend
