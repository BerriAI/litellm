INSERT INTO "LiteLLM_TeamMembership" (team_id, user_id, spend, total_spend)
SELECT id, member_id, spend, spend FROM jsonb_to_recordset($1::jsonb) AS input(id text, member_id text, spend double precision)
WHERE EXISTS (
    SELECT 1 FROM "LiteLLM_TeamTable" AS team
    WHERE team.team_id = input.id
      AND team.members_with_roles @> jsonb_build_array(jsonb_build_object('user_id', input.member_id))
) OR EXISTS (
    SELECT 1 FROM "LiteLLM_TeamMembership" AS membership
    WHERE membership.team_id = input.id AND membership.user_id = input.member_id
)
ORDER BY id COLLATE "C", member_id COLLATE "C"
ON CONFLICT (user_id, team_id) DO UPDATE
SET spend = "LiteLLM_TeamMembership".spend + EXCLUDED.spend,
    total_spend = greatest("LiteLLM_TeamMembership".total_spend, "LiteLLM_TeamMembership".spend) + EXCLUDED.total_spend
