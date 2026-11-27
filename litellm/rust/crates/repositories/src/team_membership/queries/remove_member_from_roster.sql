UPDATE "LiteLLM_TeamTable"
SET members_with_roles = COALESCE(
        (SELECT jsonb_agg(entry.member ORDER BY entry.position)
         FROM jsonb_array_elements(members_with_roles) WITH ORDINALITY AS entry(member, position)
         WHERE entry.member ->> 'user_id' IS DISTINCT FROM $2),
        '[]'::jsonb
    ),
    updated_at = NOW() AT TIME ZONE 'UTC'
WHERE team_id = $1
  AND jsonb_typeof(members_with_roles) = 'array'
RETURNING members_with_roles AS "members_with_roles!: RawJson"
