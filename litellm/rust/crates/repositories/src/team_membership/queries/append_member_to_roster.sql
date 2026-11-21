UPDATE "LiteLLM_TeamTable"
SET members_with_roles =
        CASE WHEN jsonb_typeof(members_with_roles) = 'array' THEN members_with_roles ELSE '[]'::jsonb END
        || jsonb_build_array(jsonb_build_object('user_id', $2::text, 'user_email', $3::text, 'role', 'user')),
    updated_at = NOW() AT TIME ZONE 'UTC'
WHERE team_id = $1
RETURNING members_with_roles AS "members_with_roles!: Json<serde_json::Value>"
