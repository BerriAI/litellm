SELECT EXISTS (
    SELECT 1
    FROM jsonb_array_elements(
        CASE WHEN jsonb_typeof(t.members_with_roles) = 'array' THEN t.members_with_roles ELSE '[]'::jsonb END
    ) AS member
    WHERE member ->> 'user_id' = $2
       OR member ->> 'user_email' = (SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = $2)
) AS "listed!"
FROM "LiteLLM_TeamTable" AS t
WHERE t.team_id = $1
