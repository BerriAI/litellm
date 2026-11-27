SELECT EXISTS (
    SELECT 1
    FROM "LiteLLM_UserTable"
    WHERE lower(user_email) = lower($1)
      AND user_id <> $2
) AS "taken!"
