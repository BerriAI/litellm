WITH free AS (
    SELECT $3::bool OR NOT EXISTS (
        SELECT 1
        FROM "LiteLLM_UserTable"
        WHERE lower(user_email) = lower($2)
          AND user_id <> $1
    ) AS free
), filled AS (
    UPDATE "LiteLLM_UserTable"
    SET user_email = $2,
        updated_at = CURRENT_TIMESTAMP
    WHERE user_id = $1
      AND user_email IS NULL
      AND (SELECT free FROM free)
    RETURNING user_email
)
SELECT user_email AS "user_email!"
FROM filled
UNION ALL
SELECT user_email
FROM "LiteLLM_UserTable"
WHERE user_id = $1
  AND user_email IS NOT NULL
  AND (SELECT free FROM free)
