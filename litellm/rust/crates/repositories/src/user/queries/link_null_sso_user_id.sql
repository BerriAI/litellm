UPDATE "LiteLLM_UserTable"
SET sso_user_id = $2,
    updated_at = CURRENT_TIMESTAMP
WHERE user_id = $1
  AND sso_user_id IS NULL
