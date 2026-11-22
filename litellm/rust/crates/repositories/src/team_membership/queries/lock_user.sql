SELECT user_email
FROM "LiteLLM_UserTable"
WHERE user_id = $1
FOR UPDATE
