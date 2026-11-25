SELECT active_token_id,
       revoke_at
FROM "LiteLLM_DeprecatedVerificationToken"
WHERE token = $1
  AND revoke_at > $2
LIMIT 1
