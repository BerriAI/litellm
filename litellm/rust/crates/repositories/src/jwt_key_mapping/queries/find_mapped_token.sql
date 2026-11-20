SELECT token
FROM "LiteLLM_JWTKeyMapping"
WHERE jwt_claim_name = $1
  AND jwt_claim_value = $2
  AND jwt_issuer = $3
  AND is_active = true
LIMIT 1
