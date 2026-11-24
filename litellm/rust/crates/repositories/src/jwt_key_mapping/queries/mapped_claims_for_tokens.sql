SELECT jwt_issuer,
       jwt_claim_name,
       jwt_claim_value
FROM "LiteLLM_JWTKeyMapping"
WHERE token = ANY($1)
