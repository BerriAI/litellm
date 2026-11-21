INSERT INTO "LiteLLM_JWTKeyMapping" (
    id,
    jwt_issuer,
    jwt_claim_name,
    jwt_claim_value,
    token,
    created_by,
    updated_by,
    created_at,
    updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $6, NOW() AT TIME ZONE 'UTC', NOW() AT TIME ZONE 'UTC')
ON CONFLICT (jwt_issuer, jwt_claim_name, jwt_claim_value) DO NOTHING
RETURNING token
