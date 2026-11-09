SELECT to_jsonb(i) AS "data!: AuthData" FROM "LiteLLM_VerifiedSubject" i WHERE i.issuer=$1 AND i.tenant_id=$2 AND i.oid=$3
