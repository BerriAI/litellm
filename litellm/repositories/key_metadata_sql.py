from typing import Final

ACTIVE_TOKEN_DIGEST_SQL: Final = """
SELECT encode(sha256(convert_to(token, 'UTF8')), 'hex') AS digest, key_alias, team_id, user_id
FROM "LiteLLM_VerificationToken"
WHERE encode(sha256(convert_to(token, 'UTF8')), 'hex') = ANY($1::text[])
"""

DELETED_TOKEN_DIGEST_SQL: Final = """
SELECT DISTINCT ON (token)
    encode(sha256(convert_to(token, 'UTF8')), 'hex') AS digest, key_alias, team_id, user_id
FROM "LiteLLM_DeletedVerificationToken"
WHERE encode(sha256(convert_to(token, 'UTF8')), 'hex') = ANY($1::text[])
ORDER BY token, deleted_at DESC
"""
