import hashlib

_BEARER_PREFIX = "bearer "
_JWT_SEGMENT_COUNT = 3


def sha256_hex(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _is_jwt(token: str) -> bool:
    return len(token.split(".")) == _JWT_SEGMENT_COUNT


def hash_credential(value: str) -> str:
    """
    Replace credential material with a stable digest, leaving anything that is
    not a credential untouched.

    Covers LiteLLM virtual keys (`sk-...`, with or without a `Bearer ` prefix)
    and JWTs used to authenticate against the proxy. Values that are already a
    digest, and non-secret identifiers such as the master key alias, are
    returned unchanged so downstream grouping by key stays stable.

    Single source of truth for `UserAPIKeyAuth.api_key` and for the
    `user_api_key_hash` field of the standard logging payload, so the same
    credential yields the same value whichever boundary hashes it.
    """
    normalized = value[len(_BEARER_PREFIX) :] if value[: len(_BEARER_PREFIX)].lower() == _BEARER_PREFIX else value
    if normalized.startswith("sk-"):
        return sha256_hex(normalized)
    if _is_jwt(normalized):
        return f"hashed-jwt-{sha256_hex(normalized)}"
    return normalized


def sanitize_key_hash(value: str | None) -> str | None:
    return hash_credential(value) if isinstance(value, str) else value
