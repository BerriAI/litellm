import hashlib
import secrets
from typing import Final

HASHED_JWT_PREFIX: Final = "hashed-jwt-"
_BEARER_PREFIX: Final = "bearer "


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def is_master_key(token: str | None, master_key: str | None) -> bool:
    if token is None or master_key is None:
        return False
    return secrets.compare_digest(token.encode(), master_key.encode())


def is_jwt(token: str | None) -> bool:
    return token is not None and len(token.split(".")) == 3


def stored_api_key(api_key: str) -> str:
    token: Final = (
        api_key[len(_BEARER_PREFIX) :] if api_key[: len(_BEARER_PREFIX)].lower() == _BEARER_PREFIX else api_key
    )
    if token.startswith("sk-"):
        return hash_token(token)
    if is_jwt(token):
        return f"{HASHED_JWT_PREFIX}{hash_token(token)}"
    return token
