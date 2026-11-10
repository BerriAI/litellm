import hashlib
import secrets


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def is_master_key(token: str | None, master_key: str | None) -> bool:
    if token is None or master_key is None:
        return False
    return secrets.compare_digest(token.encode(), master_key.encode())
