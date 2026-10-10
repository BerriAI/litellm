from typing import Final


def is_jwt(token: str | None) -> bool:
    if token is None:
        return False
    parts: Final = token.split(".")
    return len(parts) == 3
