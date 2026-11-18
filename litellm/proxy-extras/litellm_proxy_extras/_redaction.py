import functools
import os
import re
from collections.abc import Callable
from typing import Final
from urllib.parse import unquote, urlsplit

_REDACTED: Final = "REDACTED"
_PASSWORD_QUERY_KEYS: Final = frozenset(("password", "sslpassword"))


@functools.cache
def _secret_shape_redactor() -> Callable[[str], str]:
    try:
        from litellm._logging import redact_secrets
    except ImportError:
        return lambda text: text
    return redact_secrets


def _url_passwords(url: str) -> frozenset[str]:
    try:
        parts: Final = urlsplit(url)
    except ValueError:
        return frozenset()
    query_pairs: Final = tuple(pair.partition("=") for pair in parts.query.split("&"))
    raw_query_passwords: Final = tuple(
        value for key, separator, value in query_pairs if separator and key.lower() in _PASSWORD_QUERY_KEYS
    )
    raw_passwords: Final = ((parts.password,) if parts.password else ()) + raw_query_passwords
    return frozenset(password for password in raw_passwords + tuple(map(unquote, raw_passwords)) if password)


def _configured_database_passwords() -> frozenset[str]:
    database_url: Final = os.getenv("DATABASE_URL")
    direct_url: Final = os.getenv("DIRECT_URL")
    database_passwords: Final = _url_passwords(database_url) if database_url else frozenset()
    direct_passwords: Final = _url_passwords(direct_url) if direct_url else frozenset()
    return database_passwords | direct_passwords


def _redact_credentials(text: str) -> str:
    """Mask configured database passwords before passing the text to LiteLLM redaction."""
    passwords: Final = sorted(_configured_database_passwords(), key=len, reverse=True)
    alternation: Final = "|".join(re.escape(password) for password in passwords)
    password_pattern: Final = (
        re.compile(rf"(?<![A-Za-z0-9_])(?:{alternation})(?![A-Za-z0-9_])", re.IGNORECASE) if passwords else None
    )
    result: Final = password_pattern.sub(_REDACTED, text) if password_pattern is not None else text
    return _secret_shape_redactor()(result)
