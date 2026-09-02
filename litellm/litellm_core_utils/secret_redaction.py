"""
Credential/secret redaction utilities.

This module owns the compiled regex and the public `redact_string` helper so
that any part of the codebase (logging, exception mapping, etc.) can scrub
secrets from strings without depending on the logging-configuration module.
"""

import re
from typing import Final

from litellm.constants import MINIMUM_CUSTOM_KEY_LENGTH

_REDACTED: Final = "REDACTED"


def _build_secret_patterns() -> "re.Pattern[str]":
    patterns: Final[list[str]] = [
        # PEM private key / certificate blocks
        r"-----BEGIN[A-Z \-]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \-]*PRIVATE KEY-----",
        # GCP OAuth2 access tokens (ya29.*)
        r"\bya29\.[A-Za-z0-9_.~+/-]+",
        # Credential %s formatting (space separator, no key= prefix)
        r"(?:client_secret|azure_password|azure_username)\s+[^\s,'\"})\]{}>]+",
        # AWS access key IDs
        r"(?:AKIA|ASIA)[0-9A-Z]{16}",
        # Bearer tokens (OAuth, JWT, etc.)
        r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
        # Basic auth headers
        r"Basic\s+[A-Za-z0-9+/]{10,}={0,2}",
        # OpenAI / Anthropic sk- prefixed keys
        rf"sk-[A-Za-z0-9\-_]{{{MINIMUM_CUSTOM_KEY_LENGTH - len('sk-')},}}",
        # Credentials passed as URL query params. Terminated by "&" like the key=
        # and sig= patterns below, so the rest of the request line survives in an
        # access log. Must precede the generic patterns to win at the same position.
        r"(?<=[?&])(?:api[_-]?key|\w*(?:token|password|passwd|client_secret|secret_key|_secret))"
        r"=[^\s&'\"]+",
        # Generic api_key / api-key / apikey (handles 'key': 'value' dict repr)
        r"(?:api[_-]?key)['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]{8,}",
        # x-api-key / api-key header values (handles 'key': 'value' dict repr)
        r"(?:x-api-key|api-key)['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # Anthropic internal header keys
        r"x-ak-[A-Za-z0-9\-_]{20,}",
        # Google API keys (bare key value)
        r"AIza[0-9A-Za-z\-_]{35}",
        # URL query-param key=VALUE (e.g. ?key=AIza... or &key=...) — catches the
        # full "key=<secret>" fragment so the value is redacted regardless of format.
        r"(?<=[?&])key=[^\s&'\"]{8,}",
        # Password / secret params (handles key=value and 'key': 'value')
        # Word boundary prevents O(n^2) backtracking on long word-char runs.
        r"(?:^|(?<=\W))\w*(?:password|passwd|client_secret|secret_key|_secret)"
        r"['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # Database connection string credentials (scheme://user:pass@host).
        # The user half stops at the ":" separator and both halves are length-capped,
        # so a long attacker-supplied URL cannot backtrack quadratically.
        r"(?<=://)[^\s'\":]{0,4096}:[^\s'\"]{1,4096}(?=@)",
        # Databricks personal access tokens
        r"dapi[0-9a-f]{32}",
        # Module-level provider keys logged as litellm.<provider>_key=<value>
        r"litellm\.[A-Za-z0-9_]*_key['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # ── Key-name-based redaction ──
        # Catches secrets inside dicts/config dumps by matching on the KEY name
        # regardless of what the value looks like.
        # e.g. 'master_key': 'any-value-here', "database_url": "postgres://..."
        # private_key with PEM-aware value capture
        r"""private_key['\"]?\s*[:=]\s*['\"]?(?:-----BEGIN[A-Z \-]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \-]*PRIVATE KEY-----|[^\s,'\"})\]{}>]+)""",
        r"(?:master_key|xai_key|database_url|db_url|connection_string|"
        r"aws_secret_access_key|aws_session_token|aws_access_key_id|"
        r"signing_key|encryption_key|"
        r"auth_token|access_token|refresh_token|"
        r"slack_webhook_url|webhook_url|"
        r"database_connection_string|"
        r"huggingface_token|jwt_secret)"
        r"""['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+""",
        # Raw JWTs (without Bearer prefix)
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*",
        # Azure SAS tokens in URLs. The delimiter is a lookbehind, like the
        # `key=` pattern above, so the `?` or `&` survives and the redacted URL
        # stays well formed (this string is often a request line in a log).
        r"(?<=[?&])sig=[A-Za-z0-9%+/=]+",
        # Full JSON service-account blobs (single-line and multi-line)
        r'\{[^{}]*"type"\s*:\s*"service_account"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
    ]
    return re.compile("|".join(patterns), re.IGNORECASE)


_SECRET_RE: Final = _build_secret_patterns()


def redact_string(value: str) -> str:
    """Scrub known secret/credential patterns from *value* and return the result."""
    return _SECRET_RE.sub(_REDACTED, value)


def _build_internal_detail_patterns() -> "re.Pattern[str]":
    patterns: Final[tuple[str, ...]] = (
        # Unix absolute filesystem paths rooted under a well-known OS/user
        # directory. Deliberately excludes bare API routes like /v1/models,
        # which never start with one of these directory names.
        r"/(?:etc|var|opt|usr|home|root|private|Users|tmp|mnt|srv)/[^\s'\"\)\]}>,]+",
        # Windows absolute filesystem paths
        r"[A-Za-z]:\\[^\s'\"\)\]}>,]+",
        # RFC 1918 private IPv4 ranges and loopback
        r"\b(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
        r"192\.168(?:\.\d{1,3}){2}|127(?:\.\d{1,3}){3})\b",
        # Hostnames under a conventionally internal-only suffix
        r"\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:internal|local|corp|lan|intra|private)\b",
    )
    return re.compile("|".join(patterns), re.IGNORECASE)


_INTERNAL_DETAIL_RE: Final = _build_internal_detail_patterns()
_TRACEBACK_MARKER: Final = "Traceback (most recent call last):"


def redact_internal_details(value: str) -> str:
    """Scrub a stack trace, filesystem paths, and internal hostnames from
    *value*, on top of the credential patterns redact_string() already covers.

    litellm's own exception mapper embeds a full traceback in some exception
    messages as a debugging aid for direct SDK callers, so the traceback is
    dropped here rather than in the mapper itself.

    Client-facing use only. An operator's own server logs must keep this
    detail to debug the underlying failure, so this must never run in the
    logging pipeline (redact_string() intentionally does not call it) — only
    at the point a message is about to leave the process in an HTTP response.
    """
    marker_index: Final = value.find(_TRACEBACK_MARKER)
    without_traceback: Final = value[:marker_index].rstrip() if marker_index != -1 else value
    return _INTERNAL_DETAIL_RE.sub(_REDACTED, redact_string(without_traceback))


def redact_structured_value(key: str | None, value: str) -> str:
    """Scrub *value* as it appeared under *key* inside a structured record.

    redact_string() replaces a whole ``key: value`` span with REDACTED, which is
    fine inside free text but destroys the surrounding syntax when the span is a
    JSON member rather than message content. This renders the pair the way a dict
    repr would, so the key-name patterns still fire, but collapses only the value
    so the caller's structure survives.
    """
    scrubbed: Final = redact_string(value)
    if scrubbed != value or key is None:
        return scrubbed
    rendered: Final = f"'{key}': '{value}'"
    return _REDACTED if redact_string(rendered) != rendered else value
