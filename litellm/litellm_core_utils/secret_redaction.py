"""
Credential/secret redaction utilities.

This module owns the compiled regex and the public `redact_string` helper so
that any part of the codebase (logging, exception mapping, etc.) can scrub
secrets from strings without depending on the logging-configuration module.
"""

import re
from typing import List

_REDACTED = "REDACTED"


def _build_secret_patterns() -> "re.Pattern[str]":
    patterns: List[str] = [
        # PEM private key / certificate blocks
        r"-----BEGIN[A-Z \-]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \-]*PRIVATE KEY-----",
        # GCP OAuth2 access tokens (ya29.*)
        r"\bya29\.[A-Za-z0-9_.~+/-]+",
        # Credential %s formatting (space separator, no key= prefix)
        r"(?:client_secret|azure_password|azure_username)\s+[^\s,'\"})\]{}>]+",
        # AWS access key IDs
        r"(?:AKIA|ASIA)[0-9A-Z]{16}",
        # AWS secrets / session tokens / access key IDs (key=value)
        r"(?:aws_secret_access_key|aws_session_token|aws_access_key_id)"
        r"\s*[:=]\s*[A-Za-z0-9/+=]{20,}",
        # Bearer tokens (OAuth, JWT, etc.)
        r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
        # Basic auth headers
        r"Basic\s+[A-Za-z0-9+/]{10,}={0,2}",
        # OpenAI / Anthropic sk- prefixed keys
        r"sk-[A-Za-z0-9\-_]{20,}",
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
        # Database connection string credentials (scheme://user:pass@host)
        r"(?<=://)[^\s'\"]*:[^\s'\"@]+(?=@)",
        # Databricks personal access tokens
        r"dapi[0-9a-f]{32}",
        # ── Key-name-based redaction ──
        # Catches secrets inside dicts/config dumps by matching on the KEY name
        # regardless of what the value looks like.
        # e.g. 'master_key': 'any-value-here', "database_url": "postgres://..."
        # private_key with PEM-aware value capture
        r"""private_key['\"]?\s*[:=]\s*['\"]?(?:-----BEGIN[A-Z \-]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \-]*PRIVATE KEY-----|[^\s,'\"})\]{}>]+)""",
        r"(?:master_key|database_url|db_url|connection_string|"
        r"signing_key|encryption_key|"
        r"auth_token|access_token|refresh_token|"
        r"slack_webhook_url|webhook_url|"
        r"database_connection_string|"
        r"huggingface_token|jwt_secret)"
        r"""['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+""",
        # Raw JWTs (without Bearer prefix)
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*",
        # Azure SAS tokens in URLs
        r"[?&]sig=[A-Za-z0-9%+/=]+",
        # Full JSON service-account blobs (single-line and multi-line)
        r'\{[^{}]*"type"\s*:\s*"service_account"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
    ]
    return re.compile("|".join(patterns), re.IGNORECASE)


_SECRET_RE = _build_secret_patterns()


def redact_string(value: str) -> str:
    """Scrub known secret/credential patterns from *value* and return the result."""
    return _SECRET_RE.sub(_REDACTED, value)
