# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

"""
Utility to redact API keys from error messages.

API keys from various providers can leak into error messages when provider
APIs return errors that include the request's authentication details.
This module provides functions to detect and mask such keys before they
reach end users.
"""

import re
from typing import Optional

# Pattern to match common API key formats in error messages
# Covers: sk-xxx, Bearer xxx, key=xxx in URLs, and common provider key patterns
_API_KEY_PATTERNS = [
    # Anthropic keys: sk-ant-<key> (must be before generic sk- pattern)
    (re.compile(r"(sk-ant-)[A-Za-z0-9_-]{20,}"), r"\g<1>****"),
    # OpenAI-style keys: sk-<anything that looks like a key>
    # Match sk- followed by at least 20 chars of key material
    (re.compile(r"(sk-)[A-Za-z0-9_-]{20,}"), r"\g<1>****"),
    # Bearer token in messages: Bearer <token>
    (re.compile(r"(Bearer\s+)[A-Za-z0-9_.+/=-]{10,}"), r"\g<1>[REDACTED]"),
    # Authorization header value patterns
    (re.compile(r"(Authorization['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9_.+/=-]{10,}"), r"\g<1>[REDACTED]"),
    # URL query param: key=<value> or api_key=<value> or apikey=<value>
    (re.compile(r"((?:api[_-]?key|key|token|secret|password|credential|auth)=)[A-Za-z0-9_.+/=-]{8,}(?=&|$|\s|['\"])"), r"\g<1>[REDACTED]"),
    # Azure API keys (32-char hex strings preceded by api-key header context)
    (re.compile(r"(api-key['\"]?\s*[:=]\s*['\"]?)[a-f0-9]{32,}"), r"\g<1>[REDACTED]"),
    # Generic long hex/base64 strings that look like keys when preceded by key-related words
    (re.compile(r"((?:api[_-]?key|secret|token|credential|auth[_-]?token)['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9_.+/=-]{16,}"), r"\g<1>[REDACTED]"),
]


def redact_api_keys(message: Optional[str]) -> Optional[str]:
    """
    Redact API keys from an error message string.

    Scans the message for patterns that look like API keys and replaces them
    with redacted placeholders. This prevents leaking sensitive credentials
    in error responses returned to users.

    Args:
        message: The error message string to redact.

    Returns:
        The message with API keys redacted, or the original message if no keys found.
        Returns None if input is None.
    """
    if message is None:
        return None

    if not isinstance(message, str):
        return message

    redacted = message
    for pattern, replacement in _API_KEY_PATTERNS:
        redacted = pattern.sub(replacement, redacted)

    return redacted
