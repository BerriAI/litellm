"""
Shared helpers for the apiToken.sale provider.
"""

APITOKEN_API_BASE = "https://api.apitoken.sale"
APITOKEN_MESSAGES_PATH = "/v1/messages"


def build_messages_url(api_base: str) -> str:
    """
    Normalize a user-supplied api_base into the full Messages API URL.

    Accepts a bare host (``https://api.apitoken.sale``), a trailing slash,
    a versioned base (``https://api.apitoken.sale/v1``) or the complete path,
    and always returns exactly one ``/v1/messages`` suffix. Naive concatenation
    would otherwise produce ``//v1/messages`` or ``/v1/v1/messages``.
    """
    base_url = api_base.rstrip("/")

    if base_url.endswith(APITOKEN_MESSAGES_PATH):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/messages"
    return f"{base_url}{APITOKEN_MESSAGES_PATH}"
