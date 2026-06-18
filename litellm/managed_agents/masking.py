"""Masking helpers for managed agents.

The agent's `litellm_api_key` is stored encrypted-at-rest by the proxy and
returned masked on every read — `sk-abc123def456` → `sk-a****`.
"""


def mask_litellm_api_key(key: str) -> str:
    """Return a masked form of a LiteLLM API key.

    Pattern: keep the first 4 characters of the key (including any
    `sk-`-style prefix), then append `****`. For very short keys, mask
    everything to avoid leaking too much.
    """
    if not key:
        return ""
    if len(key) <= 4:
        return "****"
    return f"{key[:4]}****"
