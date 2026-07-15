"""
CLI Token Utilities

SDK-level utilities for reading CLI authentication tokens.
This module has no dependencies on proxy code and can be safely imported at the SDK level.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional


def get_cli_token_file_path() -> str:
    """Get the path to the CLI token file"""
    home_dir = Path.home()
    config_dir = home_dir / ".litellm"
    return str(config_dir / "token.json")


def load_cli_token() -> Optional[dict]:
    """Load CLI token data from file"""
    token_file = get_cli_token_file_path()
    if not os.path.exists(token_file):
        return None

    try:
        with open(token_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_litellm_gateway_api_key(
    expected_base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Get the stored CLI API key for use with LiteLLM SDK.

    This function reads the token file created by `lite login`
    and returns the API key for use in Python scripts.

    Args:
        expected_base_url: When provided, the key is only returned if it was
            originally issued for this URL. Pass the target server URL to
            prevent credential leakage when the client is pointed at a
            different (possibly malicious) server.

    Returns:
        str: The API key if found (and origin matches), None otherwise

    Example:
        >>> import litellm
        >>> api_key = litellm.get_litellm_gateway_api_key()
        >>> if api_key:
        >>>     response = litellm.completion(
        >>>         model="gpt-3.5-turbo",
        >>>         messages=[{"role": "user", "content": "Hello"}],
        >>>         api_key=api_key,
        >>>         base_url="https://your-proxy.com/v1"
        >>>     )
    """
    token_data = load_cli_token()
    if not token_data or "key" not in token_data:
        return None
    if expected_base_url is not None:
        stored_url = token_data.get("base_url")
        if stored_url != expected_base_url.rstrip("/"):
            return None
    return token_data["key"]


def is_cli_token_fresh(token_data: dict, buffer_hours: float = 0.1) -> bool:
    """Check whether a cached CLI token (as stored in token.json) is still
    within its expiration window. Used by `lite auth print-token` to fail
    fast, without a network round trip, once the cached token is past
    `LITELLM_CLI_JWT_EXPIRATION_HOURS`."""
    from litellm.constants import CLI_JWT_EXPIRATION_HOURS

    timestamp = token_data.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return False
    age_hours = (time.time() - timestamp) / 3600
    return age_hours < (CLI_JWT_EXPIRATION_HOURS - buffer_hours)
