"""
Common utilities for SLNG provider integration.
"""
import os
from typing import Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class SlngException(BaseLLMException):
    """Exception raised for SLNG-specific errors."""
    pass


def get_slng_api_key(
    api_key: Optional[str] = None,
    env_var: str = "SLNG_API_KEY"
) -> str:
    """
    Retrieve SLNG API key from parameter or environment variable.

    Args:
        api_key: API key passed explicitly (takes precedence)
        env_var: Environment variable name to check (default: SLNG_API_KEY)

    Returns:
        API key string

    Raises:
        SlngException: If API key is not found
    """
    key = api_key or os.getenv(env_var)
    if not key:
        raise SlngException(
            message=f"SLNG API key not found. Set {env_var} environment variable or pass api_key parameter.",
            status_code=401
        )
    return key


def get_slng_api_base(
    api_base: Optional[str] = None,
    env_var: str = "SLNG_API_BASE"
) -> str:
    """
    Retrieve SLNG API base URL from parameter or environment variable.

    Args:
        api_base: API base URL passed explicitly (takes precedence)
        env_var: Environment variable name to check (default: SLNG_API_BASE)

    Returns:
        API base URL string
    """
    return api_base or os.getenv(env_var, "https://api.slng.ai")
