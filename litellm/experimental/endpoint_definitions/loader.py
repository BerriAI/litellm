"""
Utilities for endpoint definitions.
"""

import os
from typing import Optional


def get_api_key(env_vars: list) -> Optional[str]:
    """Get API key from environment variables."""
    for var in env_vars:
        key = os.environ.get(var)
        if key:
            return key
    return None
