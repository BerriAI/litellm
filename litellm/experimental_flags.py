"""
Experimental feature flags for LiteLLM.

All flags default OFF. Enable by setting the corresponding environment variable.
"""

import os

ENABLE_PARALLEL_ACOMPLETIONS: bool = os.getenv(
    "LITELLM_ENABLE_PARALLEL_ACOMPLETIONS", "0"
).lower() in ("1", "true", "yes", "on")
