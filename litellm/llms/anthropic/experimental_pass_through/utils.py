import os

import litellm


def is_default_reasoning_summary_disabled() -> bool:
    """Check whether the default 'summary: detailed' injection should be suppressed."""
    return (
        litellm.disable_default_reasoning_summary
        or os.getenv("LITELLM_DISABLE_DEFAULT_REASONING_SUMMARY", "false").lower()
        == "true"
    )
