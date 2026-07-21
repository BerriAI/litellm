from datetime import datetime, timezone

import litellm
from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time


def get_budget_reset_timezone():
    """
    Get the budget reset timezone from litellm_settings.
    Falls back to UTC if not specified.

    litellm_settings values are set as attributes on the litellm module
    by proxy_server.py at startup (via setattr(litellm, key, value)).
    """
    return getattr(litellm, "timezone", None) or "UTC"


def get_weekly_budget_reset_day() -> str:
    """
    Get the configured weekly budget reset day from litellm_settings.
    Falls back to "monday" if not specified.

    Set via litellm_settings in config.yaml:
        litellm_settings:
            weekly_budget_reset_day: "sunday"

    litellm_settings values are set as attributes on the litellm module
    by proxy_server.py at startup (via setattr(litellm, key, value)).
    """
    return getattr(litellm, "weekly_budget_reset_day", None) or "monday"


def get_budget_reset_time(budget_duration: str) -> datetime:
    """
    Get the budget reset time based on the configured timezone and weekly reset day.
    Falls back to UTC / Monday if not specified.
    """

    reset_at = get_next_standardized_reset_time(
        duration=budget_duration,
        current_time=datetime.now(timezone.utc),
        timezone_str=get_budget_reset_timezone(),
        weekly_reset_day=get_weekly_budget_reset_day(),
    )
    return reset_at
