from datetime import datetime, timezone

from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time


def get_budget_reset_timezone():
    """
    Get the budget reset timezone from litellm module attributes.
    Falls back to UTC if not specified.
    """
    import litellm

    return getattr(litellm, "timezone", None) or "UTC"


def get_budget_reset_time(budget_duration: str):
    """
    Get the budget reset time from general_settings.
    Falls back to UTC if not specified.
    """

    reset_at = get_next_standardized_reset_time(
        duration=budget_duration,
        current_time=datetime.now(timezone.utc),
        timezone_str=get_budget_reset_timezone(),
    )
    return reset_at
