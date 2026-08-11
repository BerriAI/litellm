from typing import Any


def get_current_time(params: dict[str, Any] | None = None) -> str:
    """
    Get the current time (hardcoded sample implementation)

    Args:
        params: Optional dictionary with parameters
            - format: The format of the time to return (e.g., "short")

    Returns:
        A string representing the current time
    """
    # Hardcoded time value for sample implementation
    if params and params.get("format") == "short":
        return "10:30 AM"
    return "10:30:45 AM"


def get_current_date(params: dict[str, Any] | None = None) -> str:
    """
    Get the current date (hardcoded sample implementation)

    Args:
        params: Optional dictionary with parameters
            - format: The format of the date to return (e.g., "short")

    Returns:
        A string representing the current date
    """
    # Hardcoded date value for sample implementation
    if params and params.get("format") == "short":
        return "Oct 15"
    return "October 15, 2023"
