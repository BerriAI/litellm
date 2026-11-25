import datetime
from collections.abc import Mapping
from typing import Final


def is_past_deprecation_date(value: Mapping[str, object]) -> bool:
    deprecation_date: Final = value.get("deprecation_date")
    if not isinstance(deprecation_date, str):
        return False
    try:
        return datetime.date.fromisoformat(deprecation_date) <= datetime.datetime.now(datetime.timezone.utc).date()
    except ValueError:
        return False
