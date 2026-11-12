from collections.abc import Mapping
from typing import Final

from litellm.constants import KEY_EXPIRY_ALERT_MAX_DAYS
from litellm.litellm_core_utils.core_helpers import as_object_list


def key_expiry_alert_days(value: object) -> tuple[int, ...] | None:
    items: Final = as_object_list(value)
    if items is None:
        return None
    if any(type(item) is not int for item in items):
        return None
    days: Final = tuple(item for item in items if isinstance(item, int))
    if any(day <= 0 or day > KEY_EXPIRY_ALERT_MAX_DAYS for day in days):
        return None
    return days


def key_expiry_alert_metadata_error(metadata: Mapping[str, object] | None) -> str | None:
    if metadata is None or "key_expiry_alert_days" not in metadata:
        return None
    if key_expiry_alert_days(metadata["key_expiry_alert_days"]) is not None:
        return None
    return f"metadata.key_expiry_alert_days must be a list of integers between 1 and {KEY_EXPIRY_ALERT_MAX_DAYS}"
