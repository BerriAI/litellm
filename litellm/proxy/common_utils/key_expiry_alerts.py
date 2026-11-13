from collections.abc import Mapping
from typing import Final

from litellm.constants import KEY_EXPIRY_ALERT_MAX_DAYS

KEY_EXPIRY_ALERT_DAYS_ERROR: Final = (
    f"key_expiry_alert_days must be an integer between 1 and {KEY_EXPIRY_ALERT_MAX_DAYS}"
)


def key_expiry_alert_days(value: object) -> int | None:
    if type(value) is not int or not 1 <= value <= KEY_EXPIRY_ALERT_MAX_DAYS:
        return None
    return value


def key_expiry_alert_metadata_error(metadata: Mapping[str, object] | None) -> str | None:
    if metadata is None or "key_expiry_alert_days" not in metadata:
        return None
    if key_expiry_alert_days(metadata["key_expiry_alert_days"]) is not None:
        return None
    return f"metadata.{KEY_EXPIRY_ALERT_DAYS_ERROR}"
