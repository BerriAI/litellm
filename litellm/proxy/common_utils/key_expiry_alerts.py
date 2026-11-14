import logging
from collections.abc import Mapping
from typing import Final

from litellm.constants import KEY_EXPIRY_ALERT_MAX_DAYS

logger: Final = logging.getLogger(__name__)
KEY_EXPIRY_ALERT_DAYS_ERROR: Final = (
    f"key_expiry_alert_days must be an integer between 1 and {KEY_EXPIRY_ALERT_MAX_DAYS}"
)


def key_expiry_alert_days(value: object) -> int | None:
    if type(value) is not int or not 1 <= value <= KEY_EXPIRY_ALERT_MAX_DAYS:
        return None
    return value


def configured_key_expiry_alert_days(general_settings: Mapping[str, object]) -> int | None:
    """The global key_expiry_alert_days from the current general settings, read each run so a UI or API edit
    applies without a restart. An invalid value is logged and treated as unset, leaving per-key values working."""
    configured: Final = general_settings.get("key_expiry_alert_days")
    days: Final = key_expiry_alert_days(configured)
    if configured is not None and days is None:
        logger.error("Key expiry alerts: ignoring general_settings.%s", KEY_EXPIRY_ALERT_DAYS_ERROR)
    return days


def key_expiry_alert_metadata_error(metadata: Mapping[str, object] | None) -> str | None:
    if metadata is None or "key_expiry_alert_days" not in metadata:
        return None
    if key_expiry_alert_days(metadata["key_expiry_alert_days"]) is not None:
        return None
    return f"metadata.{KEY_EXPIRY_ALERT_DAYS_ERROR}"
