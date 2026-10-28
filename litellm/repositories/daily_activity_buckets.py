from collections.abc import Sequence
from typing import Final

from litellm.constants import UNKNOWN_MODEL_SPEND_LOG_MODEL

UNKNOWN_MODEL_GROUP: Final = "unknown-model-group"
UNATTRIBUTED_API_KEY: Final = "unattributed"
_RAW_API_KEY_PREFIX: Final = "raw:"


def model_bucket(model: str | None) -> str:
    return model or UNKNOWN_MODEL_SPEND_LOG_MODEL


def model_group_bucket(model_group: str | None, model: str | None) -> str:
    return model_group or model or UNKNOWN_MODEL_GROUP


def api_key_bucket(api_key: str | None) -> str:
    if not api_key:
        return UNATTRIBUTED_API_KEY
    if api_key == UNATTRIBUTED_API_KEY or api_key.startswith(_RAW_API_KEY_PREFIX):
        return _RAW_API_KEY_PREFIX + api_key
    return api_key


def api_key_from_bucket(api_key: str) -> str:
    if api_key == UNATTRIBUTED_API_KEY:
        return ""
    return api_key.removeprefix(_RAW_API_KEY_PREFIX)


def raw_api_key_filter(api_key: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """String filters are reporting IDs; sequences are raw keys resolved by authorization."""
    if api_key is None or api_key == "":
        return None
    if isinstance(api_key, str):
        return (api_key_from_bucket(api_key),)
    return tuple(api_key)
