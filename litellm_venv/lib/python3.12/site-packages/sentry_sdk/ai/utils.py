from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from sentry_sdk.tracing import Span
from sentry_sdk.utils import logger


def _normalize_data(data):
    # type: (Any) -> Any

    # convert pydantic data (e.g. OpenAI v1+) to json compatible format
    if hasattr(data, "model_dump"):
        try:
            return data.model_dump()
        except Exception as e:
            logger.warning("Could not convert pydantic data to JSON: %s", e)
            return data
    if isinstance(data, list):
        if len(data) == 1:
            return _normalize_data(data[0])  # remove empty dimensions
        return list(_normalize_data(x) for x in data)
    if isinstance(data, dict):
        return {k: _normalize_data(v) for (k, v) in data.items()}
    return data


def set_data_normalized(span, key, value):
    # type: (Span, str, Any) -> None
    normalized = _normalize_data(value)
    span.set_data(key, normalized)
