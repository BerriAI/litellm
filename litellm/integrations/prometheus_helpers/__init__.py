"""
Helpers for the Prometheus integration (extracted to keep ``prometheus.py`` smaller).

``PrometheusLabelFactoryContext`` lives here so it has a dedicated module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

from litellm.types.integrations.prometheus import (
    UserAPIKeyLabelValues,
    _sanitize_prometheus_label_name,
    _sanitize_prometheus_label_value,
)

_get_end_user_id_for_cost_tracking = None


def _get_cached_end_user_id_for_cost_tracking():
    """
    Get cached get_end_user_id_for_cost_tracking function.
    Lazy imports on first call to avoid loading utils.py at import time (60MB saved).
    Subsequent calls use cached function for better performance.
    """
    global _get_end_user_id_for_cost_tracking
    if _get_end_user_id_for_cost_tracking is None:
        from litellm.utils import get_end_user_id_for_cost_tracking

        _get_end_user_id_for_cost_tracking = get_end_user_id_for_cost_tracking
    return _get_end_user_id_for_cost_tracking


class PrometheusLabelFactoryContext:
    """
    Precomputes per-request label inputs so prometheus_label_factory can subset
    per metric without repeated model_dump / tag / metadata work.
    """

    __slots__ = (
        "enum_values",
        "_sanitized_enum",
        "_custom_by_sanitized_key",
        "_tag_labels",
        "_resolved_end_user",
    )

    _END_USER_NOT_COMPUTED = object()

    def __init__(self, enum_values: UserAPIKeyLabelValues) -> None:
        self.enum_values = enum_values
        enum_dict = enum_values.model_dump()
        self._sanitized_enum: Dict[str, Optional[str]] = {
            k: _sanitize_prometheus_label_value(v) for k, v in enum_dict.items()
        }
        self._custom_by_sanitized_key: Dict[str, Optional[str]] = {}
        if enum_values.custom_metadata_labels is not None:
            for key, value in enum_values.custom_metadata_labels.items():
                sk = _sanitize_prometheus_label_name(key)
                self._custom_by_sanitized_key[sk] = _sanitize_prometheus_label_value(
                    value
                )
        self._tag_labels: Dict[str, Optional[str]] = {}
        if enum_values.tags is not None:
            # Late import avoids circular import: ``prometheus`` imports this module.
            from litellm.integrations.prometheus import get_custom_labels_from_tags

            for k, v in get_custom_labels_from_tags(enum_values.tags).items():
                self._tag_labels[k] = _sanitize_prometheus_label_value(v)
        # Use a dedicated sentinel so `None` can be cached as a computed result.
        self._resolved_end_user: Any = self._END_USER_NOT_COMPUTED

    def get_resolved_end_user(self) -> Optional[str]:
        if self._resolved_end_user is self._END_USER_NOT_COMPUTED:
            fn = _get_cached_end_user_id_for_cost_tracking()
            self._resolved_end_user = fn(
                litellm_params={"user_api_key_end_user_id": self.enum_values.end_user},
                service_type="prometheus",
            )
        return cast(Optional[str], self._resolved_end_user)
