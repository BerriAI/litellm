"""
Helpers for the Prometheus integration (extracted to keep ``prometheus.py`` smaller).

``PrometheusLabelFactoryContext`` lives here so it has a dedicated module.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final, Literal, Protocol, cast

from litellm.types.integrations.prometheus import (
    UserAPIKeyLabelValues,
    sanitize_prometheus_label_name,
    sanitize_prometheus_label_value,
)

_get_end_user_id_for_cost_tracking = None

_EMAIL_PATTERN: Final = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_UUID_PATTERN: Final = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

_PII_LABEL_KEYS: Final = frozenset(
    {
        "user",
        "user_email",
        "user_alias",
        "team_alias",
        "client_ip",
        "user_agent",
    }
)


def sanitize_pii_label_value(label_name: str, value: object | None) -> str | None:
    """
    Sanitize sensitive tenant PII before Prometheus label emission.
    Protects user emails, client IPs, user-agent automation IDs, and team aliases.
    """
    if value is None:
        return None

    str_val: Final[str] = value if isinstance(value, str) else str(value)
    if not str_val:
        return str_val

    if label_name in ("user", "user_email", "end_user"):
        if _EMAIL_PATTERN.search(str_val):
            return f"redacted_user_{hashlib.sha256(str_val.encode('utf-8')).hexdigest()[:8]}"
        return f"user_{hashlib.sha256(str_val.encode('utf-8')).hexdigest()[:8]}"

    if label_name in ("team_alias", "user_alias", "api_key_alias"):
        if _EMAIL_PATTERN.search(str_val):
            return _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", str_val)
        return str_val

    if label_name == "client_ip":
        if ":" in str_val:
            return f"ipv6_{hashlib.sha256(str_val.encode('utf-8')).hexdigest()[:8]}"
        parts = str_val.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.0.0/16"
        return f"ip_{hashlib.sha256(str_val.encode('utf-8')).hexdigest()[:8]}"

    if label_name == "user_agent":
        sanitized = _UUID_PATTERN.sub("[UUID]", str_val)
        if len(sanitized) > 64:
            return sanitized[:64] + "..."
        return sanitized

    return str_val


class _EndUserIdGetter(Protocol):
    def __call__(
        self,
        litellm_params: dict[str, object],
        service_type: Literal["litellm_logging", "prometheus"] = "litellm_logging",
    ) -> str | None: ...


def get_cached_end_user_id_for_cost_tracking() -> _EndUserIdGetter:
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


_get_cached_end_user_id_for_cost_tracking = get_cached_end_user_id_for_cost_tracking


class PrometheusLabelFactoryContext:
    """
    Precomputes per-request label inputs so prometheus_label_factory can subset
    per metric without repeated model_dump / tag / metadata work.
    """

    __slots__ = (
        "_resolved_end_user",
        "custom_by_sanitized_key",
        "enum_values",
        "sanitized_enum",
        "tag_labels",
    )

    _END_USER_NOT_COMPUTED = object()

    def __init__(self, enum_values: UserAPIKeyLabelValues) -> None:
        import litellm

        self.enum_values = enum_values
        enum_dict: Final = enum_values.model_dump()

        mask_pii = getattr(litellm, "prometheus_mask_pii_labels", None)
        if mask_pii is None:
            # Auto-sanitize PII if metrics endpoint is exposed unauthenticated
            mask_pii = (
                getattr(litellm, "allow_unauthenticated_metrics", None) is True
                or getattr(litellm, "require_auth_for_metrics_endpoint", None) is False
            )

        if mask_pii:
            self.sanitized_enum: dict[str, str | None] = {
                k: sanitize_prometheus_label_value(
                    sanitize_pii_label_value(k, v) if k in _PII_LABEL_KEYS else v
                )
                for k, v in enum_dict.items()
            }
        else:
            self.sanitized_enum: dict[str, str | None] = {
                k: sanitize_prometheus_label_value(v) for k, v in enum_dict.items()
            }
        self.custom_by_sanitized_key: dict[str, str | None] = {}
        if enum_values.custom_metadata_labels is not None:
            for key, value in enum_values.custom_metadata_labels.items():
                sk = sanitize_prometheus_label_name(key)
                self.custom_by_sanitized_key[sk] = sanitize_prometheus_label_value(value)
        self.tag_labels: dict[str, str | None] = {}
        if enum_values.tags is not None:
            # Late import avoids circular import: ``prometheus`` imports this module.
            from litellm.integrations.prometheus import get_custom_labels_from_tags

            for k, v in get_custom_labels_from_tags(enum_values.tags).items():
                self.tag_labels[k] = sanitize_prometheus_label_value(v)
        # Use a dedicated sentinel so `None` can be cached as a computed result.
        self._resolved_end_user: object = self._END_USER_NOT_COMPUTED

    @property
    def _custom_by_sanitized_key(self) -> dict[str, str | None]:
        return self.custom_by_sanitized_key

    @_custom_by_sanitized_key.setter
    def _custom_by_sanitized_key(self, value: dict[str, str | None]) -> None:
        self.custom_by_sanitized_key = value

    @property
    def _sanitized_enum(self) -> dict[str, str | None]:
        return self.sanitized_enum

    @_sanitized_enum.setter
    def _sanitized_enum(self, value: dict[str, str | None]) -> None:
        self.sanitized_enum = value

    @property
    def _tag_labels(self) -> dict[str, str | None]:
        return self.tag_labels

    @_tag_labels.setter
    def _tag_labels(self, value: dict[str, str | None]) -> None:
        self.tag_labels = value

    def get_resolved_end_user(self) -> str | None:
        if self._resolved_end_user is self._END_USER_NOT_COMPUTED:
            fn: Final = get_cached_end_user_id_for_cost_tracking()
            resolved = fn(
                litellm_params={"user_api_key_end_user_id": self.enum_values.end_user},
                service_type="prometheus",
            )
            import litellm

            mask_pii = getattr(litellm, "prometheus_mask_pii_labels", None)
            if mask_pii is None:
                mask_pii = (
                    getattr(litellm, "allow_unauthenticated_metrics", None) is True
                    or getattr(litellm, "require_auth_for_metrics_endpoint", None) is False
                )
            if mask_pii and resolved:
                resolved = sanitize_pii_label_value("end_user", resolved)
            self._resolved_end_user = resolved
        return cast(str | None, self._resolved_end_user)
