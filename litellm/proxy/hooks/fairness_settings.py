"""
Applies ``litellm_settings.fairness_settings`` to the running proxy.

``FairnessSettings`` is the admin-facing source of truth for fairness under load. The dynamic
rate limiter still reads ``litellm.priority_reservation`` and ``litellm.priority_reservation_settings``,
so enabling fairness mirrors the workload classes into those globals and registers the v3 limiter
callback when the config file never did.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.litellm_logging import (
    _init_custom_logger_compatible_class,  # pyright: ignore[reportPrivateUsage]  # only constructor the callback registry offers
    get_custom_logger_compatible_class,
)
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    _PROXY_DynamicRateLimitHandlerV3,  # pyright: ignore[reportPrivateUsage]  # registry class carries a leading underscore
)
from litellm.router import Router
from litellm.types.proxy.fairness import FairnessSettings
from litellm.types.utils import PriorityReservationDict, PriorityReservationSettings

_SETTINGS_ADAPTER: Final = TypeAdapter(FairnessSettings)
_LIMITER_CALLBACK: Final = "dynamic_rate_limiter_v3"


@dataclass(slots=True)
class _ConfigReservation:
    """The ``priority_reservation`` globals the config file set before fairness took them over."""

    priority_reservation: Mapping[str, float | PriorityReservationDict] | None = None
    settings: PriorityReservationSettings | None = None


_CONFIG_RESERVATION: Final = _ConfigReservation()


def parse_fairness_settings(value: object) -> FairnessSettings | None:
    if isinstance(value, FairnessSettings):
        return value
    try:
        return _SETTINGS_ADAPTER.validate_python(value)
    except ValidationError as error:
        verbose_proxy_logger.error("Ignoring invalid litellm_settings.fairness_settings: %s", error)
        return None


def active_fairness_limiter() -> _PROXY_DynamicRateLimitHandlerV3 | None:
    limiter: Final = get_custom_logger_compatible_class(_LIMITER_CALLBACK)
    return limiter if isinstance(limiter, _PROXY_DynamicRateLimitHandlerV3) else None


def apply_fairness_settings(
    settings: FairnessSettings,
    internal_usage_cache: DualCache | None,
    llm_router: Router | None,
) -> None:
    previous: Final = litellm.fairness_settings
    was_enabled: Final = previous is not None and previous.enabled
    litellm.fairness_settings = settings
    if not settings.enabled:
        if was_enabled:
            litellm.priority_reservation = _CONFIG_RESERVATION.priority_reservation
            litellm.priority_reservation_settings = _CONFIG_RESERVATION.settings or PriorityReservationSettings()
        return
    if not was_enabled:
        _CONFIG_RESERVATION.priority_reservation = litellm.priority_reservation
        _CONFIG_RESERVATION.settings = litellm.priority_reservation_settings
    litellm.priority_reservation = settings.reserved_shares()
    litellm.priority_reservation_settings = PriorityReservationSettings(
        default_priority=settings.default_reserved_share,
        saturation_threshold=settings.saturation_threshold,
        saturation_check_cache_ttl=settings.saturation_check_cache_ttl,
    )
    _ensure_limiter_registered(internal_usage_cache, llm_router)


def _ensure_limiter_registered(internal_usage_cache: DualCache | None, llm_router: Router | None) -> None:
    already_registered: Final = any(
        callback == _LIMITER_CALLBACK or isinstance(callback, _PROXY_DynamicRateLimitHandlerV3)
        for callback in litellm.callbacks
    )
    existing: Final = active_fairness_limiter()
    if existing is None and (llm_router is None or internal_usage_cache is None):
        if not already_registered:
            litellm.logging_callback_manager.add_litellm_callback(_LIMITER_CALLBACK)
        return
    initialized: Final = (
        existing
        if existing is not None
        else _init_custom_logger_compatible_class(
            _LIMITER_CALLBACK, internal_usage_cache=internal_usage_cache, llm_router=llm_router
        )
    )
    if not isinstance(initialized, _PROXY_DynamicRateLimitHandlerV3):
        return
    if llm_router is not None:
        initialized.update_variables(llm_router=llm_router)
    if not already_registered:
        litellm.logging_callback_manager.add_litellm_callback(initialized)
