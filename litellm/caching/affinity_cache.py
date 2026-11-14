"""Atomic affinity claims shared by deployment and tier-model selection."""

from collections.abc import Mapping
from typing import Final

from litellm._logging import verbose_router_logger
from litellm.caching.redis_batch import Batch
from litellm.redis.stores.router_session_pins import ROUTER_SESSION_PINS_TARGET, PinKey, RouterSessionPinStore

__all__ = [
    "ROUTER_SESSION_PINS_TARGET",
    "claim_affinity_pin",
    "read_affinity_pin",
    "set_local_affinity_pin",
]


def set_local_affinity_pin(pins: RouterSessionPinStore, cache_key: PinKey, value: object, ttl_seconds: int) -> None:
    pins.set_local(cache_key, value, ttl_seconds)


async def read_affinity_pin(pins: RouterSessionPinStore, cache_key: PinKey, ttl_seconds: int) -> object:
    """Refresh the authoritative pin without writing a previously observed value back."""
    if pins.has_redis:
        try:
            value: Final = await pins.read(cache_key, ttl_seconds, batch=Batch.PRE_CALL)
        except Exception as error:  # noqa: BLE001  # Redis faults retain pod-local continuity, unlike a successful miss
            verbose_router_logger.debug("Affinity cache: Redis read failed, using pod-local pin. error=%s", error)
        else:
            return value
    local_value: Final = pins.get_local(cache_key)
    if local_value is not None:
        set_local_affinity_pin(pins, cache_key, local_value, ttl_seconds)
    return local_value


async def claim_affinity_pin(
    pins: RouterSessionPinStore,
    cache_key: PinKey,
    pin_value: Mapping[str, str],
    ttl_seconds: int,
    *,
    eligible_values: tuple[Mapping[str, str], ...] | None = None,
    preserve_legacy_model_ids: bool = False,
) -> object:
    """Return the authoritative first writer, replacing it only when it becomes ineligible.

    Eligible claims refresh the returned winner. Legacy deployment claims only refresh
    a matching candidate. Resolve Redis per call because the proxy attaches it lazily.
    """
    if pins.has_redis:
        try:
            winner: Final = await pins.claim(
                cache_key,
                pin_value,
                ttl_seconds,
                eligible_values=eligible_values,
                preserve_legacy_model_ids=preserve_legacy_model_ids,
            )
        except Exception as error:  # noqa: BLE001  # Redis/Lua faults retain same-pod affinity through local claims
            verbose_router_logger.debug("Affinity cache: Redis claim failed, using pod-local claim. error=%s", error)
        else:
            return pin_value if winner is None else winner
    return pins.claim_local(
        cache_key,
        pin_value,
        ttl_seconds,
        eligible_values=eligible_values,
        preserve_legacy_model_ids=preserve_legacy_model_ids,
    )
