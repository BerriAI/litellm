"""Per-model-group concurrency limits that queue a request instead of rejecting it.

``general_settings.model_concurrency`` maps a model group to the number of
requests that may be in flight for that group at once. A request that arrives
while every slot is taken waits for a slot to free up, up to
``queue_timeout_seconds``, and only then gets a 429. The slots live in the v3
rate limiter's Redis gauge, so every proxy worker and replica shares the same
count; a deployment without Redis keeps one count per process.

Example:

.. code-block:: yaml

    general_settings:
      model_concurrency:
        gpt-4o:
          limit: 5
          queue_timeout_seconds: 60
          max_queue_size: 100

The key is the model group the client asks for (the same name it would pass as
``model``), not the deployment. Every HTTP request holds one slot for its whole
life, including the full stream of a streaming response.

The wait is bounded twice. ``queue_timeout_seconds`` (or the request's own
remaining ``total_timeout``, if shorter) limits how long one request waits, and
``max_queue_size`` limits how many requests wait at all. A request that finds
the queue full gets a 429 immediately, so a saturated group cannot pin an
unbounded number of tasks and Redis polls on the worker. ``max_queue_size``
counts waiters per process, not across replicas, because the resource it
protects is per process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.redis.stores.rate_limits import RateLimitScope

MODEL_CONCURRENCY_SETTING_KEY: Final = "model_concurrency"
MODEL_CONCURRENCY_DESCRIPTOR_KEY: Final = "model_concurrency"
MODEL_CONCURRENCY_QUEUE_POLL_INTERVAL_SECONDS: Final = 0.1
DEFAULT_MODEL_CONCURRENCY_MAX_QUEUE_SIZE: Final = 100


class ModelConcurrencyLimit(BaseModel):
    model_config = ConfigDict(frozen=True)

    limit: int = Field(gt=0)
    queue_timeout_seconds: float = Field(default=60.0, ge=0)
    max_queue_size: int = Field(default=DEFAULT_MODEL_CONCURRENCY_MAX_QUEUE_SIZE, ge=1)


class ModelConcurrencyQueueAdmission:
    """Bounds waiters per process, where the cost of a waiter is paid.

    A Redis-wide cap would throttle an unshared resource and would need a TTL
    reaper to survive a worker dying mid-wait.
    """

    def __init__(self) -> None:
        self._waiters: dict[str, int] = {}

    def try_enter(self, model_group: str, max_queue_size: int) -> bool:
        waiters: Final = self._waiters.get(model_group, 0)
        if waiters >= max_queue_size:
            return False
        self._waiters[model_group] = waiters + 1
        return True

    def leave(self, model_group: str) -> None:
        waiters: Final = self._waiters.get(model_group, 0)
        if waiters <= 1:
            self._waiters.pop(model_group, None)
        else:
            self._waiters[model_group] = waiters - 1


@dataclass(frozen=True, slots=True)
class ModelConcurrencySlot:
    scope: RateLimitScope
    slot_id: str


_MODEL_CONCURRENCY_LIMIT_ADAPTER: Final[TypeAdapter[ModelConcurrencyLimit]] = TypeAdapter(ModelConcurrencyLimit)
_RAW_LIMITS_ADAPTER: Final[TypeAdapter[Mapping[object, object]]] = TypeAdapter(Mapping[object, object])


@lru_cache(maxsize=64)
def _warn_invalid_entry(model_group: str, value: str, error: str) -> None:
    verbose_proxy_logger.warning(
        "Ignoring invalid general_settings.%s[%r]=%s: %s",
        MODEL_CONCURRENCY_SETTING_KEY,
        model_group,
        value,
        error,
    )


def _validated_limit(model_group: str, value: object) -> ModelConcurrencyLimit | None:
    try:
        return _MODEL_CONCURRENCY_LIMIT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        _warn_invalid_entry(model_group, repr(value), str(exc))
        return None


def parse_model_concurrency_limits(raw: object) -> Mapping[str, ModelConcurrencyLimit]:
    if raw is None:
        return MappingProxyType({})
    try:
        entries: Final = _RAW_LIMITS_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        _warn_invalid_entry(MODEL_CONCURRENCY_SETTING_KEY, repr(raw), str(exc))
        return MappingProxyType({})
    return _parse_limit_entries(entries)


def _parse_limit_entries(entries: Mapping[object, object]) -> Mapping[str, ModelConcurrencyLimit]:
    return MappingProxyType(
        {
            model_group: limit
            for model_group, value in entries.items()
            if isinstance(model_group, str) and (limit := _validated_limit(model_group, value)) is not None
        }
    )


def _general_settings_model_concurrency(settings: Mapping[str, object]) -> Mapping[str, ModelConcurrencyLimit]:
    return parse_model_concurrency_limits(settings.get(MODEL_CONCURRENCY_SETTING_KEY))


def model_concurrency_limits_from_general_settings() -> Mapping[str, ModelConcurrencyLimit]:
    from litellm.proxy.proxy_server import general_settings_view

    return _general_settings_model_concurrency(general_settings_view())
