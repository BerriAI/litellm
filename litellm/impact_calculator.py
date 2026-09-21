from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Protocol

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import ImpactInformation, ImpactValue


@dataclass(frozen=True, slots=True)
class ImpactRequest:
    model: str
    custom_llm_provider: str | None
    completion_tokens: int
    response_time: float | None


class ImpactEstimator(Protocol):
    """Produces an environmental impact estimate for a request."""

    @property
    def name(self) -> str: ...

    def estimate(self, request: ImpactRequest) -> ImpactInformation | None: ...


def point_impact_value(unit: str, value: float) -> ImpactValue:
    return ImpactValue(value=value, min=None, max=None, unit=unit)


def interval_impact_value(unit: str, minimum: float, maximum: float) -> ImpactValue:
    """``value`` is the midpoint, so a consumer summing it stays inside the modelled interval."""
    return ImpactValue(value=(minimum + maximum) / 2, min=minimum, max=maximum, unit=unit)


@dataclass(slots=True)
class _EstimatorRegistry:
    estimator: ImpactEstimator | None = None


_registry: Final = _EstimatorRegistry()


def register_impact_estimator(estimator: ImpactEstimator | None) -> None:
    _registry.estimator = estimator


@lru_cache(maxsize=1)
def _resolve_default_estimator() -> ImpactEstimator | None:
    from litellm.integrations.ecologits_impact import EcoLogitsImpactEstimator

    try:
        return EcoLogitsImpactEstimator()
    except Exception as e:  # noqa: BLE001  # a missing backend must never fail request logging
        verbose_logger.warning(
            "litellm.track_impact is on but no impact estimator is available, so no impact is recorded: %s",
            str(e),
        )
        return None


def get_impact_estimator() -> ImpactEstimator | None:
    if _registry.estimator is not None:
        return _registry.estimator
    return _resolve_default_estimator()


def current_estimator_name() -> str | None:
    estimator: Final = get_impact_estimator()
    return None if estimator is None else estimator.name


def reset_default_estimator_cache() -> None:
    _resolve_default_estimator.cache_clear()


def calculate_impact(request: ImpactRequest) -> ImpactInformation | None:
    """
    Estimate the environmental impact of a request.

    Raises whatever the estimator raises, so the caller can record the failure.
    """
    if litellm.track_impact is not True:
        return None
    estimator: Final = get_impact_estimator()
    if estimator is None:
        return None
    return estimator.estimate(request)
