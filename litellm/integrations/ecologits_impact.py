from collections.abc import Sequence
from types import MappingProxyType
from typing import Final, Protocol

from litellm._logging import verbose_logger
from litellm.impact_calculator import ImpactRequest, interval_impact_value, point_impact_value
from litellm.types.utils import ImpactInformation, ImpactValue

ESTIMATOR_NAME: Final = "ecologits"

# ecologits totals cover the usage phase plus embodied manufacturing, which is boundary B.
# Source: https://ecologits.ai/latest/methodology/llm_inference/ read 2026-09-20
IMPACT_BOUNDARY: Final = "B"

# litellm names a provider by the API it speaks to; ecologits names one by who trained the
# model, and keys its registry on that. Providers absent here have no ecologits data.
PROVIDER_NAMES: Final = MappingProxyType(
    {
        "anthropic": "anthropic",
        "azure": "openai",
        "azure_ai": "openai",
        "cohere": "cohere",
        "cohere_chat": "cohere",
        "gemini": "google_genai",
        "huggingface": "huggingface_hub",
        "mistral": "mistralai",
        "openai": "openai",
        "text-completion-openai": "openai",
        "vertex_ai": "google_genai",
    }
)


class _RangeLike(Protocol):
    min: float
    max: float


class _CriterionLike(Protocol):
    value: float | _RangeLike
    unit: str


class _ErrorLike(Protocol):
    message: str


class _ImpactsLike(Protocol):
    energy: _CriterionLike | None
    gwp: _CriterionLike | None
    adpe: _CriterionLike | None
    pe: _CriterionLike | None
    wcf: _CriterionLike | None
    errors: Sequence[_ErrorLike]


def _to_impact_value(criterion: _CriterionLike | None) -> ImpactValue | None:
    if criterion is None:
        return None
    raw: Final = criterion.value
    if isinstance(raw, (int, float)):
        return point_impact_value(unit=criterion.unit, value=float(raw))
    return interval_impact_value(unit=criterion.unit, minimum=float(raw.min), maximum=float(raw.max))


def _bare_model_name(model: str) -> str:
    """Strip the routing prefix litellm carries, leaving the name ecologits registers."""
    return model.rsplit("/", 1)[-1]


class EcoLogitsImpactEstimator:
    """
    Estimates request impact with EcoLogits.

    Constructing this requires the ``ecologits`` package. The import is deferred to
    ``__init__`` so that importing LiteLLM, and this module, stays free of that
    dependency and only enabling the feature can fail.
    """

    def __init__(self) -> None:
        from ecologits.tracers.utils import llm_impacts

        self._llm_impacts: Final = llm_impacts

    @property
    def name(self) -> str:
        return ESTIMATOR_NAME

    def estimate(self, request: ImpactRequest) -> ImpactInformation | None:
        if request.custom_llm_provider is None or request.response_time is None:
            return None
        if request.completion_tokens <= 0:
            return None
        provider: Final = PROVIDER_NAMES.get(request.custom_llm_provider)
        if provider is None:
            return None

        impacts: Final[_ImpactsLike | None] = self._llm_impacts(
            provider=provider,
            model_name=_bare_model_name(request.model),
            output_token_count=request.completion_tokens,
            request_latency=request.response_time,
        )
        if impacts is None or impacts.errors:
            verbose_logger.debug(
                "ecologits has no impact data for %s/%s: %s",
                provider,
                request.model,
                tuple(error.message for error in impacts.errors) if impacts is not None else "no result",
            )
            return None

        return ImpactInformation(
            energy=_to_impact_value(impacts.energy),
            gwp=_to_impact_value(impacts.gwp),
            adpe=_to_impact_value(impacts.adpe),
            pe=_to_impact_value(impacts.pe),
            water=_to_impact_value(impacts.wcf),
            boundary=IMPACT_BOUNDARY,
            estimator=ESTIMATOR_NAME,
        )
