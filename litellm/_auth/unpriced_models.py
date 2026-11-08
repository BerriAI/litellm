from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class PriceMetric:
    name: str
    scalar: float | None = None
    nested: tuple[float | None, ...] = ()


@dataclass(frozen=True, slots=True)
class PriceEntry:
    tiered_present: bool
    metrics: tuple[PriceMetric, ...]


@dataclass(frozen=True, slots=True)
class DeploymentPricing:
    parameter_keys: tuple[str, ...]
    has_id: bool
    info_loaded: bool = False
    info: PriceEntry | None = None
    cost_keys: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class UnpricedModelFacts:
    model: str | None
    router_present: bool
    group_found: bool | None = None
    deployments: tuple[DeploymentPricing, ...] | None = None
    cost_deployments: tuple[DeploymentPricing, ...] | None = None


@dataclass(frozen=True, slots=True)
class UnpricedModelDecision:
    kind: Literal["Complete", "ReadGroup", "ReadDeployments", "ReadInfo", "ReadCostDeployments", "ReadCost"]
    unpriced: bool = False
    index: int = 0


_DECISION: Final = TypeAdapter(UnpricedModelDecision)


def unpriced_model(facts: UnpricedModelFacts) -> UnpricedModelDecision:
    return _DECISION.validate_python(_native.request_auth_unpriced_model(asdict(facts)))
