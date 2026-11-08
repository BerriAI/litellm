from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class DeploymentPrice:
    has_input_price: bool = False
    has_output_price: bool = False
    input_price: float | None = None
    output_price: float | None = None
    query_price: float | None = None
    ptu_count: float | None = None
    cost_per_ptu_per_hour: float | None = None


@dataclass(frozen=True, slots=True)
class ModelPriceFacts:
    cached: bool | None = None
    lookup_failed: bool = False
    group_found: bool = False
    input_price: float | None = None
    output_price: float | None = None
    deployments: tuple[DeploymentPrice, ...] | None = None


@dataclass(frozen=True, slots=True)
class PriceDecision:
    zero_cost: bool
    cache_value: bool | None
    read_deployments: bool = False


@dataclass(frozen=True, slots=True)
class RouteBudgetFacts:
    side_effect_route: bool = False
    discovery_route: bool = False
    llm_route: bool = False
    registered_pass_through: bool | None = None
    route: str = ""
    method: str | None = None
    body_present: bool = False


RouteBudgetDecision: TypeAlias = Literal["ReadPassThrough", "ReadPricing", "Skip"]
_PRICE: Final = TypeAdapter(ModelPriceFacts)
_PRICE_DECISION: Final = TypeAdapter(PriceDecision)
_ROUTE: Final = TypeAdapter(RouteBudgetFacts)
_ROUTE_DECISION: Final = TypeAdapter[RouteBudgetDecision](RouteBudgetDecision)


def price(facts: ModelPriceFacts) -> PriceDecision:
    return _PRICE_DECISION.validate_python(_native.request_auth_budget_price(_PRICE.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary


def route(facts: RouteBudgetFacts) -> RouteBudgetDecision:
    return _ROUTE_DECISION.validate_python(_native.request_auth_budget_route(_ROUTE.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary
