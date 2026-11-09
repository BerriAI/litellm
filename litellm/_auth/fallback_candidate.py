from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.budget_pricing import ModelPriceFacts
from litellm._auth.common_budgets import BudgetFacts
from litellm._auth.failure import native_boundary
from litellm._auth.fallback_access import FallbackAuthorizationFacts
from litellm._auth.model_access import ModelAccessFacts
from litellm._auth.model_attribution import ModelGroupAttributionFacts
from litellm._auth.model_budget_data import ModelBudgetEntity, ModelBudgetSnapshot
from litellm._auth.routes import RouteContext
from litellm._auth.unpriced_models import UnpricedModelFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class CandidateBudgetScope:
    entity: ModelBudgetEntity
    identity_present: bool
    configured: bool


@dataclass(frozen=True, slots=True)
class CandidateBudgetRead:
    model_index: int
    entity: ModelBudgetEntity
    snapshot: ModelBudgetSnapshot


@dataclass(frozen=True, slots=True)
class FallbackCandidateFacts:
    router_present: bool
    user_present: bool
    team_present: bool
    membership: Literal["Unread", "Loaded", "Failed"] = "Unread"
    block_unpriced: bool = False
    route: RouteContext | None = None
    unpriced: UnpricedModelFacts | None = None
    customer_present: bool = False
    customer_grants: tuple[str, ...] | None = None
    customer: ModelAccessFacts | None = None
    price: ModelPriceFacts | None = None
    budget_models: tuple[str, ...] | None = None
    budget_scopes: tuple[CandidateBudgetScope, ...] | None = None
    budgets: tuple[CandidateBudgetRead, ...] = ()
    authorization: FallbackAuthorizationFacts | None = None
    attribution_reset: bool = False
    attribution: ModelGroupAttributionFacts | None = None
    attribution_stamped: bool = False
    group_budgets_read: bool = False
    group_budgets: tuple[BudgetFacts, ...] = ()
    io_failed: bool = False


@dataclass(frozen=True, slots=True)
class FallbackCandidateDecision:
    kind: Literal[
        "Complete",
        "ReadMembership",
        "ReadRoute",
        "ReadUnpriced",
        "ReadCustomerGrants",
        "ReadCustomer",
        "ReadPrice",
        "ReadBudgetModels",
        "ReadBudgetScopes",
        "ReadBudget",
        "ReadAuthorization",
        "ResetAttribution",
        "ReadAttribution",
        "StampAttribution",
        "ReadGroupBudgets",
    ]
    usable: bool = False
    log_failure: bool = False
    model_index: int = 0
    entity: ModelBudgetEntity = "Key"
    groups: tuple[str, ...] = ()


_DECISION: Final = TypeAdapter(FallbackCandidateDecision)


@native_boundary
def fallback_candidate(facts: FallbackCandidateFacts) -> FallbackCandidateDecision:
    return _DECISION.validate_python(_native.request_auth_fallback_candidate(asdict(facts)))
