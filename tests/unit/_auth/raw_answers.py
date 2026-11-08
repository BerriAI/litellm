from litellm._auth.decision import ModelBudgetExceeded, ModelBudgetRead
from litellm._auth.budget_pricing import ModelPriceFacts
from litellm._auth.fallback_candidate import FallbackCandidateFacts
from litellm._auth.model_attribution import ModelGroupAttributionFacts
from litellm._auth.fallback_access import FallbackAuthorizationFacts, GroupedModelScope
from litellm._auth.model_access import ModelAccessFacts
from litellm._auth.model_budget_data import ModelBudgetSnapshot


def fallback_read(model: str, authorized: bool = True) -> FallbackAuthorizationFacts:
    return FallbackAuthorizationFacts(
        key=GroupedModelScope(
            ModelAccessFacts((model,), (model,) if authorized else ("not-allowed",), object_type="key")
        )
    )


def budget_read(exceeded: ModelBudgetExceeded | None = None) -> ModelBudgetRead:
    if exceeded is None:
        return ModelBudgetRead(ModelBudgetSnapshot())
    return ModelBudgetRead(
        ModelBudgetSnapshot(max_budget=exceeded.max_budget, local_spend=exceeded.current_cost),
        exceeded.message,
        exceeded.entity_type,
        exceeded.entity_id,
    )


def candidate_read(model: str, usable: bool = True) -> FallbackCandidateFacts:
    return FallbackCandidateFacts(
        router_present=True,
        user_present=True,
        team_present=True,
        membership="Loaded",
        price=ModelPriceFacts(cached=True),
        authorization=fallback_read(model, authorized=usable),
        attribution_reset=True,
        attribution=ModelGroupAttributionFacts(),
        attribution_stamped=True,
    )
