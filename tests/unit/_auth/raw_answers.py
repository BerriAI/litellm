from litellm._auth.decision import ModelBudgetExceeded, ModelBudgetRead
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
