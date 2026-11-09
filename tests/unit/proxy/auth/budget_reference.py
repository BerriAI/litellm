from collections.abc import Mapping
from typing import Final

import litellm
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
    entry_names_model,
    resolve_model_budget,
)


async def model_allowed(
    limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter,
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    team_id: str | None,
    budgets: Mapping[str, object] | None,
    model: str,
    message: str,
) -> bool:
    if not budgets:
        return True
    resolved: Final = resolve_model_budget(model, budgets, limiter._model_access_groups, team_id)
    if resolved is None:
        return True
    limit: Final = resolved.budget_config.max_budget
    if limit is None or limit < 0:
        return True
    local: Final = await limiter._get_spend_for_model_budget(entity_type, entity_id, model, resolved)
    current: Final = (
        local if local >= limit else await limiter._with_shared_spend(entity_type, entity_id, resolved, local)
    )
    if current >= limit:
        raise litellm.BudgetExceededError(
            message=message
            if entry_names_model(resolved.budget_model, model)
            else f"{message}, budget_model={resolved.budget_model}",
            current_cost=current,
            max_budget=limit,
            entity_type=entity_type.value,
            entity_id=entity_id,
        )
    return True
