from collections.abc import Mapping
from typing import Final

from litellm.proxy.auth.core.answers import (
    BudgetExemption,
    FallbackTarget,
    ModelBudgetScope,
    TeamMemberModelBudgetScope,
)
from litellm.proxy.auth.core.host import ModelBudgetHost, TeamMemberModelBudgetHost
from litellm.proxy.auth.core.outcome import (
    BudgetsSkipped,
    ModelBudgetExceeded,
    ModelBudgetResult,
    WithinModelBudgets,
)


def skips_model_budgets(exemption: BudgetExemption) -> bool:
    return exemption.route_skips_budgets or exemption.mcp_discovery or exemption.model_is_free


def skip_model_budgets() -> BudgetsSkipped:
    return BudgetsSkipped()


def within_model_budgets(exceeded: ModelBudgetExceeded | None) -> ModelBudgetResult:
    return WithinModelBudgets() if exceeded is None else exceeded


async def check_key_and_user_model_budgets(host: ModelBudgetHost) -> ModelBudgetExceeded | None:
    scope: Final = await host.read_model_budget_scope()
    if scope.key is None or not scope.models:
        return await _check_user_budgets(host, scope)
    exceeded: Final = await _check_key_budgets(host, scope.key.fallbacks, scope.models, scope.rewritable_model)
    if exceeded is not None:
        return exceeded
    return await _check_user_budgets(host, await host.read_model_budget_scope())


async def check_user_model_budgets(host: ModelBudgetHost) -> ModelBudgetExceeded | None:
    return await _check_user_budgets(host, await host.read_model_budget_scope())


async def check_team_member_model_budgets(host: TeamMemberModelBudgetHost) -> ModelBudgetExceeded | None:
    scope: Final = await host.read_team_member_model_budget_scope()
    if not scope.has_model_budgets:
        return None
    return await _check_team_member_budgets(host, scope)


async def _check_key_budgets(
    host: ModelBudgetHost,
    fallbacks: Mapping[str, tuple[str, ...]],
    models: tuple[str, ...],
    rewritable_model: str | None,
) -> ModelBudgetExceeded | None:
    request_model: str | None = rewritable_model
    for model in models:
        if (exceeded := await host.check_key_model_budget(model)) is None:
            continue
        if (
            model != request_model
            or (fallback := await _authorized_key_fallback(host, fallbacks.get(model, ()))) is None
        ):
            return exceeded
        await host.rewrite_request_model(fallback)
        request_model = fallback
    return None


async def _authorized_key_fallback(host: ModelBudgetHost, candidates: tuple[str, ...]) -> str | None:
    fallback: Final = await _first_key_fallback_within_budget(host, candidates)
    if fallback is None or not await host.authorize_budget_fallback(fallback):
        return None
    return fallback


async def _first_key_fallback_within_budget(host: ModelBudgetHost, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if await host.check_key_model_budget(candidate) is None:
            return candidate
    return None


async def _check_user_budgets(host: ModelBudgetHost, scope: ModelBudgetScope) -> ModelBudgetExceeded | None:
    if not scope.user_has_model_budgets:
        return None
    for model in scope.models:
        if (exceeded := await host.check_user_model_budget(model)) is not None:
            return exceeded
    return None


async def _check_team_member_budgets(
    host: TeamMemberModelBudgetHost, scope: TeamMemberModelBudgetScope
) -> ModelBudgetExceeded | None:
    request_model: str | None = scope.rewritable_model
    for model in scope.models:
        if (exceeded := await host.check_team_member_model_budget(model)) is None:
            continue
        if (
            model != request_model
            or not scope.can_fall_back
            or (target := await _usable_team_member_fallback(host, model)) is None
        ):
            return exceeded
        await host.reroute_team_member_request(target.model)
        request_model = target.model
    return None


async def _usable_team_member_fallback(host: TeamMemberModelBudgetHost, model: str) -> FallbackTarget | None:
    targets: Final = await host.read_team_member_fallbacks(model)
    return await _first_usable_fallback(host, tuple(target for target in targets if target.model != model))


async def _first_usable_fallback(
    host: TeamMemberModelBudgetHost, targets: tuple[FallbackTarget, ...]
) -> FallbackTarget | None:
    for target in targets:
        if await host.vet_team_member_fallback(target):
            return target
    return None
