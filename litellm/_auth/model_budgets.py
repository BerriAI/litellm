from collections.abc import Callable
from typing import Final

from litellm._auth.decision import (
    AuthorizeBudgetFallback,
    CheckKeyModelBudget,
    CheckUserModelBudget,
    Deny,
    ModelBudgetExceeded,
    ModelBudgetScope,
    ReadModelBudgetScope,
    RewriteRequestModel,
    Step,
)


def check_model_budgets(then: Callable[[], Step]) -> Step:
    return ReadModelBudgetScope(resume=lambda scope: _check_key_budgets(scope, then))


def check_user_model_budgets(then: Callable[[], Step]) -> Step:
    return ReadModelBudgetScope(resume=lambda scope: _check_user_budgets(scope, scope.models, then))


def _check_key_budgets(scope: ModelBudgetScope, then: Callable[[], Step]) -> Step:
    if not scope.key_has_model_budgets or not scope.models:
        return _check_user_budgets(scope, scope.models, then)
    return _check_key_budget_of(
        scope,
        scope.models,
        scope.rewritable_model,
        then=lambda: ReadModelBudgetScope(resume=lambda rescoped: _check_user_budgets(rescoped, rescoped.models, then)),
    )


def _check_key_budget_of(
    scope: ModelBudgetScope, models: tuple[str, ...], rewritable_model: str | None, then: Callable[[], Step]
) -> Step:
    if not models:
        return then()
    model: Final = models[0]
    rest: Final = models[1:]

    def after(exceeded: ModelBudgetExceeded | None) -> Step:
        if exceeded is None:
            return _check_key_budget_of(scope, rest, rewritable_model, then)
        if model != rewritable_model:
            return Deny(exceeded)
        return _first_fallback_within_budget(
            scope.key_budget_fallbacks.get(model, ()),
            found=lambda fallback: _reroute(fallback, exceeded, then=lambda: _check_key_budget_of(scope, rest, fallback, then)),
            exhausted=lambda: Deny(exceeded),
        )

    return CheckKeyModelBudget(model=model, resume=after)


def _first_fallback_within_budget(
    candidates: tuple[str, ...], found: Callable[[str], Step], exhausted: Callable[[], Step]
) -> Step:
    if not candidates:
        return exhausted()
    candidate: Final = candidates[0]
    return CheckKeyModelBudget(
        model=candidate,
        resume=lambda exceeded: (
            found(candidate)
            if exceeded is None
            else _first_fallback_within_budget(candidates[1:], found, exhausted)
        ),
    )


def _reroute(fallback: str, exceeded: ModelBudgetExceeded, then: Callable[[], Step]) -> Step:
    return AuthorizeBudgetFallback(
        model=fallback,
        resume=lambda authorized: RewriteRequestModel(model=fallback, resume=then) if authorized else Deny(exceeded),
    )


def _check_user_budgets(scope: ModelBudgetScope, models: tuple[str, ...], then: Callable[[], Step]) -> Step:
    if not scope.user_has_model_budgets or not models:
        return then()
    return CheckUserModelBudget(
        model=models[0],
        resume=lambda exceeded: _check_user_budgets(scope, models[1:], then) if exceeded is None else Deny(exceeded),
    )
