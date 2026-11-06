from dataclasses import dataclass, replace
from typing import Final

import pytest

from litellm._auth.decision import Allow, Deny, ModelBudgetExceeded, ModelBudgetScope, Outcome, Step, WithinModelBudgets
from litellm._auth.driver import drive_model_budgets
from litellm._auth.python_core import PythonCore

Call = tuple[str, str | None]

SCOPE: Final = ModelBudgetScope(
    models=("gpt-4o",),
    rewritable_model="gpt-4o",
    key_has_model_budgets=True,
    key_budget_fallbacks={"gpt-4o": ("gpt-4o-mini", "haiku")},
    user_has_model_budgets=True,
)
ALLOWED: Final = Allow(WithinModelBudgets())


def _exceeded(entity_type: str, model: str) -> ModelBudgetExceeded:
    return ModelBudgetExceeded(
        message=f"{entity_type} over budget for {model}",
        current_cost=2.0,
        max_budget=1.0,
        entity_type=entity_type,
        entity_id="id",
    )


@dataclass
class FakeRequest:
    scope: ModelBudgetScope
    key_over: frozenset[str] = frozenset()
    user_over: frozenset[str] = frozenset()
    unauthorized: frozenset[str] = frozenset()
    calls: tuple[Call, ...] = ()

    def _record(self, name: str, model: str | None = None) -> None:
        self.calls = (*self.calls, (name, model))

    async def read_model_budget_scope(self) -> ModelBudgetScope:
        self._record("read")
        return self.scope

    async def check_key_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        self._record("key", model)
        return _exceeded("key", model) if model in self.key_over else None

    async def authorize_budget_fallback(self, model: str) -> bool:
        self._record("authorize", model)
        return model not in self.unauthorized

    async def rewrite_request_model(self, model: str) -> None:
        self._record("rewrite", model)
        self.scope = replace(
            self.scope,
            models=tuple(model if m == self.scope.rewritable_model else m for m in self.scope.models),
            rewritable_model=model,
        )

    async def check_user_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        self._record("user", model)
        return _exceeded("user", model) if model in self.user_over else None


async def _run(step: Step, request: FakeRequest) -> tuple[Outcome, tuple[Call, ...]]:
    outcome: Final = await drive_model_budgets(step, request)
    return outcome, request.calls


@pytest.mark.asyncio
async def test_within_budget_checks_the_key_then_the_user_without_rerouting():
    assert await _run(PythonCore().check_model_budgets(), FakeRequest(SCOPE)) == (
        ALLOWED,
        (("read", None), ("key", "gpt-4o"), ("read", None), ("user", "gpt-4o")),
    )


@pytest.mark.asyncio
async def test_over_budget_reroutes_to_the_first_fallback_within_budget_and_checks_the_user_on_it():
    request: Final = FakeRequest(SCOPE, key_over=frozenset({"gpt-4o", "gpt-4o-mini"}))
    assert await _run(PythonCore().check_model_budgets(), request) == (
        ALLOWED,
        (
            ("read", None),
            ("key", "gpt-4o"),
            ("key", "gpt-4o-mini"),
            ("key", "haiku"),
            ("authorize", "haiku"),
            ("rewrite", "haiku"),
            ("read", None),
            ("user", "haiku"),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "key_over", "unauthorized"),
    [
        (replace(SCOPE, rewritable_model=None), {"gpt-4o"}, set()),
        (replace(SCOPE, rewritable_model="other"), {"gpt-4o"}, set()),
        (replace(SCOPE, key_budget_fallbacks={}), {"gpt-4o"}, set()),
        (SCOPE, {"gpt-4o", "gpt-4o-mini", "haiku"}, set()),
        (SCOPE, {"gpt-4o"}, {"gpt-4o-mini"}),
    ],
    ids=["nested_model", "different_top_level_model", "no_fallbacks", "every_fallback_over", "fallback_unauthorized"],
)
async def test_an_unreroutable_over_budget_model_is_denied_with_its_own_budget_error(
    scope: ModelBudgetScope, key_over: set[str], unauthorized: set[str]
):
    request: Final = FakeRequest(scope, key_over=frozenset(key_over), unauthorized=frozenset(unauthorized))
    outcome, calls = await _run(PythonCore().check_model_budgets(), request)
    assert outcome == Deny(_exceeded("key", "gpt-4o"))
    assert not any(name in ("rewrite", "user") for name, _ in calls)


@pytest.mark.asyncio
async def test_a_later_model_is_checked_after_an_earlier_one_was_rerouted():
    scope: Final = replace(SCOPE, models=("gpt-4o", "embed"))
    request: Final = FakeRequest(scope, key_over=frozenset({"gpt-4o", "embed"}))
    outcome, calls = await _run(PythonCore().check_model_budgets(), request)
    assert outcome == Deny(_exceeded("key", "embed"))
    assert calls[-2:] == (("rewrite", "gpt-4o-mini"), ("key", "embed"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "expected_calls"),
    [
        (
            replace(SCOPE, key_has_model_budgets=False),
            (("read", None), ("user", "gpt-4o")),
        ),
        (
            replace(SCOPE, user_has_model_budgets=False),
            (("read", None), ("key", "gpt-4o"), ("read", None)),
        ),
        (replace(SCOPE, models=()), (("read", None),)),
    ],
    ids=["no_key_budgets", "no_user_budgets", "no_models"],
)
async def test_only_budgets_that_exist_are_checked(scope: ModelBudgetScope, expected_calls: tuple[Call, ...]):
    assert await _run(PythonCore().check_model_budgets(), FakeRequest(scope)) == (ALLOWED, expected_calls)


@pytest.mark.asyncio
async def test_a_user_over_budget_is_denied_on_every_path():
    over: Final = frozenset({"gpt-4o"})
    for step in (PythonCore().check_model_budgets(), PythonCore().check_user_model_budgets()):
        outcome, _ = await _run(step, FakeRequest(SCOPE, user_over=over))
        assert outcome == Deny(_exceeded("user", "gpt-4o"))


@pytest.mark.asyncio
async def test_the_user_only_check_never_reads_key_budgets():
    request: Final = FakeRequest(SCOPE, key_over=frozenset({"gpt-4o"}))
    assert await _run(PythonCore().check_user_model_budgets(), request) == (
        ALLOWED,
        (("read", None), ("user", "gpt-4o")),
    )
