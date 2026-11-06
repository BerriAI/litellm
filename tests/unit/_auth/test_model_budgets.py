from dataclasses import dataclass, replace
from typing import Final

import pytest

from litellm._auth.decision import (
    Allow,
    Deny,
    FallbackTarget,
    ModelBudgetExceeded,
    ModelBudgetScope,
    Outcome,
    Step,
    TeamMemberModelBudgetScope,
    WithinModelBudgets,
)
from litellm._auth.driver import drive_model_budgets, drive_team_member_model_budgets
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


MEMBER_SCOPE: Final = TeamMemberModelBudgetScope(
    models=("gpt-4o",), rewritable_model="gpt-4o", has_model_budgets=True, can_fall_back=True
)


def _target(model: str) -> FallbackTarget:
    return FallbackTarget(requested=model, model=model, served_model=model)


@dataclass
class FakeTeamMemberRequest:
    scope: TeamMemberModelBudgetScope
    over: frozenset[str] = frozenset()
    fallbacks: tuple[str, ...] = ("gpt-4o", "unusable", "gpt-4o-mini", "haiku")
    unusable: frozenset[str] = frozenset({"unusable"})
    calls: tuple[Call, ...] = ()

    def _record(self, name: str, model: str | None = None) -> None:
        self.calls = (*self.calls, (name, model))

    async def read_team_member_model_budget_scope(self) -> TeamMemberModelBudgetScope:
        self._record("read")
        return self.scope

    async def check_team_member_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        self._record("member", model)
        return _exceeded("member", model) if model in self.over else None

    async def read_team_member_fallbacks(self, model: str) -> tuple[FallbackTarget, ...]:
        self._record("fallbacks", model)
        return tuple(_target(fallback) for fallback in self.fallbacks)

    async def vet_team_member_fallback(self, target: FallbackTarget) -> bool:
        self._record("vet", target.model)
        return target.model not in self.unusable

    async def reroute_team_member_request(self, model: str) -> None:
        self._record("reroute", model)


async def _run_member(request: FakeTeamMemberRequest) -> tuple[Outcome, tuple[Call, ...]]:
    outcome: Final = await drive_team_member_model_budgets(PythonCore().check_team_member_model_budgets(), request)
    return outcome, request.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "expected_calls"),
    [
        (MEMBER_SCOPE, (("read", None), ("member", "gpt-4o"))),
        (replace(MEMBER_SCOPE, has_model_budgets=False), (("read", None),)),
        (replace(MEMBER_SCOPE, models=()), (("read", None),)),
    ],
    ids=["within_budget", "no_member_budgets", "no_models"],
)
async def test_a_team_member_within_budget_is_allowed(
    scope: TeamMemberModelBudgetScope, expected_calls: tuple[Call, ...]
):
    assert await _run_member(FakeTeamMemberRequest(scope)) == (ALLOWED, expected_calls)


@pytest.mark.asyncio
async def test_an_over_budget_team_member_reroutes_to_the_first_usable_fallback_other_than_itself():
    request: Final = FakeTeamMemberRequest(replace(MEMBER_SCOPE, models=("gpt-4o", "embed")), over=frozenset({"gpt-4o"}))
    assert await _run_member(request) == (
        ALLOWED,
        (
            ("read", None),
            ("member", "gpt-4o"),
            ("fallbacks", "gpt-4o"),
            ("vet", "unusable"),
            ("vet", "gpt-4o-mini"),
            ("reroute", "gpt-4o-mini"),
            ("member", "embed"),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "unusable"),
    [
        (replace(MEMBER_SCOPE, rewritable_model=None), frozenset()),
        (replace(MEMBER_SCOPE, can_fall_back=False), frozenset()),
        (MEMBER_SCOPE, frozenset({"unusable", "gpt-4o-mini", "haiku"})),
    ],
    ids=["nested_model", "fallbacks_unavailable", "no_usable_fallback"],
)
async def test_an_over_budget_team_member_without_a_usable_fallback_is_denied_with_its_own_error(
    scope: TeamMemberModelBudgetScope, unusable: frozenset[str]
):
    outcome, calls = await _run_member(FakeTeamMemberRequest(scope, over=frozenset({"gpt-4o"}), unusable=unusable))
    assert outcome == Deny(_exceeded("member", "gpt-4o"))
    assert not any(name == "reroute" for name, _ in calls)


@pytest.mark.asyncio
async def test_a_later_model_is_not_rerouted_once_the_request_model_changed():
    scope: Final = replace(MEMBER_SCOPE, models=("gpt-4o", "gpt-4o"))
    request: Final = FakeTeamMemberRequest(scope, over=frozenset({"gpt-4o"}))
    outcome, calls = await _run_member(request)
    assert outcome == Deny(_exceeded("member", "gpt-4o"))
    assert [name for name, _ in calls].count("reroute") == 1
