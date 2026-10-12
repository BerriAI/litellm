from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.pass_through.caller_model_checks import _caller_within_team_member_model_budget
from litellm.proxy._types import UserAPIKeyAuth

CLASSIFIER: Final = "classifier"
CLASSIFIER_BUDGET: Final = {CLASSIFIER: {"max_budget": 0.1, "budget_duration": "1d"}}


class _RecordingLimiter:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: tuple[Mapping[str, object], ...] = ()

    async def is_team_member_within_model_budget(
        self, user_id: str, team_id: str, team_member_model_max_budget: Mapping[str, object], model: str
    ) -> bool:
        self.calls = (
            *self.calls,
            {
                "user_id": user_id,
                "team_id": team_id,
                "team_member_model_max_budget": team_member_model_max_budget,
                "model": model,
            },
        )
        if self.failure is not None:
            raise self.failure
        return True


def _member_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="member-1", team_id="team-1", team_member_model_max_budget=CLASSIFIER_BUDGET)


@pytest.mark.asyncio
async def test_member_over_their_classifier_budget_is_denied():
    limiter: Final = _RecordingLimiter(litellm.BudgetExceededError(current_cost=0.2, max_budget=0.1))

    allowed: Final = await _caller_within_team_member_model_budget(
        user_api_key_auth=_member_auth(), model_max_budget_limiter=limiter, model=CLASSIFIER
    )

    assert allowed is False
    assert [call["model"] for call in limiter.calls] == [CLASSIFIER]
    assert (limiter.calls[0]["user_id"], limiter.calls[0]["team_id"]) == ("member-1", "team-1")
    assert limiter.calls[0]["team_member_model_max_budget"] == _member_auth().team_member_model_max_budget


@pytest.mark.asyncio
async def test_member_within_their_classifier_budget_is_allowed():
    limiter: Final = _RecordingLimiter()

    allowed: Final = await _caller_within_team_member_model_budget(
        user_api_key_auth=_member_auth(), model_max_budget_limiter=limiter, model=CLASSIFIER
    )

    assert allowed is True
    assert len(limiter.calls) == 1


@pytest.mark.asyncio
async def test_a_limiter_failure_denies_like_the_other_budget_scopes():
    limiter: Final = _RecordingLimiter(RuntimeError("cache unreachable"))

    allowed: Final = await _caller_within_team_member_model_budget(
        user_api_key_auth=_member_auth(), model_max_budget_limiter=limiter, model=CLASSIFIER
    )

    assert allowed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth",
    [
        UserAPIKeyAuth(user_id="member-1", team_id="team-1"),
        UserAPIKeyAuth(user_id="member-1", team_id="team-1", team_member_model_max_budget={}),
        UserAPIKeyAuth(team_id="team-1", team_member_model_max_budget=CLASSIFIER_BUDGET),
        UserAPIKeyAuth(user_id="member-1", team_member_model_max_budget=CLASSIFIER_BUDGET),
    ],
    ids=["no-member-budget", "empty-member-budget", "no-user", "no-team"],
)
async def test_callers_without_a_member_budget_skip_the_limiter(auth: UserAPIKeyAuth):
    limiter: Final = _RecordingLimiter(litellm.BudgetExceededError(current_cost=0.2, max_budget=0.1))

    allowed: Final = await _caller_within_team_member_model_budget(
        user_api_key_auth=auth, model_max_budget_limiter=limiter, model=CLASSIFIER
    )

    assert allowed is True
    assert limiter.calls == ()
