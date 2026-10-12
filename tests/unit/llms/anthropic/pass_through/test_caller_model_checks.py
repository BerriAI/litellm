from collections.abc import Mapping, Sequence
from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.pass_through.caller_model_checks import (
    _caller_within_team_member_model_budget,
    caller_model_verdict,
    rate_limit_verdict,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RateLimitDescriptor,
    RateLimitResponse,
    RateLimitUnverifiableError,
)

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

    verdict: Final = await _caller_within_team_member_model_budget(
        user_api_key_auth=_member_auth(), model_max_budget_limiter=limiter, model=CLASSIFIER
    )

    assert verdict == "denied"
    assert [call["model"] for call in limiter.calls] == [CLASSIFIER]
    assert (limiter.calls[0]["user_id"], limiter.calls[0]["team_id"]) == ("member-1", "team-1")
    assert limiter.calls[0]["team_member_model_max_budget"] == _member_auth().team_member_model_max_budget


@pytest.mark.asyncio
async def test_member_within_their_classifier_budget_is_allowed():
    limiter: Final = _RecordingLimiter()

    verdict: Final = await _caller_within_team_member_model_budget(
        user_api_key_auth=_member_auth(), model_max_budget_limiter=limiter, model=CLASSIFIER
    )

    assert verdict == "allowed"
    assert len(limiter.calls) == 1


@pytest.mark.asyncio
async def test_a_limiter_failure_denies_for_now_like_the_other_budget_scopes():
    limiter: Final = _RecordingLimiter(RuntimeError("cache unreachable"))

    verdict: Final = await _caller_within_team_member_model_budget(
        user_api_key_auth=_member_auth(), model_max_budget_limiter=limiter, model=CLASSIFIER
    )

    assert verdict == "denied_for_now"


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

    verdict: Final = await _caller_within_team_member_model_budget(
        user_api_key_auth=auth, model_max_budget_limiter=limiter, model=CLASSIFIER
    )

    assert verdict == "allowed"
    assert limiter.calls == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_models, verdict",
    [(["other-model"], "denied"), ([CLASSIFIER], "allowed")],
    ids=["classifier-not-on-the-key", "classifier-on-the-key"],
)
async def test_the_key_allowlist_decides_access_to_the_classifier(key_models: list[str], verdict: str) -> None:
    assert await caller_model_verdict(UserAPIKeyAuth(models=key_models), CLASSIFIER, llm_router=None) == verdict


class _RecordingRateLimiter:
    def __init__(self, overall_code: str = "OK", failure: Exception | None = None) -> None:
        self.overall_code = overall_code
        self.failure = failure
        self.read_only: bool | None = None

    def create_rate_limit_descriptors(
        self,
        *,
        user_api_key_dict: UserAPIKeyAuth,
        data: Mapping[str, str],
        rpm_limit_type: object,
        tpm_limit_type: object,
        model_has_failures: bool,
    ) -> list[RateLimitDescriptor]:
        return [{"key": "api_key", "value": "k", "rate_limit": {"requests_per_unit": 1, "window_size": 60}}]

    def _add_team_model_rate_limit_descriptor_from_metadata(
        self, *, user_api_key_dict: UserAPIKeyAuth, requested_model: str, descriptors: Sequence[RateLimitDescriptor]
    ) -> None:
        return None

    def _add_project_model_rate_limit_descriptor_from_metadata(
        self, *, user_api_key_dict: UserAPIKeyAuth, requested_model: str, descriptors: Sequence[RateLimitDescriptor]
    ) -> None:
        return None

    def create_organization_rate_limit_descriptor(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None = None
    ) -> list[RateLimitDescriptor]:
        return []

    async def should_rate_limit(
        self, *, descriptors: Sequence[RateLimitDescriptor], parent_otel_span: object, read_only: bool
    ) -> RateLimitResponse:
        self.read_only = read_only
        if self.failure is not None:
            raise self.failure
        return {"overall_code": self.overall_code, "statuses": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limiter, verdict",
    [
        (_RecordingRateLimiter("OK"), "allowed"),
        (_RecordingRateLimiter("OVER_LIMIT"), "denied_for_now"),
        (_RecordingRateLimiter(failure=RateLimitUnverifiableError()), "denied_for_now"),
        (_RecordingRateLimiter(failure=RuntimeError("redis down")), "allowed"),
    ],
    ids=["under-limit", "over-limit", "unverifiable", "limiter-crash"],
)
async def test_a_full_rate_limit_window_denies_for_now(limiter: _RecordingRateLimiter, verdict: str) -> None:
    result: Final = await rate_limit_verdict(
        user_api_key_auth=UserAPIKeyAuth(api_key="k"),
        model=CLASSIFIER,
        limiter=limiter,
        unverifiable=RateLimitUnverifiableError,
    )

    assert result == verdict
    assert limiter.read_only is True
