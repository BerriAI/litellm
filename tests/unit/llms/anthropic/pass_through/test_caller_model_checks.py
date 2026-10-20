from collections.abc import Mapping, Sequence
from typing import Final

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.llms.anthropic.pass_through.caller_model_checks import (
    _caller_within_team_member_model_budget,
    caller_model_verdict,
    rate_limit_verdict,
)
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import run_centralized_common_checks
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitResponse,
    RateLimitUnverifiableError,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.router import Router
from litellm.types.agents import AgentResponse
from tests.unit.llms.anthropic.pass_through.conftest import ProxyAuthState

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


_ProxyRow = (
    LiteLLM_TeamTableCachedObj
    | LiteLLM_UserTable
    | LiteLLM_EndUserTable
    | LiteLLM_ProjectTableCachedObj
    | LiteLLM_TeamMembership
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller, rows, verdict",
    [
        (
            UserAPIKeyAuth(api_key="sk-personal", user_id="u-1"),
            (LiteLLM_UserTable(user_id="u-1", models=["other-model"]),),
            "denied",
        ),
        (
            UserAPIKeyAuth(api_key="sk-team", user_id="u-1", team_id="t-1"),
            (
                LiteLLM_TeamTableCachedObj(team_id="t-1", models=[CLASSIFIER]),
                LiteLLM_UserTable(user_id="u-1", models=["other-model"]),
                LiteLLM_TeamMembership(user_id="u-1", team_id="t-1"),
            ),
            "allowed",
        ),
        (
            UserAPIKeyAuth(api_key="sk-team", team_id="t-1"),
            (LiteLLM_TeamTableCachedObj(team_id="t-1", models=["other-model"]),),
            "denied",
        ),
        (
            UserAPIKeyAuth(api_key="sk-team", user_id="u-1", team_id="t-1"),
            (
                LiteLLM_TeamTableCachedObj(team_id="t-1", models=[CLASSIFIER]),
                LiteLLM_TeamMembership(
                    user_id="u-1",
                    team_id="t-1",
                    litellm_budget_table=LiteLLM_BudgetTable(allowed_models=["other-model"]),
                ),
            ),
            "denied",
        ),
        (
            UserAPIKeyAuth(api_key="sk-project", project_id="p-1"),
            (LiteLLM_ProjectTableCachedObj(project_id="p-1", models=["other-model"]),),
            "denied",
        ),
        (
            UserAPIKeyAuth(api_key="sk-app", end_user_id="eu-1"),
            (LiteLLM_EndUserTable(user_id="eu-1", blocked=False, models=["other-model"]),),
            "denied",
        ),
        (
            UserAPIKeyAuth(api_key="sk-team", user_id="u-1", team_id="t-1"),
            (LiteLLM_TeamTableCachedObj(team_id="t-1", models=[CLASSIFIER]),),
            "denied_for_now",
        ),
    ],
    ids=[
        "personal-key-user-scope-excludes",
        "team-key-team-grants-though-the-users-personal-scope-does-not",
        "team-scope-excludes",
        "team-member-scope-excludes",
        "project-scope-excludes",
        "end-user-scope-excludes",
        "team-membership-unreadable",
    ],
)
async def test_the_proxys_own_access_checks_decide_the_classifier_for_every_scope(
    proxy_auth_state: ProxyAuthState, caller: UserAPIKeyAuth, rows: Sequence[_ProxyRow], verdict: str
) -> None:
    proxy_auth_state.seed(*rows)

    assert await caller_model_verdict(caller, CLASSIFIER, llm_router=None) == verdict


def _managed_agent(object_permission: Mapping[str, object] | None) -> UserAPIKeyAuth:
    agent_key: Final = UserAPIKeyAuth(api_key="sk-agent", agent_id="agent-1")
    agent_key.managed_agent_policy = AgentResponse(
        agent_id="agent-1",
        agent_name="coder",
        agent_card_params={},
        object_permission=dict(object_permission) if object_permission is not None else None,
    )
    return agent_key


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "object_permission, verdict",
    [(None, "denied"), ({"models": ["other-model"]}, "denied"), ({"models": [CLASSIFIER]}, "allowed")],
    ids=["no-model-grants", "grant-for-another-model", "grant-for-the-classifier"],
)
async def test_a_managed_agent_reaches_the_classifier_only_through_a_grant_for_it(
    proxy_auth_state: ProxyAuthState, object_permission: Mapping[str, object] | None, verdict: str
) -> None:
    assert await caller_model_verdict(_managed_agent(object_permission), CLASSIFIER, llm_router=None) == verdict


def _limiter() -> PROXY_MaxParallelRequestsHandler_v3:
    return PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(dual_cache=DualCache()),
        model_group_resolver=lambda model: None,
        fail_closed_resolver=lambda: False,
    )


async def _admit(limiter: PROXY_MaxParallelRequestsHandler_v3, caller: UserAPIKeyAuth, model: str) -> None:
    await limiter.async_pre_call_hook(
        user_api_key_dict=caller, cache=DualCache(), data={"model": model}, call_type="acompletion"
    )


@pytest.mark.asyncio
async def test_the_parent_turn_holding_the_keys_only_rpm_slot_leaves_the_classifier_allowed() -> None:
    limiter: Final = _limiter()
    caller: Final = UserAPIKeyAuth(api_key=hash_token("sk-one-rpm"), rpm_limit=1)
    await _admit(limiter, caller, "kimi-k3")
    with pytest.raises(HTTPException):
        await _admit(limiter, caller, "kimi-k3")

    verdict: Final = await rate_limit_verdict(
        user_api_key_auth=caller, model=CLASSIFIER, limiter=limiter, unverifiable=RateLimitUnverifiableError
    )

    assert verdict == "allowed"


@pytest.mark.asyncio
async def test_a_full_per_model_rpm_limit_on_the_classifier_still_denies_for_now() -> None:
    limiter: Final = _limiter()
    caller: Final = UserAPIKeyAuth(api_key=hash_token("sk-model-rpm"), metadata={"model_rpm_limit": {CLASSIFIER: 1}})
    before: Final = await rate_limit_verdict(
        user_api_key_auth=caller, model=CLASSIFIER, limiter=limiter, unverifiable=RateLimitUnverifiableError
    )
    await _admit(limiter, caller, CLASSIFIER)

    after: Final = await rate_limit_verdict(
        user_api_key_auth=caller, model=CLASSIFIER, limiter=limiter, unverifiable=RateLimitUnverifiableError
    )

    assert (before, after) == ("allowed", "denied_for_now")


def _classifier_router(cost_per_token: float) -> Router:
    return Router(
        model_list=[
            {
                "model_name": CLASSIFIER,
                "litellm_params": {
                    "model": "openai/classifier",
                    "api_key": "sk-fake",
                    "input_cost_per_token": cost_per_token,
                    "output_cost_per_token": cost_per_token,
                },
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller, rows",
    [
        (UserAPIKeyAuth(api_key="sk-spent", spend=2.0, max_budget=1.0), ()),
        (
            UserAPIKeyAuth(api_key="sk-team", team_id="t-1"),
            (LiteLLM_TeamTableCachedObj(team_id="t-1", models=[CLASSIFIER], spend=2.0, max_budget=1.0),),
        ),
    ],
    ids=["key-budget-spent", "team-budget-spent"],
)
async def test_a_spent_budget_denies_a_paid_classifier_even_when_the_parent_model_is_free(
    proxy_auth_state: ProxyAuthState, caller: UserAPIKeyAuth, rows: Sequence[_ProxyRow]
) -> None:
    router: Final = _classifier_router(cost_per_token=0.001)
    proxy_auth_state.serve(router)
    proxy_auth_state.seed(*rows)

    assert await caller_model_verdict(caller, CLASSIFIER, llm_router=router) == "denied"


@pytest.mark.asyncio
async def test_a_spent_budget_leaves_a_free_classifier_allowed(proxy_auth_state: ProxyAuthState) -> None:
    router: Final = _classifier_router(cost_per_token=0.0)
    proxy_auth_state.serve(router)
    caller: Final = UserAPIKeyAuth(api_key="sk-spent", spend=2.0, max_budget=1.0)

    assert await caller_model_verdict(caller, CLASSIFIER, llm_router=router) == "allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("sub_call, reserved", [(False, True), (True, False)], ids=["parent-request", "sub-call"])
async def test_a_sub_call_reserves_nothing_against_the_callers_budget(
    proxy_auth_state: ProxyAuthState, sub_call: bool, reserved: bool
) -> None:
    proxy_auth_state.serve(_classifier_router(cost_per_token=0.001))
    caller: Final = UserAPIKeyAuth(api_key="sk-live", spend=0.0, max_budget=10.0)
    counter_key: Final = f"spend:key:{caller.token}"
    await proxy_server.spend_counter_cache.async_set_cache(counter_key, 0.0)
    request: Final = Request(
        scope={"type": "http", "method": "POST", "path": "/v1/messages", "headers": [], "query_string": b""}
    )
    priced_body: Final[dict[str, object]] = {
        "model": CLASSIFIER,
        "messages": [{"role": "user", "content": "classify this tool call"}],
        "max_tokens": 100,
    }

    await run_centralized_common_checks(caller, request, priced_body, "/v1/messages", sub_call=sub_call)

    counter: Final = await proxy_server.spend_counter_cache.async_get_cache(counter_key)
    assert (counter > 0, caller.budget_reservation is not None) == (reserved, reserved)


@pytest.mark.asyncio
async def test_a_parent_header_the_sub_call_request_cannot_carry_denies_for_now() -> None:
    parent_metadata: Final = {"headers": {"x-client-name": "snow ☃"}}

    verdict: Final = await caller_model_verdict(
        UserAPIKeyAuth(models=[CLASSIFIER]), CLASSIFIER, llm_router=None, parent_litellm_metadata=parent_metadata
    )

    assert verdict == "denied_for_now"


@pytest.mark.asyncio
async def test_a_team_read_that_fails_mid_turn_denies_for_now(proxy_auth_state: ProxyAuthState) -> None:
    caller: Final = UserAPIKeyAuth(api_key="sk-team-key", team_id="team-not-in-cache")

    assert await caller_model_verdict(caller, CLASSIFIER, llm_router=None) == "denied_for_now"
