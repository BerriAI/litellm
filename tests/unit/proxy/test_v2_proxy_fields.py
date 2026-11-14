from collections.abc import Generator
from typing import Final

import pytest

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Field, Hook, Patch, on
from litellm.integrations.custom_logger_v2.call import CallFacts
from litellm.integrations.custom_logger_v2.host import reset_host_for_tests
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.proxy.v2_proxy_fields import limiter_request, output_cap
from litellm.types.integrations.custom_logger_v2 import (
    BudgetPlan,
    BudgetReservation,
    KeyIdentity,
    LimiterRequest,
    OutputCap,
    RateLimitStatus,
    RateLimitStatuses,
    Stringified,
)
from litellm.types.utils import ModelResponse

PLAN: Final = BudgetPlan(reservation_cost=0.25, fail_closed=True)


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    yield
    reset_host_for_tests()


@pytest.fixture
def proxy_logging() -> ProxyLogging:
    return ProxyLogging(user_api_key_cache=DualCache())


def _auth() -> UserAPIKeyAuth:
    auth: Final = UserAPIKeyAuth(api_key="sk-real", user_id="real-user", request_route="/chat/completions")
    auth.budget_plan = PLAN
    auth.budget_reservation = {"reserved_cost": 0.25}
    return auth


def _forged_request() -> dict[str, object]:
    return {
        "model": "m",
        "litellm_call_id": "call-1",
        "metadata": {"user_api_key_user_id": "forged", "user_api_key_hash": "forged"},
        "user_api_key_dict": {"user_id": "forged"},
    }


def _seen(view: object, field: Field) -> object:
    return getattr(view, field.value, None)


class _Reserver(CustomLoggerV2):
    def __init__(self) -> None:
        self.seen: list[tuple[object, ...]] = []

    @on(
        Hook.pre_call,
        reads={Field.key_identity, Field.budget_plan, Field.budget_reservation},
        writes={Field.budget_reservation, Field.rate_limit_statuses},
    )
    async def pre_call(self, view: object, ctx: CallContext[object]) -> Patch:
        self.seen.append(
            (
                _seen(view, Field.key_identity),
                _seen(view, Field.budget_plan),
                _seen(view, Field.budget_reservation),
                ctx.call,
            )
        )
        return (
            Patch()
            .set(Field.budget_reservation, BudgetReservation(finalized=True, input_tokens=12))
            .set(Field.rate_limit_statuses, RateLimitStatuses(statuses=(RateLimitStatus(code="OK", current_limit=5),)))
        )


class _Reader(CustomLoggerV2):
    def __init__(self) -> None:
        self.seen: list[tuple[object, ...]] = []

    @on(Hook.pre_call, reads={Field.budget_plan, Field.budget_reservation, Field.rate_limit_statuses}, awaited=True)
    async def pre_call(self, view: object, ctx: CallContext[object]) -> None:
        self.seen.append(
            (
                _seen(view, Field.budget_plan),
                _seen(view, Field.budget_reservation),
                _seen(view, Field.rate_limit_statuses),
            )
        )


async def test_pre_call_hands_auth_fields_to_subscribers_and_a_reservation_write_reaches_the_next_one(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    reserver: Final = _Reserver()
    reader: Final = _Reader()
    monkeypatch.setattr(litellm, "callbacks", [reserver, reader])
    auth: Final = _auth()
    shared: Final = auth.budget_reservation

    _ = await proxy_logging.pre_call_hook(user_api_key_dict=auth, data=_forged_request(), call_type="completion")

    identity, plan, reservation, call = reserver.seen[0]
    assert identity == KeyIdentity(key_hash=auth.token, api_key=auth.api_key, user_id="real-user")
    assert plan == PLAN
    assert reservation == BudgetReservation(reserved_cost=0.25)
    assert isinstance(call, CallFacts)
    assert (call.ids.call_id, call.kind.route, call.kind.call_type) == ("call-1", "/chat/completions", "completion")
    assert reader.seen == [
        (
            PLAN,
            BudgetReservation(reserved_cost=0.25, finalized=True, input_tokens=12),
            RateLimitStatuses(overall_code="OK", statuses=(RateLimitStatus(code="OK", current_limit=5),)),
        )
    ]
    assert auth.budget_reservation is shared
    assert shared == {"reserved_cost": 0.25, "finalized": True, "input_tokens": 12}


async def test_a_rerun_pre_call_with_the_same_auth_sees_no_budget_plan(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    reader: Final = _Reader()
    monkeypatch.setattr(litellm, "callbacks", [reader])
    auth: Final = _auth()

    _ = await proxy_logging.pre_call_hook(user_api_key_dict=auth, data={"model": "m"}, call_type="completion")
    _ = await proxy_logging.pre_call_hook(user_api_key_dict=auth, data={"model": "m"}, call_type="completion")

    assert [plan for plan, _, _ in reader.seen] == [PLAN, None]


class _OffPathReader(CustomLoggerV2):
    @on(Hook.pre_call, reads={Field.budget_plan})
    async def pre_call(self, view: object, ctx: CallContext[object]) -> None:
        return None


async def test_an_off_path_reader_derives_its_fields_before_the_hook_returns(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_OffPathReader()])
    auth: Final = _auth()

    _ = await proxy_logging.pre_call_hook(user_api_key_dict=auth, data={"model": "m"}, call_type="completion")

    assert auth.budget_plan is None


class _PostCall(CustomLoggerV2):
    def __init__(self) -> None:
        self.seen: list[tuple[Hook, object, CallFacts | None]] = []

    @on(Hook.post_call_success, reads={Field.key_identity}, awaited=True)
    async def post_call_success(self, view: object, ctx: CallContext[object]) -> None:
        self.seen.append((ctx.hook, _seen(view, Field.key_identity), ctx.call))

    @on(Hook.post_call_failure, reads={Field.key_identity}, awaited=True)
    async def post_call_failure(self, view: object, ctx: CallContext[object]) -> None:
        self.seen.append((ctx.hook, _seen(view, Field.key_identity), ctx.call))


async def test_post_call_hooks_get_call_facts_and_the_identity_of_the_auth_object(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    recorder: Final = _PostCall()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    auth: Final = _auth()

    _ = await proxy_logging.post_call_success_hook(
        data=_forged_request(), response=ModelResponse(), user_api_key_dict=auth
    )
    _ = await proxy_logging.post_call_failure_hook(
        request_data=_forged_request(), original_exception=RuntimeError("upstream"), user_api_key_dict=auth
    )

    identity: Final = KeyIdentity(key_hash=auth.token, api_key=auth.api_key, user_id="real-user")
    assert [(hook, seen) for hook, seen, _ in recorder.seen] == [
        (Hook.post_call_success, identity),
        (Hook.post_call_failure, identity),
    ]
    assert [
        (call.ids.call_id, call.kind.route, call.target.model) for _, _, call in recorder.seen if call is not None
    ] == [("call-1", "/chat/completions", "m")] * 2


@pytest.mark.parametrize(
    ("request_data", "explicit"),
    [
        pytest.param({"max_tokens": 64}, 64, id="small"),
        pytest.param({"max_tokens": 2**70}, 2**70, id="past_i64"),
        pytest.param(
            {"max_tokens": "1e300"},
            Stringified(
                text=str(int(1e300)), truthy=True, hashable=True, integer=True, type_name="int", hash_error=None
            ),
            id="past_i128",
        ),
        pytest.param({"max_tokens": "inf"}, None, id="overflow_is_no_cap"),
    ],
)
def test_the_output_cap_carries_the_int_the_limiter_parses_however_wide(
    request_data: dict[str, object], explicit: int | Stringified | None
) -> None:
    assert output_cap(HookArgs(request=request_data, call_type="acompletion")) == OutputCap(explicit=explicit)


@pytest.mark.parametrize(
    ("request_data", "expected"),
    [
        pytest.param(
            {"litellm_session_id": 7, "metadata": {"session_id": "m", "tags": ["a", 3, "b"]}, "tags": ["c", "a"]},
            LimiterRequest(
                fallback_on_saturation=False, fallback_walk_armed=False, tags=("a", "b", "c", "a"), session_id="7"
            ),
            id="session_from_litellm_session_id",
        ),
        pytest.param(
            {"metadata": {"session_id": ""}, "litellm_metadata": {"session_id": "l", "agent_id": "agent-l"}},
            LimiterRequest(
                fallback_on_saturation=False,
                fallback_walk_armed=False,
                tags=(),
                session_id="l",
                resolved_agent_id="agent-l",
            ),
            id="session_and_agent_from_litellm_metadata",
        ),
        pytest.param(
            {"metadata": "not json", "tags": "t"},
            LimiterRequest(
                fallback_on_saturation=False,
                fallback_walk_armed=False,
                tags=(),
                session_id_error_type="AttributeError",
                session_id_error_text="'str' object has no attribute 'get'",
                resolved_agent_id_error_type="AttributeError",
                resolved_agent_id_error_text="'str' object has no attribute 'get'",
            ),
            id="metadata_not_a_mapping",
        ),
        pytest.param(
            {"metadata": {"agent_id": ["a1"]}},
            LimiterRequest(
                fallback_on_saturation=False,
                fallback_walk_armed=False,
                tags=(),
                resolved_agent_id=Stringified(
                    text="['a1']",
                    truthy=True,
                    hashable=False,
                    integer=False,
                    type_name="list",
                    hash_error="unhashable type: 'list'",
                ),
            ),
            id="list_agent_id",
        ),
    ],
)
def test_limiter_request_serves_the_tags_session_and_agent_the_limiter_reads_from_the_body(
    request_data: dict[str, object], expected: LimiterRequest
) -> None:
    auth: Final = UserAPIKeyAuth(api_key="sk-real")

    assert limiter_request(HookArgs(auth=auth, request=request_data)) == expected


def test_limiter_request_resolves_the_key_agent_before_the_body_agent() -> None:
    auth: Final = UserAPIKeyAuth(api_key="sk-real", agent_id="key-agent")

    served: Final = limiter_request(HookArgs(auth=auth, request={"metadata": {"agent_id": "body-agent"}}))

    assert served == LimiterRequest(
        fallback_on_saturation=False, fallback_walk_armed=False, tags=(), resolved_agent_id="key-agent"
    )


class _UnprintableFlag:
    def __repr__(self) -> str:
        raise ValueError("no repr")


@pytest.mark.parametrize(
    "request_data",
    [
        pytest.param({}, id="no_override"),
        pytest.param({"router_settings_override": None}, id="override_none"),
        pytest.param({"router_settings_override": "fallback_on_saturation"}, id="override_text"),
        pytest.param({"router_settings_override": ["fallback_on_saturation"]}, id="override_list"),
        pytest.param({"router_settings_override": {1: True}}, id="override_int_key"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": True}}, id="true"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": False}}, id="false"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": None}}, id="none"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": "yes"}}, id="text"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": "true"}}, id="text_true"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": 1}}, id="one"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": 1.0}}, id="float"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": [True]}}, id="list"),
        pytest.param({"router_settings_override": {"fallback_on_saturation": {"on": True}}}, id="dict"),
    ],
)
@pytest.mark.parametrize("walk", ["off", "armed", "exhausted"])
def test_fallback_on_saturation_members_reproduce_the_capacity_callbacks_handoff_check(
    request_data: dict[str, object], walk: str, caplog: pytest.LogCaptureFixture
) -> None:
    from litellm.proxy.common_utils.proxy_rate_limit_error import (
        arm_saturation_fallback_walk,
        disarm_saturation_fallback_walk,
        fallback_on_saturation_requested,
        mark_saturation_fallbacks_exhausted,
    )

    token: Final = None if walk == "off" else arm_saturation_fallback_walk()
    if walk == "exhausted":
        mark_saturation_fallbacks_exhausted()
    try:
        served: Final = limiter_request(HookArgs(auth=UserAPIKeyAuth(api_key="sk-real"), request=request_data))
        with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
            requested: Final = fallback_on_saturation_requested(request_data)
    finally:
        if token is not None:
            disarm_saturation_fallback_walk(token)

    assert served is not None
    assert served.rejected is None
    assert served.fallback_walk_armed is (walk == "armed")
    assert (served.fallback_on_saturation is True and served.fallback_walk_armed is True) is requested
    assert [entry.getMessage() for entry in caplog.records] == (
        []
        if served.fallback_on_saturation_invalid is None
        else [
            f"router_settings.fallback_on_saturation={served.fallback_on_saturation_invalid} is not a boolean, "
            "treating it as disabled"
        ]
    )


def test_fallback_on_saturation_names_a_value_whose_repr_raises_in_rejected() -> None:
    served: Final = limiter_request(
        HookArgs(
            auth=UserAPIKeyAuth(api_key="sk-real"),
            request={"router_settings_override": {"fallback_on_saturation": _UnprintableFlag()}},
        )
    )

    assert served == LimiterRequest(
        tags=(),
        fallback_on_saturation=False,
        fallback_walk_armed=False,
        rejected=("fallback_on_saturation_invalid",),
    )
