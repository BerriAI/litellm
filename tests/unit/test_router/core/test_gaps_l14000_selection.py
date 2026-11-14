"""
Gaps in the characterization of ``router_python_core.py`` lines 14000-17669 on the selection path:
health-check routing (the cooldown bypass under an allowed_fails_policy, health-check probes, the
sync filter the pass-through pick uses), pass-through picks (paused deployments by id, priority
groups, cooling, other strategies, the async entry point), the failure dispatch to a request's
logging object, ``model_alias_map``, access groups and litellm model names, tier ``litellm_params``
(dropped params, effort carriers, restored output ceilings), routing-decision redaction and the
Claude Code session binding's guards.
"""

import asyncio
import time
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import AllowedFailsPolicy, RouterRateLimitError
from tests.unit.test_router.core.scripted_host import ScriptedHost, auth_error, deployment, rate_limit
from tests.unit.test_router.core.selection_helpers import MESSAGES, MakeCore, ids_of, picked_ids
from tests.unit.test_router.core.test_selection_characterization_prerouting import (
    HARD_MESSAGES,
    SESSION_ID,
    TIERS,
    _complexity_router,
    _session_request,
    _tiered,
)

SPREAD_PICKS: Final = 40


def _unhealthy(*ids: str) -> dict[str, dict[str, object]]:  # mutable-ok: health states
    return {i: {"is_healthy": False, "timestamp": time.time(), "reason": "probe failed"} for i in ids}


def _metadata(request: Mapping[str, object], bucket: str = "metadata") -> Mapping[str, object]:
    metadata: Final = request.get(bucket)
    assert isinstance(metadata, Mapping)
    return metadata


# --- health-check routing ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_routing_with_an_allowed_fails_policy_ignores_cooldowns_when_all_cool(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")],
        enable_health_check_routing=True,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=0),
        cooldown_time=60,
    )
    assert [core.health().record_failure(i, rate_limit(), 429) for i in ("a", "b")] == [True, True]
    assert await picked_ids(core, "g", times=SPREAD_PICKS) == {"a", "b"}


@pytest.mark.asyncio
async def test_health_check_routing_without_a_policy_keeps_honoring_cooldowns(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], enable_health_check_routing=True, cooldown_time=60
    )
    for deployment_id in ("a", "b"):
        core.health().record_failure(deployment_id, rate_limit(), 429)
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g")
    assert str(raised.value) == (
        "No deployments available for selected model, Try again in 60 seconds. All deployments for selected "
        "model are in cooldown. Passed model=g. pre-call-checks=False, cooldown_list=['a', 'b']"
    )


@pytest.mark.asyncio
async def test_a_health_check_probe_gets_no_deployments_when_all_are_unhealthy(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], enable_health_check_routing=True)
    core.health().health_checks.set_deployment_health_states(_unhealthy("a", "b"))  # pyright: ignore[reportArgumentType]  # health states
    assert ids_of(await core.select("g", kind="healthy")) == ["a", "b"]
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g", kind="healthy", health_check_probe=True)  # pyright: ignore[reportCallIssue]
    assert str(raised.value) == (
        "No deployments available for selected model, Try again in 5 seconds. Passed model=g. "
        "pre-call-checks=False, cooldown_list=[]"
    )


@pytest.mark.asyncio
async def test_a_health_check_probe_still_drops_only_the_unhealthy_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], enable_health_check_routing=True)
    core.health().health_checks.set_deployment_health_states(_unhealthy("a"))  # pyright: ignore[reportArgumentType]  # health states
    assert ids_of(await core.select("g", kind="healthy", health_check_probe=True)) == ["b"]  # pyright: ignore[reportCallIssue]


PASS_THROUGH_PAIR: Final = [
    deployment("g", "pt", use_in_pass_through=True),
    deployment("g", "pt2", use_in_pass_through=True),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({}, {"pt2"}),
        ({"allowed_fails_policy": {"RateLimitErrorAllowedFails": 1}}, {"pt", "pt2"}),
        ({"background_health_check_model_groups": ["other"]}, {"pt", "pt2"}),
        (
            {"background_health_check_model_groups": ["g"], "allowed_fails_policy": {"RateLimitErrorAllowedFails": 1}},
            {"pt2"},
        ),
    ],
    ids=["filtered", "policy-disables", "group-not-listed", "listed-group-with-policy"],
)
async def test_pass_through_selection_applies_the_health_check_filter(
    make_core: MakeCore, arguments: dict[str, object], expected: set[str]
) -> None:
    core: Final = make_core(model_list=list(PASS_THROUGH_PAIR), enable_health_check_routing=True, **arguments)
    core.health().health_checks.set_deployment_health_states(_unhealthy("pt"))  # pyright: ignore[reportArgumentType]  # health states
    assert await picked_ids(core, "g", times=SPREAD_PICKS, kind="pass_through") == expected


@pytest.mark.asyncio
async def test_pass_through_selection_uses_every_deployment_when_all_are_unhealthy(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PASS_THROUGH_PAIR), enable_health_check_routing=True)
    core.health().health_checks.set_deployment_health_states(_unhealthy("pt", "pt2"))  # pyright: ignore[reportArgumentType]  # health states
    assert await picked_ids(core, "g", times=SPREAD_PICKS, kind="pass_through") == {"pt", "pt2"}


# --- pass-through selection ----------------------------------------------------------------------

PAUSED_PASS_THROUGH: Final = [
    deployment("g", "pt", use_in_pass_through=True, model_info={"blocked": True}),
    deployment("g", "pt2", use_in_pass_through=True),
]

PAUSED_MESSAGE: Final = (
    # Oddity: the message names the deployment's litellm model, not the id that was asked for.
    "litellm.ServiceUnavailableError: Model 'openai/pt' is currently paused and cannot accept requests."
)


@pytest.mark.asyncio
async def test_pass_through_selection_of_a_paused_deployment_by_id_is_unavailable(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PAUSED_PASS_THROUGH))
    with pytest.raises(litellm.ServiceUnavailableError) as raised:
        await core.select("pt", kind="pass_through")
    assert str(raised.value) == PAUSED_MESSAGE


@pytest.mark.asyncio
async def test_async_pass_through_selection_of_a_paused_deployment_by_id_is_unavailable(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PAUSED_PASS_THROUGH))
    with pytest.raises(litellm.ServiceUnavailableError) as raised:
        await core.async_get_available_deployment_for_pass_through("pt", {})  # pyright: ignore[reportAttributeAccessIssue]
    assert str(raised.value) == PAUSED_MESSAGE


@pytest.mark.asyncio
async def test_async_pass_through_selection_by_id_returns_a_pass_through_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PAUSED_PASS_THROUGH))
    chosen: Final = await core.async_get_available_deployment_for_pass_through("pt2", {})  # pyright: ignore[reportAttributeAccessIssue]
    assert chosen["model_info"]["id"] == "pt2"


@pytest.mark.asyncio
async def test_async_pass_through_selection_by_id_refuses_a_plain_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "pt", use_in_pass_through=True), deployment("g", "plain")])
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.async_get_available_deployment_for_pass_through("plain", {})  # pyright: ignore[reportAttributeAccessIssue]
    assert str(raised.value) == (
        "litellm.BadRequestError: Deployment plain does not support pass-through endpoint (use_in_pass_through=False)"
    )


@pytest.mark.asyncio
async def test_async_pass_through_selection_picks_only_pass_through_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "pt", use_in_pass_through=True), deployment("g", "plain")])
    for _ in range(10):
        chosen = await core.async_get_available_deployment_for_pass_through("g", {})  # pyright: ignore[reportAttributeAccessIssue]
        assert chosen["model_info"]["id"] == "pt"


@pytest.mark.asyncio
async def test_pass_through_selection_with_a_selector_strategy_picks_through_it(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PASS_THROUGH_PAIR), routing_strategy="least-busy")
    assert await picked_ids(core, "g", kind="pass_through") == {"pt"}
    chosen: Final = await core.async_get_available_deployment_for_pass_through("g", {})  # pyright: ignore[reportAttributeAccessIssue]
    assert chosen["model_info"]["id"] == "pt"


PRIORITY_PASS_THROUGH: Final = [
    deployment("m1", "a", use_in_pass_through=True),
    deployment("m2", "b", use_in_pass_through=True),
]
PRIORITY_GROUP: Final = {
    "group_name": "pg",
    "models": ["m1", "m2"],
    "routing_strategy": "priority",
    "model_priorities": {"m1": 1, "m2": 2},
}


@pytest.mark.asyncio
async def test_pass_through_selection_of_a_priority_group_keeps_the_lowest_priority(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PRIORITY_PASS_THROUGH), routing_groups=[dict(PRIORITY_GROUP)])
    assert await picked_ids(core, "pg", kind="pass_through") == {"a"}
    assert await picked_ids(core, "pg", request_kwargs={"_target_order": 2}, kind="pass_through") == {"b"}


@pytest.mark.asyncio
async def test_pass_through_selection_of_a_priority_order_nothing_has_rate_limits(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PRIORITY_PASS_THROUGH), routing_groups=[dict(PRIORITY_GROUP)])
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("pg", {"_target_order": 3}, kind="pass_through")
    assert str(raised.value) == (
        "No deployments available for selected model, Try again in 5 seconds. Passed model=pg. "
        "pre-call-checks=False, cooldown_list=[]"
    )


@pytest.mark.asyncio
async def test_pass_through_selection_with_every_pass_through_deployment_cooling_rate_limits(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "pt", use_in_pass_through=True), deployment("g", "plain")], cooldown_time=60
    )
    assert core.health().record_failure("pt", auth_error(), 401) is True
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g", kind="pass_through")
    assert str(raised.value) == (
        "No deployments available for selected model, Try again in 60 seconds. All deployments for selected "
        "model are in cooldown. Passed model=g. pre-call-checks=False, cooldown_list=['pt']"
    )


# --- failure dispatch to the request's logging object --------------------------------------------


class _RecordingLoggingObject:
    """Stands in for a request's ``litellm_logging_obj``: records the failure dispatches."""

    def __init__(self) -> None:
        self.dispatched: list[dict[str, object]] = []  # mutable-ok: the dispatches, in order

    async def dispatch_failure_handlers(self, **kwargs: object) -> None:  # kwargs-ok: the dispatch arguments
        self.dispatched.append(kwargs)


async def _let_tasks_run() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_refused_pick_is_dispatched_to_the_requests_logging_object(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    logging_obj: Final = _RecordingLoggingObject()
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("nope", {"litellm_logging_obj": logging_obj})
    await _let_tasks_run()
    (dispatched,) = logging_obj.dispatched
    assert dispatched["exception"] is raised.value
    assert dispatched["prefer_async_handlers"] is True
    assert "Traceback" in str(dispatched["traceback_exception"])


@pytest.mark.asyncio
async def test_a_refused_async_pass_through_pick_is_dispatched_to_the_requests_logging_object(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    logging_obj: Final = _RecordingLoggingObject()
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.async_get_available_deployment_for_pass_through("g", {"litellm_logging_obj": logging_obj})  # pyright: ignore[reportAttributeAccessIssue]
    await _let_tasks_run()
    assert [entry["exception"] for entry in logging_obj.dispatched] == [raised.value]


# --- model_alias_map, access groups and litellm model names --------------------------------------


@pytest.mark.asyncio
async def test_the_process_model_alias_map_renames_the_model_in_a_no_deployments_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(model_alias_map={"g": "renamed"})
    core: Final = make_core(model_list=[deployment("g", "a")], cooldown_time=60)
    assert (await core.select("g"))["model_info"]["id"] == "a"  # pyright: ignore[reportIndexIssue]
    core.health().record_failure("a", auth_error(), 401)
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g")
    assert "Passed model=renamed." in str(raised.value)


ACCESS_GROUPED: Final = [
    deployment("g", "in_a", model_info={"access_groups": ["grpA"]}),
    deployment("g", "in_b", model_info={"access_groups": ["grpB"]}),
    deployment("h", "hh"),
]


@pytest.mark.asyncio
async def test_an_access_group_only_caller_reaches_any_deployment_by_its_litellm_model(make_core: MakeCore) -> None:
    # Oddity: the access-group narrowing looks up access groups by the requested name, and a
    # litellm model string names no group, so a grpA-only caller reaches the grpB deployment.
    core: Final = make_core(model_list=list(ACCESS_GROUPED), default_fallbacks=["h"])
    auth: Final = {"metadata": {"user_api_key_auth": UserAPIKeyAuth(models=["grpA"])}}  # pyright: ignore[reportCallIssue]
    assert (await core.select("openai/in_b", dict(auth)))["model_info"]["id"] == "in_b"  # pyright: ignore[reportIndexIssue]
    assert (await core.select("openai/in_a", dict(auth)))["model_info"]["id"] == "in_a"  # pyright: ignore[reportIndexIssue]
    assert (await core.select("unknown", dict(auth)))["model_info"]["id"] == "hh"  # pyright: ignore[reportIndexIssue]


# --- tier litellm_params -------------------------------------------------------------------------


def _pinned(**tier_params: object) -> dict[str, object]:  # mutable-ok: tiers  # kwargs-ok: tier litellm_params
    return {**TIERS, "SIMPLE": {"model_name": "small", "litellm_params": tier_params}}


@pytest.mark.asyncio
async def test_a_tier_param_no_deployment_supports_is_dropped(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("small", "s", model="anthropic/claude-3-5-sonnet-20240620"),
            deployment("big", "b"),
            _complexity_router(tiers=_pinned(logit_bias={"1": 2}, temperature=0.2)),
        ]
    )
    request: Final[dict[str, object]] = {}
    chosen: Final = await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "s"  # pyright: ignore[reportIndexIssue]
    assert request["temperature"] == 0.2
    assert "logit_bias" not in request


@pytest.mark.asyncio
async def test_a_tier_param_the_request_allowlists_is_kept(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("small", "s", model="anthropic/claude-3-5-sonnet-20240620"),
            deployment("big", "b"),
            _complexity_router(tiers=_pinned(logit_bias={"1": 2})),
        ]
    )
    request: Final[dict[str, object]] = {"allowed_openai_params": ["logit_bias"]}
    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert request["logit_bias"] == {"1": 2}


@pytest.mark.asyncio
async def test_a_tier_reasoning_effort_removes_the_callers_other_effort_carriers(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered(tiers=_pinned(reasoning_effort="low")))
    request: Final[dict[str, object]] = {
        "thinking": {"type": "enabled"},
        "output_config": {"effort": "high", "format": "json"},
        "reasoning": {"effort": "high"},
        "max_completion_tokens": 100,
    }
    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert request["reasoning_effort"] == "low"
    assert "thinking" not in request
    assert request["output_config"] == {"format": "json"}
    assert "reasoning" not in request
    assert request["max_completion_tokens"] == 100


@pytest.mark.asyncio
async def test_a_tier_without_an_output_ceiling_leaves_the_callers_ceiling(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered(tiers=_pinned(temperature=0.1)))
    request: Final[dict[str, object]] = {"max_completion_tokens": 100}
    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert request["temperature"] == 0.1
    assert request["max_completion_tokens"] == 100
    assert "max_tokens" not in request
    # The caller's ceiling is stamped on every routed pick, pinned or not.
    assert _metadata(request)["_client_output_ceiling"] == {"max_completion_tokens": 100}


@pytest.mark.asyncio
async def test_a_later_pick_restores_the_callers_ceiling_a_tier_pin_replaced(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered(tiers=_pinned(max_tokens=7)))
    request: Final[dict[str, object]] = {"max_completion_tokens": 100}
    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert request["max_tokens"] == 7
    assert "max_completion_tokens" not in request
    assert _metadata(request)["_client_output_ceiling"] == {"max_completion_tokens": 100}

    chosen: Final = await core.select("smart", request, messages=HARD_MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]
    assert request["max_completion_tokens"] == 100
    assert "max_tokens" not in request


@pytest.mark.asyncio
async def test_a_second_pick_leaves_the_first_picks_forwarded_marker_params_untracked(make_core: MakeCore) -> None:
    # Oddity: forwarded marker params already on the request are not "newly forwarded" on the
    # next attempt, so the tracking key is dropped while the params themselves stay behind.
    core: Final = make_core(model_list=_tiered(temperature=0.3))
    request: Final[dict[str, object]] = {}
    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    forwarded: Final = request["_alias_marker_forwarded_params"]
    assert isinstance(forwarded, tuple)
    assert "temperature" in forwarded
    assert request["temperature"] == 0.3

    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert "_alias_marker_forwarded_params" not in request
    assert request["temperature"] == 0.3


# --- routing decision redaction ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_routing_decision_keeps_its_prompt_signals_by_default(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    request: Final[dict[str, object]] = {}
    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    decision: Final = _metadata(request)["routing_decision"]
    assert isinstance(decision, Mapping)
    assert "signals" in decision


@pytest.mark.asyncio
async def test_redacted_message_logging_drops_the_prompt_quoting_routing_decision_fields(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    request: Final[dict[str, object]] = {"standard_callback_dynamic_params": {"turn_off_message_logging": True}}
    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    decision: Final = _metadata(request)["routing_decision"]
    assert isinstance(decision, Mapping)
    assert "signals" not in decision
    assert (decision["router_model_name"], decision["tier"], decision["routed_model"]) == ("smart", "SIMPLE", "small")


@pytest.mark.asyncio
async def test_routing_metadata_goes_to_litellm_metadata_when_the_request_has_it(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    request: Final[dict[str, object]] = {"litellm_metadata": {}}
    await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert "routing_decision" in _metadata(request, "litellm_metadata")
    assert "metadata" not in request


# --- the Claude Code session binding's guards ----------------------------------------------------


async def _subagent_pick(core: object) -> str:
    chosen: Final = await core.select("big", _session_request(agent_id="agent-1"), messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    return str(chosen["model_info"]["id"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("depth", "expected"), [(1, "b"), (0, "s")], ids=["fallback-attempt", "first-attempt"])
async def test_only_a_first_attempt_binds_its_session(make_core: MakeCore, depth: int, expected: str) -> None:
    core: Final = make_core(model_list=_tiered())
    first: Final = _session_request()
    first["fallback_depth"] = depth
    await core.select("smart", first, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert await _subagent_pick(core) == expected


@pytest.mark.asyncio
async def test_a_request_without_a_caller_hash_never_binds(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    first: Final = _session_request()
    first["metadata"] = {}
    await core.select("smart", first, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert await _subagent_pick(core) == "b"


@pytest.mark.asyncio
async def test_the_caller_hash_is_read_from_litellm_metadata_when_present(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    first: Final = _session_request()
    first["metadata"] = {}
    first["litellm_metadata"] = {"user_api_key_hash": "caller-hash"}
    await core.select("smart", first, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert await _subagent_pick(core) == "s"


@pytest.mark.asyncio
async def test_the_session_header_name_is_matched_case_insensitively(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    first: Final = _session_request()
    first["proxy_server_request"] = {"headers": {"X-Claude-Code-Session-Id": SESSION_ID}}
    await core.select("smart", first, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert await _subagent_pick(core) == "s"
