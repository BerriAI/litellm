"""The post-call hook feeds spend and completion signals from LiteLLM's logging payload into the router."""

import asyncio

import pytest

from litellm.router_strategy.oracle_router.config import (
    CHOSEN_MODEL_METADATA_KEY,
    PROGRAM_ID_METADATA_KEY,
    RESPONSE_HEADER,
)
from litellm.router_strategy.oracle_router.hooks import OracleRouterPostCallHook
from litellm.router_strategy.oracle_router.oracle_router import OracleRouter
from litellm.router_strategy.oracle_router.verifier import ReportedVerifier
from litellm.types.router import OracleRouterConfig
from litellm.types.utils import Choices, Message, ModelResponse


class _Maker:
    def __init__(self) -> None:
        self.updates = []

    @property
    def models(self):
        return ("smart", "fast")

    async def select(self, context):
        return "smart"

    def update(self, context, model, score):
        self.updates.append((model, score))

    def snapshot(self):
        return {}


ROUTED = {"router_model_name": "oracle", "router_type": "oracle"}  # what the Router stamps for this router


def _kwargs(program_id="t1", cost=0.01, routing_decision=ROUTED, **metadata):
    stamped = {PROGRAM_ID_METADATA_KEY: program_id, **metadata}
    if routing_decision is not None:
        stamped["routing_decision"] = routing_decision
    return {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "task"}],
        "litellm_params": {"metadata": stamped},
        "response_cost": cost,
    }


def _response(text="answer") -> ModelResponse:
    return ModelResponse(choices=[Choices(message=Message(content=text, role="assistant"))])


@pytest.fixture
def bound():
    maker = _Maker()
    router = OracleRouter("oracle", OracleRouterConfig(available_models=["smart", "fast"]), maker, ReportedVerifier())
    asyncio.run(
        router.async_pre_routing_hook(
            model="oracle",
            request_kwargs={"metadata": {"program_id": "t1"}},
            messages=[{"role": "user", "content": "task"}],
        )
    )
    return router, maker, OracleRouterPostCallHook(router)


@pytest.mark.asyncio
async def test_success_events_accumulate_spend_and_the_latest_answer(bound):
    router, _, hook = bound
    await hook.async_log_success_event(_kwargs(cost=0.01), _response("first"), None, None)
    await hook.async_log_success_event(_kwargs(cost=0.02), _response("second"), None, None)
    binding = router.binding("t1")
    assert binding.requests == 2 and binding.cost == pytest.approx(0.03) and binding.last_response_text == "second"
    assert list(binding.last_messages) == [{"role": "user", "content": "task"}]


@pytest.mark.asyncio
async def test_program_done_completes_the_program_with_the_reported_score(bound):
    router, maker, hook = bound
    await hook.async_log_success_event(
        _kwargs(cost=0.05, program_done=True, program_score="1"), _response(), None, None
    )
    assert router.binding("t1") is None
    await router.drain()
    assert maker.updates == [("smart", 1.0)]


@pytest.mark.asyncio
async def test_a_non_finite_program_score_is_treated_as_no_score(bound):
    router, maker, hook = bound
    await hook.async_log_success_event(
        _kwargs(cost=0.05, program_done=True, program_score="nan"), _response(), None, None
    )
    assert router.binding("t1") is None
    await router.drain()
    assert maker.updates == [("smart", 0.0)]  # the reported verifier's default, never NaN


@pytest.mark.asyncio
async def test_failure_event_does_not_add_spend_but_still_honours_program_done(bound):
    router, maker, hook = bound
    await hook.async_log_failure_event(_kwargs(cost=0.5), None, None, None)
    assert router.binding("t1").cost == 0.0
    await hook.async_log_failure_event(_kwargs(program_done=True), None, None, None)
    await router.drain()
    assert maker.updates == [("smart", 0.0)]


@pytest.mark.asyncio
async def test_events_for_unknown_or_untagged_programs_are_ignored(bound):
    router, _, hook = bound
    await hook.async_log_success_event(
        {"litellm_params": {"metadata": {}}, "response_cost": 1.0}, _response(), None, None
    )
    await hook.async_log_success_event(_kwargs(program_id="other"), _response(), None, None)
    assert router.binding("t1").requests == 0 and router.binding("other") is None


@pytest.mark.asyncio
async def test_events_from_requests_this_router_did_not_route_are_ignored(bound):
    router, maker, hook = bound
    unrouted = _kwargs(cost=0.5, program_done=True, program_score="1", routing_decision=None)
    await hook.async_log_success_event(unrouted, _response(), None, None)
    elsewhere = _kwargs(
        cost=0.5, program_done=True, program_score="1", routing_decision={**ROUTED, "router_model_name": "x"}
    )
    await hook.async_log_success_event(elsewhere, _response(), None, None)
    assert router.binding("t1").requests == 0 and maker.updates == []


@pytest.mark.asyncio
async def test_events_only_reach_the_program_of_the_key_that_sent_them(bound):
    router, maker, hook = bound  # t1 was started without a key
    stranger = _kwargs(cost=0.5, program_done=True, program_score="1", user_api_key_hash="other-key")
    await hook.async_log_success_event(stranger, _response(), None, None)
    assert router.binding("t1").requests == 0 and maker.updates == []
    await router.async_pre_routing_hook(
        model="oracle",
        request_kwargs={"metadata": {"program_id": "t1", "user_api_key_hash": "other-key"}},
        messages=[{"role": "user", "content": "task"}],
    )
    await hook.async_log_success_event(stranger, _response(), None, None)
    assert router.binding("t1", "other-key") is None and router.binding("t1").requests == 0
    await router.drain()
    assert maker.updates == [("smart", 1.0)]


@pytest.mark.asyncio
async def test_response_header_names_the_bound_model(bound):
    _, _, hook = bound
    headers = await hook.async_post_call_response_headers_hook(
        {"metadata": {CHOSEN_MODEL_METADATA_KEY: "smart", "routing_decision": ROUTED}}, None, None
    )
    assert headers == {RESPONSE_HEADER: "smart"}
    assert (
        await hook.async_post_call_response_headers_hook({"metadata": {"routing_decision": ROUTED}}, None, None) is None
    )
    forged = {"metadata": {CHOSEN_MODEL_METADATA_KEY: "smart"}}  # a request this router did not route
    assert await hook.async_post_call_response_headers_hook(forged, None, None) is None


@pytest.mark.asyncio
async def test_response_header_reads_the_litellm_metadata_bucket_too(bound):
    _, _, hook = bound
    data = {"metadata": {}, "litellm_metadata": {CHOSEN_MODEL_METADATA_KEY: "fast", "routing_decision": ROUTED}}
    assert await hook.async_post_call_response_headers_hook(data, None, None) == {RESPONSE_HEADER: "fast"}
