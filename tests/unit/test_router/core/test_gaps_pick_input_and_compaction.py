"""
Parity gaps differential fuzzing found (regress5 R2, U4, N10): the ``messages`` and ``input`` each
endpoint hands its pick, which decide whether the pre-call checks run and what they count;
``print_deployment`` on a ``Deployment``; and the context compaction checks of an auto-router.
"""

import io
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.router_api import RoutingContext
from litellm.router_core import RouterCore
from litellm.router_strategy.complexity_router.context_compaction import CompactionState
from litellm.types.router import Deployment, LiteLLM_Params, RouterRateLimitError
from tests.unit.test_router.core.construction_precall_helpers import windowed
from tests.unit.test_router.core.endpoints_helpers import EndpointHost
from tests.unit.test_router.core.scripted_host import deployment

MakeCore = Callable[..., RouterCore]

MESSAGES: Final = [{"role": "user", "content": "hi"}]
LONG_TEXT: Final = "word " * 200
REGIONAL: Final = [deployment("g", "eu", region_name="eu"), deployment("g", "ap", region_name="ap")]
EU_ONLY: Final = [deployment("g", "eu", region_name="eu")]
BATCH: Final = {"input_file_id": "f", "endpoint": "/v1/chat/completions", "completion_window": "24h"}


@pytest.fixture
def host() -> EndpointHost:
    return EndpointHost()


# --- R2: the pre-call checks of a prompt-only request ---------------------------------------


@pytest.mark.asyncio
async def test_async_text_completion_is_refused_outside_the_allowed_region(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=EU_ONLY, enable_pre_call_checks=True, num_retries=0)
    with pytest.raises(RouterRateLimitError):
        await core.call("atext_completion", {"model": "g", "prompt": "hi", "allowed_model_region": "ap"})
    assert host.calls_to("atext_completion") == []


def test_sync_text_completion_is_refused_outside_the_allowed_region(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(model_list=EU_ONLY, enable_pre_call_checks=True, num_retries=0)
    with pytest.raises(RouterRateLimitError):
        core.text_completion(model="g", prompt="hi", allowed_model_region="ap")  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls_to("text_completion") == []


@pytest.mark.asyncio
async def test_async_text_completion_keeps_the_deployment_in_the_allowed_region(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=REGIONAL, enable_pre_call_checks=True, num_retries=0)
    await core.call("atext_completion", {"model": "g", "prompt": "hi", "allowed_model_region": "ap"})
    assert host.deployments_called("atext_completion") == ["ap"]


def test_sync_text_completion_keeps_the_deployment_in_the_allowed_region(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=REGIONAL, enable_pre_call_checks=True, num_retries=0)
    core.text_completion(model="g", prompt="hi", allowed_model_region="ap")  # pyright: ignore[reportAttributeAccessIssue]
    # The sync text path sends no ``model_info``: the params name the deployment.
    assert [call.kwargs["region_name"] for call in host.calls_to("text_completion")] == ["ap"]


@pytest.mark.asyncio
async def test_async_text_completion_counts_the_prompt_against_the_context_window(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=[windowed("g", "small", 5)], enable_pre_call_checks=True, num_retries=0)
    with pytest.raises(litellm.ContextWindowExceededError, match="Context Window exceeded for given call"):
        await core.call("atext_completion", {"model": "g", "prompt": LONG_TEXT})
    assert host.calls_to("atext_completion") == []


def test_sync_text_completion_counts_the_prompt_against_the_context_window(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=[windowed("g", "small", 5)], enable_pre_call_checks=True, num_retries=0)
    with pytest.raises(litellm.ContextWindowExceededError, match="Context Window exceeded for given call"):
        core.text_completion(model="g", prompt=LONG_TEXT)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls_to("text_completion") == []


@pytest.mark.parametrize(
    ("endpoint", "request_fields"),
    [
        ("aimage_generation", {"prompt": "a cat"}),
        ("image_generation", {"prompt": "a cat"}),
        ("atranscription", {"file": io.BytesIO(b"x")}),
        ("aspeech", {"input": "hi", "voice": "alloy"}),
        ("aadapter_completion", {"adapter_id": "a", "messages": MESSAGES}),
        ("acreate_batch", BATCH),
    ],
)
@pytest.mark.asyncio
async def test_endpoints_picking_with_a_stand_in_message_run_the_region_check(
    make_core: MakeCore, host: EndpointHost, endpoint: str, request_fields: Mapping[str, object]
) -> None:
    core: Final = make_core(model_list=EU_ONLY, enable_pre_call_checks=True, num_retries=0)
    with pytest.raises(RouterRateLimitError):
        await core.call(endpoint, {"model": "g", "allowed_model_region": "ap", **request_fields})
    assert host.calls == []


@pytest.mark.parametrize(
    ("endpoint", "request_fields"),
    [("arerank", {"query": "q", "documents": ["a"]}), ("amoderation", {"input": "text"})],
)
@pytest.mark.asyncio
async def test_endpoints_picking_with_no_input_skip_the_region_check(
    make_core: MakeCore, host: EndpointHost, endpoint: str, request_fields: Mapping[str, object]
) -> None:
    core: Final = make_core(model_list=EU_ONLY, enable_pre_call_checks=True, num_retries=0)
    await core.call(endpoint, {"model": "g", "allowed_model_region": "ap", **request_fields})
    assert host.deployments_called(endpoint) == ["eu"]


@pytest.mark.asyncio
async def test_speech_counts_its_stand_in_message_and_not_its_input(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(model_list=[windowed("g", "small", 50)], enable_pre_call_checks=True, num_retries=0)
    await core.call("aspeech", {"model": "g", "input": LONG_TEXT, "voice": "alloy"})
    assert host.deployments_called("aspeech") == ["small"]


# --- U4: print_deployment --------------------------------------------------------------------


def test_print_deployment_masks_the_key_of_a_deployment_and_returns_a_deployment(make_core: MakeCore) -> None:
    given: Final = Deployment(model_name="g", litellm_params=LiteLLM_Params(model="openai/x", api_key="sk-obj-123456"))
    core: Final = make_core(model_list=[deployment("g", "a")])
    printed: Final = core.print_deployment(given)  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(printed, Deployment)
    assert printed.litellm_params.api_key == "sk**********"
    assert given.litellm_params.api_key == "sk-obj-123456"


def test_print_deployment_refuses_a_list_of_pairs(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    with pytest.raises(TypeError):
        core.print_deployment([["litellm_params", {"api_key": "sk-pairs-1234"}]])  # pyright: ignore[reportAttributeAccessIssue]


# --- N10: context compaction -----------------------------------------------------------------


def _complexity_router_over_a_small_window() -> list[dict[str, object]]:  # mutable-ok: a model_list
    return [
        windowed("mini", "mini", 60),
        deployment("o47", "o47", model="anthropic/claude-opus-4-7"),
        {
            "model_name": "smart",
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_config": {
                    "tiers": {"SIMPLE": "mini", "REASONING": "o47"},
                    "enable_context_window_escalation": True,
                },
                "complexity_router_default_model": "mini",
            },
            "model_info": {"id": "smart"},
        },
    ]


@pytest.mark.asyncio
async def test_an_auto_router_refuses_a_prompt_compaction_cannot_fit(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(model_list=_complexity_router_over_a_small_window(), num_retries=0)
    messages: Final = [
        {"role": "user", "content": "context line about the project history and many details " * 30},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "and next?"},
    ]
    with pytest.raises(litellm.ContextWindowExceededError, match="Context compaction"):
        await core.call("acompletion", {"model": "smart", "messages": messages})
    assert host.calls_to("acompletion") == []


@pytest.mark.asyncio
async def test_the_async_chat_attempt_carries_the_compaction_state(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=0)
    await core.call("acompletion", {"model": "g", "messages": MESSAGES})
    state: Final = host.calls_to("acompletion")[0].kwargs["_context_compaction_state"]
    assert isinstance(state, CompactionState)
    assert state.surface == "chat"
    assert state.config is None


@pytest.mark.parametrize(
    ("endpoint", "request_fields"),
    [("completion", {"messages": MESSAGES}), ("atext_completion", {"prompt": "hi"})],
)
@pytest.mark.asyncio
async def test_other_attempts_carry_no_compaction_state(
    make_core: MakeCore, host: EndpointHost, endpoint: str, request_fields: Mapping[str, object]
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=0)
    await core.call(endpoint, {"model": "g", **request_fields})
    assert "_context_compaction_state" not in host.calls_to(endpoint)[0].kwargs


def test_the_routing_context_counts_messages_and_tools(make_core: MakeCore) -> None:
    ctx: Final = RoutingContext(make_core(model_list=[deployment("g", "a")]))  # pyright: ignore[reportArgumentType]
    tool: Final = {"type": "function", "function": {"name": "lookup", "description": "look it up", "parameters": {}}}
    plain: Final = ctx.count_pre_call_check_tokens(messages=MESSAGES, input=None)
    with_tools: Final = ctx.count_pre_call_check_tokens(messages=MESSAGES, input=None, request_kwargs={"tools": [tool]})
    assert 0 < plain < with_tools


def test_the_routing_context_counts_responses_input(make_core: MakeCore) -> None:
    ctx: Final = RoutingContext(make_core(model_list=[deployment("g", "a")]))  # pyright: ignore[reportArgumentType]
    assert ctx.count_pre_call_check_tokens(messages=None, input="hello there") > 0


def test_the_routing_context_needs_messages_or_input_to_count(make_core: MakeCore) -> None:
    ctx: Final = RoutingContext(make_core(model_list=[deployment("g", "a")]))  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="Either messages or input must be provided to count tokens"):
        ctx.count_pre_call_check_tokens(messages=None, input=None)


def test_the_routing_context_pre_call_checks_refuse_a_count_over_the_window(make_core: MakeCore) -> None:
    small: Final = windowed("g", "small", 5)
    ctx: Final = RoutingContext(make_core(model_list=[small]))  # pyright: ignore[reportArgumentType]
    with pytest.raises(litellm.ContextWindowExceededError, match="Context Window exceeded for given call"):
        ctx.pre_call_checks(
            model="g",
            healthy_deployments=[small],
            messages=MESSAGES,  # pyright: ignore[reportArgumentType]
            request_kwargs={},
            input_token_count=100,
            skip_inline_token_count=True,
        )


def test_the_routing_context_pre_call_checks_drop_a_deployment_outside_the_region(make_core: MakeCore) -> None:
    ctx: Final = RoutingContext(make_core(model_list=REGIONAL))  # pyright: ignore[reportArgumentType]
    kept: Final = ctx.pre_call_checks(
        model="g",
        healthy_deployments=list(REGIONAL),
        messages=MESSAGES,  # pyright: ignore[reportArgumentType]
        request_kwargs={"allowed_model_region": "ap"},
        input_token_count=1,
        skip_inline_token_count=True,
    )
    assert [record["model_info"]["id"] for record in kept] == ["ap"]  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]
