"""
Characterization of the inference endpoints (completion, text completion, embedding, image,
speech, transcription, rerank, adapter completion, moderation): which host op each invokes and
which keyword arguments the core builds for it.

Every test builds its core through ``make_core`` on the ``host`` fixture, so it runs once per core.
"""

import asyncio
import io
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from tests.unit.test_router.core.endpoints_helpers import EndpointHost
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, timeout

MESSAGES: Final = [{"role": "user", "content": "hi"}]
MakeCore = Callable[..., RouterCore]


@pytest.fixture
def host() -> EndpointHost:
    return EndpointHost()


def _sent(host: ScriptedHost, index: int = -1) -> Mapping[str, object]:
    return host.calls[index].kwargs


def _bucket(kwargs: Mapping[str, object], name: str = "metadata") -> Mapping[str, object]:
    bucket: Final = kwargs[name]
    assert isinstance(bucket, Mapping)
    return bucket


# --- host op per endpoint -------------------------------------------------------------------


async def test_aadapter_completion_invokes_the_adapter_host_op_with_the_adapter_id(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.aadapter_completion(adapter_id="my-adapter", model="g", messages=MESSAGES)
    call: Final = host.calls[0]
    assert (call.op, call.model, call.kwargs["adapter_id"], call.deployment_id, call.model_group) == (
        "aadapter_completion",
        "openai/d1",
        "my-adapter",
        "d1",
        "g",
    )


async def test_amoderation_routes_a_known_group_through_its_deployment(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", api_base="https://d1.example")])
    await core.amoderation(model="g", input="x")
    call: Final = host.calls[0]
    assert (call.op, call.model, call.kwargs["api_base"], call.deployment_id, call.model_group) == (
        "amoderation",
        "openai/d1",
        "https://d1.example",
        "d1",
        "g",
    )
    assert "litellm_metadata" not in call.kwargs


async def test_amoderation_calls_the_host_directly_when_the_model_is_not_a_group(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.amoderation(model="omni-moderation-latest", input="x")
    call: Final = host.calls[0]
    assert (call.op, call.model, call.deployment_id) == ("amoderation", "omni-moderation-latest", None)
    assert _bucket(call.kwargs)["model_group"] == "omni-moderation-latest"


async def test_amoderation_deployment_params_overwrite_the_request_params(make_core: MakeCore, host: EndpointHost):
    """The deployment's litellm_params are applied after the request's (kwargs.update(data))."""
    core: Final = make_core(model_list=[deployment("g", "d1", temperature=0.5)])
    await core.amoderation(model="g", input="x", temperature=1.0)
    assert _sent(host)["temperature"] == 0.5


async def test_amoderation_with_a_request_api_key_keeps_the_deployment_key_under_a_request_scoped_id(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.amoderation(model="g", input="x", api_key="sk-request")
    sent: Final = _sent(host)
    model_info: Final = _bucket(sent, "model_info")
    assert sent["api_key"] == "sk-test"
    assert (model_info["original_model_id"], model_info["id"] != "d1") == ("d1", True)


async def test_amoderation_is_not_retried(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", litellm.InternalServerError(message="boom", llm_provider="openai", model="d1"))
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=3)
    with pytest.raises(litellm.InternalServerError):
        await core.amoderation(model="g", input="x")
    assert len(host.calls) == 1
    assert host.sleeps == []


async def test_arerank_sends_only_the_request_fields_and_no_input(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.arerank(model="g", query="q", documents=["a"])
    sent: Final = _sent(host)
    assert (sent["query"], sent["documents"], "input" in sent) == ("q", ["a"], False)


@pytest.mark.parametrize(("request_voice", "expected"), [(None, "deployment-voice"), ("alloy", "alloy")])
async def test_aspeech_voice_defaults_to_the_deployment_voice(
    make_core: MakeCore, host: EndpointHost, request_voice: str | None, expected: str
):
    core: Final = make_core(model_list=[deployment("g", "d1", voice="deployment-voice")])
    await core.aspeech(model="g", input="hello", voice=request_voice)
    assert (_sent(host)["voice"], _sent(host)["input"]) == (expected, "hello")


async def test_aspeech_sends_no_caching_flag(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], cache_responses=True)
    await core.aspeech(model="g", input="hello", voice="alloy")
    assert "caching" not in _sent(host)


async def test_atranscription_forwards_the_same_file_object(make_core: MakeCore, host: EndpointHost):
    audio: Final = io.BytesIO(b"audio")
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.atranscription(model="g", file=audio)
    assert _sent(host)["file"] is audio


@pytest.mark.parametrize(
    "endpoint",
    ["acompletion", "atext_completion", "aembedding", "aimage_generation", "atranscription", "arerank", "completion"],
)
async def test_cache_responses_reaches_the_host_as_caching(make_core: MakeCore, host: EndpointHost, endpoint: str):
    core: Final = make_core(model_list=[deployment("g", "d1")], cache_responses=True)
    requests: Final[dict[str, dict[str, object]]] = {
        "acompletion": {"messages": MESSAGES},
        "completion": {"messages": MESSAGES},
        "atext_completion": {"prompt": "hi"},
        "aembedding": {"input": ["a"]},
        "aimage_generation": {"prompt": "a cat"},
        "atranscription": {"file": io.BytesIO(b"x")},
        "arerank": {"query": "q", "documents": ["a"]},
    }
    await core.call(endpoint, {"model": "g", **requests[endpoint]})
    assert _sent(host)["caching"] is True


@pytest.mark.parametrize("endpoint", ["acompletion", "completion"])
async def test_cache_responses_turned_on_after_construction_reaches_the_host(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    """The proxy turns ``cache_responses`` on through ``apply`` once its cache exists."""
    from litellm.router_api import ChangeSet

    core: Final = make_core(model_list=[deployment("g", "d1")])
    _ = core.apply(ChangeSet(assign={"cache_responses": True}))
    await core.call(endpoint, {"model": "g", "messages": MESSAGES})
    assert core.settings().cache_responses is True
    assert _sent(host)["caching"] is True


# --- sync text completion and image generation ----------------------------------------------


async def test_sync_text_completion_sends_the_router_num_retries_and_no_routing_fields(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=4)
    core.text_completion(model="g", prompt="p")
    sent: Final = _sent(host)
    assert sent["num_retries"] == 4
    assert "model_info" not in sent
    assert "litellm_trace_id" not in sent
    assert sent.get("timeout") is None
    metadata: Final = _bucket(sent)
    assert metadata["model_group"] == "g"
    assert "deployment" not in metadata


async def test_sync_text_completion_request_num_retries_wins(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=4)
    core.text_completion(model="g", prompt="p", num_retries=1)
    assert _sent(host)["num_retries"] == 1


async def test_sync_text_completion_merges_default_metadata_into_request_metadata(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(
        model_list=[deployment("g", "d1")], default_litellm_params={"metadata": {"x": 1}, "user": "u"}
    )
    core.text_completion(model="g", prompt="p", metadata={"y": 2})
    sent: Final = _sent(host)
    metadata: Final = _bucket(sent)
    assert (metadata["x"], metadata["y"], metadata["model_group"], sent["user"]) == (1, 2, "g", "u")


async def test_sync_text_completion_sends_the_model_group_name_not_the_deployment_model(
    make_core: MakeCore, host: EndpointHost
):
    """The request's model="g" is spread after the deployment params, so it replaces "openai/d1"."""
    core: Final = make_core(model_list=[deployment("g", "d1", api_base="https://d1.example")])
    core.text_completion(model="g", prompt="p")
    call: Final = host.calls[0]
    assert (call.model, call.kwargs["api_base"]) == ("g", "https://d1.example")


async def test_sync_text_completion_is_not_retried(make_core: MakeCore, host: EndpointHost):
    host.answer("g", litellm.InternalServerError(message="boom", llm_provider="openai", model="d1"))
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=3)
    with pytest.raises(litellm.InternalServerError):
        core.text_completion(model="g", prompt="p")
    assert len(host.calls) == 1
    assert host.sleeps == []


async def test_sync_image_generation_sends_routing_metadata_without_a_trace_id(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    core.image_generation(model="g", prompt="p")
    sent: Final = _sent(host)
    assert "litellm_trace_id" not in sent
    assert (_bucket(sent)["model_group"], _bucket(sent)["deployment"], sent["prompt"]) == ("g", "openai/d1", "p")


@pytest.mark.parametrize("endpoint", ["acompletion", "atext_completion", "aembedding", "aresponses"])
async def test_async_endpoints_generate_a_trace_id_unless_the_request_sends_one(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    request: Final[dict[str, object]] = (
        {"messages": MESSAGES}
        if endpoint == "acompletion"
        else {"prompt": "p"}
        if endpoint == "atext_completion"
        else {"input": ["a"]}
        if endpoint == "aembedding"
        else {"input": "hi"}
    )
    await core.call(endpoint, {"model": "g", **request})
    await core.call(endpoint, {"model": "g", "litellm_trace_id": "trace-1", **request})
    generated: Final = _sent(host, 0)["litellm_trace_id"]
    assert isinstance(generated, str) and len(generated) == 36
    assert _sent(host, 1)["litellm_trace_id"] == "trace-1"


# --- routing metadata ---------------------------------------------------------------------


async def test_a_model_group_alias_is_recorded_as_the_model_group_and_its_alias(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")], model_group_alias={"al": "g"})
    await core.acompletion(model="al", messages=MESSAGES)
    metadata: Final = _bucket(_sent(host))
    assert (metadata["model_group"], metadata["model_group_alias"], metadata["deployment_model_name"]) == (
        "al",
        "al",
        "g",
    )


async def test_a_request_for_a_deployment_id_routes_to_that_deployment(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1"), deployment("g", "d2")])
    for _ in range(4):
        await core.acompletion(model="d2", messages=MESSAGES)
    assert host.deployments_called() == ["d2"] * 4
    assert _bucket(_sent(host))["model_group"] == "d2"


async def test_deployment_tags_and_credential_name_join_the_request_tags(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", tags=["t1"], litellm_credential_name="cred")])
    await core.acompletion(model="g", messages=MESSAGES, metadata={"tags": ["r1"]})
    sent: Final = _sent(host)
    assert _bucket(sent)["tags"] == ["r1", "t1", "Credential: cred"]
    assert sent["litellm_credential_name"] == "cred"


async def test_daily_reset_time_is_not_sent_to_the_host(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.acompletion(model="g", messages=MESSAGES, daily_reset_time="00:00")
    assert "daily_reset_time" not in _sent(host)


async def test_deployment_model_id_param_is_sent_as_model_id(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", model_id="arn:pinned")])
    await core.acompletion(model="g", messages=MESSAGES)
    core.text_completion(model="g", prompt="p")
    assert [call.kwargs["model_id"] for call in host.calls] == ["arn:pinned", "arn:pinned"]


async def test_deployment_model_id_reaches_the_host_even_when_clientside_auth_allows_model_id(
    make_core: MakeCore, host: EndpointHost
):
    """The pin is skipped, but the deployment's litellm_params (with model_id) are still spread into the call."""
    core: Final = make_core(
        model_list=[deployment("g", "d1", model_id="arn:pinned", configurable_clientside_auth_params=["model_id"])]
    )
    await core.acompletion(model="g", messages=MESSAGES)
    assert _sent(host)["model_id"] == "arn:pinned"


async def test_request_model_id_wins_over_a_clientside_configurable_deployment_model_id(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(
        model_list=[deployment("g", "d1", model_id="arn:pinned", configurable_clientside_auth_params=["model_id"])]
    )
    await core.acompletion(model="g", messages=MESSAGES, model_id="arn:request")
    assert _sent(host)["model_id"] == "arn:request"


# --- default litellm params -------------------------------------------------------------------


async def test_default_litellm_params_metadata_merges_into_request_metadata(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], default_litellm_params={"metadata": {"team": "t"}})
    await core.acompletion(model="g", messages=MESSAGES, metadata={"trace": "x"})
    metadata: Final = _bucket(_sent(host))
    assert (metadata["team"], metadata["trace"], metadata["model_group"]) == ("t", "x", "g")


async def test_default_litellm_params_metadata_overwrites_the_same_request_metadata_key(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")], default_litellm_params={"metadata": {"team": "t"}})
    await core.acompletion(model="g", messages=MESSAGES, metadata={"team": "request"})
    assert _bucket(_sent(host))["team"] == "t"


async def test_default_litellm_params_metadata_goes_to_litellm_metadata_for_generic_endpoints(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")], default_litellm_params={"metadata": {"team": "t"}})
    await core.aresponses(model="g", input="hi")
    sent: Final = _sent(host)
    assert _bucket(sent, "litellm_metadata")["team"] == "t"
    assert "metadata" not in sent


async def test_default_litellm_params_with_none_values_are_not_sent(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], default_litellm_params={"user": None})
    await core.acompletion(model="g", messages=MESSAGES)
    assert "user" not in _sent(host)


# --- tools ----------------------------------------------------------------------------------

DEPLOYMENT_TOOL: Final = {"type": "function", "function": {"name": "dep", "parameters": {}}}
REQUEST_TOOL: Final = {"type": "function", "function": {"name": "req", "parameters": {}}}


async def test_deployment_tools_come_before_request_tools(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", tools=[DEPLOYMENT_TOOL])])
    await core.acompletion(model="g", messages=MESSAGES, tools=[REQUEST_TOOL])
    assert _sent(host)["tools"] == [DEPLOYMENT_TOOL, REQUEST_TOOL]


async def test_deployment_tools_are_sent_when_the_request_has_none(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", tools=[DEPLOYMENT_TOOL])])
    await core.acompletion(model="g", messages=MESSAGES)
    assert _sent(host)["tools"] == [DEPLOYMENT_TOOL]


@pytest.mark.parametrize(("request_choice", "expected"), [(None, "required"), ("none", "none")])
async def test_tool_choice_comes_from_the_request_else_the_deployment(
    make_core: MakeCore, host: EndpointHost, request_choice: str | None, expected: str
):
    core: Final = make_core(model_list=[deployment("g", "d1", tools=[DEPLOYMENT_TOOL], tool_choice="required")])
    extra: Final = {} if request_choice is None else {"tool_choice": request_choice}
    await core.acompletion(model="g", messages=MESSAGES, **extra)
    assert _sent(host)["tool_choice"] == expected


async def test_merging_tools_does_not_change_the_deployment_tools(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", tools=[DEPLOYMENT_TOOL])])
    await core.acompletion(model="g", messages=MESSAGES, tools=[REQUEST_TOOL])
    await core.acompletion(model="g", messages=MESSAGES)
    assert _sent(host)["tools"] == [DEPLOYMENT_TOOL]


# --- reasoning effort -----------------------------------------------------------------------


async def test_request_reasoning_effort_drops_the_deployment_thinking_param(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", thinking={"type": "enabled", "budget_tokens": 100})])
    await core.acompletion(model="g", messages=MESSAGES, reasoning_effort="low")
    sent: Final = _sent(host)
    assert (sent["reasoning_effort"], "thinking" in sent) == ("low", False)


async def test_deployment_thinking_is_kept_without_a_request_reasoning_effort(make_core: MakeCore, host: EndpointHost):
    thinking: Final = {"type": "enabled", "budget_tokens": 100}
    core: Final = make_core(model_list=[deployment("g", "d1", thinking=thinking)])
    await core.acompletion(model="g", messages=MESSAGES)
    assert _sent(host)["thinking"] == thinking


async def test_request_reasoning_effort_strips_effort_from_deployment_extra_body(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(
        model_list=[
            deployment(
                "g",
                "d1",
                extra_body={"reasoning_effort": "high", "thinking": {"type": "enabled"}, "other": 1},
                reasoning={"effort": "high", "summary": "auto"},
            )
        ]
    )
    await core.acompletion(model="g", messages=MESSAGES, reasoning_effort="low")
    sent: Final = _sent(host)
    assert sent["extra_body"] == {"other": 1}
    assert sent["reasoning"] == {"summary": "auto"}


async def test_request_reasoning_effort_drops_an_extra_body_left_empty(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", extra_body={"reasoning_effort": "high"})])
    await core.acompletion(model="g", messages=MESSAGES, reasoning_effort="low")
    assert "extra_body" not in _sent(host)


async def test_reasoning_override_does_not_change_the_deployment_config(make_core: MakeCore, host: EndpointHost):
    thinking: Final = {"type": "enabled", "budget_tokens": 100}
    core: Final = make_core(model_list=[deployment("g", "d1", thinking=thinking)])
    await core.acompletion(model="g", messages=MESSAGES, reasoning_effort="low")
    await core.acompletion(model="g", messages=MESSAGES)
    assert _sent(host)["thinking"] == thinking


CLASSIFIER_METADATA: Final = {"internal_call_origin": "autorouter_classifier"}


async def test_classifier_call_drops_reasoning_effort_the_deployment_does_not_support(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", model="openai/gpt-3.5-turbo")])
    body: Final[dict[str, object]] = {"reasoning_effort": "low", "keep": 1}
    await core.acompletion(
        model="g",
        messages=MESSAGES,
        reasoning_effort="low",
        metadata=dict(CLASSIFIER_METADATA),
        proxy_server_request={"body": body},
    )
    sent: Final = _sent(host)
    assert "reasoning_effort" not in sent
    assert sent["proxy_server_request"] == {"body": {"keep": 1}}


async def test_non_classifier_call_keeps_reasoning_effort_on_a_deployment_without_support(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", model="openai/gpt-3.5-turbo")])
    await core.acompletion(model="g", messages=MESSAGES, reasoning_effort="low")
    assert _sent(host)["reasoning_effort"] == "low"


async def test_classifier_call_keeps_reasoning_effort_on_a_deployment_with_support(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", model="openai/o3-mini")])
    await core.acompletion(
        model="g", messages=MESSAGES, reasoning_effort="low", litellm_metadata=dict(CLASSIFIER_METADATA)
    )
    assert _sent(host)["reasoning_effort"] == "low"


# --- silent experiments ---------------------------------------------------------------------


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def test_silent_model_mirrors_acompletion_to_the_silent_group(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", silent_model="shadow"), deployment("shadow", "s1")])
    await core.acompletion(model="g", messages=MESSAGES, metadata={"trace": "t"})
    await _settle()
    assert sorted(host.deployments_called("acompletion")) == ["d1", "s1"]
    primary: Final = next(call for call in host.calls if call.deployment_id == "d1")
    shadow: Final = next(call for call in host.calls if call.deployment_id == "s1")
    assert "silent_model" not in primary.kwargs
    assert "is_silent_experiment" not in _bucket(primary.kwargs)
    shadow_metadata: Final = _bucket(shadow.kwargs)
    assert (shadow_metadata["is_silent_experiment"], shadow_metadata["model_group"], shadow_metadata["trace"]) == (
        True,
        "shadow",
        "t",
    )


async def test_silent_model_list_mirrors_to_every_listed_group(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(
        model_list=[
            deployment("g", "d1", silent_model=["shadow-a", "shadow-b"]),
            deployment("shadow-a", "sa"),
            deployment("shadow-b", "sb"),
        ]
    )
    await core.acompletion(model="g", messages=MESSAGES)
    await _settle()
    assert sorted(host.deployments_called("acompletion")) == ["d1", "sa", "sb"]


async def test_a_failing_silent_experiment_does_not_fail_the_primary_call(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/s1", litellm.BadRequestError(message="shadow broke", llm_provider="openai", model="s1"))
    core: Final = make_core(model_list=[deployment("g", "d1", silent_model="shadow"), deployment("shadow", "s1")])
    response: Final = await core.acompletion(model="g", messages=MESSAGES)
    await _settle()
    assert response is not None
    assert "s1" in host.deployments_called("acompletion")


async def test_a_malformed_silent_model_is_ignored(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", silent_model=5)])
    await core.acompletion(model="g", messages=MESSAGES)
    await _settle()
    assert host.deployments_called() == ["d1"]
    assert "silent_model" not in _sent(host)


# --- timeouts -------------------------------------------------------------------------------


@pytest.mark.parametrize(("stream", "expected"), [(False, (99.0, None)), (True, (99.0, 33.0))])
async def test_service_tier_timeouts_come_from_the_deployment_tier_maps(
    make_core: MakeCore, host: EndpointHost, stream: bool, expected: tuple[float, float | None]
):
    core: Final = make_core(
        model_list=[
            deployment(
                "g", "d1", timeout=11, timeout_by_service_tier={"flex": 99}, stream_timeout_by_service_tier={"flex": 33}
            )
        ]
    )
    await core.acompletion(model="g", messages=MESSAGES, service_tier="flex", stream=stream)
    sent: Final = _sent(host)
    assert (sent["timeout"], sent.get("stream_timeout")) == expected


async def test_router_stream_timeout_applies_to_streaming_acompletion(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11)], stream_timeout=5)
    await core.acompletion(model="g", messages=MESSAGES, stream=True)
    sent: Final = _sent(host)
    assert (sent["timeout"], sent["stream_timeout"]) == (11.0, 5.0)


async def test_non_streaming_acompletion_sends_no_stream_timeout(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", stream_timeout=3)])
    await core.acompletion(model="g", messages=MESSAGES)
    assert "stream_timeout" not in _sent(host) or _sent(host)["stream_timeout"] == 3


async def test_aembedding_uses_the_request_timeout_over_the_deployment_timeout(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11)])
    await core.aembedding(model="g", input=["a"], request_timeout=4)
    assert _sent(host)["timeout"] == 4


async def test_timeout_message_carries_deployment_timeouts_when_router_debug_is_exposed(
    make_core: MakeCore, host: EndpointHost
):
    host.set_dynamic(expose_router_debug_in_errors=True)
    host.answer("openai/d1", timeout("d1"))
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11, stream_timeout=3)], num_retries=0)
    with pytest.raises(litellm.Timeout) as raised:
        await core.acompletion(model="g", messages=MESSAGES)
    assert "Deployment Info: request_timeout: None\ntimeout: 11.0\nstream_timeout: 3.0" in str(raised.value)


async def test_timeout_message_has_no_deployment_info_by_default(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", timeout("d1"))
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11)], num_retries=0)
    with pytest.raises(litellm.Timeout) as raised:
        await core.acompletion(model="g", messages=MESSAGES)
    assert "Deployment Info" not in str(raised.value)


# --- clientside credentials -------------------------------------------------------------------


@pytest.mark.parametrize("credential", [{"api_key": "sk-client"}, {"api_base": "https://client.example"}])
async def test_clientside_credentials_get_a_request_scoped_deployment_id(
    make_core: MakeCore, host: EndpointHost, credential: dict[str, str]
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.acompletion(model="g", messages=MESSAGES, **credential)
    sent: Final = _sent(host)
    model_info: Final = _bucket(sent, "model_info")
    assert model_info["original_model_id"] == "d1"
    assert isinstance(model_info["id"], str) and len(model_info["id"]) == 64
    assert _bucket(_bucket(sent), "model_info")["id"] == model_info["id"]


async def test_clientside_credential_id_is_stable_for_the_same_credentials(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.acompletion(model="g", messages=MESSAGES, api_key="sk-a")
    await core.acompletion(model="g", messages=MESSAGES, api_key="sk-a")
    await core.acompletion(model="g", messages=MESSAGES, api_key="sk-b")
    ids: Final = [call.deployment_id for call in host.calls]
    assert ids[0] == ids[1] != ids[2]


async def test_clientside_credentials_do_not_register_a_deployment(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.acompletion(model="g", messages=MESSAGES, api_key="sk-client")
    await core.acompletion(model="g", messages=MESSAGES)
    assert host.deployments_called()[-1] == "d1"
    assert [d["model_info"]["id"] for d in core.registry().deployments("g")] == ["d1"]  # pyright: ignore[reportIndexIssue, reportUnknownMemberType]  # registry rows are deployment dicts


# --- embedding batch size --------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["aembedding", "embedding"])
async def test_embedding_over_the_deployment_batch_size_raises_without_calling_the_host(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    core: Final = make_core(
        model_list=[deployment("g", "d1", model_info={"max_embedding_batch_size": 2})], num_retries=0
    )
    with pytest.raises(
        litellm.BadRequestError,
        match="Embedding request has 3 inputs, which exceeds max_embedding_batch_size=2 for model=g",
    ):
        await core.call(endpoint, {"model": "g", "input": ["a", "b", "c"]})
    assert host.calls == []


@pytest.mark.parametrize("endpoint", ["aembedding", "embedding"])
async def test_embedding_within_the_deployment_batch_size_is_sent(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    core: Final = make_core(model_list=[deployment("g", "d1", model_info={"max_embedding_batch_size": 2})])
    await core.call(endpoint, {"model": "g", "input": ["a", "b"]})
    assert host.calls[0].kwargs["input"] == ["a", "b"]
