"""
What a router core sends to its host for each endpoint, and what it hands back.

Every test builds its core through ``make_core`` on the ``host`` fixture, so it runs once per core.
"""

import datetime
import io
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router_core import RouterCore
from litellm.types.utils import EmbeddingResponse, ModelResponse, ModelResponseStream
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    bad_request,
    completion_response,
    deployment,
    embedding_response,
)

MESSAGES: Final = [{"role": "user", "content": "hi"}]
MakeCore = Callable[..., RouterCore]

EndpointCall = Callable[[RouterCore], object]

ENDPOINTS: Final[dict[str, tuple[str, EndpointCall]]] = {
    "acompletion": ("acompletion", lambda core: core.acompletion(model="g", messages=MESSAGES)),
    "completion": ("completion", lambda core: core.completion(model="g", messages=MESSAGES)),
    "aembedding": ("aembedding", lambda core: core.aembedding(model="g", input=["a"])),
    "embedding": ("embedding", lambda core: core.embedding(model="g", input=["a"])),
    "atext_completion": ("atext_completion", lambda core: core.atext_completion(model="g", prompt="hi")),
    "aimage_generation": ("aimage_generation", lambda core: core.aimage_generation(model="g", prompt="a cat")),
    "image_generation": ("image_generation", lambda core: core.image_generation(model="g", prompt="a cat")),
    "arerank": ("arerank", lambda core: core.arerank(model="g", query="q", documents=["a"])),
    "aspeech": ("aspeech", lambda core: core.aspeech(model="g", input="hi", voice="alloy")),
    "atranscription": ("atranscription", lambda core: core.atranscription(model="g", file=io.BytesIO(b"x"))),
    "aresponses": ("aresponses", lambda core: core.aresponses(model="g", input="hi")),
    "aanthropic_messages": (
        "anthropic_messages",
        lambda core: core.aanthropic_messages(model="g", messages=MESSAGES, max_tokens=5),
    ),
    "call-acompletion": ("acompletion", lambda core: core.call("acompletion", {"model": "g", "messages": MESSAGES})),
    "call-completion": ("completion", lambda core: core.call("completion", {"model": "g", "messages": MESSAGES})),
}
SYNC_ENDPOINTS: Final = frozenset({"completion", "embedding", "image_generation"})
LITELLM_METADATA_ENDPOINTS: Final = frozenset({"aresponses", "aanthropic_messages"})
EMBEDDING_ENDPOINTS: Final = frozenset({"aembedding", "embedding"})


async def _invoke(core: RouterCore, endpoint: str) -> object:
    result: Final = ENDPOINTS[endpoint][1](core)
    if endpoint in SYNC_ENDPOINTS:
        return result
    return await result  # pyright: ignore[reportGeneralTypeIssues]  # the async endpoints return awaitables


def _one_group(
    make_core: MakeCore, **litellm_params: object
) -> RouterCore:  # kwargs-ok: the deployment's litellm_params
    return make_core(model_list=[deployment("g", "d1", **litellm_params)])


def _metadata(kwargs: Mapping[str, object], endpoint: str) -> Mapping[str, object]:
    bucket: Final = kwargs["litellm_metadata" if endpoint in LITELLM_METADATA_ENDPOINTS else "metadata"]
    assert isinstance(bucket, Mapping)
    return bucket


@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
async def test_each_endpoint_sends_the_deployment_litellm_params_to_the_host(
    make_core: MakeCore, host: ScriptedHost, endpoint: str
):
    core: Final = _one_group(make_core, api_base="https://d1.example", temperature=0.3, seed=7)
    await _invoke(core, endpoint)
    sent: Final = host.calls[0].kwargs
    assert (sent["model"], sent["api_key"], sent["api_base"], sent["temperature"], sent["seed"]) == (
        "openai/d1",
        "sk-test",
        "https://d1.example",
        0.3,
        7,
    )


@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
async def test_each_endpoint_invokes_its_host_op_once_with_routing_metadata(
    make_core: MakeCore, host: ScriptedHost, endpoint: str
):
    core: Final = _one_group(make_core, api_base="https://d1.example")
    await _invoke(core, endpoint)
    assert [call.op for call in host.calls] == [ENDPOINTS[endpoint][0]]
    call: Final = host.calls[0]
    metadata: Final = _metadata(call.kwargs, endpoint)
    assert (call.model_group, call.deployment_id) == ("g", "d1")
    assert (metadata["model_group"], metadata["deployment"], metadata["api_base"]) == (
        "g",
        "openai/d1",
        "https://d1.example",
    )
    assert metadata["model_info"]["id"] == "d1"  # pyright: ignore[reportIndexIssue]  # metadata values are untyped


@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
async def test_each_endpoint_returns_the_host_response_object_with_routing_hidden_params(
    make_core: MakeCore, host: ScriptedHost, endpoint: str
):
    answer: Final[ModelResponse | EmbeddingResponse] = (
        embedding_response() if endpoint in EMBEDDING_ENDPOINTS else completion_response("scripted")
    )
    host.answer("openai/d1", answer)
    core: Final = _one_group(make_core)
    response: Final = await _invoke(core, endpoint)
    assert response is answer
    assert response._hidden_params["model_id"] == "d1"
    assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "g"


@pytest.mark.parametrize("endpoint", sorted(SYNC_ENDPOINTS | {"call-completion"}))
async def test_sync_endpoints_return_the_response_not_an_awaitable(
    make_core: MakeCore, host: ScriptedHost, endpoint: str
):
    answer: Final = embedding_response() if endpoint == "embedding" else completion_response("sync")
    host.answer("openai/d1", answer)
    core: Final = _one_group(make_core)
    if endpoint == "call-completion":
        assert await core.call("completion", {"model": "g", "messages": MESSAGES}) is answer
    else:
        assert ENDPOINTS[endpoint][1](core) is answer


@pytest.mark.parametrize(
    "endpoint, request_body",
    [("acompletion", {"messages": MESSAGES}), ("completion", {"messages": MESSAGES})],
)
async def test_router_only_include_fallback_errors_is_not_sent_to_the_host(
    make_core: MakeCore, host: ScriptedHost, endpoint: str, request_body: dict[str, object]
):
    core: Final = _one_group(make_core)
    await core.call(endpoint, {"model": "g", "include_fallback_errors": True, **request_body})
    assert host.calls[0].op == endpoint
    assert "include_fallback_errors" not in host.calls[0].kwargs


@pytest.mark.parametrize("router_only_param", ["rpm", "tpm", "max_parallel_requests"])
async def test_deployment_limit_params_are_forwarded_with_the_rest_of_litellm_params(
    make_core: MakeCore, host: ScriptedHost, router_only_param: str
):
    core: Final = _one_group(make_core, **{router_only_param: 50})
    await core.acompletion(model="g", messages=MESSAGES)
    assert host.calls[0].kwargs[router_only_param] == 50


@pytest.mark.parametrize("endpoint", ["acompletion", "aembedding", "completion", "embedding"])
async def test_request_metadata_reaches_the_host_beside_the_routing_metadata(
    make_core: MakeCore, host: ScriptedHost, endpoint: str
):
    core: Final = _one_group(make_core)
    request: Final[dict[str, object]] = {"messages": MESSAGES} if "completion" in endpoint else {"input": ["a"]}
    await core.call(endpoint, {"model": "g", "metadata": {"trace": "t-1"}, **request})
    metadata: Final = _metadata(host.calls[0].kwargs, endpoint)
    assert (metadata["trace"], metadata["model_group"]) == ("t-1", "g")


async def test_aresponses_sends_request_metadata_as_metadata_and_routing_metadata_as_litellm_metadata(
    make_core: MakeCore, host: ScriptedHost
):
    core: Final = _one_group(make_core)
    await core.aresponses(model="g", input="hi", metadata={"trace": "t-1"})
    sent: Final = host.calls[0].kwargs
    assert sent["metadata"] == {"trace": "t-1"}
    assert sent["litellm_metadata"]["model_group"] == "g"  # pyright: ignore[reportIndexIssue]  # untyped kwargs


@pytest.mark.parametrize(
    "router_arguments, deployment_params, expected_timeout",
    [
        ({}, {}, 6000.0),
        ({"timeout": 42}, {}, 42),
        ({"timeout": 42}, {"timeout": 11}, 11.0),
        ({}, {"timeout": 11}, 11.0),
    ],
)
async def test_acompletion_timeout_comes_from_deployment_then_router_then_process_setting(
    make_core: MakeCore,
    host: ScriptedHost,
    router_arguments: dict[str, object],
    deployment_params: dict[str, object],
    expected_timeout: float,
):
    core: Final = make_core(model_list=[deployment("g", "d1", **deployment_params)], **router_arguments)
    await core.acompletion(model="g", messages=MESSAGES)
    assert host.calls[0].kwargs["timeout"] == expected_timeout


@pytest.mark.parametrize("endpoint", ["acompletion", "aembedding", "aresponses", "aanthropic_messages"])
async def test_request_timeout_overrides_router_and_deployment_timeouts(
    make_core: MakeCore, host: ScriptedHost, endpoint: str
):
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11)], timeout=42)
    request: Final[dict[str, object]] = (
        {"input": ["a"]}
        if endpoint == "aembedding"
        else {"input": "hi"}
        if endpoint == "aresponses"
        else {"messages": MESSAGES, "max_tokens": 5}
    )
    await core.call(endpoint, {"model": "g", "timeout": 7, **request})
    assert host.calls[0].kwargs["timeout"] == 7


async def test_aresponses_timeout_falls_back_to_router_timeout(make_core: MakeCore, host: ScriptedHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], timeout=42)
    await core.aresponses(model="g", input="hi")
    assert host.calls[0].kwargs["timeout"] == 42


@pytest.mark.parametrize(
    "request_stream_timeout, expected_stream_timeout",
    [(None, 3.0), (2, 2)],
)
async def test_streaming_acompletion_sends_deployment_or_request_stream_timeout(
    make_core: MakeCore, host: ScriptedHost, request_stream_timeout: int | None, expected_stream_timeout: float
):
    core: Final = _one_group(make_core, timeout=11, stream_timeout=3)
    extra: Final = {} if request_stream_timeout is None else {"stream_timeout": request_stream_timeout}
    await core.acompletion(model="g", messages=MESSAGES, stream=True, **extra)
    sent: Final = host.calls[0].kwargs
    assert (sent["stream"], sent["timeout"], sent["stream_timeout"]) == (True, 11.0, expected_stream_timeout)


@pytest.mark.parametrize(
    "request_params, expected",
    [
        ({}, ("u1", 0.9)),
        ({"temperature": 0.1}, ("u1", 0.1)),
        ({"user": "u2"}, ("u2", 0.9)),
    ],
)
async def test_default_litellm_params_fill_what_the_request_leaves_unset(
    make_core: MakeCore, host: ScriptedHost, request_params: dict[str, object], expected: tuple[str, float]
):
    core: Final = make_core(
        model_list=[deployment("g", "d1")], default_litellm_params={"user": "u1", "temperature": 0.9}
    )
    await core.acompletion(model="g", messages=MESSAGES, **request_params)
    sent: Final = host.calls[0].kwargs
    assert (sent["user"], sent["temperature"]) == expected


@pytest.mark.parametrize(
    "request_credentials, expected_api_base",
    [
        ({"api_key": "sk-client"}, "https://d1.example"),
        ({"api_key": "sk-client", "api_base": "https://client.example"}, "https://client.example"),
    ],
)
async def test_request_level_api_key_replaces_the_deployment_key_on_the_host_call(
    make_core: MakeCore, host: ScriptedHost, request_credentials: dict[str, str], expected_api_base: str
):
    core: Final = _one_group(make_core, api_base="https://d1.example")
    await core.acompletion(model="g", messages=MESSAGES, **request_credentials)
    call: Final = host.calls[0]
    assert (call.kwargs["api_key"], call.kwargs["api_base"], call.model_group) == (
        "sk-client",
        expected_api_base,
        "g",
    )


async def test_construction_registers_one_success_and_one_failure_callback_per_mode(
    make_core: MakeCore, host: ScriptedHost
):
    _one_group(make_core)
    registered: Final = {name: len(callbacks) for name, callbacks in host.lists.items()}
    assert registered == {
        "callbacks": 0,
        "success_callback": 1,
        "_async_success_callback": 1,
        "failure_callback": 1,
        "_async_failure_callback": 1,
        "input_callback": 0,
    }
    assert all(callable(callbacks[0]) for callbacks in host.lists.values() if callbacks)


async def test_retire_removes_the_callbacks_the_core_registered(make_core: MakeCore, host: ScriptedHost):
    core: Final = _one_group(make_core)
    core.retire()
    assert {name: callbacks for name, callbacks in host.lists.items() if callbacks} == {}


async def test_sync_completion_counts_its_tokens_and_request_in_group_usage(make_core: MakeCore, host: ScriptedHost):
    core: Final = _one_group(make_core, tpm=100, rpm=5)
    assert await core.health().usage("g") == (None, None)
    core.completion(model="g", messages=MESSAGES)
    assert await core.health().usage("g") == (10, 1)
    assert await core.health().usage("g", kind="remaining") == {
        "x-ratelimit-remaining-tokens": 90,
        "x-ratelimit-limit-tokens": 100,
        "x-ratelimit-remaining-requests": 4,
        "x-ratelimit-limit-requests": 5,
    }


class DeferredLoggingHost(ScriptedHost):
    """``ScriptedHost`` that, like litellm, runs async success callbacks after the endpoint returns."""

    __slots__ = ("pending",)

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[Coroutine[object, object, None]] = []

    async def _answer_async(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        callbacks: Final = self.lists["_async_success_callback"]
        self.lists["_async_success_callback"] = []
        try:
            response: Final = await super()._answer_async(op, args, kwargs)
        finally:
            self.lists["_async_success_callback"] = callbacks
        call: Final = self.calls[-1]
        logged: Final = self._logging_kwargs(call, None, response)
        self.pending.extend(callback(logged, response, 0.0, 0.0) for callback in callbacks if callable(callback))
        return response

    async def drain(self) -> None:
        while self.pending:
            await self.pending.pop(0)


class TestUsageWhenTheHostLogsAfterTheResponse:
    """litellm runs async success callbacks after the endpoint has returned; this host does the same."""

    @pytest.fixture
    def host(self) -> DeferredLoggingHost:
        return DeferredLoggingHost()

    async def test_one_acompletion_counts_its_tokens_and_request_once(
        self, make_core: MakeCore, host: DeferredLoggingHost
    ):
        core: Final = _one_group(make_core, tpm=100, rpm=5)
        await core.acompletion(model="g", messages=MESSAGES)
        assert await core.health().usage("g") == (10, 1)
        await host.drain()
        assert await core.health().usage("g") == (10, 1)


async def test_usage_based_routing_registers_its_logger_on_the_host(make_core: MakeCore, host: ScriptedHost):
    make_core(model_list=[deployment("g", "d1", rpm=1)], routing_strategy="usage-based-routing-v2")
    assert [type(callback).__name__ for callback in host.lists["callbacks"]] == ["LowestTPMLoggingHandler_v2"]


async def test_a_successful_call_uses_up_the_deployment_rpm_for_the_next_selection(
    make_core: MakeCore, host: ScriptedHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", rpm=1)], routing_strategy="usage-based-routing-v2")
    await core.acompletion(model="g", messages=MESSAGES)
    with pytest.raises(litellm.RateLimitError, match="No deployments available"):
        await core.acompletion(model="g", messages=MESSAGES)
    assert host.deployments_called() == ["d1"]


async def test_host_error_reaches_the_caller_as_raised(make_core: MakeCore, host: ScriptedHost):
    error: Final = bad_request(message="scripted bad input")
    host.answer("openai/d1", error)
    core: Final = _one_group(make_core)
    with pytest.raises(litellm.BadRequestError, match="scripted bad input"):
        await core.acompletion(model="g", messages=MESSAGES)
    assert host.deployments_called("acompletion")[0] == "d1"


@pytest.mark.parametrize("endpoint", ["acompletion", "completion"])
async def test_unknown_model_group_raises_without_calling_the_host(
    make_core: MakeCore, host: ScriptedHost, endpoint: str
):
    core: Final = _one_group(make_core)
    with pytest.raises(litellm.BadRequestError, match="model=missing"):
        await core.call(endpoint, {"model": "missing", "messages": MESSAGES})
    assert host.calls == []


@pytest.mark.parametrize("op", ["no_such_endpoint", "acompletions", ""])
async def test_call_with_an_unknown_op_raises_attribute_error_without_calling_the_host(
    make_core: MakeCore, host: ScriptedHost, op: str
):
    core: Final = _one_group(make_core)
    with pytest.raises(AttributeError):
        await core.call(op, {"model": "g"})
    assert host.calls == []


async def test_sync_text_completion_invokes_the_sync_host_op_with_deployment_credentials(
    make_core: MakeCore, host: ScriptedHost
):
    core: Final = _one_group(make_core, api_base="https://d1.example")
    core.text_completion(model="g", prompt="hi")
    call: Final = host.calls[0]
    assert (call.op, call.kwargs["api_base"], call.kwargs["prompt"], call.model_group) == (
        "text_completion",
        "https://d1.example",
        "hi",
        "g",
    )


def _scripted_stream(*texts: str) -> Callable[..., CustomStreamWrapper]:
    def answer(**_request: object) -> CustomStreamWrapper:  # kwargs-ok: the host passes the request kwargs
        async def chunks() -> AsyncIterator[ModelResponseStream]:
            for text in texts:
                yield ModelResponseStream(choices=[{"index": 0, "delta": {"role": "assistant", "content": text}}])

        logging_obj: Final = LiteLLMLogging(
            model="d1",
            messages=MESSAGES,
            stream=True,
            call_type="acompletion",
            start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            litellm_call_id="stream-test",
            function_id="stream-test",
        )
        logging_obj.update_environment_variables(model="d1", user=None, optional_params={}, litellm_params={})
        return CustomStreamWrapper(
            completion_stream=chunks(), model="d1", custom_llm_provider="openai", logging_obj=logging_obj
        )

    return answer


async def test_streaming_acompletion_yields_the_host_stream_chunks_with_routing_hidden_params(
    make_core: MakeCore, host: ScriptedHost
):
    host.answer("openai/d1", _scripted_stream("he", "llo"))
    core: Final = _one_group(make_core)
    stream: Final = await core.acompletion(model="g", messages=MESSAGES, stream=True)
    texts: Final = [chunk.choices[0].delta.content async for chunk in stream]  # pyright: ignore[reportUnknownMemberType]  # stream chunks are untyped
    assert "".join(text for text in texts if text) == "hello"
    assert stream._hidden_params["model_id"] == "d1"
    assert stream._hidden_params["additional_headers"]["x-litellm-model-group"] == "g"
    assert (host.calls[0].op, host.calls[0].kwargs["stream"]) == ("acompletion", True)
