import asyncio
import json
from collections.abc import AsyncIterator, Callable, Generator, Mapping
from contextlib import contextmanager
from datetime import datetime
from types import MappingProxyType
from typing import Final, cast
from uuid import uuid4

import httpx
import pytest
import respx
from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget, TokenCounter
from litellm.proxy.hooks.autorouter_baseline_cache import AutoRouterBaselineCache, CapturedBaselineObservation
from litellm.router import Router
from litellm.types.router import RetryPolicy
from litellm.types.utils import CallTypes, StandardLoggingRoutingDecision

pytestmark: Final = pytest.mark.asyncio
_STARTED: Final = datetime(2026, 1, 1)


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


_OBJECTS: Final = TypeAdapter(dict[str, object])


_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])


_MESSAGES_JSON: Final = """[{"role":"user","content":[
    {"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}},
    {"type":"text","text":"question"}]}]"""


_MODELS: Final = _MESSAGES.validate_json("""[
    {"model_name":"test-router","litellm_params":{"model":"auto_router/complexity_router",
      "complexity_router_config":{"tiers":{"SIMPLE":"sonnet","MEDIUM":"sonnet","COMPLEX":"sonnet",
      "REASONING":"opus"},"session_affinity":false,
      "keyword_tier_rules":[{"keywords":["USE_OPUS"],"tier":"REASONING"}]}}},
    {"model_name":"sonnet","litellm_params":{"model":"anthropic/claude-sonnet-5","api_key":"test-selected"},
      "model_info":{"id":"selected"}},
    {"model_name":"opus","litellm_params":{"model":"anthropic/claude-opus-5","api_key":"test-selected"},
      "model_info":{"id":"baseline"}}]""")


def _message(completed: bool, model: str) -> Mapping[str, JsonValue]:
    return _JSON_OBJECT.validate_json(f"""{{
        "id":"msg_baseline_test","type":"message","role":"assistant","model":{json.dumps(model)},
        "content":{'[{"type":"text","text":"OK"}]' if completed else "[]"},
        "stop_reason":{'"end_turn"' if completed else "null"},"stop_sequence":null,
        "usage":{{"input_tokens":1000,"output_tokens":{10 if completed else 0},
          "cache_creation_input_tokens":5000,"cache_read_input_tokens":0,
          "cache_creation":{{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":5000}}}}}}""")


_EVENTS: Final = _MESSAGES.validate_json("""[
    {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}},
    {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK"}},
    {"type":"content_block_stop","index":0},
    {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}},
    {"type":"message_stop"}
]""")


async def _count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
    assert model == "claude-opus-5"
    return 6000 if "question" in json.dumps(_JSON_OBJECT.validate_python(body)) else 5000


class _CallContext(TypedDict):
    litellm_logging_obj: NotRequired[ReadOnly[Logging]]
    litellm_call_id: ReadOnly[str]
    litellm_metadata: ReadOnly[Mapping[str, object]]
    litellm_session_id: ReadOnly[str]


def _kwargs(logging_obj: Logging, trusted: bool = True, *, explicit_logging: bool = True) -> _CallContext:
    context: Final = _OBJECTS.validate_json('{"litellm_metadata":{"user_api_key_hash":"test-caller-hash"}}')
    Router._record_routing_decision(  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # production trusted stamp owner
        context,
        StandardLoggingRoutingDecision(
            router_model_name="test-router",
            router_type="complexity",
            routed_model="sonnet",
            cause="heuristic_scorer",
            conversation_continuing=True,
            savings_baseline_model="anthropic/claude-opus-5",
            savings_baseline_deployment_id="baseline",
        ),
    )
    metadata: Final = _OBJECTS.validate_python(context["litellm_metadata"])
    if not trusted:
        metadata["_autorouter_baseline_route"] = _JSON_OBJECT.validate_json(
            '{"router_name":"test-router","baseline_model":"anthropic/claude-opus-5","baseline_deployment_id":"baseline"}'
        )
    envelope: Final[_CallContext] = {
        "litellm_call_id": logging_obj.litellm_call_id,
        "litellm_session_id": "baseline-session",
        "litellm_metadata": metadata,
    }
    supplied: Final[_CallContext] = {**envelope, "litellm_logging_obj": logging_obj}
    return supplied if explicit_logging else envelope


def _stream(logging_obj: Logging) -> bool:
    return logging_obj.stream is True  # pyright: ignore[reportUnknownMemberType]  # normalize the legacy Logging flag


def _sse(completed: bool = True, model: str = "claude-sonnet-5") -> tuple[bytes, ...]:
    events: Final = (
        {
            "type": "message_start",
            "message": _message(False, model),
        },
        *_EVENTS,
    )
    return tuple(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        for event in (events if completed else events[:-1])
    )


def _upstream(request: httpx.Request) -> httpx.Response:
    body: Final = _JSON_OBJECT.validate_json(request.content)
    model: Final = body.get("model")
    assert isinstance(model, str)
    stream: Final = body.get("stream") is True
    content: Final = b"".join(_sse(model=model)) if stream else json.dumps(_message(True, model)).encode()
    return httpx.Response(
        200,
        content=content,
        request=request,
        headers=MappingProxyType({"content-type": "text/event-stream" if stream else "application/json"}),
    )


def _error(request: httpx.Request, code: int, message: str) -> httpx.Response:
    return httpx.Response(
        code,
        text='{"type":"error","error":{"type":"rate_limit_error","message":' + json.dumps(message) + "}}",
        headers=MappingProxyType({"retry-after": "0"}),
        request=request,
    )


@contextmanager
def _transport(upstream: Callable[[httpx.Request], httpx.Response]) -> Generator[respx.Route]:
    with respx.mock() as transport:
        yield transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=upstream)


class _NativeOptions(TypedDict):
    api_key: NotRequired[ReadOnly[str]]
    num_retries: NotRequired[ReadOnly[int]]


async def _call(
    target: Router | None,
    logging_obj: Logging,
    *,
    trusted: bool = True,
    messages: str = _MESSAGES_JSON,
    explicit_logging: bool = True,
) -> None:
    invoke: Final = target.anthropic_messages if target else litellm.anthropic_messages  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy native call signatures
    options: Final = _NativeOptions() if target else _NativeOptions(api_key="test-selected", num_retries=0)
    response: Final[object] = await invoke(  # pyright: ignore[reportUnknownVariableType]  # native Router returns an opaque SDK result
        model="test-router" if target else "anthropic/claude-sonnet-5",
        max_tokens=16,
        stream=_stream(logging_obj),
        messages=_MESSAGES.validate_json(messages),
        **options,
        **_kwargs(logging_obj, trusted, explicit_logging=explicit_logging),
    )
    assert response is not None
    if _stream(logging_obj):
        assert isinstance(response, AsyncIterator)
        stream: Final = cast(AsyncIterator[object], response)  # cast-ok: iterator checked; all items satisfy object
        assert tuple([chunk async for chunk in stream])


class _Capture(CustomLogger):
    def __init__(self, call_id: str) -> None:
        self.call_id: Final = call_id
        self.payloads: Final[asyncio.Queue[Mapping[str, object]]] = asyncio.Queue()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        payload: Final = _OBJECTS.validate_python(kwargs.get("standard_logging_object"))
        if payload.get("litellm_call_id") == self.call_id:
            self.payloads.put_nowait(payload)

    async def payload(self) -> Mapping[str, object]:
        return await asyncio.wait_for(self.payloads.get(), timeout=20)


class _Rig:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        retries: int = 0,
        count: TokenCounter = _count,
        models: list[dict[str, JsonValue]] = _MODELS,
    ) -> None:
        self.router: Final = Router(
            model_list=models,
            num_retries=retries,
            retry_policy=RetryPolicy(RateLimitErrorRetries=retries),
            disable_cooldowns=True,
        )

        def router() -> Router:
            return self.router

        self.hook: Final = AutoRouterBaselineCache(None, router=router, token_counter=count)
        self.call_id: Final = uuid4().hex
        self.capture: Final = _Capture(self.call_id)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        for name in ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(litellm, "callbacks", [self.hook])
        for name in ("success_callback", "failure_callback", "_async_failure_callback"):
            monkeypatch.setattr(litellm, name, [])
        monkeypatch.setattr(litellm, "_async_success_callback", [self.capture])

    def logging(self, stream: bool = False) -> Logging:
        return Logging(
            model="anthropic/claude-sonnet-5",
            messages=_MESSAGES.validate_json(_MESSAGES_JSON),
            stream=stream,
            call_type=CallTypes.anthropic_messages.value,
            start_time=_STARTED,
            litellm_call_id=self.call_id,
            function_id=self.call_id,
            kwargs={"litellm_session_id": "baseline-session"},
        )


def _observation(payload: Mapping[str, object]) -> CapturedBaselineObservation:
    from litellm.proxy.db.baseline_accounting import BaselineAccountingRecord
    from litellm.proxy.spend_tracking.savings import BaselineCostSnapshot

    encoded: Final = payload["autorouter_baseline_observation"]
    assert isinstance(encoded, str)
    assert "test-selected" not in encoded and "stable" not in encoded and "x-api-key" not in encoded
    captured: Final = CapturedBaselineObservation.model_validate_json(encoded)
    record: Final = BaselineAccountingRecord(
        scope=captured.scope,
        api_key=captured.api_key,
        session_id=captured.session_id,
        router_name=captured.router_name,
        baseline_model=captured.baseline_model,
        observation=captured.observation,
        pricing=BaselineCostSnapshot(
            model=captured.model,
            provider=captured.provider,
            prices=captured.prices,
            actual_spend=0.0,
            actual_token_cost=None,
        ),
        turn=None,
        daily=None,
    )
    assert record.observation == captured.observation
    return captured


@pytest.mark.parametrize("stream,baseline", ((False, False), (True, False), (False, True), (True, True)))
async def test_native_logging_captures_usage_without_publishing_hypothetical_savings(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    baseline: bool,
) -> None:
    rig: Final = _Rig(monkeypatch)
    messages: Final = _MESSAGES_JSON.replace("question", "question USE_OPUS") if baseline else _MESSAGES_JSON
    with _transport(_upstream):
        await _call(rig.router, rig.logging(stream), messages=messages)
        payload: Final = await rig.capture.payload()
    captured: Final = _observation(payload)
    assert payload["autorouter_savings"] is None
    assert _OBJECTS.validate_python(payload["autorouter_savings_estimate"])["reason"] == "pending_projection"
    assert captured.observation.outcome == "complete"
    assert captured.observation.baseline_equivalent == baseline
    assert captured.observation.usage is not None and captured.observation.usage.completion_tokens == 10
    assert captured.observation.plan is not None and captured.observation.plan.total_tokens == 6000


async def test_count_failure_preserves_initial_observed_equivalence(monkeypatch: pytest.MonkeyPatch) -> None:
    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        return None

    rig: Final = _Rig(monkeypatch, count=count)
    with _transport(_upstream):
        await _call(rig.router, rig.logging(), messages=_MESSAGES_JSON.replace("question", "question USE_OPUS"))
        captured: Final = _observation(await rig.capture.payload())
    assert captured.observation.baseline_equivalent and captured.observation.usage is not None
    assert captured.observation.plan is None and captured.observation.reason == "token_count_unavailable"


async def test_native_retry_is_uncertain_even_when_final_response_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    rig: Final = _Rig(monkeypatch, retries=1)

    def upstream(request: httpx.Request) -> httpx.Response:
        return _upstream(request) if route.call_count else _error(request, 429, "retry")

    with _transport(upstream) as route:
        await _call(rig.router, rig.logging())
        captured: Final = _observation(await rig.capture.payload())
        assert route.call_count == 2
    assert captured.observation.outcome == "uncertain"
    assert captured.observation.reason == "retried_request"


async def test_caller_cannot_forge_an_observation_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    rig: Final = _Rig(monkeypatch)
    with _transport(_upstream):
        await _call(None, rig.logging(), trusted=False)
        payload: Final = await rig.capture.payload()
    assert payload["autorouter_baseline_observation"] is None
    assert payload["autorouter_savings"] is None


@pytest.mark.parametrize(
    "model,key,endpoint",
    (
        ("claude-sonnet-5", "test-first", None),
        ("claude-opus-5", "test-second", None),
        ("claude-opus-5", "test-first", "https://example.test"),
    ),
)
async def test_count_memo_is_scoped_to_provider_recipient(model: str, key: str, endpoint: str | None) -> None:
    counts: Final = iter((5000, 6000))

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        return next(counts)

    collector: Final = AutoRouterBaselineCache(None, token_counter=count)
    original: Final = NativePredictionTarget("claude-opus-5", "test-first")
    other: Final = NativePredictionTarget(model, key, endpoint)
    assert await collector._count(original, {}) == 5000  # pyright: ignore[reportPrivateUsage]
    assert await collector._count(other, {}) == 6000  # pyright: ignore[reportPrivateUsage]
    assert await collector._count(original, {}) == 5000  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("stream", (False, True))
async def test_provider_counting_does_not_hold_the_inference_response(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    counting: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        counting.set()
        await release.wait()
        return await _count(model, api_key, body)

    rig: Final = _Rig(monkeypatch, count=count)
    try:
        with _transport(_upstream):
            await asyncio.wait_for(_call(rig.router, rig.logging(stream)), timeout=2)
            await asyncio.wait_for(counting.wait(), timeout=2)
            assert rig.capture.payloads.empty()
            release.set()
            assert _observation(await rig.capture.payload()).observation.plan is not None
    finally:
        release.set()


@pytest.mark.parametrize("surface", ("chat", "responses", "normalized_responses", "messages"))
async def test_direct_openai_baseline_collects_provider_usage_across_api_surfaces(surface: str) -> None:
    from litellm.proxy.hooks.autorouter_baseline_cache import finalize_baseline_cache
    from litellm.proxy.spend_tracking.baseline_accounting import BaselineHistory, advance_baseline_history
    from litellm.types.utils import ModelResponse, Usage

    call_type: Final = {
        "chat": CallTypes.acompletion,
        "responses": CallTypes.aresponses,
        "normalized_responses": CallTypes.aresponses,
        "messages": CallTypes.anthropic_messages,
    }[surface]
    prompt: Final = "shared stable content " * 2000
    body: Final = (
        {"input": [{"role": "user", "content": prompt}]}
        if "responses" in surface
        else {"messages": [{"role": "user", "content": prompt}]}
    )
    request: Final[dict[str, object]] = {
        **body,
        "litellm_metadata": {"user_api_key_hash": "test-key", "session_id": "test-session"},
    }
    Router._record_routing_decision(
        request,
        StandardLoggingRoutingDecision(
            router_model_name="test-router",
            router_type="complexity",
            routed_model="openai/gpt-6.1-sol",
            savings_baseline_model="openai/gpt-6-astra",
        ),
    )
    now: Final = datetime(2026, 1, 1)
    logging: Final = Logging(
        model="openai/gpt-6.1-sol",
        messages=[],
        stream=False,
        call_type=call_type.value,
        start_time=now,
        litellm_call_id="openai-observation",
        function_id="test",
        kwargs={},
    )
    collector: Final = AutoRouterBaselineCache(None, router=lambda: None, clock=lambda: now.timestamp() + 1)
    await collector.async_pre_call_deployment_hook({**request, "litellm_logging_obj": logging}, call_type)
    assert logging.baseline_cache_context is not None
    if surface == "messages":
        await collector.async_pre_call_deployment_hook(
            {**request, "litellm_logging_obj": logging}, CallTypes.aresponses
        )
    usage: Final = Usage(
        prompt_tokens=10000,
        completion_tokens=10,
        total_tokens=10010,
        prompt_tokens_details={"cached_tokens": 4000, "cache_creation_tokens": 6000},
    )
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

    normalized: Final = ResponsesAPIResponse(
        id="resp-test",
        created_at=1,
        output=[],
        usage=ResponseAPIUsage(input_tokens=10000, output_tokens=10, total_tokens=10010),
    ).model_copy(update={"usage": usage})
    response: Final = (
        normalized
        if surface == "normalized_responses"
        else (
            {
                "usage": {
                    "input_tokens": 10000,
                    "output_tokens": 10,
                    "total_tokens": 10010,
                    "input_tokens_details": {"cached_tokens": 4000, "cache_creation_tokens": 6000},
                }
            }
            if surface == "responses"
            else ModelResponse(usage=usage)
        )
    )
    await finalize_baseline_cache(logging, response)
    captured: Final = logging.baseline_observation
    assert captured is not None and captured.provider == "openai"
    assert captured.observation.outcome == "complete", captured.observation.reason
    assert captured.observation.usage is not None and captured.observation.usage.prompt_tokens == 10000
    assert captured.observation.usage.prompt_tokens_details.cached_tokens == 4000
    assert captured.observation.plan is not None
    assert prompt not in captured.model_dump_json()
    _, estimates = advance_baseline_history(BaselineHistory(), (captured.observation,))
    assert estimates[0].usage is not None and estimates[0].usage.prompt_tokens_details.cached_tokens == 0
    assert estimates[0].usage.prompt_tokens_details.cache_creation_tokens == 10000


async def test_openai_chat_success_logging_publishes_captured_history(monkeypatch: pytest.MonkeyPatch) -> None:
    call_id: Final = uuid4().hex
    capture: Final = _Capture(call_id)
    collector: Final = AutoRouterBaselineCache(None, router=lambda: None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [collector])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    for name in ("success_callback", "failure_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    request: Final[dict[str, object]] = {
        "litellm_metadata": {"user_api_key_hash": "test-key", "session_id": "test-session"}
    }
    Router._record_routing_decision(
        request,
        StandardLoggingRoutingDecision(
            router_model_name="test-router",
            router_type="complexity",
            routed_model="openai/gpt-6.1-sol",
            savings_baseline_model="openai/gpt-6-astra",
        ),
    )
    with respx.mock(assert_all_called=True) as transport:
        transport.post("https://api.openai.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-baseline",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-6.1-sol",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10000,
                    "completion_tokens": 10,
                    "total_tokens": 10010,
                    "prompt_tokens_details": {"cached_tokens": 4000, "cache_creation_tokens": 6000},
                },
            },
        )
        await litellm.acompletion(
            model="openai/gpt-6.1-sol",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            messages=[{"role": "user", "content": "stable content " * 3000}],
            litellm_call_id=call_id,
            **request,
        )
        payload: Final = await capture.payload()
    captured: Final = _observation(payload)
    assert captured.observation.outcome == "complete" and captured.observation.plan is not None
    assert (
        captured.observation.usage is not None
        and captured.observation.usage.prompt_tokens_details.cache_creation_tokens == 6000
    )
    assert _OBJECTS.validate_python(payload["autorouter_savings_estimate"])["reason"] == "pending_projection"
    assert payload["autorouter_savings"] is None


async def test_estimated_capture_defers_token_counting_until_completion_off_event_loop() -> None:
    from queue import SimpleQueue
    from threading import get_ident

    from litellm.proxy.hooks.autorouter_baseline_cache import finalize_baseline_cache
    from litellm.types.utils import ModelResponse, Usage

    threads: Final[SimpleQueue[int]] = SimpleQueue()

    def count(model: str, text: str) -> int:
        threads.put(get_ident())
        return len(text)

    collector: Final = AutoRouterBaselineCache(
        None, router=lambda: None, prefix_token_counter=count, clock=lambda: _STARTED.timestamp() + 1
    )
    logging: Final = Logging(
        model="gpt-6.1-sol",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        call_type=CallTypes.acompletion.value,
        start_time=_STARTED,
        litellm_call_id=uuid4().hex,
        function_id=uuid4().hex,
    )
    request: Final[dict[str, object]] = {
        "messages": [{"role": "user", "content": "a long stable prompt " * 1000}],
        "litellm_logging_obj": logging,
        "litellm_metadata": {"user_api_key_hash": "test-key", "session_id": "test-session"},
    }
    Router._record_routing_decision(
        request,
        StandardLoggingRoutingDecision(
            router_model_name="test-router",
            router_type="complexity",
            routed_model="openai/gpt-6.1-sol",
            savings_baseline_model="openai/gpt-6-astra",
        ),
    )
    await collector.async_pre_call_deployment_hook(request, CallTypes.acompletion)
    assert logging.baseline_cache_context is not None
    assert threads.empty()
    response: Final = ModelResponse(usage=Usage(prompt_tokens=5000, completion_tokens=1, total_tokens=5001))
    await asyncio.gather(*(finalize_baseline_cache(logging, response) for _ in range(3)))
    assert not threads.empty()
    counted: Final = tuple(threads.get_nowait() for _ in range(threads.qsize()))
    assert len(counted) == 4
    assert all(thread != get_ident() for thread in counted)
    captured: Final = logging.baseline_observation
    assert captured is not None and captured.observation.plan is not None
    assert captured.observation.plan.total_tokens == 5000
    await finalize_baseline_cache(logging, response)
    assert logging.baseline_observation is captured and threads.empty()


async def test_estimator_capacity_remains_bounded_when_caller_is_cancelled() -> None:
    from threading import Event

    from litellm.proxy.spend_tracking.cache_history import prepare_cache_request

    loop: Final = asyncio.get_running_loop()
    entered: Final = asyncio.Queue[None]()
    release: Final = Event()

    def count(model: str, text: str) -> int:
        loop.call_soon_threadsafe(entered.put_nowait, None)
        assert release.wait(timeout=5), "test did not release token counting"
        return len(text)

    collector: Final = AutoRouterBaselineCache(None, prefix_token_counter=count)
    prepared: Final = prepare_cache_request(
        {"messages": [{"role": "user", "content": "text"}]}, "model", "other", None, {}
    )
    assert prepared is not None
    pending: Final = tuple(asyncio.create_task(collector.estimate(prepared, "model")) for _ in range(8))
    try:
        await asyncio.wait_for(entered.get(), timeout=5)
        await asyncio.wait_for(entered.get(), timeout=5)
        assert await collector.estimate(prepared, "model") is None
        pending[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending[0]
        assert await collector.estimate(prepared, "model") is None
    finally:
        release.set()
        await asyncio.gather(*pending, return_exceptions=True)
    assert all(task.result() is not None for task in pending[1:])


@pytest.mark.parametrize("stream", (False, True))
async def test_saturated_estimator_preserves_success_spend_payloads_before_logging_deadline(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    from threading import Event

    from litellm.litellm_core_utils.logging_worker import LoggingWorker
    from litellm.proxy.hooks.autorouter_baseline_cache import finalize_baseline_cache
    from litellm.types.utils import ModelResponse, Usage

    release: Final = Event()

    def count(model: str, text: str) -> int:
        assert release.wait(timeout=5), "test did not release estimator"
        return len(text)

    collector: Final = AutoRouterBaselineCache(
        None, router=lambda: None, prefix_token_counter=count, clock=lambda: _STARTED.timestamp() + 1
    )
    worker: Final = LoggingWorker(timeout=1.0, concurrency=8)
    logs: Final = tuple(
        Logging(
            model="openai/gpt-6.1-sol",
            messages=[{"role": "user", "content": "prompt"}],
            stream=stream,
            call_type=CallTypes.acompletion.value,
            start_time=_STARTED,
            litellm_call_id=uuid4().hex,
            function_id=uuid4().hex,
        )
        for _ in range(8)
    )
    captures: Final = tuple(_Capture(log.litellm_call_id) for log in logs)
    first_observations: Final[asyncio.Queue[tuple[str, CapturedBaselineObservation | None]]] = asyncio.Queue()

    async def duplicate(log: Logging, response: ModelResponse) -> None:
        await finalize_baseline_cache(log, response)
        first_observations.put_nowait((log.litellm_call_id, log.baseline_observation))

    async def publish(log: Logging, response: ModelResponse) -> None:
        await asyncio.gather(duplicate(log, response), log.async_success_handler(result=response))

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "_async_success_callback", list(captures))
    monkeypatch.setattr(litellm, "success_callback", [])
    for log in logs:
        request: Final[dict[str, object]] = {
            "messages": [{"role": "user", "content": "prompt"}],
            "litellm_logging_obj": log,
            "litellm_metadata": {"user_api_key_hash": "test-key", "session_id": "test-session"},
        }
        Router._record_routing_decision(
            request,
            StandardLoggingRoutingDecision(
                router_model_name="test-router",
                router_type="complexity",
                routed_model="openai/gpt-6.1-sol",
                savings_baseline_model="openai/gpt-6-astra",
            ),
        )
        await collector.async_pre_call_deployment_hook(request, CallTypes.acompletion)
        response: Final = ModelResponse(
            model="gpt-6.1-sol",
            usage=Usage(
                prompt_tokens=5000,
                completion_tokens=10,
                total_tokens=5010,
            ),
        )
        response._hidden_params["response_cost"] = 0.75
        worker.ensure_initialized_and_enqueue(publish(log, response))
    try:
        payloads: Final = await asyncio.wait_for(asyncio.gather(*(capture.payload() for capture in captures)), 3)
        observations: Final = tuple(log.baseline_observation for log in logs)
        assert len(payloads) == len(logs)
        assert all(payload["response_cost"] == 0.75 for payload in payloads)
        assert all(_observation(payload).observation.reason == "baseline_estimation_timeout" for payload in payloads)
        assert not release.is_set()
        assert worker._timeout_total == 0
        first: Final = dict(first_observations.get_nowait() for _ in logs)
        assert all(log.baseline_observation is first[log.litellm_call_id] for log in logs)
    finally:
        release.set()
        await asyncio.gather(*(log.baseline_cache_context.finalization for log in logs if log.baseline_cache_context))
        await worker.stop()
    assert tuple(log.baseline_observation for log in logs) == observations
    assert all(capture.payloads.empty() for capture in captures)


@pytest.mark.parametrize("worker_timeout", (8.0, 20.0))
@pytest.mark.parametrize("stream", (False, True))
async def test_native_count_timeout_preserves_observed_usage_and_session_equivalence(
    monkeypatch: pytest.MonkeyPatch,
    worker_timeout: float,
    stream: bool,
) -> None:
    from litellm.litellm_core_utils.logging_worker import LoggingWorker
    from litellm.proxy.spend_tracking.baseline_accounting import BaselineHistory, advance_baseline_history

    release: Final = asyncio.Event()

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        await release.wait()
        return await _count(model, api_key, body)

    from litellm.litellm_core_utils import logging_worker

    worker: Final = LoggingWorker(timeout=worker_timeout)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", worker)
    rig: Final = _Rig(monkeypatch, count=count)
    logging: Final = rig.logging(stream)
    try:
        with _transport(_upstream):
            await _call(rig.router, logging, messages=_MESSAGES_JSON.replace("question", "question USE_OPUS"))
            payload: Final = await rig.capture.payload()
        captured: Final = _observation(payload).observation
        assert captured.outcome == "complete" and captured.baseline_equivalent
        assert captured.usage is not None and captured.usage.completion_tokens == 10
        assert captured.plan is None and captured.reason == "token_count_timeout"
        assert payload["response_cost"] is not None and worker._timeout_total == 0
        history, estimates = advance_baseline_history(BaselineHistory(), (captured,))
        assert history.equivalent and estimates[0].provenance == "observed_identical"
        _, subsequent = advance_baseline_history(
            history,
            (
                captured.model_copy(
                    update={
                        "request_id": "next",
                        "started_at": captured.available_at + 1,
                        "available_at": captured.available_at + 2,
                    }
                ),
            ),
        )
        assert subsequent[0].provenance == "observed_identical" and subsequent[0].usage == captured.usage
        first: Final = logging.baseline_observation
    finally:
        release.set()
        if logging.baseline_cache_context and logging.baseline_cache_context.finalization:
            await logging.baseline_cache_context.finalization
        await worker.stop()
    assert logging.baseline_observation is first


@pytest.mark.parametrize("deployment_baseline", (False, True))
@pytest.mark.parametrize("setting", ("reasoning_effort", "verbosity"))
async def test_router_tier_effort_switch_reuses_baseline_prefix(
    monkeypatch: pytest.MonkeyPatch,
    deployment_baseline: bool,
    setting: str,
) -> None:
    from datetime import timedelta

    from litellm.proxy.spend_tracking.baseline_accounting import BaselineHistory, advance_baseline_history

    baseline: Final = "baseline" if deployment_baseline else "openai/gpt-6-astra"
    models: Final = _MESSAGES.validate_python(
        [
            {
                "model_name": "test-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {
                            "SIMPLE": {"model_name": "selected", "litellm_params": {setting: "low"}},
                            "MEDIUM": {"model_name": "selected", "litellm_params": {setting: "low"}},
                            "COMPLEX": {"model_name": "selected", "litellm_params": {setting: "high"}},
                            "REASONING": baseline,
                        },
                        "session_affinity": False,
                        "keyword_tier_rules": [{"keywords": ["ESCALATE"], "tier": "COMPLEX"}],
                    },
                },
            },
            {"model_name": "selected", "litellm_params": {"model": "openai/gpt-6.1-sol", "api_key": "test-key"}},
            *(
                [
                    {
                        "model_name": "baseline",
                        "litellm_params": {
                            "model": "openai/gpt-6-astra",
                            "api_key": "test-key",
                        },
                        "model_info": {"id": "baseline"},
                    }
                ]
                if deployment_baseline
                else []
            ),
        ]
    )
    router: Final = Router(model_list=models, num_retries=0)
    collector: Final = AutoRouterBaselineCache(None, router=lambda: router, clock=lambda: _STARTED.timestamp() + 10)
    captures: Final = tuple(_Capture(uuid4().hex) for _ in range(2))
    monkeypatch.setattr(litellm, "callbacks", [collector])
    monkeypatch.setattr(litellm, "_async_success_callback", list(captures))
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    observations: Final[asyncio.Queue[CapturedBaselineObservation]] = asyncio.Queue()
    prefix: Final[list[dict[str, JsonValue]]] = [{"role": "user", "content": "stable content " * 2000}]
    with respx.mock() as transport:
        route: Final = transport.post("https://api.openai.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-tier",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-6.1-sol",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9000, "completion_tokens": 1, "total_tokens": 9001},
            },
        )
        for index, capture in enumerate(captures):
            messages: Final = (
                prefix
                if index == 0
                else [
                    *prefix,
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": "ESCALATE"},
                ]
            )
            logging: Final = Logging(
                model="gpt-6.1-sol",
                messages=messages,
                stream=False,
                call_type=CallTypes.acompletion.value,
                start_time=_STARTED + timedelta(seconds=index * 20),
                litellm_call_id=capture.call_id,
                function_id=capture.call_id,
            )
            await router.acompletion(
                model="test-router",
                messages=messages,
                litellm_logging_obj=logging,
                litellm_call_id=capture.call_id,
                litellm_metadata={"user_api_key_hash": "test-caller-hash"},
                litellm_session_id="tier-switch",
            )
            observations.put_nowait(_observation(await capture.payload()))
        efforts: Final = tuple(_JSON_OBJECT.validate_json(call.request.content)[setting] for call in route.calls)
    assert efforts == ("low", "high")
    first, second = (observations.get_nowait() for _ in captures)
    assert first.scope == second.scope
    history, initial = advance_baseline_history(BaselineHistory(), (first.observation,))
    _, reused = advance_baseline_history(history, (second.observation,))
    assert initial[0].usage is not None and initial[0].usage.prompt_tokens_details.cached_tokens == 0
    assert reused[0].usage is not None and reused[0].usage.prompt_tokens_details.cached_tokens > 0


@pytest.mark.parametrize("baseline_effort", (None, "medium"))
async def test_native_tier_switch_uses_baseline_settings_and_preserves_history(
    monkeypatch: pytest.MonkeyPatch,
    baseline_effort: str | None,
) -> None:
    from litellm.proxy.spend_tracking.baseline_accounting import BaselineHistory, advance_baseline_history

    models: Final = _MESSAGES.validate_python(
        [
            {
                "model_name": "test-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {
                            "SIMPLE": {"model_name": "sonnet", "litellm_params": {"reasoning_effort": "low"}},
                            "MEDIUM": {"model_name": "sonnet", "litellm_params": {"reasoning_effort": "low"}},
                            "COMPLEX": {"model_name": "sonnet", "litellm_params": {"reasoning_effort": "high"}},
                            "REASONING": "opus",
                        },
                        "session_affinity": False,
                        "keyword_tier_rules": [{"keywords": ["ESCALATE"], "tier": "COMPLEX"}],
                    },
                },
            },
            _MODELS[1],
            {
                "model_name": "opus",
                "model_info": {"id": "baseline"},
                "litellm_params": {
                    "model": "anthropic/claude-opus-5",
                    "api_key": "test-selected",
                    **({"reasoning_effort": baseline_effort} if baseline_effort else {}),
                },
            },
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models)
    captures: Final[asyncio.Queue[CapturedBaselineObservation]] = asyncio.Queue()
    with _transport(_upstream) as route:
        for suffix in ("", " ESCALATE"):
            log: Final = rig.logging()
            await rig.router.anthropic_messages(
                model="test-router",
                max_tokens=4096,
                messages=_MESSAGES.validate_json(_MESSAGES_JSON.replace("question", "question" + suffix)),
                litellm_logging_obj=log,
                litellm_call_id=rig.call_id,
                litellm_metadata={"user_api_key_hash": "test-caller-hash"},
                litellm_session_id="native-tiers",
            )
            captures.put_nowait(_observation(await rig.capture.payload()))
        first_wire, second_wire = (_JSON_OBJECT.validate_json(call.request.content) for call in route.calls)
    assert (first_wire.get("thinking"), first_wire.get("output_config")) != (
        second_wire.get("thinking"),
        second_wire.get("output_config"),
    )
    first, second = (captures.get_nowait() for _ in range(2))
    assert first.scope == second.scope
    assert first.observation.plan is not None and second.observation.plan is not None
    assert first.observation.plan.breakpoints == second.observation.plan.breakpoints
    history, _ = advance_baseline_history(
        BaselineHistory(),
        (first.observation.model_copy(update={"request_id": "first", "started_at": 1000.0, "available_at": 1001.0}),),
    )
    _, result = advance_baseline_history(
        history,
        (second.observation.model_copy(update={"request_id": "second", "started_at": 1020.0, "available_at": 1021.0}),),
    )
    assert result[0].usage is not None and result[0].usage.prompt_tokens_details.cached_tokens == 5000


@pytest.mark.parametrize("call_type", (CallTypes.acompletion, CallTypes.aresponses, CallTypes.anthropic_messages))
async def test_plain_requests_do_not_initialize_or_warn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    call_type: CallTypes,
) -> None:
    rig: Final = _Rig(monkeypatch)
    logging: Final = rig.logging()
    await rig.hook.async_pre_call_deployment_hook(
        {
            "litellm_logging_obj": logging,
            "litellm_metadata": {"session_id": "ordinary"},
        },
        call_type,
    )
    assert logging.baseline_cache_context is None
    assert "baseline observation could not be initialized" not in caplog.text
    assert not rig.hook.counts


async def test_plain_fallback_invalidates_existing_autorouter_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    rig: Final = _Rig(monkeypatch)
    logging: Final = rig.logging()
    await rig.hook.async_pre_call_deployment_hook(_kwargs(logging), CallTypes.anthropic_messages)
    assert logging.baseline_cache_context is not None
    await rig.hook.async_pre_call_deployment_hook({"litellm_logging_obj": logging}, CallTypes.anthropic_messages)
    assert logging.baseline_observation is not None
    assert logging.baseline_observation.observation.reason == "retried_request"


async def test_native_count_finishing_after_quarter_worker_budget_keeps_plan_and_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.litellm_core_utils import logging_worker
    from litellm.litellm_core_utils.logging_worker import LoggingWorker

    release: Final = asyncio.Event()

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        if not release.is_set():
            asyncio.get_running_loop().call_later(2.3, release.set)
            await release.wait()
        return await _count(model, api_key, body)

    worker: Final = LoggingWorker(timeout=8.0)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", worker)
    rig: Final = _Rig(monkeypatch, count=count)
    try:
        with _transport(_upstream):
            await _call(rig.router, rig.logging())
            payload: Final = await rig.capture.payload()
        observed: Final = _observation(payload).observation
        assert observed.outcome == "complete" and observed.reason is None
        assert observed.plan is not None and observed.plan.breakpoints[0].prefix_tokens == 5000
        assert payload["response_cost"] is not None and worker._timeout_total == 0
    finally:
        release.set()
        await worker.stop()
