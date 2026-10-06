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


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


_OBJECTS: Final = TypeAdapter(dict[str, object])


_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])


_MESSAGES_JSON: Final = """[{"role":"user","content":[
    {"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}},
    {"type":"text","text":"question"}]}]"""


_MODELS: Final = _MESSAGES.validate_json("""[
    {"model_name":"test-router","litellm_params":{"model":"auto_router/complexity_router",
      "complexity_router_config":{"tiers":{"SIMPLE":"sonnet","MEDIUM":"sonnet","COMPLEX":"sonnet",
      "REASONING":{"model_name":"opus","litellm_params":{"max_tokens":16}}},"session_affinity":false,
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
    context: Final = _OBJECTS.validate_json(
        '{"max_tokens":16,"litellm_metadata":{"user_api_key_hash":"test-caller-hash"}}'
    )
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
            start_time=datetime.now(),
            litellm_call_id=self.call_id,
            function_id=self.call_id,
            kwargs={"litellm_session_id": "baseline-session"},
        )


def _observation(payload: Mapping[str, object]) -> CapturedBaselineObservation:
    encoded: Final = payload["autorouter_baseline_observation"]
    assert isinstance(encoded, str)
    assert "test-selected" not in encoded and "stable" not in encoded and "x-api-key" not in encoded
    return CapturedBaselineObservation.model_validate_json(encoded)


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


@pytest.mark.parametrize(
    "options",
    (
        {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        {"extra_body": {"speed": "fast", "output_config": {"effort": "high"}}},
    ),
)
async def test_native_baseline_identity_keeps_the_actual_transformed_body(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, JsonValue]
) -> None:
    rig: Final = _Rig(monkeypatch)
    log: Final = rig.logging()
    with _transport(_upstream):
        await rig.router.anthropic_messages(
            model="test-router",
            max_tokens=16,
            messages=_MESSAGES.validate_json(_MESSAGES_JSON.replace("question", "question USE_OPUS")),
            litellm_logging_obj=log,
            litellm_call_id=rig.call_id,
            litellm_metadata={"user_api_key_hash": "test-caller-hash"},
            litellm_session_id="native-identical",
            **options,
        )
        observed: Final = _observation(await rig.capture.payload()).observation
    assert observed.outcome == "complete"
    assert log.baseline_cache_context is not None
    assert observed.baseline_equivalent and observed.usage is not None, (
        log.baseline_cache_context.baseline_parameters,
        log.baseline_cache_context.selected_parameters,
    )


@pytest.mark.parametrize("tier_limit", (8, 16))
@pytest.mark.parametrize("extra", ({}, {"max_tokens": 8}))
async def test_native_baseline_identity_respects_caller_limit_and_tier_override(
    monkeypatch: pytest.MonkeyPatch, tier_limit: int, extra: dict[str, int]
) -> None:
    models: Final = _MESSAGES.validate_python(
        [
            {
                "model_name": "test-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {
                            "SIMPLE": {"model_name": "opus", "litellm_params": {"max_tokens": tier_limit}},
                            "MEDIUM": {"model_name": "opus", "litellm_params": {"max_tokens": tier_limit}},
                            "COMPLEX": "opus",
                            "REASONING": "opus",
                        },
                        "session_affinity": False,
                    },
                },
            },
            {**_MODELS[2], "litellm_params": {**_MODELS[2]["litellm_params"], "max_tokens": 64}},
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models)
    with _transport(_upstream) as route:
        await rig.router.anthropic_messages(
            model="test-router",
            max_tokens=8,
            messages=_MESSAGES.validate_json(_MESSAGES_JSON),
            litellm_logging_obj=rig.logging(),
            litellm_call_id=rig.call_id,
            litellm_metadata={"user_api_key_hash": "test-caller-hash"},
            litellm_session_id="native-limits",
            extra_body=extra,
        )
        observed: Final = _observation(await rig.capture.payload()).observation
        wire: Final = _JSON_OBJECT.validate_json(route.calls.last.request.content)
    assert wire["max_tokens"] == tier_limit
    assert observed.baseline_equivalent == (tier_limit == 8)
