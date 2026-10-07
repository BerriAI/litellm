import asyncio
import json
from collections.abc import AsyncIterator, Callable, Generator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta
from itertools import product
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
    logging._update_completion_start_time(now + timedelta(seconds=0.25))
    collector: Final = AutoRouterBaselineCache(None, router=lambda: None, clock=lambda: now.timestamp() + 10)
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
    history, estimates = advance_baseline_history(BaselineHistory(), (captured.observation,))
    assert estimates[0].usage is not None and estimates[0].usage.prompt_tokens_details.cached_tokens == 0
    assert estimates[0].usage.prompt_tokens_details.cache_creation_tokens == 10000
    _, followup = advance_baseline_history(
        history,
        (
            captured.observation.model_copy(
                update={
                    "request_id": "followup",
                    "started_at": now.timestamp() + 1,
                    "available_at": now.timestamp() + 2,
                }
            ),
        ),
    )
    assert followup[0].usage is not None and followup[0].usage.prompt_tokens_details.cached_tokens == 10000


@pytest.mark.parametrize("baseline_effort", (None, "medium"))
@pytest.mark.parametrize(
    "automatic_system, caching",
    (
        (None, "explicit"),
        ("stable system", "request"),
        ([{"type": "text", "text": "stable system"}], "request"),
        ("stable system", "global"),
        ([{"type": "text", "text": "stable system"}], "configured"),
    ),
)
async def test_native_tier_switch_uses_baseline_settings_and_preserves_history(
    monkeypatch: pytest.MonkeyPatch,
    baseline_effort: str | None,
    automatic_system: str | list[dict[str, str]] | None,
    caching: str,
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
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", caching == "global")
    controls: Final = (
        {
            "cache_control_injection_points": [
                {"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}},
                {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "1h"}},
            ]
        }
        if caching == "configured"
        else {"enable_prompt_caching": caching == "request"}
    )
    captures: Final[asyncio.Queue[CapturedBaselineObservation]] = asyncio.Queue()
    with _transport(_upstream) as route:
        for suffix in ("", " ESCALATE"):
            log: Final = rig.logging()
            await rig.router.anthropic_messages(
                model="test-router",
                max_tokens=4096,
                messages=(
                    [{"role": "user", "content": "question" + suffix}]
                    if automatic_system is not None
                    else _MESSAGES.validate_json(_MESSAGES_JSON.replace("question", "question" + suffix))
                ),
                system=automatic_system,
                **controls,
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
    assert first.observation.plan.breakpoints[0] == second.observation.plan.breakpoints[0]
    assert len(first.observation.plan.breakpoints) == (2 if automatic_system is not None else 1)
    history, _ = advance_baseline_history(
        BaselineHistory(first_at=0.0),
        (first.observation.model_copy(update={"request_id": "first", "started_at": 10000.0, "available_at": 10001.0}),),
    )
    _, result = advance_baseline_history(
        history,
        (
            second.observation.model_copy(
                update={"request_id": "second", "started_at": 10020.0, "available_at": 10021.0}
            ),
        ),
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
            asyncio.get_running_loop().call_later(2.05, release.set)
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
    "options,on_deployment",
    (
        ({"thinking": {"type": "enabled", "budget_tokens": 2048}}, False),
        ({"extra_body": {"speed": "fast", "output_config": {"effort": "high"}}}, False),
        *product(
            (
                {"container": {"id": "container_test"}},
                {"mcp_servers": [{"type": "url", "name": "test", "url": "https://example.com/mcp"}]},
                {"inference_geo": "us"},
                {"safeguards": [{"type": "default"}]},
            ),
            (False, True),
        ),
    ),
)
async def test_native_baseline_identity_keeps_the_actual_transformed_body(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, JsonValue], on_deployment: bool
) -> None:
    models: Final = _MESSAGES.validate_python(
        [
            *_MODELS[:2],
            {
                **_MODELS[2],
                "litellm_params": {
                    **_JSON_OBJECT.validate_python(_MODELS[2]["litellm_params"]),
                    **(options if on_deployment else {}),
                },
            },
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models)
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
            **({} if on_deployment else options),
        )
        observed: Final = _observation(await rig.capture.payload()).observation
    assert observed.outcome == "complete"
    assert log.baseline_cache_context is not None
    assert observed.baseline_equivalent and observed.usage is not None, (
        log.baseline_cache_context.baseline_body,
        log.baseline_cache_context.selected_body_digest,
    )

    from litellm.proxy.spend_tracking.baseline_accounting import BaselineHistory, advance_baseline_history

    _, estimates = advance_baseline_history(BaselineHistory(), (observed,))
    assert estimates[0].provenance == "observed_identical" and estimates[0].usage == observed.usage


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


@pytest.mark.parametrize("nested", (False, True))
async def test_native_baseline_projection_matches_wire_parameter_placement(
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
) -> None:
    counted: Final[asyncio.Queue[Mapping[str, JsonValue]]] = asyncio.Queue()

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        counted.put_nowait(body)
        return await _count(model, api_key, body)

    models: Final = _MESSAGES.validate_python(
        [
            {
                **entry,
                "litellm_params": {
                    **_JSON_OBJECT.validate_python(entry["litellm_params"]),
                    "model": "anthropic/claude-opus-5",
                },
            }
            if entry["model_name"] == "sonnet"
            else entry
            for entry in _MODELS
        ]
    )
    rig: Final = _Rig(monkeypatch, count=count, models=models)
    settings: Final = {"speed": "standard", "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    with _transport(_upstream) as route:
        await rig.router.anthropic_messages(
            model="test-router",
            max_tokens=4096,
            messages=_MESSAGES.validate_json(_MESSAGES_JSON),
            litellm_logging_obj=rig.logging(),
            litellm_call_id=rig.call_id,
            litellm_metadata={"user_api_key_hash": "test-caller-hash"},
            litellm_session_id="native-placement",
            **({"extra_body": settings} if nested else settings),
        )
        captured: Final = _observation(await rig.capture.payload())
        wire: Final = _JSON_OBJECT.validate_json(route.calls.last.request.content)
    assert captured.observation.plan is not None and not captured.observation.baseline_equivalent
    projected: Final = counted.get_nowait()
    assert {key: projected[key] for key in settings if key in projected} == {
        key: wire[key] for key in settings if key in wire
    }
    assert {key: wire[key] for key in settings if key in wire} == ({} if nested else settings)


_PARITY_TOOL: Final = {"name": "custom", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}
_PARITY_SYSTEM: Final = [{"type": "text", "text": "stable system", "cache_control": {"type": "ephemeral"}}]
_PARITY_POINTS: Final = [{"location": "message", "role": "system"}, {"location": "message", "index": -1}]


@pytest.mark.parametrize(
    "caller,selected,baseline,summary",
    (
        pytest.param({"extra_body": {"cache_control": {"type": "ephemeral"}}}, {}, {}, False, id="envelope-control"),
        pytest.param({"extra_body": {"system": _PARITY_SYSTEM}}, {}, {}, False, id="envelope-system"),
        pytest.param(
            {"extra_body": {"messages": _MESSAGES.validate_json(_MESSAGES_JSON)}}, {}, {}, False, id="envelope-messages"
        ),
        pytest.param({}, {"tools": [_PARITY_TOOL]}, {}, False, id="selected-tool-mark"),
        pytest.param({}, {}, {"tools": [_PARITY_TOOL]}, False, id="baseline-tool-mark"),
        pytest.param({}, {"system": _PARITY_SYSTEM}, {"system": "baseline system"}, False, id="selected-system-mark"),
        pytest.param({}, {"system": "selected system"}, {"system": _PARITY_SYSTEM}, False, id="baseline-system-mark"),
        pytest.param({"system": None}, {}, {"system": "configured system"}, False, id="null-system"),
        pytest.param({"thinking": None}, {}, {"thinking": {"type": "adaptive"}}, False, id="null-thinking"),
        pytest.param({"tools": None}, {}, {"tools": [_PARITY_TOOL]}, False, id="null-tools"),
        pytest.param({"verbosity": "low", "instructions": "ignored"}, {}, {}, False, id="ignored-native-options"),
        pytest.param(
            {"messages": _MESSAGES.validate_json(_MESSAGES_JSON.replace("stable", " "))},
            {},
            {},
            False,
            id="empty-marked-block",
        ),
        pytest.param({"thinking": {"type": "adaptive"}}, {}, {}, True, id="reasoning-summary"),
        pytest.param(
            {"thinking": {"type": "adaptive"}, "additional_drop_params": ["thinking.display"]},
            {},
            {},
            True,
            id="drop-nested-option",
        ),
        pytest.param(
            {"cache_control_injection_points": _PARITY_POINTS},
            {"tools": [{**_PARITY_TOOL, "name": f"custom_{index}"} for index in range(4)]},
            {},
            False,
            id="configured-cap",
        ),
    ),
)
async def test_native_baseline_projection_matches_direct_baseline_request(
    monkeypatch: pytest.MonkeyPatch,
    caller: dict[str, JsonValue],
    selected: dict[str, JsonValue],
    baseline: dict[str, JsonValue],
    summary: bool,
) -> None:
    counted: Final[asyncio.Queue[Mapping[str, JsonValue]]] = asyncio.Queue()

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        counted.put_nowait(body)
        return await _count(model, api_key, body)

    def upstream(request: httpx.Request) -> httpx.Response:
        body: Final = _JSON_OBJECT.validate_json(request.content)
        model: Final = body.get("model")
        assert isinstance(model, str)
        return httpx.Response(
            200,
            request=request,
            json={
                **_message(True, model),
                "usage": {"input_tokens": 6000, "output_tokens": 10},
            },
        )

    models: Final = _MESSAGES.validate_python(
        [
            _MODELS[0],
            {
                **_MODELS[1],
                "litellm_params": {**_JSON_OBJECT.validate_python(_MODELS[1]["litellm_params"]), **selected},
            },
            {
                **_MODELS[2],
                "litellm_params": {**_JSON_OBJECT.validate_python(_MODELS[2]["litellm_params"]), **baseline},
            },
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models, count=count)
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    monkeypatch.setattr(litellm, "reasoning_auto_summary", summary)
    monkeypatch.delenv("LITELLM_REASONING_AUTO_SUMMARY", raising=False)
    request: Final = {
        "messages": [{"role": "user", "content": "question"}],
        **({"system": "stable system"} if "system" not in selected and "system" not in baseline else {}),
        "max_tokens": 4096,
        "enable_prompt_caching": True,
        **caller,
    }
    with _transport(upstream) as route:
        await rig.router.anthropic_messages(model="opus", **_JSON_OBJECT.validate_python(request))
        direct: Final = _JSON_OBJECT.validate_json(route.calls.last.request.content)
        await rig.router.anthropic_messages(
            model="test-router",
            litellm_logging_obj=rig.logging(),
            litellm_call_id=rig.call_id,
            litellm_metadata={"user_api_key_hash": "test-caller-hash"},
            litellm_session_id="native-projection-parity",
            **_JSON_OBJECT.validate_python(request),
        )
        captured: Final = _observation(await rig.capture.payload())
    assert captured.observation.plan is not None, captured.observation.reason
    projected: Final = counted.get_nowait()
    assert {key: value for key, value in projected.items() if key not in ("metadata", "stream")} == {
        key: value for key, value in direct.items() if key not in ("metadata", "stream")
    }


@pytest.mark.parametrize(
    "selected,baseline,usage_field,observed_value,multiplier",
    (
        ({}, {"speed": "fast"}, "speed", "standard", 3.0),
        ({"inference_geo": "us"}, {}, "inference_geo", "us", 1.0),
    ),
)
@pytest.mark.parametrize("estimated", (False, True))
async def test_native_baseline_prices_projected_settings_without_changing_actual_spend(
    monkeypatch: pytest.MonkeyPatch,
    estimated: bool,
    selected: dict[str, JsonValue],
    baseline: dict[str, JsonValue],
    usage_field: str,
    observed_value: str,
    multiplier: float,
) -> None:
    from litellm.proxy.spend_tracking.baseline_accounting import BaselineHistory, advance_baseline_history
    from litellm.proxy.spend_tracking.savings import baseline_cost_snapshot, price_baseline_comparison
    from litellm.types.utils import ModelInfo

    def upstream(request: httpx.Request) -> httpx.Response:
        body: Final = _JSON_OBJECT.validate_json(request.content)
        model: Final = body.get("model")
        assert isinstance(model, str)
        assert body.get(usage_field) == selected.get(usage_field)
        return httpx.Response(
            200,
            request=request,
            json={
                **_message(True, model),
                "usage": {
                    "input_tokens": 6000,
                    "output_tokens": 10,
                    usage_field: observed_value,
                    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                },
            },
        )

    rig: Final = _Rig(
        monkeypatch,
        models=_MESSAGES.validate_python(
            [
                _MODELS[0],
                {
                    **_MODELS[1],
                    "litellm_params": {
                        **_JSON_OBJECT.validate_python(_MODELS[1]["litellm_params"]),
                        **selected,
                    },
                },
                {
                    **_MODELS[2],
                    "litellm_params": {
                        **_JSON_OBJECT.validate_python(_MODELS[2]["litellm_params"]),
                        **baseline,
                        **({"extra_body": {}} if estimated else {}),
                    },
                },
            ]
        ),
    )
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    with _transport(upstream):
        response: Final = _JSON_OBJECT.validate_python(
            await rig.router.anthropic_messages(
                model="test-router",
                max_tokens=16,
                messages=[{"role": "user", "content": "question"}],
                litellm_logging_obj=rig.logging(),
                litellm_call_id=rig.call_id,
                litellm_metadata={"user_api_key_hash": "test-caller-hash"},
                litellm_session_id="baseline-speed",
            )
        )
        payload: Final = await rig.capture.payload()
    captured: Final = _observation(payload)
    _, estimates = advance_baseline_history(BaselineHistory(), (captured.observation,))
    estimate: Final = estimates[0]
    assert captured.prices is not None
    prices: Final[ModelInfo] = {
        **captured.prices,
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "provider_specific_entry": {"fast": 3.0, "us": 2.0},
    }
    actual: Final = payload["response_cost"]
    assert isinstance(actual, float)
    snapshot: Final = baseline_cost_snapshot(
        captured.model,
        prices,
        actual,
        _OBJECTS.validate_python(payload["cost_breakdown"]),
        None,
    )
    comparison: Final = price_baseline_comparison(snapshot, estimate.usage, estimate.provenance)
    assert comparison is not None and snapshot.actual_token_cost is not None, estimate.reason
    assert comparison.baseline == pytest.approx(
        actual + (6000 * 1e-6 + 10 * 2e-6) * multiplier - snapshot.actual_token_cost
    )
    assert comparison.actual == actual
    assert _JSON_OBJECT.validate_python(response["usage"])[usage_field] == observed_value


async def test_native_request_rewritten_after_capture_preserves_spend_without_guessing_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RewriteSystem(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: Mapping[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            return {**kwargs, "system": "hook system"}

    rig: Final = _Rig(monkeypatch)
    monkeypatch.setattr(litellm, "callbacks", [rig.hook, RewriteSystem()])
    with _transport(_upstream) as route:
        await _call(rig.router, rig.logging())
        payload: Final = await rig.capture.payload()
        wire: Final = _JSON_OBJECT.validate_json(route.calls.last.request.content)
    observed: Final = _observation(payload).observation
    assert wire["system"] == "hook system"
    assert observed.reason == "unsupported_request_transformation" and observed.plan is None
    assert observed.usage is not None
    actual: Final = payload["response_cost"]
    assert isinstance(actual, float) and actual > 0


@pytest.mark.parametrize("history", ("long_session", "non_ascii"))
async def test_native_baseline_models_long_and_non_ascii_history(monkeypatch: pytest.MonkeyPatch, history: str) -> None:
    rounds: Final = tuple(
        message
        for index in range(1200)
        for message in (
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{index}", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{index}", "content": "ok"}]},
        )
    )
    prefix: Final = (
        [{"role": "user", "content": "start"}, *rounds]
        if history == "long_session"
        else [{"role": "user", "content": "a" * 400_000 + "é"}, {"role": "assistant", "content": "ok"}]
    )
    messages: Final = json.dumps([*prefix, *_MESSAGES.validate_json(_MESSAGES_JSON)])
    rig: Final = _Rig(monkeypatch)
    with _transport(_upstream):
        await _call(rig.router, rig.logging(), messages=messages)
        observed: Final = _observation(await rig.capture.payload()).observation
    assert observed.outcome == "complete" and observed.plan is not None


async def test_native_baseline_abstains_after_selected_tier_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.router_strategy.complexity_router.context_compaction import compaction_executor

    monkeypatch.setitem(
        litellm.model_cost,
        "summary-fixture",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "max_input_tokens": 32000,
            "max_output_tokens": 4096,
            "supports_anthropic_compaction": True,
        },
    )
    models: Final = _MESSAGES.validate_python(
        [
            {
                "model_name": "test-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": "sonnet", "MEDIUM": "opus", "COMPLEX": "opus", "REASONING": "opus"},
                        "keyword_tier_rules": [{"keywords": ["answer"], "tier": "SIMPLE"}],
                        "session_affinity": False,
                        "enable_context_window_escalation": False,
                        "max_tokens_from_tier_model": False,
                        "context_compaction": {"model": "compactor", "max_tokens": 512},
                    },
                },
            },
            {
                "model_name": "sonnet",
                "litellm_params": {"model": "anthropic/claude-sonnet-5", "api_key": "test-selected"},
                "model_info": {"id": "selected", "max_input_tokens": 512, "max_output_tokens": 64},
            },
            {
                "model_name": "opus",
                "litellm_params": {"model": "anthropic/claude-opus-5", "api_key": "test-selected"},
                "model_info": {"id": "baseline", "max_input_tokens": 200000, "max_output_tokens": 4096},
            },
            {
                "model_name": "compactor",
                "litellm_params": {"model": "anthropic/summary-fixture", "api_key": "test-compactor"},
                "model_info": {"id": "compactor"},
            },
        ]
    )

    async def summarize(protocol: object, request: object, parent_model: object = None) -> Mapping[str, object]:
        return {
            "stop_reason": "compaction",
            "content": [{"type": "compaction", "content": "compacted", "signature": "s"}],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    messages: Final = json.dumps(
        [
            {"role": "user", "content": "Background detail. " * 300},
            {"role": "assistant", "content": "Recorded"},
            {"role": "user", "content": "Answer briefly"},
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models)
    token: Final = compaction_executor.set(summarize)
    try:
        with _transport(_upstream) as route:
            await _call(rig.router, rig.logging(), messages=messages)
            observed: Final = _observation(await rig.capture.payload()).observation
            wire: Final = route.calls.last.request.content.decode()
    finally:
        compaction_executor.reset(token)
    assert "compacted" in wire and "Background detail" not in wire
    assert observed.reason == "unsupported_request_transformation" and observed.plan is None
    assert observed.usage is not None


async def test_selected_tier_cache_markers_do_not_hide_an_unmarked_baseline_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: Final = _MESSAGES.validate_python(
        [
            _MODELS[0],
            {
                "model_name": "sonnet",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5",
                    "api_key": "test-selected",
                    "cache_control_injection_points": [{"location": "message", "role": "user", "index": -1}],
                },
                "model_info": {"id": "selected"},
            },
            _MODELS[2],
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models)
    with _transport(_upstream) as route:
        await _call(
            rig.router,
            rig.logging(),
            messages='[{"role":"user","content":[{"type":"text","text":"stable"},{"type":"text","text":"question"}]}]',
        )
        observed: Final = _observation(await rig.capture.payload()).observation
        wire: Final = route.calls.last.request.content.decode()
    assert "cache_control" in wire
    assert observed.outcome == "complete" and not observed.baseline_equivalent
    assert observed.reason is None and observed.plan is not None and not observed.plan.breakpoints


@pytest.mark.parametrize("recovery", ("retry", "fallback"))
async def test_tier_pins_never_enter_the_caller_snapshot_on_later_routing_passes(
    monkeypatch: pytest.MonkeyPatch, recovery: str
) -> None:
    pinned: Final = {"model_name": "first", "litellm_params": {"reasoning_effort": "high", "max_tokens": 777}}
    models: Final = _MESSAGES.validate_python(
        [
            {
                "model_name": "test-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": pinned, "MEDIUM": pinned, "COMPLEX": pinned, "REASONING": "opus"},
                        "session_affinity": False,
                    },
                },
            },
            {
                "model_name": "fallback-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": "sonnet", "MEDIUM": "sonnet", "COMPLEX": "sonnet", "REASONING": "opus"},
                        "session_affinity": False,
                    },
                },
            },
            {
                "model_name": "first",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5" if recovery == "retry" else "anthropic/claude-haiku-5",
                    "api_key": "test-selected",
                },
                "model_info": {"id": "first"},
            },
            *_MODELS[1:],
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models, retries=1 if recovery == "retry" else 0)
    rig.router.fallbacks = [{"test-router": ["fallback-router"]}]

    def upstream(request: httpx.Request) -> httpx.Response:
        return _upstream(request) if route.call_count else _error(request, 429, "first attempt")

    log: Final = rig.logging()
    with _transport(upstream) as route:
        await _call(rig.router, log)
        await rig.capture.payload()
        first_wire: Final = _JSON_OBJECT.validate_json(route.calls[0].request.content)
    assert first_wire.get("output_config") == {"effort": "high"} and first_wire.get("max_tokens") == 777
    context: Final = log.baseline_cache_context
    assert context is not None and context.baseline_body is not None
    assert context.baseline_body.get("max_tokens") == 16
    assert "output_config" not in context.baseline_body and "thinking" not in context.baseline_body


@pytest.mark.parametrize("chat_adapter", (False, True))
async def test_messages_estimate_keeps_adapted_baseline_extra_body_retention(
    monkeypatch: pytest.MonkeyPatch, chat_adapter: bool
) -> None:
    settings: Final = {"prompt_cache_retention": "24h", "prompt_cache_key": "configured-baseline"}
    models: Final = _MESSAGES.validate_python(
        [
            *_MODELS[:2],
            {
                "model_name": "opus",
                "model_info": {"id": "baseline"},
                "litellm_params": {
                    "model": "openai/gpt-6-astra",
                    "api_key": "test-selected",
                    "api_base": "https://api.openai.com/v1",
                    "extra_body": settings,
                },
            },
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models)
    monkeypatch.setattr(litellm, "use_chat_completions_url_for_anthropic_messages", chat_adapter)

    def openai_response(request: httpx.Request) -> httpx.Response:
        body: Final = _JSON_OBJECT.validate_json(request.content)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-baseline" if chat_adapter else "resp_baseline",
                "object": "chat.completion" if chat_adapter else "response",
                "model": body["model"],
                **(
                    {
                        "created": 1,
                        "choices": [
                            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}
                        ],
                        "usage": {"prompt_tokens": 6000, "completion_tokens": 10, "total_tokens": 6010},
                    }
                    if chat_adapter
                    else {
                        "created_at": 1,
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 6000, "output_tokens": 10, "total_tokens": 6010},
                    }
                ),
            },
        )

    with respx.mock() as transport:
        transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=_upstream)
        adapted: Final = transport.post(
            "https://api.openai.com/v1/" + ("chat/completions" if chat_adapter else "responses")
        ).mock(side_effect=openai_response)
        await rig.router.anthropic_messages(
            model="opus", max_tokens=16, messages=[{"role": "user", "content": "question"}]
        )
        wire: Final = _JSON_OBJECT.validate_json(adapted.calls.last.request.content)
        log: Final = rig.logging()
        await _call(rig.router, log, messages='[{"role":"user","content":"question"}]')
        captured: Final = _observation(await rig.capture.payload())
    assert {key: wire.get(key) for key in settings} == settings
    assert log.baseline_cache_context is not None
    projected: Final = log.baseline_cache_context.estimated_request
    assert projected is not None and {key: projected.get(key) for key in settings} == settings
    assert captured.observation.plan is not None and captured.observation.plan.breakpoints
    assert {marker.ttl_seconds for marker in captured.observation.plan.breakpoints} == {24 * 60 * 60}


@pytest.mark.parametrize("chat_adapter", (False, True))
async def test_adapted_messages_tier_estimates_native_anthropic_baseline(
    monkeypatch: pytest.MonkeyPatch, chat_adapter: bool
) -> None:
    models: Final = _MESSAGES.validate_python(
        [
            _MODELS[0],
            {
                "model_name": "sonnet",
                "model_info": {"id": "selected"},
                "litellm_params": {
                    "model": "openai/gpt-6-astra",
                    "api_key": "test-selected",
                    "api_base": "https://api.openai.com/v1",
                },
            },
            _MODELS[2],
        ]
    )
    rig: Final = _Rig(monkeypatch, models=models)
    monkeypatch.setattr(litellm, "use_chat_completions_url_for_anthropic_messages", chat_adapter)

    def openai_response(request: httpx.Request) -> httpx.Response:
        body: Final = _JSON_OBJECT.validate_json(request.content)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-selected",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 6000, "completion_tokens": 10, "total_tokens": 6010},
            }
            if chat_adapter
            else {
                "id": "resp_selected",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": body["model"],
                "output": [],
                "usage": {"input_tokens": 6000, "output_tokens": 10, "total_tokens": 6010},
            },
        )

    with respx.mock(assert_all_called=False) as transport:
        anthropic: Final = transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=_upstream)
        transport.post("https://api.openai.com/v1/" + ("chat/completions" if chat_adapter else "responses")).mock(
            side_effect=openai_response
        )
        log: Final = rig.logging()
        await _call(rig.router, log)
        captured: Final = _observation(await rig.capture.payload())
    assert not anthropic.calls
    assert log.baseline_cache_context is not None and log.baseline_cache_context.estimated
    assert captured.observation.cache_policy == "estimated" and captured.observation.outcome == "complete"
    assert captured.observation.plan is not None and captured.observation.reason is None


@pytest.mark.parametrize("passthrough", (False, True))
async def test_messages_estimate_does_not_flatten_native_baseline_extra_body(
    monkeypatch: pytest.MonkeyPatch, passthrough: bool
) -> None:
    settings: Final = {"prompt_cache_retention": "24h", "prompt_cache_key": "native-ignored"}
    rig: Final = _Rig(
        monkeypatch,
        models=_MESSAGES.validate_python(
            [
                *_MODELS[:2],
                {
                    "model_name": "opus",
                    "model_info": {
                        "id": "baseline",
                        **({"supported_endpoints": ["/v1/messages"]} if passthrough else {}),
                    },
                    "litellm_params": {
                        "model": "openai/gpt-6-astra" if passthrough else "anthropic/claude-opus-5",
                        "api_key": "test-selected",
                        "api_base": "https://api.anthropic.com",
                        "extra_body": settings,
                    },
                },
            ]
        ),
    )
    with _transport(_upstream) as route:
        await rig.router.anthropic_messages(
            model="opus", max_tokens=16, messages=[{"role": "user", "content": "question"}]
        )
        wire: Final = _JSON_OBJECT.validate_json(route.calls.last.request.content)
        log: Final = rig.logging()
        await _call(rig.router, log)
        await rig.capture.payload()
    assert log.baseline_cache_context is not None and log.baseline_cache_context.estimated
    projected: Final = log.baseline_cache_context.estimated_request
    assert projected is not None
    assert (
        {key: projected.get(key) for key in settings}
        == {key: wire.get(key) for key in settings}
        == {key: None for key in settings}
    )


@pytest.mark.parametrize("unresolved_input", (False, True))
async def test_estimated_baseline_alias_resolves_threshold_and_preserves_usage_for_unsupported_input(
    monkeypatch: pytest.MonkeyPatch,
    unresolved_input: bool,
) -> None:
    from litellm import utils
    from litellm.proxy.hooks.autorouter_baseline_cache import finalize_baseline_cache
    from litellm.proxy.spend_tracking.baseline_accounting import BaselineHistory, advance_baseline_history
    from litellm.types.utils import ModelResponse, Usage

    model: Final = "gemini/cache-minimum-test"
    minimum: Final = 8192
    monkeypatch.setattr(utils, "MINIMUM_PROMPT_CACHE_TOKEN_COUNT_OVERRIDE", None)
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "gemini",
            "mode": "chat",
            "supports_prompt_caching": True,
            "prompt_cache_min_tokens": minimum,
            "max_tokens": minimum * 2,
            "max_input_tokens": minimum * 2,
            "max_output_tokens": minimum,
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-7,
        },
    )
    rig: Final = _Rig(
        monkeypatch,
        models=_MESSAGES.validate_python(
            [
                *_MODELS[:2],
                {
                    **_MODELS[2],
                    "litellm_params": {"model": model, "api_key": "test-selected"},
                    "model_info": {**litellm.get_model_info(model=model), "id": "baseline"},
                },
            ]
        ),
    )
    logging: Final = rig.logging()
    request: Final[dict[str, object]] = {
        **_kwargs(logging),
        "messages": None if unresolved_input else _MESSAGES.validate_json(_MESSAGES_JSON),
    }
    Router._record_routing_decision(
        request,
        StandardLoggingRoutingDecision(
            router_model_name="test-router",
            router_type="complexity",
            routed_model="anthropic/claude-sonnet-5",
            savings_baseline_model="opus",
            savings_baseline_deployment_id="baseline",
        ),
    )
    await rig.hook.async_pre_call_deployment_hook(request, CallTypes.anthropic_messages)
    await finalize_baseline_cache(
        logging, ModelResponse(usage=Usage(prompt_tokens=6000, completion_tokens=10, total_tokens=6010))
    )
    captured: Final = logging.baseline_observation
    assert captured is not None
    assert captured.baseline_model == "opus" and captured.provider == "gemini"
    assert captured.observation.minimum_cache_tokens == minimum
    assert captured.observation.outcome == "complete" and captured.observation.usage is not None
    if unresolved_input:
        assert captured.observation.plan is None and captured.observation.reason == "unsupported_cache_request"
        assert captured.observation.usage.prompt_tokens == 6000
        return
    history, cold = advance_baseline_history(BaselineHistory(), (captured.observation,))
    _, warm = advance_baseline_history(
        history,
        (
            captured.observation.model_copy(
                update={
                    "request_id": "cache-minimum-followup",
                    "started_at": captured.observation.available_at + 1,
                    "available_at": captured.observation.available_at + 2,
                }
            ),
        ),
    )
    for estimate in (*cold, *warm):
        assert estimate.usage is not None, estimate.reason
        assert estimate.usage.prompt_tokens_details.cached_tokens == 0
        assert estimate.usage.prompt_tokens_details.cache_creation_tokens == 0
