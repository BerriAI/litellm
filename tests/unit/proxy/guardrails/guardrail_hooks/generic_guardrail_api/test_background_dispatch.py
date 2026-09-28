import asyncio
import contextvars
import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace

import httpx
import pydantic
import pytest

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPI,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.background_dispatch import (
    DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT,
    FIRE_AND_FORGET_DISPATCHED_REASON,
    FIRE_AND_FORGET_DROPPED_REASON,
    FIRE_AND_FORGET_NOT_DISPATCHED_REASON,
    FIRE_AND_FORGET_POST_TIMEOUT_SECONDS,
    BackgroundDispatcher,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.guardrails import LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIOptionalParams,
)
from litellm.types.utils import Delta, ModelResponseStream

API_BASE = "https://api.test.guardrail.com"
CLIENT_TIMEOUT_SECONDS = 600.0

_request_scoped = contextvars.ContextVar("request_scoped", default=None)


class _Endpoint:
    """The guardrail server behind a MockTransport. Each request is recorded, then waits on ``gate``."""

    def __init__(self, *, body=None, status_code=200, error=None, gate_open=True):
        self.gate = asyncio.Event()
        if gate_open:
            self.gate.set()
        self.payloads: list[dict] = []
        self.read_timeouts: list[float | None] = []
        self.seen_request_scoped: list[object] = []
        self.completed = 0
        self._body = body or {"action": "NONE"}
        self._status_code = status_code
        self._error = error

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        self.read_timeouts.append(request.extensions["timeout"]["read"])
        self.seen_request_scoped.append(_request_scoped.get())
        await self.gate.wait()
        if self._error is not None:
            raise self._error
        self.completed += 1
        return httpx.Response(self._status_code, json=self._body)

    def handler(self) -> AsyncHTTPHandler:
        return AsyncHTTPHandler(timeout=CLIENT_TIMEOUT_SECONDS, transport=httpx.MockTransport(self))


def _logging_obj(call_id="call-123"):
    return SimpleNamespace(litellm_call_id=call_id, litellm_trace_id="trace-123", model_call_details={})


def _guardrail(endpoint, *, name="ff-guardrail", event_hook="pre_call", **options):
    return GenericGuardrailAPI(
        api_base=API_BASE,
        guardrail_name=name,
        event_hook=event_hook,
        default_on=True,
        async_handler=endpoint.handler(),
        **options,
    )


def _fire_and_forget(endpoint, *, max_inflight=10, **options):
    dispatcher = BackgroundDispatcher(guardrail_name="ff-guardrail", max_inflight=max_inflight)
    return _guardrail(endpoint, dispatcher=dispatcher, fire_and_forget=True, **options), dispatcher


def _request_data():
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"user_api_key_hash": "hash-1", "user_api_key_team_id": "team-1"},
    }


@pytest.fixture
def captured_warnings() -> Iterator[list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = records.append
    previous_level = verbose_proxy_logger.level
    verbose_proxy_logger.addHandler(handler)
    verbose_proxy_logger.setLevel(logging.WARNING)
    yield records
    verbose_proxy_logger.removeHandler(handler)
    verbose_proxy_logger.setLevel(previous_level)


def _messages(records, needle):
    return [m for m in (r.getMessage() for r in records) if needle in m]


async def test_returns_before_the_post_completes():
    endpoint = _Endpoint(gate_open=False)
    guardrail, dispatcher = _fire_and_forget(endpoint)
    inputs = {"texts": ["hello"], "structured_messages": [{"role": "user", "content": "hello"}]}

    result = await asyncio.wait_for(
        guardrail.apply_guardrail(
            inputs=inputs, request_data=_request_data(), input_type="request", logging_obj=_logging_obj()
        ),
        timeout=5,
    )

    assert result == inputs
    assert endpoint.completed == 0
    assert dispatcher.pending_count == 1

    endpoint.gate.set()
    await dispatcher.wait_for_pending()

    assert endpoint.completed == 1
    assert dispatcher.pending_count == 0


async def test_endpoint_receives_the_same_payload_as_the_awaited_path():
    awaited_endpoint = _Endpoint()
    background_endpoint = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(background_endpoint)
    inputs = {"texts": ["hello"], "images": ["data:image/png;base64,AAAA"], "model": "gpt-4o"}

    for target in (_guardrail(awaited_endpoint), guardrail):
        await target.apply_guardrail(
            inputs=dict(inputs), request_data=_request_data(), input_type="request", logging_obj=_logging_obj()
        )
    await dispatcher.wait_for_pending()

    assert background_endpoint.payloads[0]["texts"] == ["hello"]
    assert background_endpoint.payloads[0]["request_data"]["user_api_key_team_id"] == "team-1"
    assert background_endpoint.payloads == awaited_endpoint.payloads


async def test_background_post_uses_its_own_timeout():
    awaited_endpoint = _Endpoint()
    background_endpoint = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(background_endpoint)

    for target in (_guardrail(awaited_endpoint), guardrail):
        await target.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")
    await dispatcher.wait_for_pending()

    assert background_endpoint.read_timeouts == [FIRE_AND_FORGET_POST_TIMEOUT_SECONDS]
    assert awaited_endpoint.read_timeouts == [CLIENT_TIMEOUT_SECONDS]


async def test_background_post_does_not_inherit_request_context():
    endpoint = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(endpoint)
    token = _request_scoped.set("request-1")
    try:
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")
    finally:
        _request_scoped.reset(token)
    await dispatcher.wait_for_pending()

    assert endpoint.completed == 1
    assert endpoint.seen_request_scoped == [None]


@pytest.mark.parametrize(
    "body",
    [
        {"action": "BLOCKED", "blocked_reason": "nope"},
        {"action": "GUARDRAIL_INTERVENED", "texts": ["MASKED"]},
    ],
)
async def test_verdict_is_ignored(body):
    endpoint = _Endpoint(body=body)
    guardrail, dispatcher = _fire_and_forget(endpoint)

    result = await guardrail.apply_guardrail(inputs={"texts": ["my ssn is 123"]}, request_data={}, input_type="request")
    await dispatcher.wait_for_pending()

    assert result == {"texts": ["my ssn is 123"]}
    assert endpoint.completed == 1


@pytest.mark.parametrize(
    "endpoint_options",
    [
        {"error": httpx.ConnectError("connection refused")},
        {"status_code": 500},
    ],
)
async def test_failing_endpoint_is_logged_not_raised(endpoint_options, captured_warnings):
    endpoint = _Endpoint(**endpoint_options)
    guardrail, dispatcher = _fire_and_forget(endpoint, fail_on_error=True, unreachable_fallback="fail_closed")

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["hello"]},
        request_data={},
        input_type="response",
        logging_obj=_logging_obj(call_id="call-failing"),
    )
    await dispatcher.wait_for_pending()

    assert result == {"texts": ["hello"]}
    failures = _messages(captured_warnings, "call failed")
    assert len(failures) == 1
    assert "ff-guardrail" in failures[0]
    assert "input_type=response" in failures[0]
    assert "litellm_call_id=call-failing" in failures[0]


@pytest.mark.parametrize("fail_on_error", [True, False])
@pytest.mark.parametrize(
    ("inputs", "make_request_data"),
    [
        ({"texts": ["hi"], "tools": [{"function": {"name": "f"}}]}, _request_data),
        ({"texts": ["hi"]}, lambda: {"messages": [], "metadata": None}),
    ],
    ids=["tool_without_type", "malformed_request_metadata"],
)
async def test_failure_before_dispatch_is_logged_and_passes_through(
    inputs, make_request_data, fail_on_error, captured_warnings
):
    request_data = make_request_data()
    endpoint = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(endpoint, fail_on_error=fail_on_error)

    result = await guardrail.apply_guardrail(
        inputs=inputs, request_data=request_data, input_type="request", logging_obj=_logging_obj("call-bad")
    )
    await dispatcher.wait_for_pending()

    assert result == inputs
    assert endpoint.payloads == []
    assert _recorded_outcomes(request_data) == [("not_run", FIRE_AND_FORGET_NOT_DISPATCHED_REASON)]
    warnings = _messages(captured_warnings, "not dispatched")
    assert len(warnings) == 1
    assert "litellm_call_id=call-bad" in warnings[0]


async def test_inflight_cap_drops_and_counts_excess_calls(captured_warnings):
    endpoint = _Endpoint(gate_open=False)
    guardrail, dispatcher = _fire_and_forget(endpoint, max_inflight=2)

    results = [
        await guardrail.apply_guardrail(inputs={"texts": [f"t{i}"]}, request_data={}, input_type="request")
        for i in range(5)
    ]

    assert results == [{"texts": [f"t{i}"]} for i in range(5)]
    assert dispatcher.pending_count == 2
    assert dispatcher.dropped_count == 3
    assert len(_messages(captured_warnings, "dropped")) == 1

    endpoint.gate.set()
    await dispatcher.wait_for_pending()

    assert [p["texts"] for p in endpoint.payloads] == [["t0"], ["t1"]]


def _recorded_outcomes(request_data):
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    return [(entry["guardrail_status"], entry["guardrail_response"]) for entry in entries]


async def test_dispatched_and_dropped_calls_are_recorded():
    endpoint = _Endpoint(gate_open=False)
    guardrail, dispatcher = _fire_and_forget(endpoint, max_inflight=1)
    dispatched, dropped = _request_data(), _request_data()

    for request_data in (dispatched, dropped):
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request")
    endpoint.gate.set()
    await dispatcher.wait_for_pending()

    assert _recorded_outcomes(dispatched) == [("success", FIRE_AND_FORGET_DISPATCHED_REASON)]
    assert _recorded_outcomes(dropped) == [("not_run", FIRE_AND_FORGET_DROPPED_REASON)]


async def test_finished_task_frees_its_slot():
    endpoint = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(endpoint, max_inflight=1)

    for i in range(3):
        await guardrail.apply_guardrail(inputs={"texts": [f"t{i}"]}, request_data={}, input_type="request")
        await dispatcher.wait_for_pending()
        assert dispatcher.pending_count == 0

    assert dispatcher.dropped_count == 0
    assert endpoint.completed == 3


def _stream_chunks():
    words = ("Hello", " ", "world", "!", " Bye")
    return [
        ModelResponseStream(
            model="gpt-4",
            choices=[
                litellm.StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content=word),
                    finish_reason="stop" if i == len(words) - 1 else None,
                )
            ],
        )
        for i, word in enumerate(words)
    ]


async def _run_stream(guardrail):
    async def stream():
        for chunk in _stream_chunks():
            yield chunk

    return [
        chunk
        async for chunk in UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test", request_route="/chat/completions"),
            response=stream(),
            request_data={
                "messages": [{"role": "user", "content": "hi"}],
                "guardrail_to_apply": guardrail,
                "metadata": {"guardrails": ["ff-guardrail"]},
            },
        )
    ]


async def test_stream_dispatches_one_call():
    per_chunk_endpoint = _Endpoint()
    await _run_stream(_guardrail(per_chunk_endpoint, event_hook="post_call", streaming_sampling_rate=1))

    background_endpoint = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(
        background_endpoint, event_hook="post_call", streaming_end_of_stream_only=False, streaming_sampling_rate=1
    )
    streamed = await _run_stream(guardrail)
    await dispatcher.wait_for_pending()

    assert len(per_chunk_endpoint.payloads) > 1
    assert len(streamed) == len(_stream_chunks())
    assert len(background_endpoint.payloads) == 1
    assert guardrail.streaming_end_of_stream_only is True


def test_observe_only_warning_only_when_enabled(captured_warnings):
    _guardrail(_Endpoint(), name="enforcing")
    _guardrail(_Endpoint(), name="observer", fire_and_forget=True)

    warnings = _messages(captured_warnings, "observe-only")
    assert len(warnings) == 1
    assert "observer" in warnings[0]


@pytest.mark.parametrize("value", ["false", "true", 1])
def test_non_bool_fire_and_forget_is_rejected(value):
    with pytest.raises(ValueError, match="fire_and_forget must be a bool"):
        _guardrail(_Endpoint(), fire_and_forget=value)


@pytest.mark.parametrize("max_inflight", ["5", 2.5, True])
def test_non_int_max_inflight_is_rejected(max_inflight):
    with pytest.raises(ValueError, match="fire_and_forget_max_inflight must be an int"):
        _guardrail(_Endpoint(), fire_and_forget=True, fire_and_forget_max_inflight=max_inflight)


@pytest.mark.parametrize("max_inflight", [0, -1])
def test_max_inflight_below_one_is_rejected(max_inflight):
    with pytest.raises(ValueError, match="fire_and_forget_max_inflight"):
        _guardrail(_Endpoint(), fire_and_forget=True, fire_and_forget_max_inflight=max_inflight)
    with pytest.raises(pydantic.ValidationError):
        GenericGuardrailAPIOptionalParams(fire_and_forget_max_inflight=max_inflight)


async def test_configured_max_inflight_bounds_dispatch():
    endpoint = _Endpoint(gate_open=False)
    guardrail = _guardrail(endpoint, fire_and_forget=True, fire_and_forget_max_inflight=1)
    first, second = _request_data(), _request_data()

    for request_data in (first, second):
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request")
    endpoint.gate.set()
    await guardrail._dispatcher.wait_for_pending()

    assert _recorded_outcomes(first) == [("success", FIRE_AND_FORGET_DISPATCHED_REASON)]
    assert _recorded_outcomes(second) == [("not_run", FIRE_AND_FORGET_DROPPED_REASON)]
    assert len(endpoint.payloads) == 1


async def test_default_max_inflight_admits_concurrent_calls():
    endpoint = _Endpoint(gate_open=False)
    guardrail = _guardrail(endpoint, fire_and_forget=True)
    calls = [_request_data() for _ in range(DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT + 1)]

    for request_data in calls:
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request")
    endpoint.gate.set()
    await guardrail._dispatcher.wait_for_pending()

    outcomes = [_recorded_outcomes(request_data)[0][0] for request_data in calls]
    assert outcomes == ["success"] * DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT + ["not_run"]


async def test_initialize_guardrail_forwards_fire_and_forget():
    litellm_params = LitellmParams(
        guardrail="generic_guardrail_api", mode="pre_call", api_base=API_BASE, default_on=True
    )
    litellm_params.fire_and_forget = True
    litellm_params.fire_and_forget_max_inflight = 1
    gate = asyncio.Event()

    guardrail = initialize_guardrail(litellm_params, {"guardrail_name": "from-config"})
    try:
        assert guardrail.fire_and_forget is True
        assert guardrail.streaming_end_of_stream_only is True
        assert guardrail._dispatcher.dispatch(gate.wait, context="first") is True
        assert guardrail._dispatcher.dispatch(gate.wait, context="second") is False
    finally:
        gate.set()
        await guardrail._dispatcher.wait_for_pending()
        litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)


async def test_default_awaits_the_endpoint_and_blocks():
    endpoint = _Endpoint(body={"action": "BLOCKED", "blocked_reason": "nope"})
    guardrail = _guardrail(endpoint)

    assert guardrail.fire_and_forget is False
    assert guardrail.streaming_end_of_stream_only is False
    with pytest.raises(GuardrailRaisedException, match="nope"):
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")
    assert endpoint.completed == 1
