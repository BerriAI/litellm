import asyncio
import copy
import json
from collections.abc import AsyncIterator, Callable, Mapping
from types import SimpleNamespace
from typing import Final

import httpx
import pydantic
import pytest

import litellm
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
from litellm.types.utils import Delta, GenericGuardrailAPIInputs, ModelResponseStream

API_BASE: Final = "https://api.test.guardrail.com"
CLIENT_TIMEOUT_SECONDS: Final = 600.0
_STREAM_WORDS: Final = ("Hello", " ", "world", "!", " Bye")


class _Endpoint:
    def __init__(
        self,
        *,
        body: Mapping[str, object] | None = None,
        status_code: int = 200,
        error: Exception | None = None,
        gate_open: bool = True,
    ) -> None:
        self.gate: Final = asyncio.Event()
        if gate_open:
            self.gate.set()
        self.requests: Final[list[tuple[str, dict[str, str]]]] = []  # mutable-ok: records each request
        self.payloads: Final[list[dict[str, object]]] = []  # mutable-ok: records each request body
        self.read_timeouts: Final[list[float | None]] = []  # mutable-ok: records each request timeout
        self.completed: int = 0
        self._finished: Final = asyncio.Condition()
        self._body: Final = dict(body or {"action": "NONE"})
        self._status_code: Final = status_code
        self._error: Final = error

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((str(request.url), dict(request.headers)))
        self.payloads.append(json.loads(request.content))
        self.read_timeouts.append(request.extensions["timeout"]["read"])
        await self.gate.wait()
        async with self._finished:
            self.completed += 1
            self._finished.notify_all()
        if self._error is not None:
            raise self._error
        return httpx.Response(self._status_code, json=self._body)

    async def wait_for_completed(self, count: int) -> None:
        async with self._finished:
            await asyncio.wait_for(self._finished.wait_for(lambda: self.completed >= count), timeout=5)

    def handler(self) -> AsyncHTTPHandler:
        return AsyncHTTPHandler(timeout=CLIENT_TIMEOUT_SECONDS, transport=httpx.MockTransport(self))


def _logging_obj(call_id: str = "call-123") -> SimpleNamespace:
    return SimpleNamespace(litellm_call_id=call_id, litellm_trace_id="trace-123", model_call_details={})


def _guardrail(
    endpoint: _Endpoint, *, name: str = "ff-guardrail", event_hook: str = "pre_call", **options: object
) -> GenericGuardrailAPI:
    return GenericGuardrailAPI(
        api_base=API_BASE,
        guardrail_name=name,
        event_hook=event_hook,
        default_on=True,
        async_handler=endpoint.handler(),
        **options,
    )


def _fire_and_forget(
    endpoint: _Endpoint, *, max_inflight: int = 10, **options: object
) -> tuple[GenericGuardrailAPI, BackgroundDispatcher]:
    dispatcher: Final = BackgroundDispatcher(guardrail_name="ff-guardrail", max_inflight=max_inflight)
    return _guardrail(endpoint, dispatcher=dispatcher, fire_and_forget=True, **options), dispatcher


def _request_data() -> dict[str, object]:  # mutable-ok: apply_guardrail records entries into it
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"user_api_key_hash": "hash-1", "user_api_key_team_id": "team-1"},
    }


def _recorded_outcomes(request_data: Mapping[str, object]) -> list[tuple[str, str]]:
    metadata: Final = request_data["metadata"]
    assert isinstance(metadata, dict)
    return [
        (entry["guardrail_status"], entry["guardrail_response"])
        for entry in metadata["standard_logging_guardrail_information"]
    ]


async def test_returns_before_the_post_completes() -> None:
    endpoint: Final = _Endpoint(gate_open=False)
    guardrail, dispatcher = _fire_and_forget(endpoint)
    inputs: Final = GenericGuardrailAPIInputs(
        texts=["hello"], structured_messages=[{"role": "user", "content": "hello"}]
    )

    result: Final = await asyncio.wait_for(
        guardrail.apply_guardrail(
            inputs=inputs, request_data=_request_data(), input_type="request", logging_obj=_logging_obj()
        ),
        timeout=5,
    )

    assert (result, endpoint.completed, dispatcher.pending_count) == (inputs, 0, 1)
    endpoint.gate.set()
    await dispatcher.wait_for_pending()
    assert (endpoint.completed, dispatcher.pending_count) == (1, 0)


async def test_background_post_reaches_the_same_url_with_the_same_headers_and_payload() -> None:
    awaited_endpoint: Final = _Endpoint()
    background_endpoint: Final = _Endpoint()
    auth: Final = {"api_key": "audit-key", "headers": {"x-static": "static-value"}}
    guardrail, dispatcher = _fire_and_forget(background_endpoint, **auth)
    inputs: Final = GenericGuardrailAPIInputs(texts=["hello"], images=["data:image/png;base64,AAAA"], model="gpt-4o")

    for target in (_guardrail(awaited_endpoint, **auth), guardrail):
        await target.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(**inputs),
            request_data=_request_data(),
            input_type="request",
            logging_obj=_logging_obj(),
        )
    await dispatcher.wait_for_pending()

    assert background_endpoint.requests == awaited_endpoint.requests
    assert background_endpoint.payloads == awaited_endpoint.payloads
    url, headers = background_endpoint.requests[0]
    assert (url, headers["x-api-key"], headers["x-static"]) == (
        f"{API_BASE}/beta/litellm_basic_guardrail_api",
        "audit-key",
        "static-value",
    )


@pytest.mark.parametrize(
    ("configured_timeout", "expected_background_timeout"),
    [(None, FIRE_AND_FORGET_POST_TIMEOUT_SECONDS), (5.0, 5.0)],
)
async def test_background_post_honors_the_configured_timeout(
    configured_timeout: float | None, expected_background_timeout: float
) -> None:
    endpoint: Final = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(endpoint, timeout=configured_timeout)

    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")
    await dispatcher.wait_for_pending()

    assert endpoint.read_timeouts == [expected_background_timeout]


@pytest.mark.parametrize(
    "body",
    [{"action": "BLOCKED", "blocked_reason": "nope"}, {"action": "GUARDRAIL_INTERVENED", "texts": ["MASKED"]}],
)
async def test_the_endpoint_verdict_is_ignored(body: Mapping[str, object]) -> None:
    endpoint: Final = _Endpoint(body=body)
    guardrail, dispatcher = _fire_and_forget(endpoint)

    result: Final = await guardrail.apply_guardrail(
        inputs={"texts": ["my ssn is 123"]}, request_data={}, input_type="request"
    )
    await dispatcher.wait_for_pending()

    assert (result, endpoint.completed) == ({"texts": ["my ssn is 123"]}, 1)


@pytest.mark.parametrize(
    "endpoint_options",
    [{"error": httpx.ConnectError("connection refused")}, {"status_code": 500}],
)
async def test_a_failing_endpoint_is_logged_not_raised(
    endpoint_options: Mapping[str, object], warning_messages: Callable[[str], list[str]]
) -> None:
    endpoint: Final = _Endpoint(**endpoint_options)
    guardrail, dispatcher = _fire_and_forget(endpoint, fail_on_error=True, unreachable_fallback="fail_closed")

    result: Final = await guardrail.apply_guardrail(
        inputs={"texts": ["hello"]},
        request_data={},
        input_type="response",
        logging_obj=_logging_obj(call_id="call-failing"),
    )
    await dispatcher.wait_for_pending()

    assert result == {"texts": ["hello"]}
    failures: Final = warning_messages("call failed")
    assert len(failures) == 1
    assert "ff-guardrail" in failures[0]
    assert "input_type=response litellm_call_id=call-failing" in failures[0]


@pytest.mark.parametrize("fail_on_error", [True, False])
@pytest.mark.parametrize(
    ("inputs", "request_data"),
    [
        ({"texts": ["hi"], "tools": [{"function": {"name": "f"}}]}, _request_data()),
        ({"texts": ["hi"]}, {"messages": [], "metadata": None}),
    ],
    ids=["tool_without_type", "malformed_request_metadata"],
)
async def test_a_failure_before_dispatch_is_logged_and_passes_through(
    inputs: GenericGuardrailAPIInputs,
    request_data: dict[str, object],  # mutable-ok: apply_guardrail records entries into it
    fail_on_error: bool,
    warning_messages: Callable[[str], list[str]],
) -> None:
    endpoint: Final = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(endpoint, fail_on_error=fail_on_error)
    request: Final = copy.deepcopy(request_data)

    result: Final = await guardrail.apply_guardrail(
        inputs=inputs, request_data=request, input_type="request", logging_obj=_logging_obj("call-bad")
    )
    await dispatcher.wait_for_pending()

    assert (result, endpoint.payloads) == (inputs, [])
    assert _recorded_outcomes(request) == [("not_run", FIRE_AND_FORGET_NOT_DISPATCHED_REASON)]
    warnings: Final = warning_messages("not dispatched")
    assert len(warnings) == 1
    assert "litellm_call_id=call-bad" in warnings[0]


async def test_a_payload_that_cannot_be_serialized_is_recorded_as_not_dispatched() -> None:
    endpoint: Final = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(endpoint, additional_provider_specific_params={"bad": object()})
    request: Final = _request_data()

    result: Final = await guardrail.apply_guardrail(
        inputs={"texts": ["hello"]}, request_data=request, input_type="request"
    )
    await dispatcher.wait_for_pending()

    assert (result, endpoint.payloads) == ({"texts": ["hello"]}, [])
    assert _recorded_outcomes(request) == [("not_run", FIRE_AND_FORGET_NOT_DISPATCHED_REASON)]


async def test_a_call_dropped_by_the_cap_never_builds_its_payload() -> None:
    endpoint: Final = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(
        endpoint, max_inflight=1, additional_provider_specific_params={"bad": object()}
    )
    gate: Final = asyncio.Event()
    dispatcher.dispatch(lambda: gate.wait, context="occupies the only slot")
    request: Final = _request_data()

    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request, input_type="request")
    gate.set()
    await dispatcher.wait_for_pending()

    assert _recorded_outcomes(request) == [("not_run", FIRE_AND_FORGET_DROPPED_REASON)]


async def test_dispatched_calls_are_recorded_as_success_and_dropped_ones_as_not_run() -> None:
    endpoint: Final = _Endpoint(gate_open=False)
    guardrail, dispatcher = _fire_and_forget(endpoint, max_inflight=1)
    dispatched, dropped = _request_data(), _request_data()

    for request_data in (dispatched, dropped):
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request")
    endpoint.gate.set()
    await dispatcher.wait_for_pending()

    assert _recorded_outcomes(dispatched) == [("success", FIRE_AND_FORGET_DISPATCHED_REASON)]
    assert _recorded_outcomes(dropped) == [("not_run", FIRE_AND_FORGET_DROPPED_REASON)]


async def test_a_configured_max_inflight_bounds_dispatch() -> None:
    endpoint: Final = _Endpoint(gate_open=False)
    guardrail: Final = _guardrail(endpoint, fire_and_forget=True, fire_and_forget_max_inflight=1)
    first, second = _request_data(), _request_data()

    for request_data in (first, second):
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request")
    endpoint.gate.set()
    await endpoint.wait_for_completed(1)

    assert _recorded_outcomes(first) + _recorded_outcomes(second) == [
        ("success", FIRE_AND_FORGET_DISPATCHED_REASON),
        ("not_run", FIRE_AND_FORGET_DROPPED_REASON),
    ]
    assert len(endpoint.payloads) == 1


@pytest.mark.parametrize("max_inflight", [None, 0, "many"])
async def test_an_unset_or_invalid_max_inflight_admits_the_default_number_of_calls(max_inflight: object) -> None:
    endpoint: Final = _Endpoint(gate_open=False)
    guardrail: Final = _guardrail(endpoint, fire_and_forget=True, fire_and_forget_max_inflight=max_inflight)
    calls: Final = [_request_data() for _ in range(DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT + 1)]

    for request_data in calls:
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request")
    endpoint.gate.set()
    await endpoint.wait_for_completed(DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT)

    outcomes: Final = [_recorded_outcomes(request_data)[0][0] for request_data in calls]
    assert outcomes == ["success"] * DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT + ["not_run"]


def _stream_chunks() -> list[ModelResponseStream]:
    return [
        ModelResponseStream(
            model="gpt-4",
            choices=[
                litellm.StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content=word),
                    finish_reason="stop" if i == len(_STREAM_WORDS) - 1 else None,
                )
            ],
        )
        for i, word in enumerate(_STREAM_WORDS)
    ]


async def _streamed_texts(guardrail: GenericGuardrailAPI) -> list[str]:
    async def stream() -> AsyncIterator[ModelResponseStream]:
        for chunk in _stream_chunks():
            yield chunk

    return [
        chunk.choices[0].delta.content or ""
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


@pytest.mark.parametrize("streaming_transform_mode", ["block_only", "incremental_diff"])
async def test_a_stream_is_emitted_live_and_sends_one_call_with_the_whole_text(
    streaming_transform_mode: str,
) -> None:
    endpoint: Final = _Endpoint()
    guardrail, dispatcher = _fire_and_forget(
        endpoint,
        event_hook="post_call",
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=1,
        streaming_transform_mode=streaming_transform_mode,
    )

    streamed: Final = await _streamed_texts(guardrail)
    await dispatcher.wait_for_pending()

    assert streamed == list(_STREAM_WORDS), "every chunk must be emitted as it arrives"
    assert [payload["texts"] for payload in endpoint.payloads] == [["Hello world! Bye"]]


async def test_an_awaited_stream_is_checked_per_sampled_chunk() -> None:
    endpoint: Final = _Endpoint()

    await _streamed_texts(_guardrail(endpoint, event_hook="post_call", streaming_sampling_rate=1))

    assert len(endpoint.payloads) > 1


def test_the_observe_only_warning_is_logged_only_when_enabled(warning_messages: Callable[[str], list[str]]) -> None:
    _guardrail(_Endpoint(), name="enforcing")
    _guardrail(_Endpoint(), name="observer", fire_and_forget=True)

    warnings: Final = warning_messages("observe-only")
    assert len(warnings) == 1
    assert "observer" in warnings[0]


@pytest.mark.parametrize("value", ["false", "maybe", 2])
async def test_a_quoted_false_or_unparseable_fire_and_forget_keeps_the_guardrail_enforcing(value: object) -> None:
    endpoint: Final = _Endpoint(body={"action": "BLOCKED", "blocked_reason": "nope"})
    guardrail: Final = _guardrail(endpoint, fire_and_forget=value)

    with pytest.raises(GuardrailRaisedException, match="nope"):
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")


async def test_a_quoted_true_fire_and_forget_turns_on_observe_only() -> None:
    endpoint: Final = _Endpoint(body={"action": "BLOCKED", "blocked_reason": "nope"})
    dispatcher: Final = BackgroundDispatcher(guardrail_name="ff-guardrail", max_inflight=1)
    quoted: Final = _guardrail(endpoint, dispatcher=dispatcher, fire_and_forget="true")

    result: Final = await quoted.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")
    await dispatcher.wait_for_pending()

    assert (result, endpoint.completed) == ({"texts": ["hello"]}, 1)


@pytest.mark.parametrize("max_inflight", [0, -1])
def test_the_config_form_rejects_a_max_inflight_below_one(max_inflight: int) -> None:
    with pytest.raises(pydantic.ValidationError):
        GenericGuardrailAPIOptionalParams(fire_and_forget_max_inflight=max_inflight)


async def test_initialize_guardrail_forwards_fire_and_forget_and_max_inflight() -> None:
    endpoint: Final = _Endpoint(gate_open=False)
    guardrail: Final = initialize_guardrail(
        LitellmParams(
            guardrail="generic_guardrail_api",
            mode="pre_call",
            api_base=API_BASE,
            default_on=True,
            fire_and_forget=True,
            fire_and_forget_max_inflight=1,
        ),
        {"guardrail_name": "from-config"},
    )
    guardrail.async_handler = endpoint.handler()
    first, second = _request_data(), _request_data()

    try:
        for request_data in (first, second):
            await asyncio.wait_for(
                guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data=request_data, input_type="request"),
                timeout=5,
            )
        endpoint.gate.set()
        await endpoint.wait_for_completed(1)
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)

    assert _recorded_outcomes(first) + _recorded_outcomes(second) == [
        ("success", FIRE_AND_FORGET_DISPATCHED_REASON),
        ("not_run", FIRE_AND_FORGET_DROPPED_REASON),
    ]


async def test_by_default_the_endpoint_is_awaited_and_can_block() -> None:
    endpoint: Final = _Endpoint(body={"action": "BLOCKED", "blocked_reason": "nope"})
    guardrail: Final = _guardrail(endpoint)

    with pytest.raises(GuardrailRaisedException, match="nope"):
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")
    assert endpoint.completed == 1
