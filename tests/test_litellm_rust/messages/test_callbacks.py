import asyncio
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextvars import ContextVar
from datetime import datetime
from typing import Final, Protocol, cast  # noqa: TID251  # narrows public exception attributes from the legacy mapper

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.argilla import ArgillaLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.integrations.argilla import ArgillaItem
from litellm.types.utils import ModelResponse, StandardLoggingPayload
from tests.test_litellm_rust.messages.data import BAD_REQUEST, MESSAGES_EVENTS, MESSAGES_RESPONSE, RATE_LIMITED, call
from tests.test_litellm_rust.support.callback_recorder import HookRecorder, RecordingLogger, request_body
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, relaxed
from tests.test_litellm_rust.support.results import response_id, with_id
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

_OBJECT: Final = TypeAdapter(dict[str, object])
_HEADERS: Final = TypeAdapter(dict[str, str])


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook,expected",
    (
        pytest.param("log_pre_api_call", 1),
        pytest.param("log_post_api_call", 1),
        pytest.param("async_log_success_event", 1),
        pytest.param("async_logging_hook", 1),
        pytest.param("redact_standard_logging_payload_from_model_call_details", 1, id="redact"),
        pytest.param("async_pre_call_deployment_hook", 1),
        pytest.param("async_post_call_success_deployment_hook", 1),
        pytest.param("log_success_event", 0),
        pytest.param("logging_hook", 0),
    ),
)
async def test_success_hook_fires_the_expected_number_of_times(
    server: RecordingServer, hook: str, expected: int
) -> None:
    recorder: Final = HookRecorder()

    with rebound(litellm, "callbacks", [recorder]):
        response: Final = await call(server)
        await recorder.wait_for_async("async_log_success_event")

    assert_served_natively(server, response)
    assert recorder.names.count(hook) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook,expected",
    (
        pytest.param("async_log_failure_event", 1),
        pytest.param("async_post_call_failure_deployment_hook", 1),
        pytest.param("log_model_group_rate_limit_error", 1),
        pytest.param("log_failure_event", 0),
    ),
)
async def test_failure_hook_fires_the_expected_number_of_times(
    server: RecordingServer, hook: str, expected: int
) -> None:
    server.default_response = ResponseSpec(body=RATE_LIMITED, status=429)
    recorder: Final = HookRecorder()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.RateLimitError):
            await call(server, metadata={"model_group_size": 1})
        await recorder.wait_for_async("async_log_failure_event")

    assert_served_natively(server)
    assert recorder.names.count(hook) == expected


@pytest.mark.asyncio
async def test_failure_events_carry_the_error_raised_to_the_caller(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body=BAD_REQUEST, status=400)
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.BadRequestError) as raised:
            await call(server)
        events: Final = await recorder.wait_for_async("async_log_failure_event")

    assert_served_natively(server)
    assert isinstance(events[0].kwargs, dict)
    assert events[0].kwargs["exception"] is raised.value


@pytest.mark.asyncio
async def test_pre_api_call_sees_the_wire_body(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    response: Final = await call(server, callbacks=[recorder])

    assert_served_natively(server, response)
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    assert request_body(pre_call) == server.requests[0].body


@pytest.mark.asyncio
async def test_pre_api_call_edits_reach_the_provider(server: RecordingServer) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = 0.25

    response: Final = await call(server, callbacks=[Edit()])

    assert_served_natively(server, response)
    assert isinstance(server.requests[0].body, dict)
    assert server.requests[0].body["temperature"] == 0.25


@pytest.mark.asyncio
async def test_pre_api_call_sees_the_shaped_optional_params(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await call(server, callbacks=[recorder], temperature=0.2, top_k=3, drop_params=True)

    sent: Final = server.requests[0].body
    assert isinstance(sent, dict)
    assert not {"temperature", "top_k"} & sent.keys()
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    optional_params: Final = pre_call["optional_params"]
    assert isinstance(optional_params, dict)
    assert not {"model", "messages", "temperature", "top_k"} & optional_params.keys()
    assert optional_params["max_tokens"] == sent["max_tokens"]


@pytest.mark.asyncio
async def test_logging_hook_and_redaction_edits_reach_the_async_success_event(server: RecordingServer) -> None:
    class Rewrite(HookRecorder):
        async def async_logging_hook(
            self, kwargs: dict[str, object], result: object, call_type: str
        ) -> tuple[dict[str, object], object]:
            return {**kwargs, "logging_hook_ran": True}, result

        def redact_standard_logging_payload_from_model_call_details(
            self, model_call_details: dict[str, object]
        ) -> dict[str, object]:
            return {**model_call_details, "redacted": True}

    recorder: Final = Rewrite()

    with rebound(litellm, "callbacks", [recorder]):
        await call(server)
        success: Final = await recorder.wait_for_async("async_log_success_event")

    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["logging_hook_ran"] is True
    assert success[0].kwargs["redacted"] is True


@pytest.mark.asyncio
async def test_pre_call_deployment_hooks_compose_the_wire_request_in_registration_order(
    server: RecordingServer,
) -> None:
    class Rewrite(CustomLogger):
        def __init__(self, marker: str) -> None:
            super().__init__()
            self.marker = marker

        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            headers: Final = _HEADERS.validate_python(kwargs["extra_headers"])
            return {
                **kwargs,
                "extra_headers": {**headers, "x-deployment-hook": headers["x-deployment-hook"] + self.marker},
            }

    with rebound(litellm, "callbacks", [Rewrite("first"), Rewrite("-second")]):
        response: Final = await call(server, extra_headers={"x-deployment-hook": "caller-", "x-kept": "unchanged"})

    assert_served_natively(server, response)
    assert server.requests[0].headers["x-deployment-hook"] == "caller-first-second"
    assert server.requests[0].headers["x-kept"] == "unchanged"


@pytest.mark.asyncio
async def test_pre_call_deployment_rejection_logs_failure_without_provider_io(server: RecordingServer) -> None:
    rejection: Final = litellm.BadRequestError(
        message="deployment rejected", model="test-model", llm_provider="anthropic"
    )
    recorder: Final = RecordingLogger()

    class Reject(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            raise rejection

    with rebound(litellm, "callbacks", [Reject(), recorder]):
        with pytest.raises(litellm.BadRequestError, match="deployment rejected") as raised:
            await call(server)
        failures: Final = await recorder.wait_for_async("async_log_failure_event")

    assert raised.value is rejection
    assert not server.requests
    assert len(failures) == 1
    assert _OBJECT.validate_python(failures[0].kwargs)["exception"] is raised.value
    assert not {"log_pre_api_call", "log_success_event", "async_log_success_event"} & set(recorder.names)


@pytest.mark.asyncio
async def test_pre_call_deployment_hook_can_delete_keywords(server: RecordingServer) -> None:
    class DropKeywords(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {name: value for name, value in kwargs.items() if name not in ("system", "extra_headers")}

    with rebound(litellm, "callbacks", [DropKeywords()]):
        response: Final = await call(
            server,
            system="remove this instruction",
            extra_headers={"x-deleted": "remove this header"},
            headers={"x-kept": "keep this header"},
        )

    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert isinstance(sent.body, dict)
    assert "system" not in sent.body
    assert "x-deleted" not in sent.headers
    assert sent.headers["x-kept"] == "keep this header"


@pytest.mark.asyncio
async def test_post_call_success_deployment_hook_replaces_the_response(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    class Replace(CustomLogger):
        async def async_post_call_success_deployment_hook(
            self, request_data: dict[str, object], response: object, call_type: object
        ) -> object:
            return with_id(response, "replaced")

    with rebound(litellm, "callbacks", [Replace(), recorder]):
        response: Final = await call(server)
        success: Final = await recorder.wait_for_async("async_log_success_event")

    assert_served_natively(server, response)
    assert response_id(response) == "replaced"
    assert len(success) == 1
    assert isinstance(success[0].response, ModelResponse)
    assert success[0].response.id == response_id(response)


@pytest.mark.asyncio
async def test_concurrent_calls_keep_callback_context_metadata_and_responses_isolated(server: RecordingServer) -> None:
    context: Final = ContextVar("messages-callback-context", default="outside")
    first_entered: Final = asyncio.Event()
    second_entered: Final = asyncio.Event()
    server.enqueue(ResponseSpec(body={**MESSAGES_RESPONSE, "id": "msg_first"}))
    server.enqueue(ResponseSpec(body={**MESSAGES_RESPONSE, "id": "msg_second"}))

    class Observe(RecordingLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            arrived, other = (
                (first_entered, second_entered)
                if kwargs["litellm_call_id"] == "first"
                else (second_entered, first_entered)
            )
            arrived.set()
            await asyncio.wait_for(other.wait(), timeout=5)
            return kwargs

        def log_pre_api_call(self, model: object, messages: object, kwargs: dict[str, object]) -> None:
            self._record("log_pre_api_call", {**kwargs, "caller_context": context.get()})

        async def async_log_success_event(
            self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            self._record("async_log_success_event", {**kwargs, "caller_context": context.get()}, response_obj)

    recorder: Final = Observe()

    async def invoke(marker: str) -> object:
        token: Final = context.set(marker)
        try:
            return await call(server, litellm_call_id=marker, metadata={"marker": marker})
        finally:
            context.reset(token)

    with rebound(litellm, "callbacks", [recorder]):
        responses: Final = await asyncio.gather(invoke("first"), invoke("second"))
        success: Final = await recorder.wait_for_async("async_log_success_event", count=2)

    assert_served_natively(server, *responses)
    expected: Final = {"first": responses[0], "second": responses[1]}
    pre_call: Final = recorder.wait_for("log_pre_api_call", count=2)
    assert len(pre_call) == len(success) == 2
    assert {_OBJECT.validate_python(event.kwargs)["litellm_call_id"] for event in success} == expected.keys()
    for event in (*pre_call, *success):
        details: Final = _OBJECT.validate_python(event.kwargs)
        call_id: Final = str(details["litellm_call_id"])
        params: Final = _OBJECT.validate_python(details["litellm_params"])
        assert details["caller_context"] == call_id
        assert _OBJECT.validate_python(params["metadata"])["marker"] == call_id
    for event in success:
        logged_call_id: Final = str(_OBJECT.validate_python(event.kwargs)["litellm_call_id"])
        assert isinstance(event.response, ModelResponse)
        assert event.response.id == response_id(expected[logged_call_id])
    assert context.get() == "outside"


@pytest.mark.asyncio
async def test_a_failing_logger_does_not_fail_the_call(server: RecordingServer) -> None:
    class Broken(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            raise RuntimeError("logger exploded")

    response: Final = await call(server, callbacks=[Broken()])

    assert_served_natively(server, response)
    assert response_id(response) == MESSAGES_RESPONSE["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event,asynchronous,dynamic,expected",
    (
        pytest.param("input", False, False, 1, id="global-input"),
        pytest.param("success", True, False, 1, id="global-async-success"),
        pytest.param("success", True, True, 1, id="request-async-success"),
        pytest.param("success", False, False, 0, id="global-sync-success"),
        pytest.param("success", False, True, 0, id="request-sync-success"),
        pytest.param("failure", True, False, 1, id="global-async-failure"),
        pytest.param("failure", False, False, 1, id="global-sync-failure"),
        pytest.param("failure", False, True, 1, id="request-sync-failure"),
    ),
)
async def test_function_callback_fires_the_expected_number_of_times(
    server: RecordingServer, event: str, asynchronous: bool, dynamic: bool, expected: int
) -> None:
    recorder: Final = RecordingLogger()

    def callback(kwargs: dict[str, object], *_: object) -> None:
        recorder._record(event, kwargs)

    async def async_callback(kwargs: dict[str, object], *_: object) -> None:
        recorder._record(event, kwargs)

    registered: Final = async_callback if asynchronous else callback
    global_list: Final = f"_async_{event}_callback" if asynchronous else f"{event}_callback"
    request_callbacks: Final = {f"{event}_callback": [registered]} if dynamic else {}
    if event == "failure":
        server.default_response = ResponseSpec(body=BAD_REQUEST, status=400)

    with rebound(litellm, "callbacks", [recorder]), rebound(litellm, global_list, [] if dynamic else [registered]):
        try:
            await call(server, **request_callbacks)
        except litellm.BadRequestError:
            assert event == "failure"
        await recorder.wait_for_async(f"async_log_{'failure' if event == 'failure' else 'success'}_event")

    assert_served_natively(server)
    assert recorder.names.count(event) == expected


@pytest.mark.asyncio
async def test_pre_call_rule_rejects_before_provider_io(server: RecordingServer) -> None:
    checked: Final[list[str]] = []

    def reject(text: str) -> bool:
        checked.append(text)
        return False

    with rebound(litellm, "pre_call_rules", [reject]), pytest.raises(litellm.APIResponseValidationError):
        await call(server)

    assert checked == ["Hello"]
    assert not server.requests


@pytest.mark.asyncio
async def test_dataset_hook_returning_none_drops_the_record_from_the_success_sink(server: RecordingServer) -> None:
    datasets: Final = RecordingLogger()

    class Drop(CustomLogger):
        async def async_dataset_hook(
            self, logged_data: ArgillaItem, standard_logging_payload: StandardLoggingPayload | None
        ) -> None:
            datasets._record("async_dataset_hook", response=logged_data)

    server.enqueue(ResponseSpec(body={"items": [{"id": "callback-dataset"}]}))
    with rebound(litellm, "argilla_transformation_object", {"prompt": "messages", "answer": "response"}):
        sink: Final = ArgillaLogger(
            argilla_api_key="test-key",
            argilla_base_url=server.base_url,
            argilla_dataset_name="callback-dataset",
            flush_interval=3600,
        )
        with rebound(litellm, "callbacks", [sink, Drop()]):
            await call(server)
            await datasets.wait_for_async("async_dataset_hook")
            await sink.flush_queue()

    assert [request.method for request in server.requests] == ["GET", "POST"]


@pytest.mark.asyncio
async def test_sync_call_raises_like_python_without_logging(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    with pytest.raises(ValueError, match="anthropic_messages_handler is not implemented for sync calls"):
        await call(server, asynchronous=False, callbacks=[recorder])

    assert not server.requests
    assert not recorder.names


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_provider", (False, True))
@pytest.mark.parametrize("status", (401, 429))
async def test_inferred_provider_is_preserved_in_native_public_errors(
    server: RecordingServer, explicit_provider: bool, status: int
) -> None:
    server.default_response = ResponseSpec(
        body={"error": {"type": "provider_error", "message": "synthetic refusal"}}, status=status
    )
    error_type: Final = litellm.AuthenticationError if status == 401 else litellm.RateLimitError
    with pytest.raises(error_type) as raised:
        await call(
            server, model="deepseek/deepseek-v4-flash", custom_llm_provider="deepseek" if explicit_provider else None
        )
    assert_served_natively(server)
    public_error: Final = cast(
        _ProviderError, raised.value
    )  # cast-ok: the exception mapper supplies these public attributes
    assert public_error.llm_provider == "deepseek"
    assert public_error.model == "deepseek/deepseek-v4-flash"
    assert "synthetic refusal" in str(raised.value)


class _ProviderError(Protocol):
    @property
    def llm_provider(self) -> str: ...

    @property
    def model(self) -> str: ...


@pytest.mark.asyncio
@pytest.mark.parametrize("value", (2**64 - 1, -(2**63), "9" * 4096, 1e30))
async def test_native_response_preserves_supported_numbers_and_numeric_strings(
    server: RecordingServer, value: int | str | float
) -> None:
    server.default_response = ResponseSpec(body={**MESSAGES_RESPONSE, "future_extension": {"wide_integer": value}})
    response: Final = await call(server)
    assert_served_natively(server, response)
    body: Final = _OBJECT.validate_python(response)
    actual: Final = _OBJECT.validate_python(body["future_extension"])["wide_integer"]
    assert actual == value
    assert type(actual) is type(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", (2**64 + 1, -(2**63) - 1))
async def test_native_response_rejects_oversized_integers_instead_of_returning_rounded_success(
    server: RecordingServer, value: int
) -> None:
    server.default_response = ResponseSpec(body={**MESSAGES_RESPONSE, "future_extension": {"wide_integer": value}})
    with pytest.raises(litellm.APIError, match="response integer outside") as raised:
        await call(server)
    assert raised.value.status_code == 500
    assert not isinstance(raised.value, litellm.APIConnectionError)
    assert str(value) not in str(raised.value)
    assert_served_natively(server)


class _MessagesRouter(Protocol):
    def aanthropic_messages(self, **kwargs: object) -> Awaitable[object]: ...


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("provider_prefix", "explicit_provider", "router_alias"))
@pytest.mark.parametrize("streaming", (False, True), ids=("unary", "stream"))
async def test_platform_billing_identity_survives_wire_model_rewriting(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, entry: str, streaming: bool
) -> None:
    model: Final = "claude-billing-identity-test"
    backend: Final = f"bedrock/claude_platform/{model}"
    input_price: Final = 0.002
    output_price: Final = 0.003
    prices: Final = cast(
        dict[str, object], getattr(litellm, "model_cost")
    )  # cast-ok: model_cost is the mutable runtime registry
    monkeypatch.setitem(
        prices,
        backend,
        {
            "litellm_provider": "bedrock",
            "mode": "chat",
            "input_cost_per_token": input_price,
            "output_cost_per_token": output_price,
            "max_input_tokens": 200000,
            "max_output_tokens": 64000,
        },
    )
    provider_response: Final = {**MESSAGES_RESPONSE, "model": model}
    events: Final = (
        (
            "message_start",
            {"type": "message_start", "message": {**provider_response, "content": [], "stop_reason": None}},
        ),
        *MESSAGES_EVENTS[1:],
    )
    server.default_response = ResponseSpec(body=provider_response, events=events if streaming else ())
    recorder: Final = RecordingLogger()
    metadata: Final = {"caller_marker": "cost-context", "model_group": "billing-alias"}

    async def invoke() -> object:
        if entry == "router_alias":
            factory: Final = cast(
                Callable[..., _MessagesRouter], litellm.Router
            )  # cast-ok: Router exposes its public constructor and endpoint dynamically
            router: Final = factory(
                model_list=[
                    {
                        "model_name": "billing-alias",
                        "litellm_params": {
                            "model": backend,
                            "api_base": server.base_url,
                            "api_key": "synthetic",
                            "workspace_id": "billing-workspace",
                        },
                    }
                ],
                num_retries=0,
                disable_cooldowns=True,
            )
            return await router.aanthropic_messages(
                model="billing-alias",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                litellm_metadata=metadata,
                stream=streaming,
            )
        return await call(
            server,
            model=f"claude_platform/{model}" if entry == "explicit_provider" else backend,
            custom_llm_provider="bedrock" if entry == "explicit_provider" else None,
            workspace_id="billing-workspace",
            litellm_metadata=metadata,
            stream=streaming,
        )

    with rebound(litellm, "callbacks", [recorder]):
        response: Final = await invoke()
        if streaming:
            stream: Final = cast(AsyncIterator[bytes], response)  # cast-ok: native Messages streams yield bytes
            chunks: Final = [chunk async for chunk in stream]
            assert b"".join(chunks) == b"".join(server.default_response.payloads())
        successes: Final = await recorder.wait_for_async("async_log_success_event")
    if streaming:
        assert_served_natively(server)
    else:
        assert_served_natively(server, response)
        assert _OBJECT.validate_python(response)["model"] == model
    usage: Final = _OBJECT.validate_python(provider_response["usage"])
    expected: Final = (
        TypeAdapter(int).validate_python(usage["input_tokens"]) * input_price
        + TypeAdapter(int).validate_python(usage["output_tokens"]) * output_price
    )
    logged: Final = _OBJECT.validate_python(successes[0].kwargs)
    standard: Final = _OBJECT.validate_python(logged["standard_logging_object"])
    assert math.isclose(TypeAdapter(float).validate_python(logged["response_cost"]), expected)
    assert math.isclose(TypeAdapter(float).validate_python(standard["response_cost"]), expected)
    pre_call: Final = _OBJECT.validate_python(recorder.wait_for("log_pre_api_call")[0].kwargs)
    assert pre_call["model"] == f"claude_platform/{model}"
    assert _OBJECT.validate_python(server.requests[0].body)["model"] == model
    params: Final = _OBJECT.validate_python(logged["litellm_params"])
    assert _OBJECT.validate_python(params["metadata"])["caller_marker"] == "cost-context"
