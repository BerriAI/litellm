import json
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

import httpx
import openai
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator, SyncResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)
Provider: TypeAlias = Literal["anthropic", "gemini"]
Operation: TypeAlias = Literal["delete", "get", "cancel"]
CancelProvider: TypeAlias = Literal["openai", "azure"]


def _invoke_unsupported_response_operation(provider: Provider, operation: Operation) -> object:
    match operation:
        case "delete":
            return litellm.delete_responses(
                response_id="resp_unsupported",
                custom_llm_provider=provider,
                api_key="sk-test",
            )
        case "get":
            return litellm.get_responses(
                response_id="resp_unsupported",
                custom_llm_provider=provider,
                api_key="sk-test",
            )
        case "cancel":
            return litellm.cancel_responses(
                response_id="resp_unsupported",
                custom_llm_provider=provider,
                api_key="sk-test",
            )


async def _invoke_cancel_response(sync_mode: bool, call_kwargs: dict[str, object]) -> object:
    if sync_mode:
        return litellm.cancel_responses(**call_kwargs)
    return await litellm.acancel_responses(**call_kwargs)


async def _invoke_get_response(sync_mode: bool, call_kwargs: dict[str, object]) -> object:
    if sync_mode:
        return litellm.get_responses(**call_kwargs)
    return await litellm.aget_responses(**call_kwargs)


@pytest.mark.parametrize(
    ("provider", "operation"),
    (
        ("anthropic", "delete"),
        ("anthropic", "get"),
        ("anthropic", "cancel"),
        ("gemini", "delete"),
        ("gemini", "get"),
        ("gemini", "cancel"),
    ),
)
def test_unsupported_response_lifecycle_operation_fails_before_http(provider: Provider, operation: Operation) -> None:
    with respx.mock() as mock_router:
        with pytest.raises(litellm.APIConnectionError) as exc_info:
            _invoke_unsupported_response_operation(provider, operation)
        calls: Final = tuple(mock_router.calls)

    assert exc_info.value.status_code == 500
    assert f"not supported for {provider}" in str(exc_info.value)
    assert calls == ()


@pytest.mark.parametrize(
    ("provider", "sync_mode"),
    (("openai", True), ("openai", False), ("azure", True), ("azure", False)),
)
@pytest.mark.asyncio
async def test_cancel_responses_404_surfaces_openai_api_error_with_exact_url(
    provider: CancelProvider, sync_mode: bool
) -> None:
    response_id: Final = "resp_missing"
    error_body: Final = {
        "error": {
            "message": "Response was not found",
            "type": "invalid_request_error",
            "code": "not_found",
        }
    }
    api_base: Final = (
        "https://api.openai.com/v1" if provider == "openai" else "https://example-resource.openai.azure.com"
    )
    expected_url: Final = (
        f"{api_base}/responses/{response_id}/cancel"
        if provider == "openai"
        else f"{api_base}/openai/responses/{response_id}/cancel?api-version=2025-03-01-preview"
    )

    with respx.mock() as mock_router:
        route: Final = mock_router.post(expected_url).mock(
            return_value=httpx.Response(status_code=404, json=error_body)
        )
        call_kwargs: Final = {
            "custom_llm_provider": provider,
            "api_key": "sk-test",
            "api_base": api_base,
            "api_version": "2025-03-01-preview",
            "response_id": response_id,
        }
        with pytest.raises(openai.APIError) as exc_info:
            await _invoke_cancel_response(sync_mode, call_kwargs)
        requests: Final[tuple[respx.models.Call, ...]] = tuple(route.calls)

    assert exc_info.value.status_code == 404
    assert len(requests) == 1
    assert str(requests[0].request.url) == expected_url


StreamProvider: TypeAlias = Literal["openai", "azure", "azure_ai"]
_STORED_RESPONSE_ID: Final = "resp_background"
_STREAM_HEADERS: Final = {"content-type": "text/event-stream"}


def _stored_response(status: str) -> dict[str, object]:
    return {
        "id": _STORED_RESPONSE_ID,
        "object": "response",
        "created_at": 1761600000,
        "status": status,
        "background": True,
        "model": "gpt-5.5",
        "output": [],
        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
    }


def _sse(event: dict[str, object]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _delta_event(sequence_number: int) -> dict[str, object]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": sequence_number,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hi",
        "logprobs": [],
    }


def _replay_body(*events: dict[str, object]) -> bytes:
    return b"".join(_sse(event) for event in events)


_REPLAY_AFTER_3: Final = _replay_body(
    _delta_event(4),
    {"type": "response.completed", "sequence_number": 5, "response": _stored_response("completed")},
)


def _retrieve_route(mock_router: respx.MockRouter, provider: StreamProvider) -> respx.Route:
    match provider:
        case "openai":
            return mock_router.get(f"https://api.openai.com/v1/responses/{_STORED_RESPONSE_ID}")
        case "azure":
            return mock_router.get(f"https://example-resource.openai.azure.com/openai/responses/{_STORED_RESPONSE_ID}")
        case "azure_ai":
            return mock_router.get(
                f"https://example-resource.services.ai.azure.com/openai/v1/responses/{_STORED_RESPONSE_ID}"
            )


def _retrieve_kwargs(provider: StreamProvider) -> dict[str, object]:
    match provider:
        case "openai":
            return {"custom_llm_provider": "openai", "api_key": "sk-test", "api_base": "https://api.openai.com/v1"}
        case "azure":
            return {
                "custom_llm_provider": "azure",
                "api_key": "sk-test",
                "api_base": "https://example-resource.openai.azure.com",
                "api_version": "2025-03-01-preview",
            }
        case "azure_ai":
            return {
                "custom_llm_provider": "azure_ai",
                "api_key": "sk-test",
                "api_base": "https://example-resource.services.ai.azure.com",
            }


async def _retrieve(sync_mode: bool, **kwargs: object) -> object:
    if sync_mode:
        return litellm.get_responses(response_id=_STORED_RESPONSE_ID, **kwargs)
    return await litellm.aget_responses(response_id=_STORED_RESPONSE_ID, **kwargs)


async def _event_types(stream: object) -> tuple[str, ...]:
    if isinstance(stream, SyncResponsesAPIStreamingIterator):
        return tuple(event.type for event in stream)
    assert isinstance(stream, ResponsesAPIStreamingIterator)
    return tuple([event.type async for event in stream])


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.parametrize("provider", ("openai", "azure", "azure_ai"))
@pytest.mark.parametrize(
    ("resume_kwargs", "expected_query"),
    (
        ({"stream": True, "starting_after": 3}, {"stream": "true", "starting_after": "3"}),
        ({"stream": True, "starting_after": 0}, {"stream": "true", "starting_after": "0"}),
        ({"stream": True}, {"stream": "true"}),
        ({"extra_query": {"stream": "true", "starting_after": "3"}}, {"stream": "true", "starting_after": "3"}),
        (
            {"stream": True, "starting_after": 1, "extra_query": {"starting_after": 3, "include[]": "usage"}},
            {"stream": "true", "starting_after": "3", "include[]": "usage"},
        ),
        (
            {"stream": True, "extra_query": {"include": "usage", "limit": "1", "api-version": "2020-01-01"}},
            {"stream": "true", "include": "usage"},
        ),
    ),
)
@pytest.mark.asyncio
async def test_streamed_retrieve_replays_the_stored_events_after_the_cursor(
    provider: StreamProvider,
    sync_mode: bool,
    resume_kwargs: dict[str, object],
    expected_query: dict[str, str],
) -> None:
    with respx.mock() as mock_router:
        route: Final = _retrieve_route(mock_router, provider).mock(
            return_value=httpx.Response(200, headers=_STREAM_HEADERS, content=_REPLAY_AFTER_3)
        )
        stream: Final = await _retrieve(sync_mode, **_retrieve_kwargs(provider), **resume_kwargs)
        event_types: Final = await _event_types(stream)
        (call,) = route.calls

    sent_query: Final = dict(call.request.url.params)
    api_version: Final = {"api-version": "2025-03-01-preview"} if provider == "azure" else {}
    assert sent_query == {**expected_query, **api_version}
    assert event_types == ("response.output_text.delta", "response.completed")


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.parametrize("starting_after", (3, -1, "abc"))
@pytest.mark.asyncio
async def test_retrieve_without_stream_still_returns_the_response_object(
    sync_mode: bool, starting_after: object
) -> None:
    with respx.mock() as mock_router:
        route: Final = _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(200, json=_stored_response("completed"))
        )
        retrieved: Final = await _retrieve(sync_mode, starting_after=starting_after, **_retrieve_kwargs("openai"))
        (call,) = route.calls

    assert isinstance(retrieved, ResponsesAPIResponse)
    assert retrieved.status == "completed"
    assert dict(call.request.url.params) == {}


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_a_streamed_retrieve_answered_with_json_returns_the_response_object(sync_mode: bool) -> None:
    with respx.mock() as mock_router:
        route: Final = _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(200, json=_stored_response("completed"))
        )
        retrieved: Final = await _retrieve(sync_mode, stream=True, starting_after=3, **_retrieve_kwargs("openai"))
        (call,) = route.calls

    assert isinstance(retrieved, ResponsesAPIResponse)
    assert retrieved.status == "completed"
    assert dict(call.request.url.params) == {"stream": "true", "starting_after": "3"}


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_resuming_at_the_end_of_a_finished_stream_ends_without_events(sync_mode: bool) -> None:
    with respx.mock() as mock_router:
        _retrieve_route(mock_router, "openai").mock(return_value=httpx.Response(200, headers=_STREAM_HEADERS))
        stream: Final = await _retrieve(sync_mode, stream=True, starting_after=5, **_retrieve_kwargs("openai"))
        event_types: Final = await _event_types(stream)

    assert event_types == ()


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_resumed_stream_cut_before_its_terminal_event_raises(sync_mode: bool) -> None:
    with respx.mock() as mock_router:
        _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(200, headers=_STREAM_HEADERS, content=_replay_body(_delta_event(4)))
        )
        stream: Final = await _retrieve(sync_mode, stream=True, starting_after=3, **_retrieve_kwargs("openai"))
        with pytest.raises(litellm.APIConnectionError, match="before any terminal event"):
            await _event_types(stream)


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_a_resumed_stream_relays_the_stored_response_without_inventing_usage(sync_mode: bool) -> None:
    stored_without_usage: Final = {key: value for key, value in _stored_response("completed").items() if key != "usage"}
    with respx.mock() as mock_router:
        _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(
                200,
                headers=_STREAM_HEADERS,
                content=_replay_body(
                    _delta_event(4),
                    {"type": "response.completed", "sequence_number": 5, "response": stored_without_usage},
                ),
            )
        )
        stream: Final = await _retrieve(sync_mode, stream=True, starting_after=3, **_retrieve_kwargs("openai"))
        if isinstance(stream, SyncResponsesAPIStreamingIterator):
            events: Final = tuple(stream)
        else:
            assert isinstance(stream, ResponsesAPIStreamingIterator)
            events = tuple([event async for event in stream])

    completed: Final = events[-1]
    assert completed.type == "response.completed"
    assert completed.response.usage is None


@dataclass(frozen=True, slots=True)
class _ClosableReplay(httpx.SyncByteStream, httpx.AsyncByteStream):
    body: bytes
    closed: threading.Event = field(default_factory=threading.Event)

    def __iter__(self) -> Iterator[bytes]:
        yield self.body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body

    def close(self) -> None:
        self.closed.set()

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_closing_a_resumed_stream_early_releases_the_upstream_connection(sync_mode: bool) -> None:
    upstream: Final = _ClosableReplay(_REPLAY_AFTER_3)
    with respx.mock() as mock_router:
        _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(200, headers=_STREAM_HEADERS, stream=upstream)
        )
        stream: Final = await _retrieve(sync_mode, stream=True, starting_after=3, **_retrieve_kwargs("openai"))
        if isinstance(stream, SyncResponsesAPIStreamingIterator):
            first_sync: Final = next(stream)
            stream.close()
            first_type: Final = first_sync.type
        else:
            assert isinstance(stream, ResponsesAPIStreamingIterator)
            first_async: Final = await anext(stream)
            await stream.aclose()
            first_type = first_async.type

    assert first_type == "response.output_text.delta"
    assert upstream.closed.is_set()


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_a_rejected_resume_keeps_the_provider_error(sync_mode: bool) -> None:
    error_message: Final = "Response cannot be streamed, it is more than 5 minutes old."
    with respx.mock() as mock_router:
        route: Final = _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(
                400, json={"error": {"message": error_message, "type": "invalid_request_error", "param": None}}
            )
        )
        with pytest.raises(openai.BadRequestError, match=error_message) as exc_info:
            await _retrieve(sync_mode, stream=True, starting_after=3, **_retrieve_kwargs("openai"))
        (call,) = route.calls

    assert exc_info.value.status_code == 400
    assert dict(call.request.url.params) == {"stream": "true", "starting_after": "3"}


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_a_resume_that_times_out_raises_a_timeout(sync_mode: bool) -> None:
    with respx.mock() as mock_router:
        route: Final = _retrieve_route(mock_router, "openai").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(litellm.Timeout):
            await _retrieve(sync_mode, stream=True, timeout=1.0, **_retrieve_kwargs("openai"))
        (call,) = route.calls

    assert dict(call.request.url.params) == {"stream": "true"}


@pytest.mark.parametrize(
    "resume_kwargs",
    (
        {"stream": "maybe"},
        {"stream": True, "starting_after": -1},
        {"extra_query": {"stream": "true", "starting_after": "abc"}},
        {"extra_query": {"stream": "1"}},
        {"extra_query": {"stream": "true", "starting_after": "9" * 5000}},
    ),
)
@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_a_malformed_resume_query_is_rejected_before_any_upstream_call(
    sync_mode: bool, resume_kwargs: dict[str, object]
) -> None:
    with respx.mock(assert_all_called=False) as mock_router:
        route: Final = _retrieve_route(mock_router, "openai")
        with pytest.raises(litellm.BadRequestError) as exc_info:
            await _retrieve(sync_mode, **_retrieve_kwargs("openai"), **resume_kwargs)

    assert exc_info.value.status_code == 400
    assert route.call_count == 0


@pytest.mark.parametrize(
    ("resume_kwargs", "expected_query"),
    (
        ({"stream": "maybe", "starting_after": 3}, {}),
        ({"extra_query": {"stream": "1"}}, {}),
        ({"stream": True, "starting_after": -1}, {"stream": "true"}),
        ({"extra_query": {"stream": "true", "starting_after": "abc"}}, {"stream": "true"}),
        ({"extra_query": {"stream": "true", "starting_after": "9" * 5000}}, {"stream": "true"}),
    ),
)
@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_a_malformed_resume_query_is_dropped_under_drop_params(
    sync_mode: bool, resume_kwargs: dict[str, object], expected_query: dict[str, str]
) -> None:
    upstream: Final = (
        httpx.Response(200, headers=_STREAM_HEADERS, content=_REPLAY_AFTER_3)
        if expected_query
        else httpx.Response(200, json=_stored_response("completed"))
    )
    with respx.mock() as mock_router:
        route: Final = _retrieve_route(mock_router, "openai").mock(return_value=upstream)
        retrieved: Final = await _retrieve(sync_mode, drop_params=True, **_retrieve_kwargs("openai"), **resume_kwargs)
        streamed_types: Final = None if isinstance(retrieved, ResponsesAPIResponse) else await _event_types(retrieved)
        (call,) = route.calls

    assert dict(call.request.url.params) == expected_query
    assert (streamed_types is not None) == bool(expected_query)


_UNSUPPORTED_PROVIDERS: Final = ("xai", "manus", "volcengine")


def _unsupported_provider_kwargs(provider: str) -> dict[str, object]:
    return {
        "custom_llm_provider": provider,
        "api_key": f"{provider}-test",
        "api_base": f"https://{provider}.example/v1",
    }


def _unsupported_provider_route(mock_router: respx.MockRouter, provider: str) -> respx.Route:
    return mock_router.get(url__regex=rf"^https://{provider}\.example/.*/responses/{_STORED_RESPONSE_ID}$")


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.parametrize("provider", _UNSUPPORTED_PROVIDERS)
@pytest.mark.asyncio
async def test_streamed_retrieve_on_an_unsupported_provider_is_a_400_without_drop_params(
    sync_mode: bool, provider: str
) -> None:
    with respx.mock(assert_all_called=False) as mock_router:
        route: Final = _unsupported_provider_route(mock_router, provider)
        with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
            await _retrieve(sync_mode, stream=True, starting_after=3, **_unsupported_provider_kwargs(provider))

    assert exc_info.value.status_code == 400
    assert route.call_count == 0


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.parametrize("provider", _UNSUPPORTED_PROVIDERS)
@pytest.mark.asyncio
async def test_streamed_retrieve_on_an_unsupported_provider_drops_the_stream_under_drop_params(
    sync_mode: bool, provider: str
) -> None:
    with respx.mock() as mock_router:
        route: Final = _unsupported_provider_route(mock_router, provider).mock(
            return_value=httpx.Response(200, json=_stored_response("completed"))
        )
        retrieved: Final = await _retrieve(
            sync_mode, stream=True, starting_after=3, drop_params=True, **_unsupported_provider_kwargs(provider)
        )
        (call,) = route.calls

    assert isinstance(retrieved, ResponsesAPIResponse)
    assert dict(call.request.url.params) == {}


@pytest.mark.asyncio
async def test_a_routed_resume_requested_in_extra_query_gets_the_deployment_stream_timeout() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "background-model",
                "litellm_params": {
                    "model": "openai/gpt-5.5",
                    "api_key": "sk-test",
                    "api_base": "https://api.openai.com/v1",
                    "timeout": 2,
                    "stream_timeout": 60,
                },
            }
        ]
    )
    with respx.mock() as mock_router:
        route: Final = _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(200, headers=_STREAM_HEADERS, content=_REPLAY_AFTER_3)
        )
        stream: Final = await router.aget_responses(
            model="background-model",
            response_id=_STORED_RESPONSE_ID,
            extra_query={"stream": "true", "starting_after": "3"},
        )
        event_types: Final = await _event_types(stream)
        (call,) = route.calls

    assert event_types == ("response.output_text.delta", "response.completed")
    assert call.request.extensions["timeout"]["read"] == 60


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_stream", "router_defaults"),
    [({"stream": True}, None), ({}, {"stream": True})],
    ids=["deployment_forces_stream", "router_default_forces_stream"],
)
async def test_a_routed_plain_retrieve_where_config_forces_stream_answers_the_response_object(
    deployment_stream: dict[str, bool], router_defaults: dict[str, bool] | None
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "background-model",
                "litellm_params": {
                    "model": "openai/gpt-5.5",
                    "api_key": "sk-test",
                    "api_base": "https://api.openai.com/v1",
                    **deployment_stream,
                },
            }
        ],
        default_litellm_params=router_defaults,
    )
    with respx.mock() as mock_router:
        route: Final = _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(200, json=_stored_response("completed"))
        )
        retrieved: Final = await router.aget_responses(model="background-model", response_id=_STORED_RESPONSE_ID)
        (call,) = route.calls

    assert isinstance(retrieved, ResponsesAPIResponse)
    assert dict(call.request.url.params) == {}


@pytest.mark.asyncio
async def test_a_routed_create_on_a_deployment_that_forces_stream_still_streams() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "background-model",
                "litellm_params": {
                    "model": "openai/gpt-5.5",
                    "api_key": "sk-test",
                    "api_base": "https://api.openai.com/v1",
                    "stream": True,
                },
            }
        ]
    )
    with respx.mock() as mock_router:
        route: Final = mock_router.post("https://api.openai.com/v1/responses").mock(
            return_value=httpx.Response(200, headers=_STREAM_HEADERS, content=_REPLAY_AFTER_3)
        )
        stream: Final = await router.aresponses(model="background-model", input="hi", background=True)
        event_types: Final = await _event_types(stream)
        (call,) = route.calls

    assert event_types == ("response.output_text.delta", "response.completed")
    assert json.loads(call.request.content)["stream"] is True

class _SuccessLog(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: queue.Queue[object] = queue.Queue()

    def log_success_event(self, kwargs: object, response_obj: object, start_time: object, end_time: object) -> None:
        self.logged.put(response_obj)


def test_a_sync_streamed_retrieve_logs_success_with_the_finished_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success_log: Final = _SuccessLog()
    monkeypatch.setattr(litellm, "callbacks", [success_log])
    with respx.mock() as mock_router:
        _retrieve_route(mock_router, "openai").mock(
            return_value=httpx.Response(200, headers=_STREAM_HEADERS, content=_REPLAY_AFTER_3)
        )
        stream: Final = litellm.get_responses(
            response_id=_STORED_RESPONSE_ID, stream=True, starting_after=3, **_retrieve_kwargs("openai")
        )
        assert isinstance(stream, SyncResponsesAPIStreamingIterator)
        consumed: Final = tuple(stream)

    logged: Final = success_log.logged.get(timeout=10)

    assert len(consumed) == 2
    assert isinstance(logged, ResponsesAPIResponse)
    assert logged.status == "completed"


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_cancel_responses_reply_carries_the_id_retrieve_gives_for_the_same_run(sync_mode: bool) -> None:
    upstream_id: Final = "resp_upstream123"
    reply: Final = {
        "id": upstream_id,
        "object": "response",
        "created_at": 0,
        "status": "cancelled",
        "model": "gpt-6.1-terra",
        "output": [],
    }
    call_kwargs: Final = {
        "custom_llm_provider": "openai",
        "api_key": "sk-test",
        "response_id": upstream_id,
        "litellm_metadata": {"model_info": {"id": "deployment-1"}},
    }

    with respx.mock() as mock_router:
        mock_router.post(f"https://api.openai.com/v1/responses/{upstream_id}/cancel").mock(
            return_value=httpx.Response(status_code=200, json=reply)
        )
        mock_router.get(f"https://api.openai.com/v1/responses/{upstream_id}").mock(
            return_value=httpx.Response(status_code=200, json=reply)
        )
        cancelled: Final = await _invoke_cancel_response(sync_mode, call_kwargs)
        retrieved: Final = await _invoke_get_response(sync_mode, call_kwargs)

    assert isinstance(cancelled, ResponsesAPIResponse)
    assert isinstance(retrieved, ResponsesAPIResponse)
    assert cancelled.id == retrieved.id
    assert cancelled.id != upstream_id
