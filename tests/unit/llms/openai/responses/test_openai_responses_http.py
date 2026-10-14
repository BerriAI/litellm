import asyncio
import json
import threading
from collections.abc import Iterable
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
import respx
from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import (
    IncompleteDetails,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
)
from litellm.types.utils import StandardLoggingPayload, Usage
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)
_OPENAI_URL: Final = "https://api.openai.com/v1/responses"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_STANDARD_LOGGING_PAYLOAD: Final = TypeAdapter(StandardLoggingPayload)
_OUTPUT_TEXT: Final = "Hello from the mocked response"
_CREATED_AT: Final = 1750000000
_AsyncLoggingMode: TypeAlias = Literal["non_stream", "stream"]
_RESPONSE_FIELD_TYPES: Final = MappingProxyType(
    {
        "error": (dict, type(None)),
        "incomplete_details": (IncompleteDetails, type(None)),
        "instructions": (str, type(None)),
        "metadata": (dict,),
        "model": (str,),
        "object": (str,),
        "parallel_tool_calls": (bool, type(None)),
        "temperature": (int, float, type(None)),
        "tool_choice": (dict, str, type(None)),
        "tools": (list, type(None)),
        "top_p": (int, float, type(None)),
        "max_output_tokens": (int, type(None)),
        "previous_response_id": (str, type(None)),
        "reasoning": (dict, type(None)),
        "status": (str,),
        "text": (dict,),
        "truncation": (str, type(None)),
        "user": (str, type(None)),
        "store": (bool, type(None)),
    }
)
_STREAM_EVENT_FIELDS: Final = MappingProxyType(
    {
        "response.created": ("response",),
        "response.in_progress": ("response",),
        "response.output_item.added": ("output_index", "item"),
        "response.content_part.added": ("item_id", "output_index", "content_index", "part"),
        "response.output_text.delta": ("item_id", "output_index", "content_index", "delta"),
        "response.output_text.done": ("item_id", "output_index", "content_index", "text"),
        "response.content_part.done": ("item_id", "output_index", "content_index", "part"),
        "response.output_item.done": ("output_index", "item"),
        "response.completed": ("response",),
    }
)


class _ResponsesLoggingCapture(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.completed: Final = threading.Event()
        self.payload: StandardLoggingPayload | None = None
        self.usage: object = None

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        assert isinstance(response_obj, ResponsesAPIResponse)
        self.payload = _STANDARD_LOGGING_PAYLOAD.validate_python(kwargs["standard_logging_object"])
        self.usage = response_obj.usage
        self.completed.set()


def _response_body(response_id: str, store: bool = False, created_at: float = _CREATED_AT) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-4o",
        "output": [
            {
                "id": f"msg_{response_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _OUTPUT_TEXT, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": store,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 5,
        },
        "user": None,
        "metadata": {},
    }


def _sse(events: Iterable[dict[str, object]]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _response_events(response_id: str) -> tuple[dict[str, object], ...]:
    completed_response: Final = _response_body(response_id)
    in_progress_response: Final = {**completed_response, "status": "in_progress", "output": [], "usage": None}
    item_id: Final = f"msg_{response_id}"
    text_part: Final = {"type": "output_text", "text": _OUTPUT_TEXT, "annotations": []}
    position: Final = {"item_id": item_id, "output_index": 0, "content_index": 0}
    return (
        {"type": "response.created", "sequence_number": 0, "response": in_progress_response},
        {"type": "response.in_progress", "sequence_number": 1, "response": in_progress_response},
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 3,
            **position,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {"type": "response.output_text.delta", "sequence_number": 4, **position, "delta": _OUTPUT_TEXT},
        {"type": "response.output_text.done", "sequence_number": 5, **position, "text": _OUTPUT_TEXT},
        {"type": "response.content_part.done", "sequence_number": 6, **position, "part": text_part},
        {
            "type": "response.output_item.done",
            "sequence_number": 7,
            "output_index": 0,
            "item": completed_response["output"][0],
        },
        {"type": "response.completed", "sequence_number": 8, "response": completed_response},
    )


def _response_sse(response_id: str) -> str:
    return _sse(_response_events(response_id))


def _sse_reply(body: str) -> httpx.Response:
    return httpx.Response(status_code=200, content=body, headers={"content-type": "text/event-stream"})


def _assert_valid_response(response: object, final_chunk: bool) -> None:
    assert isinstance(response, ResponsesAPIResponse)
    assert isinstance(response.id, str)
    assert isinstance(response.created_at, int)
    assert isinstance(response.usage, ResponseAPIUsage if final_chunk else type(None))
    mistyped: Final = {
        field: type(response[field]).__name__
        for field, expected in _RESPONSE_FIELD_TYPES.items()
        if not isinstance(response[field], expected)
    }
    assert mistyped == {}
    if final_chunk and response.status == "completed":
        assert len(response.output) > 0


def _json_object(value: object) -> dict[str, object]:
    return _JSON_OBJECT.validate_python(value.model_dump(mode="json") if isinstance(value, BaseModel) else value)


def _assert_valid_stream(events: tuple[object, ...], item_id: str) -> None:
    event_types: Final = tuple(getattr(event, "type", None) for event in events)
    assert event_types == tuple(_STREAM_EVENT_FIELDS)
    missing_fields: Final = {
        event_type: tuple(name for name in fields if getattr(event, name, None) is None)
        for (event_type, fields), event in zip(_STREAM_EVENT_FIELDS.items(), events)
    }
    assert all(fields == () for fields in missing_fields.values()), missing_fields
    created: Final = getattr(events[0], "response", None)
    _assert_valid_response(created, final_chunk=False)
    _assert_valid_response(getattr(events[1], "response", None), final_chunk=False)
    completed: Final = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    _assert_valid_response(completed.response, final_chunk=True)
    assert completed.response.id == getattr(created, "id", None)
    assert completed.response.output_text == _OUTPUT_TEXT
    assert tuple(getattr(event, "sequence_number", None) for event in events) == tuple(range(len(events)))
    item_ids: Final = (
        getattr(getattr(events[2], "item", None), "id", None),
        *(getattr(event, "item_id", None) for event in events[3:7]),
        getattr(getattr(events[7], "item", None), "id", None),
    )
    assert item_ids == (item_id,) * 6
    assert tuple(getattr(event, "output_index", None) for event in events[2:8]) == (0,) * 6
    assert tuple(getattr(event, "content_index", None) for event in events[3:7]) == (0,) * 4
    assert _json_object(getattr(events[3], "part", None)) == {
        "type": "output_text",
        "text": "",
        "annotations": [],
    }
    assert getattr(events[4], "delta", None) == _OUTPUT_TEXT
    assert getattr(events[5], "text", None) == _OUTPUT_TEXT
    assert _json_object(getattr(events[6], "part", None)) == {
        "type": "output_text",
        "text": _OUTPUT_TEXT,
        "annotations": [],
    }
    assert getattr(getattr(events[7], "item", None), "type", None) == "message"
    assert tuple(item.id for item in completed.response.output) == (item_id,)


def _completed_response(events: tuple[object, ...]) -> ResponsesAPIResponse:
    completed_events: Final = tuple(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert len(completed_events) == 1
    return completed_events[0].response


def _install_capture(monkeypatch: pytest.MonkeyPatch) -> _ResponsesLoggingCapture:
    capture: Final = _ResponsesLoggingCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    monkeypatch.setattr(litellm, "success_callback", [capture])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    return capture


def _assert_logged_payload_matches(capture: _ResponsesLoggingCapture, response: ResponsesAPIResponse) -> None:
    payload: Final = capture.payload
    usage: Final = capture.usage
    assert payload is not None
    assert response.usage is not None
    assert payload["prompt_tokens"] == response.usage.input_tokens
    assert payload["completion_tokens"] == response.usage.output_tokens
    assert payload["total_tokens"] == response.usage.input_tokens + response.usage.output_tokens
    assert payload["response_cost"] > 0
    assert payload["id"] == response.id
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == [{"content": "hi", "role": "user"}]
    callback_usage: Final = usage.model_dump() if isinstance(usage, Usage) else _JSON_OBJECT.validate_python(usage)
    assert callback_usage["prompt_tokens"] == response.usage.input_tokens
    assert callback_usage["completion_tokens"] == response.usage.output_tokens
    logged_response: Final = _JSON_OBJECT.validate_python(payload["response"])
    final_response: Final = response.model_dump(mode="json")
    assert {key: value for key, value in logged_response.items() if key != "usage"} == {
        key: value for key, value in final_response.items() if key != "usage"
    }
    logged_usage: Final = _JSON_OBJECT.validate_python(logged_response["usage"])
    assert logged_usage["prompt_tokens"] == response.usage.input_tokens
    assert logged_usage["completion_tokens"] == response.usage.output_tokens
    assert logged_usage["total_tokens"] == response.usage.total_tokens


async def _async_logged_openai_response(mode: _AsyncLoggingMode) -> ResponsesAPIResponse:
    match mode:
        case "stream":
            response_stream: Final = await litellm.aresponses(
                model="openai/gpt-4o-mini", api_key="sk-test", input="hi", stream=True
            )
            return _completed_response(tuple([event async for event in response_stream]))
        case "non_stream":
            response: Final = await litellm.aresponses(model="openai/gpt-4o-mini", api_key="sk-test", input="hi")
            assert isinstance(response, ResponsesAPIResponse)
            return response


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_responses_exposes_provider_rate_limit_headers(sync_mode: bool) -> None:
    response_headers: Final = {
        "x-ratelimit-limit-requests": "500",
        "x-ratelimit-remaining-requests": "499",
        "x-ratelimit-reset-requests": "1s",
    }

    with respx.mock() as mock_router:
        mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(
                status_code=200,
                json=_response_body("resp_rate_limit"),
                headers=response_headers,
            )
        )
        response: Final = (
            litellm.responses(model="openai/gpt-4o", api_key="sk-test", input="hi")
            if sync_mode
            else await litellm.aresponses(model="openai/gpt-4o", api_key="sk-test", input="hi")
        )
        requests: Final = tuple(mock_router.calls)

    assert isinstance(response, ResponsesAPIResponse)
    assert len(requests) == 1
    additional_headers: Final = _JSON_OBJECT.validate_python(response.hidden_params["additional_headers"])
    raw_headers: Final = _JSON_OBJECT.validate_python(response.hidden_params["headers"])
    assert {name: additional_headers[f"llm_provider-{name}"] for name in response_headers} == response_headers
    assert {name: raw_headers[name] for name in response_headers} == response_headers


@pytest.mark.asyncio
async def test_responses_converts_created_at_to_int_and_forwards_store() -> None:
    response_body: Final = _response_body("resp_store", store=True, created_at=_CREATED_AT + 0.75)

    with respx.mock() as mock_router:
        mock_router.post(_OPENAI_URL).mock(return_value=httpx.Response(status_code=200, json=response_body))
        response: Final = await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input="hi",
            store=True,
        )
        requests: Final = tuple(mock_router.calls)

    assert isinstance(response, ResponsesAPIResponse)
    assert type(response.created_at) is int
    assert response.created_at == _CREATED_AT
    assert response.store is True
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["store"] is True


def test_chat_completion_bridges_responses_only_model_tools() -> None:
    tools: Final = [
        {"type": "web_search_preview"},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ]

    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(status_code=200, json=_response_body("resp_bridge"))
        )
        response: Final = litellm.completion(
            model="openai/gpt-5.5-pro",
            api_key="sk-test",
            messages=[{"role": "user", "content": "Summarize this page."}],
            tools=tools,
        )
        requests: Final = tuple(route.calls)

    assert response.choices[0].message.content == _OUTPUT_TEXT
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["model"] == "gpt-5.5-pro"
    assert request_body["tools"] == tools
    assert request_body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Summarize this page."}],
        }
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_codex_chat_completion_uses_responses_endpoint(sync_mode: bool) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "openai-codex-mini-latest",
                "litellm_params": {"model": "openai/gpt-5.3-codex", "api_key": "sk-test"},
            }
        ]
    )
    messages: Final = [{"role": "user", "content": "Hey!"}]

    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(status_code=200, json=_response_body("resp_codex"))
        )
        response: Final = (
            router.completion(model="openai-codex-mini-latest", messages=messages)
            if sync_mode
            else await router.acompletion(model="openai-codex-mini-latest", messages=messages)
        )
        requests: Final = tuple(route.calls)

    assert response.choices[0].message.content == _OUTPUT_TEXT
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["model"] == "gpt-5.3-codex"
    assert request_body["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hey!"}]}
    ]


def test_codex_chat_completion_stream_strips_cache_control_and_transforms_tools() -> None:
    messages: Final = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Use tools carefully.", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "user", "content": "Find the value."},
    ]
    tools: Final = [
        {
            "type": "function",
            "cache_control": {"type": "ephemeral"},
            "function": {
                "name": "lookup_value",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            },
        }
    ]

    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(return_value=_sse_reply(_response_sse("resp_codex_stream")))
        stream: Final = litellm.completion(
            model="openai/gpt-5.3-codex",
            api_key="sk-test",
            messages=messages,
            tools=tools,
            stream=True,
        )
        chunks: Final = tuple(stream)
        requests: Final = tuple(route.calls)

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == _OUTPUT_TEXT
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["stream"] is True
    assert request_body["input"] == [
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "Use tools carefully."}],
        },
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Find the value."}]}
    ]
    assert request_body["tools"] == [
        {
            "type": "function",
            "name": "lookup_value",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            "strict": None,
        }
    ]


@pytest.mark.asyncio
async def test_responses_mcp_followup_forwards_approval_and_previous_id() -> None:
    mcp_tools: Final = [
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer mcp-test-token"},
        }
    ]
    approval: Final = [{"type": "mcp_approval_response", "approve": True, "approval_request_id": "approval_123"}]

    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(status_code=200, json=_response_body("resp_mcp"))
        )
        first_response: Final = await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input="Search for recent weather",
            tools=mcp_tools,
        )
        assert isinstance(first_response, ResponsesAPIResponse)
        second_response: Final = await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input=approval,
            tools=mcp_tools,
            previous_response_id=first_response.id,
        )
        requests: Final = tuple(route.calls)

    assert isinstance(second_response, ResponsesAPIResponse)
    assert len(requests) == 2
    first_request: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    second_request: Final = _JSON_OBJECT.validate_json(requests[1].request.content)
    assert first_request["tools"] == mcp_tools
    assert second_request["tools"] == mcp_tools
    assert second_request["input"] == approval
    assert second_request["previous_response_id"] == "resp_mcp"


@pytest.mark.parametrize("mode", ("non_stream", "stream"))
@pytest.mark.asyncio
async def test_responses_standard_logging_matches_final_response(
    mode: _AsyncLoggingMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture: Final = _install_capture(monkeypatch)

    with respx.mock() as mock_router:
        mock_router.post(_OPENAI_URL).mock(
            return_value=(
                _sse_reply(_response_sse("resp_logging"))
                if mode == "stream"
                else httpx.Response(status_code=200, json=_response_body("resp_logging"))
            )
        )
        response: Final = await _async_logged_openai_response(mode)
        requests: Final = tuple(mock_router.calls)
        logged: Final = await asyncio.to_thread(capture.completed.wait, 10)

    assert logged
    assert len(requests) == 1
    _assert_logged_payload_matches(capture, response)


def test_sync_stream_standard_logging_matches_final_response(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: Final = _install_capture(monkeypatch)

    with respx.mock() as mock_router:
        mock_router.post(_OPENAI_URL).mock(return_value=_sse_reply(_response_sse("resp_sync_logging")))
        stream: Final = litellm.responses(model="openai/gpt-4o-mini", api_key="sk-test", input="hi", stream=True)
        response: Final = _completed_response(tuple(stream))
        requests: Final = tuple(mock_router.calls)
        logged: Final = capture.completed.wait(10)

    assert logged
    assert len(requests) == 1
    _assert_logged_payload_matches(capture, response)


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_responses_stream_emits_valid_events(sync_mode: bool) -> None:
    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(return_value=_sse_reply(_response_sse("resp_stream")))
        events: Final = (
            tuple(litellm.responses(model="openai/gpt-4o", api_key="sk-test", input="hi", stream=True))
            if sync_mode
            else tuple(
                [
                    event
                    async for event in await litellm.aresponses(
                        model="openai/gpt-4o", api_key="sk-test", input="hi", stream=True
                    )
                ]
            )
        )
        requests: Final = tuple(route.calls)

    assert len(requests) == 1
    _assert_valid_stream(events, "msg_resp_stream")


@pytest.mark.asyncio
async def test_responses_stream_error_event_raises_bad_request() -> None:
    message: Final = "Your input exceeds the context window of this model."
    created: Final = _response_events("resp_too_long")[0]
    error_event: Final = {
        "type": "error",
        "sequence_number": 1,
        "error": {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": message,
            "param": "input",
        },
    }

    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(return_value=_sse_reply(_sse((created, error_event))))
        stream: Final = await litellm.aresponses(
            model="openai/gpt-5-mini", api_key="sk-test", input="oversized prompt", stream=True
        )
        with pytest.raises(litellm.BadRequestError) as exc_info:
            _ = [event async for event in stream]
        requests: Final = tuple(route.calls)

    assert len(requests) == 1
    assert exc_info.value.status_code == 400
    assert message in str(exc_info.value)


def _alias_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "openai-offline-alias",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", (True, False))
async def test_router_responses_alias_uses_underlying_model(sync_mode: bool) -> None:
    router: Final = _alias_router()
    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(status_code=200, json=_response_body("resp_router_alias"))
        )
        response: Final = (
            router.responses(model="openai-offline-alias", input="hi")
            if sync_mode
            else await router.aresponses(model="openai-offline-alias", input="hi")
        )
        requests: Final = tuple(route.calls)

    _assert_valid_response(response, final_chunk=True)
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["model"] == "gpt-4o"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", (True, False))
async def test_router_responses_alias_stream_uses_underlying_model(sync_mode: bool) -> None:
    router: Final = _alias_router()
    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(return_value=_sse_reply(_response_sse("resp_router_stream")))
        events: Final = (
            tuple(router.responses(model="openai-offline-alias", input="hi", stream=True))
            if sync_mode
            else tuple(
                [
                    event
                    async for event in await router.aresponses(model="openai-offline-alias", input="hi", stream=True)
                ]
            )
        )
        requests: Final = tuple(route.calls)

    _assert_valid_stream(events, "msg_resp_router_stream")
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["model"] == "gpt-4o"
