import asyncio
import json
from typing import Final, TypeAlias

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse
from litellm.types.utils import StandardLoggingPayload, Usage
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)
_OPENAI_URL: Final = "https://api.openai.com/v1/responses"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_SyncMode: TypeAlias = bool


class _ResponsesLoggingCapture(CustomLogger):
    def __init__(self) -> None:
        self.completed: Final = asyncio.Event()
        self.payload: StandardLoggingPayload | None = None
        self.usage: object | None = None

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self.payload = TypeAdapter(StandardLoggingPayload).validate_python(kwargs["standard_logging_object"])
        if isinstance(response_obj, ResponseCompletedEvent):
            self.usage = response_obj.response.usage
        elif isinstance(response_obj, ResponsesAPIResponse):
            self.usage = response_obj.usage
        self.completed.set()


def _response_sse(response_id: str) -> str:
    completed_response: Final = _response_body(response_id)
    output_item: Final = {
        "id": f"msg_{response_id}",
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    content_part: Final = {"type": "output_text", "text": "", "annotations": []}
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**completed_response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.in_progress",
            "sequence_number": 1,
            "response": {**completed_response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": output_item,
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 3,
            "item_id": f"msg_{response_id}",
            "output_index": 0,
            "content_index": 0,
            "part": content_part,
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 4,
            "item_id": f"msg_{response_id}",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello from the mocked response",
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 5,
            "item_id": f"msg_{response_id}",
            "output_index": 0,
            "content_index": 0,
            "text": "Hello from the mocked response",
        },
        {
            "type": "response.content_part.done",
            "sequence_number": 6,
            "item_id": f"msg_{response_id}",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "Hello from the mocked response",
                "annotations": [],
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 7,
            "output_index": 0,
            "item": completed_response["output"][0],
        },
        {
            "type": "response.completed",
            "sequence_number": 8,
            "response": completed_response,
        },
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _response_body(response_id: str, store: bool = False) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1750000000,
        "status": "completed",
        "model": "gpt-4o",
        "output": [
            {
                "id": f"msg_{response_id}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from the mocked response",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        "store": store,
    }


async def _logged_openai_response(
    stream_mode: bool,
) -> ResponsesAPIResponse:
    if stream_mode:
        response_stream: Final = await litellm.aresponses(
            model="openai/gpt-4o-mini",
            api_key="sk-test",
            input="hi",
            stream=True,
        )
        events: Final = tuple([event async for event in response_stream])
        completed_events: Final = tuple(event for event in events if isinstance(event, ResponseCompletedEvent))
        assert len(completed_events) == 1
        return completed_events[0].response
    response: Final = await litellm.aresponses(
        model="openai/gpt-4o-mini",
        api_key="sk-test",
        input="hi",
    )
    assert isinstance(response, ResponsesAPIResponse)
    return response


@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_responses_exposes_provider_rate_limit_headers(sync_mode: _SyncMode) -> None:
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
        if sync_mode:
            response: Final = litellm.responses(
                model="openai/gpt-4o",
                api_key="sk-test",
                input="hi",
            )
        else:
            response: Final = await litellm.aresponses(
                model="openai/gpt-4o",
                api_key="sk-test",
                input="hi",
            )
        requests: Final = tuple(mock_router.calls)

    assert len(requests) == 1
    additional_headers: Final = _JSON_OBJECT.validate_python(response._hidden_params["additional_headers"])
    assert {f"llm_provider-{name}": additional_headers[f"llm_provider-{name}"] for name in response_headers} == {
        f"llm_provider-{name}": value for name, value in response_headers.items()
    }


@pytest.mark.asyncio
async def test_responses_preserves_created_at_and_requested_store_value() -> None:
    response_body: Final = _response_body("resp_store", store=True)

    with respx.mock() as mock_router:
        mock_router.post(_OPENAI_URL).mock(return_value=httpx.Response(status_code=200, json=response_body))
        response: Final = await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input="hi",
            store=True,
        )
        requests: Final = tuple(mock_router.calls)

    assert isinstance(response.created_at, int)
    assert response.store is True
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["store"] is True


@pytest.mark.asyncio
async def test_responses_mcp_followup_forwards_approval_and_previous_id() -> None:
    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(
                status_code=200,
                json=_response_body("resp_mcp"),
            )
        )
        first_response: Final = await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input="Search for recent weather",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": "https://mcp.example.test",
                }
            ],
        )
        second_response: Final = await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input=[
                {
                    "type": "mcp_approval_response",
                    "approve": True,
                    "approval_request_id": "approval_123",
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": "https://mcp.example.test",
                }
            ],
            previous_response_id=first_response.id,
        )
        requests: Final = tuple(route.calls)

    assert isinstance(first_response, ResponsesAPIResponse)
    assert isinstance(second_response, ResponsesAPIResponse)
    assert len(requests) == 2
    first_request: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    second_request: Final = _JSON_OBJECT.validate_json(requests[1].request.content)
    assert first_request["tools"] == [
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "https://mcp.example.test",
        }
    ]
    assert second_request["input"] == [
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": "approval_123",
        }
    ]
    assert second_request["previous_response_id"] == "resp_mcp"


@pytest.mark.parametrize("stream_mode", (False, True))
@pytest.mark.asyncio
async def test_responses_standard_logging_matches_final_response(
    stream_mode: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture: Final = _ResponsesLoggingCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    monkeypatch.setattr(litellm, "success_callback", [capture])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])

    with respx.mock() as mock_router:
        mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(
                status_code=200,
                content=_response_sse("resp_logging") if stream_mode else None,
                json=None if stream_mode else _response_body("resp_logging"),
                headers={"content-type": "text/event-stream"} if stream_mode else {},
            )
        )
        response: Final = await _logged_openai_response(stream_mode)
        requests: Final = tuple(mock_router.calls)
        await capture.completed.wait()

    payload: Final = capture.payload
    usage: Final = capture.usage
    assert payload is not None
    assert usage is not None
    assert payload["prompt_tokens"] == response.usage.input_tokens
    assert payload["completion_tokens"] == response.usage.output_tokens
    assert payload["total_tokens"] == response.usage.input_tokens + response.usage.output_tokens
    assert payload["response_cost"] > 0
    assert payload["id"] == response.id
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == [{"content": "hi", "role": "user"}]
    if isinstance(usage, Usage):
        assert usage.prompt_tokens == response.usage.input_tokens
        assert usage.completion_tokens == response.usage.output_tokens
    else:
        callback_usage: Final = _JSON_OBJECT.validate_python(usage)
        assert callback_usage["prompt_tokens"] == response.usage.input_tokens
        assert callback_usage["completion_tokens"] == response.usage.output_tokens
    assert len(requests) == 1
    logged_response: Final = _JSON_OBJECT.validate_python(payload["response"])
    final_response: Final = response.model_dump(mode="json")
    assert {key: value for key, value in logged_response.items() if key != "usage"} == {
        key: value for key, value in final_response.items() if key != "usage"
    }
    logged_usage: Final = _JSON_OBJECT.validate_python(logged_response["usage"])
    assert logged_usage["prompt_tokens"] == response.usage.input_tokens
    assert logged_usage["completion_tokens"] == response.usage.output_tokens
    assert logged_usage["total_tokens"] == response.usage.input_tokens + response.usage.output_tokens


@pytest.mark.asyncio
async def test_responses_stream_emits_valid_events_for_sync_sdk() -> None:
    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(
                status_code=200,
                content=_response_sse("resp_stream_sync"),
                headers={"content-type": "text/event-stream"},
            )
        )
        stream: Final = litellm.responses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input="hi",
            stream=True,
        )
        events: Final = tuple(stream)
        requests: Final = tuple(route.calls)

    assert len(requests) == 1
    assert tuple(event.type for event in events) == (
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    )
    assert events[0].response.id == events[-1].response.id
    assert events[0].sequence_number == 0
    assert events[2].output_index == 0
    assert events[2].item.id == "msg_resp_stream_sync"
    assert events[3].content_index == 0
    assert events[3].part.type == "output_text"
    assert events[4].delta == "Hello from the mocked response"
    assert events[5].text == "Hello from the mocked response"
    part: Final = _JSON_OBJECT.validate_python(events[6].part)
    assert part["text"] == "Hello from the mocked response"
    assert events[7].item.type == "message"
    assert isinstance(events[-1], ResponseCompletedEvent)
    assert events[-1].response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", (True, False))
async def test_router_responses_alias_uses_underlying_model(sync_mode: _SyncMode) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "openai-offline-alias",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-test",
                },
            }
        ]
    )
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

    assert isinstance(response, ResponsesAPIResponse)
    assert isinstance(response.id, str)
    assert isinstance(response.created_at, int)
    assert isinstance(response.output, list)
    assert response.status == "completed"
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_router_responses_alias_stream_uses_underlying_model() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "openai-offline-alias",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-test",
                },
            }
        ]
    )
    with respx.mock() as mock_router:
        route: Final = mock_router.post(_OPENAI_URL).mock(
            return_value=httpx.Response(
                status_code=200,
                content=_response_sse("resp_router_alias"),
                headers={"content-type": "text/event-stream"},
            )
        )
        stream: Final = await router.aresponses(
            model="openai-offline-alias",
            input="hi",
            stream=True,
        )
        events: Final = tuple([event async for event in stream])
        requests: Final = tuple(route.calls)

    assert len(events) == 9
    assert isinstance(events[-1], ResponseCompletedEvent)
    assert isinstance(events[-1].response, ResponsesAPIResponse)
    assert events[-1].response.created_at == 1750000000
    assert events[-1].response.usage.input_tokens == 2
    assert events[-1].response.usage.output_tokens == 3
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["model"] == "gpt-4o"
