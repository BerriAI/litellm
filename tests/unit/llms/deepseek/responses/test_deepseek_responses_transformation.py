import json
import math
from collections.abc import Sequence
from typing import Final

import httpx
import pytest
import respx
from openai.types.responses import ToolParam
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.deepseek.responses.transformation import DeepSeekResponsesConfig
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator, SyncResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse, ResponsesAPIStreamingResponse
from litellm.utils import ProviderConfigManager

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
API_URL: Final = "https://api.deepseek.com/responses"
API_KEY: Final = "synthetic-deepseek-key"
TOOLS: Final[list[ToolParam]] = [
    {"type": "function", "name": "read_file", "parameters": {"type": "object", "properties": {}}, "strict": False},
    {"type": "custom", "name": "apply_patch", "format": {"type": "text"}},
]
OUTPUT: Final[list[JsonValue]] = [
    {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Inspect first"}],
    },
    {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "read_file",
        "arguments": "{}",
        "status": "completed",
    },
    {
        "type": "custom_tool_call",
        "id": "ct_1",
        "call_id": "call_2",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
        "status": "completed",
    },
    {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Applying patch", "annotations": []}],
    },
]
USAGE: Final = {
    "input_tokens": 12,
    "output_tokens": 7,
    "total_tokens": 19,
    "input_tokens_details": {"cached_tokens": 5},
    "output_tokens_details": {"reasoning_tokens": 3},
}


def _response(model: str) -> dict[str, object]:
    return {
        "id": "resp_deepseek",
        "object": "response",
        "created_at": 1,
        "model": model,
        "status": "completed",
        "output": OUTPUT,
        "usage": USAGE,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": TOOLS,
        "store": False,
    }


@pytest.fixture(autouse=True)
def deepseek_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", API_KEY)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro"])
def test_native_responses_preserves_tools_reasoning_and_usage(model: str, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(API_URL).mock(return_value=httpx.Response(200, json=_response(model)))
    response: Final = litellm.responses(
        model=f"deepseek/{model}",
        input="Inspect and patch",
        instructions="Use the supplied tools",
        tools=TOOLS,
        reasoning={"effort": "high"},
        max_output_tokens=8,
        temperature=0.4,
        top_p=0.99,
    )
    assert route.call_count == 1
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert JSON_OBJECT.validate_json(request.content) == {
        "model": model,
        "input": "Inspect and patch",
        "instructions": "Use the supplied tools",
        "tools": TOOLS,
        "reasoning": {"effort": "high"},
        "max_output_tokens": 8,
        "temperature": 0.4,
        "top_p": 0.99,
    }
    assert isinstance(response, ResponsesAPIResponse)
    body: Final = response.model_dump(exclude_none=True)
    assert body["output"] == OUTPUT
    assert body["usage"] == USAGE
    cost: Final = litellm.completion_cost(
        completion_response=response,
        model=f"deepseek/{model}",
        custom_llm_provider="deepseek",
        call_type="responses",
        custom_cost_per_token={
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "cache_read_input_token_cost": 0.001,
        },
    )
    assert cost is not None and math.isclose(cost, 7 * 0.01 + 5 * 0.001 + 7 * 0.02)


@pytest.mark.asyncio
async def test_async_native_responses_replays_function_and_custom_tool_results(respx_mock: respx.MockRouter) -> None:
    model: Final = "deepseek-flash"
    route: Final = respx_mock.post(API_URL).mock(return_value=httpx.Response(200, json=_response(model)))
    history: Final[ResponseInputParam] = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Inspect first"}],
        },
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "file content"},
        {
            "type": "custom_tool_call",
            "id": "ct_1",
            "call_id": "call_2",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
        },
        {"type": "custom_tool_call_output", "call_id": "call_2", "output": "patch applied"},
    ]
    response: Final = await litellm.aresponses(model=f"deepseek/{model}", input=history, tools=TOOLS)
    assert route.call_count == 1
    assert JSON_OBJECT.validate_json(route.calls.last.request.content) == {
        "model": model,
        "input": history,
        "tools": TOOLS,
    }
    assert isinstance(response, ResponsesAPIResponse)
    assert response.model_dump(exclude_none=True)["output"] == OUTPUT


def _events(model: str) -> tuple[dict[str, object], ...]:
    return (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**_response(model), "status": "in_progress", "output": [], "usage": None},
        },
        {
            "type": "response.reasoning_text.delta",
            "sequence_number": 1,
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Inspect first",
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "item_id": "fc_1",
            "output_index": 1,
            "delta": "{}",
        },
        {
            "type": "response.custom_tool_call_input.delta",
            "sequence_number": 3,
            "item_id": "ct_1",
            "output_index": 2,
            "delta": "*** Begin Patch\n*** End Patch",
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 4,
            "item_id": "msg_1",
            "output_index": 3,
            "content_index": 0,
            "delta": "Applying patch",
        },
        {"type": "response.completed", "sequence_number": 5, "response": _response(model)},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_native_stream_keeps_provider_events(is_async: bool, respx_mock: respx.MockRouter) -> None:
    model: Final = "deepseek-flash"
    events: Final = _events(model)
    route: Final = respx_mock.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            content="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
            headers={"content-type": "text/event-stream"},
        )
    )
    chunks: Final = await _stream_chunks(model, is_async)
    assert route.call_count == 1
    assert JSON_OBJECT.validate_json(route.calls.last.request.content) == {
        "model": model,
        "input": "Patch",
        "stream": True,
        "tools": TOOLS,
    }
    assert [chunk.model_dump()["type"] for chunk in chunks] == [event["type"] for event in events]
    assert [chunk.model_dump(exclude_none=True)["delta"] for chunk in chunks[1:-1]] == [
        event["delta"] for event in events[1:-1]
    ]
    terminal: Final = JSON_OBJECT.validate_python(chunks[-1].model_dump(exclude_none=True)["response"])
    assert terminal["output"] == OUTPUT
    usage: Final = JSON_OBJECT.validate_python(terminal["usage"])
    assert {key: value for key, value in usage.items() if key != "cost"} == USAGE


async def _stream_chunks(model: str, is_async: bool) -> Sequence[ResponsesAPIStreamingResponse]:
    if is_async:
        stream: Final = await litellm.aresponses(model=f"deepseek/{model}", input="Patch", stream=True, tools=TOOLS)
        assert isinstance(stream, ResponsesAPIStreamingIterator)
        return [chunk async for chunk in stream]
    sync_stream: Final = litellm.responses(model=f"deepseek/{model}", input="Patch", stream=True, tools=TOOLS)
    assert isinstance(sync_stream, SyncResponsesAPIStreamingIterator)
    return tuple(sync_stream)


@pytest.mark.parametrize(
    "base", [None, "https://api.deepseek.com", "https://api.deepseek.com/beta", "https://api.deepseek.com/v1/"]
)
def test_official_chat_base_resolves_to_native_responses(base: str | None) -> None:
    assert DeepSeekResponsesConfig().get_complete_url(base, {}) == API_URL


@pytest.mark.parametrize("base", ["https://proxy.example/deepseek/v1", "https://proxy.example/deepseek/v1/responses/"])
def test_custom_base_retains_its_path(base: str) -> None:
    assert DeepSeekResponsesConfig().get_complete_url(base, {}) == "https://proxy.example/deepseek/v1/responses"


@pytest.mark.parametrize("model", [None, "deepseek-chat", "deepseek-reasoner", "deepseek-v3.2", "deepseek-unknown"])
def test_legacy_and_unknown_models_keep_chat_bridge(model: str | None) -> None:
    assert ProviderConfigManager.get_provider_responses_api_config(provider="deepseek", model=model) is None


def test_stateless_api_rejects_previous_response_id(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="previous_response_id"):
        litellm.responses(model="deepseek/deepseek-flash", input="continue", previous_response_id="resp_previous")
    assert len(respx_mock.calls) == 0


def test_stateless_api_accepts_store_false_without_forwarding_it(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(API_URL).mock(return_value=httpx.Response(200, json=_response("deepseek-flash")))
    response: Final = litellm.responses(model="deepseek/deepseek-flash", input="hello", store=False)
    assert isinstance(response, ResponsesAPIResponse)
    assert JSON_OBJECT.validate_json(route.calls.last.request.content) == {"model": "deepseek-flash", "input": "hello"}


def test_stateless_api_rejects_store_true(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="store=True"):
        litellm.responses(model="deepseek/deepseek-flash", input="hello", store=True)
    assert len(respx_mock.calls) == 0


def test_explicit_parameter_dropping_removes_unsupported_storage(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(API_URL).mock(return_value=httpx.Response(200, json=_response("deepseek-flash")))
    response: Final = litellm.responses(model="deepseek/deepseek-flash", input="hello", store=True, drop_params=True)
    assert isinstance(response, ResponsesAPIResponse)
    assert JSON_OBJECT.validate_json(route.calls.last.request.content) == {"model": "deepseek-flash", "input": "hello"}


def test_explicit_credentials_and_custom_endpoint_override_environment(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://gateway.example/tenant/responses").mock(
        return_value=httpx.Response(200, json=_response("deepseek-flash"))
    )
    response: Final = litellm.responses(
        model="deepseek/deepseek-flash",
        input="hello",
        api_base="https://gateway.example/tenant",
        api_key="explicit-key",
    )
    assert isinstance(response, ResponsesAPIResponse)
    assert route.calls.last.request.headers["authorization"] == "Bearer explicit-key"
    assert JSON_OBJECT.validate_json(route.calls.last.request.content) == {"model": "deepseek-flash", "input": "hello"}


@pytest.mark.asyncio
async def test_router_responses_uses_native_endpoint_for_chat_mode_deployment(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(API_URL).mock(return_value=httpx.Response(200, json=_response("deepseek-flash")))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "deepseek-native",
                "litellm_params": {"model": "deepseek/deepseek-flash", "api_key": API_KEY},
                "model_info": {"mode": "chat"},
            }
        ],
        num_retries=0,
    )
    response: Final[object] = await router.aresponses(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # Router exposes a dynamic SDK dispatch signature
        model="deepseek-native", input="hello", tools=TOOLS
    )
    assert isinstance(response, ResponsesAPIResponse)
    assert JSON_OBJECT.validate_json(route.calls.last.request.content) == {
        "model": "deepseek-flash",
        "input": "hello",
        "tools": TOOLS,
    }
    assert response.model_dump(exclude_none=True)["output"] == OUTPUT


@pytest.mark.parametrize(
    "status,error", [(400, litellm.BadRequestError), (401, litellm.AuthenticationError), (429, litellm.RateLimitError)]
)
def test_native_provider_errors_preserve_status(
    status: int, error: type[Exception], respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(API_URL).mock(
        return_value=httpx.Response(
            status,
            json={
                "error": {
                    "message": "synthetic upstream error",
                    "type": {400: "invalid_request_error", 401: "authentication_error", 429: "rate_limit_error"}[
                        status
                    ],
                }
            },
        )
    )
    with pytest.raises(error, match="synthetic upstream error"):
        litellm.responses(model="deepseek/deepseek-flash", input="test", num_retries=0)
    assert route.call_count == 1
