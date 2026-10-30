import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ADDRESSED_RESPONSE_ID_FIELD


class _StopForwarding(Exception):
    """Raised by the mocked (a)completion once the forwarded kwargs are captured."""


_CODEX_ADDITIONAL_TOOLS_ITEM = {
    "type": "additional_tools",
    "id": "at_codex",
    "role": "developer",
    "tools": [
        {
            "type": "namespace",
            "name": "functions",
            "description": "",
            "tools": [
                {
                    "type": "custom",
                    "name": "exec",
                    "description": "Runs a shell command.",
                    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                },
                {
                    "type": "function",
                    "name": "wait",
                    "description": "Waits for a background command.",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                },
            ],
        }
    ],
}
_CODEX_INPUT = [_CODEX_ADDITIONAL_TOOLS_ITEM, {"type": "message", "role": "user", "content": "Run ls"}]


def test_sync_fallback_hoists_additional_tools_input_items_into_chat_tools():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):  # test-quality-ok: no DI seam; the file stubs this same boundary
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="bedrock/us.openai.gpt-5.6",
                input=_CODEX_INPUT,
                responses_api_request={},
                custom_llm_provider="bedrock",
                _is_async=False,
            )

    assert [message["role"] for message in captured["messages"]] == ["user"]
    functions_by_name = {tool["function"]["name"]: tool["function"] for tool in captured["tools"]}
    assert set(functions_by_name) == {"exec", "functions__wait"}
    assert set(functions_by_name["exec"]["parameters"]["properties"]) == {"content"}


@pytest.mark.asyncio
async def test_async_fallback_returns_hoisted_nested_custom_tool_call_as_custom_tool_call():
    from litellm.responses.litellm_completion_transformation.transformation import TOOL_CALLS_CACHE
    from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

    handler = LiteLLMCompletionTransformationHandler()
    tool_call_id = "call_exec_hoisted"

    async def fake_acompletion(**kwargs):
        return ModelResponse(
            id="chatcmpl-exec",
            created=1,
            model="us.openai.gpt-5.6",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id=tool_call_id,
                                type="function",
                                function=Function(name="exec", arguments='{"content": "ls"}'),
                            )
                        ],
                    ),
                )
            ],
        )

    try:
        with patch("litellm.acompletion", fake_acompletion):  # test-quality-ok: no DI seam; file stubs this boundary
            response = await handler.response_api_handler(
                model="bedrock/us.openai.gpt-5.6",
                input=_CODEX_INPUT,
                responses_api_request={},
                custom_llm_provider="bedrock",
                _is_async=True,
            )
    finally:
        TOOL_CALLS_CACHE.delete_cache(key=tool_call_id)

    tool_calls = [(item.type, item.name, item.input) for item in response.output if item.type == "custom_tool_call"]
    assert tool_calls == [("custom_tool_call", "exec", "ls")]


class _RecordingAnthropicHandler:
    def __init__(self, reply: Mapping[str, object]) -> None:
        self.reply: Final = reply
        self.request_body: Mapping[str, object] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_body = json.loads(request.content)
        return httpx.Response(200, json=dict(self.reply), request=request)


_ANTHROPIC_MESSAGE_PAYLOAD: Final = {
    "id": "msg_turn_two",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "14"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 1},
}


@pytest.mark.asyncio
async def test_bridged_follow_up_turn_keeps_the_addressed_response_id_off_the_provider_body():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        previous_response_id="resp_turn_one",
        client=client,
        **{ADDRESSED_RESPONSE_ID_FIELD: "resp_turn_one"},
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert ADDRESSED_RESPONSE_ID_FIELD not in provider.request_body, (
        f"the addressed response id reached the provider body: {sorted(provider.request_body)}"
    )
    assert isinstance(response, ResponsesAPIResponse)
    assert [item.type for item in response.output] == ["message"]


@pytest.mark.asyncio
async def test_bridged_turn_with_a_client_field_named_previous_response_client_ids_reaches_the_provider():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        client=client,
        previous_response_client_ids=["resp_from_the_client_body"],
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert isinstance(response, ResponsesAPIResponse)


class _StoredTurns:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows: Final = rows

    async def query_raw(self, query: str, request_ids: Sequence[str]) -> list[dict[str, object]]:
        sessions: Final = {row["session_id"] for row in self.rows if row["request_id"] in request_ids}
        return [dict(row) for row in self.rows if row["session_id"] in sessions]


def _stored_mcp_turn(request_id: str) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "turn-one-session",
        "proxy_server_request": {"input": "My favorite number is 7.", "model": "claude-sonnet-4-6"},
        "response": {
            "id": request_id,
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Noted.", "annotations": []}],
                },
                {"type": "mcp_tools_fetched", "tools": [{"name": "demo-echo"}]},
            ],
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_under", ["litellm_encoded_id", "proxy_issued_id"])
async def test_bridged_next_turn_replays_a_turn_stored_under_the_id_the_client_holds(stored_under: str):
    litellm_encoded_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    proxy_issued_id: Final = "resp_proxy-issued-turn-one"
    stored_turns: Final = _StoredTurns(
        (_stored_mcp_turn(litellm_encoded_id if stored_under == "litellm_encoded_id" else proxy_issued_id),)
    )
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored_turns)):
        await litellm.aresponses(
            model="azure_ai/claude-sonnet-4-6",
            api_base="https://fake-foundry-resource.services.ai.azure.com",
            api_key="fake-api-key",
            input="Double it",
            previous_response_id=litellm_encoded_id,
            client=client,
            **{ADDRESSED_RESPONSE_ID_FIELD: proxy_issued_id},
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert [(message["role"], json.dumps(message["content"])) for message in provider.request_body["messages"]] == [
        ("user", json.dumps([{"type": "text", "text": "My favorite number is 7."}])),
        ("assistant", json.dumps([{"type": "text", "text": "Noted."}])),
        ("user", json.dumps([{"type": "text", "text": "Double it"}])),
    ]


_LOGGED_ROW: Final = TypeAdapter(dict[str, object])


class _SuccessRowRecorder(CustomLogger):
    rows: tuple[dict[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.rows = (*self.rows, _LOGGED_ROW.validate_python(kwargs["standard_logging_object"]))


@pytest.mark.asyncio
async def test_bridged_responses_stream_logs_one_responses_row_billed_from_provider_usage(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder: Final = _SuccessRowRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    prices: Final = {"input_cost_per_token": 1e-6, "output_cost_per_token": 4e-6, "cache_read_input_token_cost": 1e-7}
    litellm.register_model(
        {"deepseek/bridge-stream-model": {"litellm_provider": "deepseek", "mode": "chat", "max_tokens": 100, **prices}}
    )
    chunk: Final[dict[str, object]] = {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": 1}
    chunks: Final[tuple[dict[str, object], ...]] = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "streamed"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **chunk,
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        },
    )
    sse: Final = ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()

    with respx.mock(assert_all_called=False) as upstream:
        chat_route: Final = upstream.post("https://example.invalid/v1/chat/completions").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)
        )
        stream: Final = await litellm.aresponses(
            model="deepseek/bridge-stream-model",
            api_base="https://example.invalid/v1",
            api_key="x",
            input="hi",
            stream=True,
        )
        assert isinstance(stream, ResponsesAPIStreamingIterator)
        async for _ in stream:
            pass
        for _ in range(100):
            if recorder.rows:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)

    assert stream.completed_response is not None
    assert chat_route.call_count == 1
    assert len(recorder.rows) == 1
    row: Final = recorder.rows[0]
    logged: Final = _LOGGED_ROW.validate_python(row["response"])
    assert logged["object"] == "response"
    assert logged["status"] == "completed"
    assert row["prompt_tokens"] == 100
    assert row["completion_tokens"] == 20
    cost: Final = row["response_cost"]
    assert isinstance(cost, float)
    assert math.isclose(
        cost,
        60 * prices["input_cost_per_token"]
        + 40 * prices["cache_read_input_token_cost"]
        + 20 * prices["output_cost_per_token"],
    )
