from collections.abc import Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import ModelResponse


def test_response_api_handler_merges_metadata_and_service_tier_without_error():
    """Sync path must merge kwargs like async; double-splat raises TypeError."""
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.completion", new_callable=MagicMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        handler.response_api_handler(
            model="test",
            input="hi",
            responses_api_request={},
            metadata={"trace": "abc"},
            service_tier="auto",
        )
        assert mock_completion.call_count == 1
        assert mock_completion.call_args.kwargs["metadata"] == {"trace": "abc"}
        assert mock_completion.call_args.kwargs["service_tier"] == "auto"


@pytest.mark.asyncio
async def test_async_response_api_handler_merges_trace_id_without_error():
    handler = LiteLLMCompletionTransformationHandler()

    async def fake_session_handler(
        previous_response_id: str, litellm_completion_request: dict[str, object], instructions: str | None = None
    ) -> dict[str, object]:
        litellm_completion_request["litellm_trace_id"] = "session-trace"
        return litellm_completion_request

    with patch.object(
        LiteLLMCompletionResponsesConfig,
        "async_responses_api_session_handler",
        side_effect=fake_session_handler,
    ):
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = ModelResponse(
                id="id", created=0, model="test", object="chat.completion", choices=[]
            )
            await handler.async_response_api_handler(
                litellm_completion_request={"model": "test"},
                request_input="hi",
                responses_api_request={"previous_response_id": "123"},
                litellm_trace_id="original-trace",
            )

            assert mock_acompletion.call_count == 1
            assert mock_acompletion.call_args.kwargs["litellm_trace_id"] == "session-trace"


@pytest.mark.asyncio
async def test_aresponses_forwards_timeout_to_acompletion():
    """Regression test: timeout passed to aresponses() must reach acompletion()
    on the completion transformation path (Anthropic, Bedrock, Vertex etc.).

    Previously, `timeout` was a named param of `responses()` but was NOT
    forwarded to `litellm_completion_transformation_handler.response_api_handler`,
    so it was silently dropped — `Router(timeout=N)` was a no-op for Anthropic
    and similar providers, with calls falling back to the provider SDK default
    (~600s for Anthropic).
    """
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            id="id",
            created=0,
            model="anthropic/claude-sonnet-4-5",
            object="chat.completion",
            choices=[],
        )

        await litellm.aresponses(
            model="anthropic/claude-sonnet-4-5",
            input="hello",
            timeout=42,
            api_key="sk-ant-fake",
        )

    assert mock_acompletion.call_count == 1
    forwarded_timeout = mock_acompletion.call_args.kwargs.get("timeout")
    assert forwarded_timeout == 42, (
        f"timeout was not forwarded to acompletion (got {forwarded_timeout!r}); "
        "this means Router(timeout=N) silently fails for providers on the "
        "completion transformation path."
    )


class _FakeSpendLogsDB:
    def __init__(self, spend_logs: Sequence[Mapping[str, object]]) -> None:
        self._spend_logs = spend_logs

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        return self._spend_logs


class _FakePrismaClient:
    def __init__(self, spend_logs: Sequence[Mapping[str, object]]) -> None:
        self.db = _FakeSpendLogsDB(spend_logs)


class _AnthropicBlock(BaseModel):
    type: str
    id: str | None = None
    tool_use_id: str | None = None
    text: str | None = None


class _AnthropicMessage(BaseModel):
    role: str
    content: tuple[_AnthropicBlock, ...]


class _AnthropicRequest(BaseModel):
    messages: tuple[_AnthropicMessage, ...]
    system: tuple[_AnthropicBlock, ...]


@pytest.mark.asyncio
async def test_previous_response_id_tool_output_with_new_instructions_builds_valid_anthropic_request():
    """
    A continuation that resends `instructions` must not land a system message between the replayed
    tool_use and its tool_result, and the previous turn's instructions do not carry over (OpenAI semantics)
    """
    model: Final = "anthropic/claude-sonnet-5-5"
    first_turn: Final = {
        "request_id": "chatcmpl-first-turn",
        "call_type": "aresponses",
        "session_id": "session-1",
        "proxy_server_request": {
            "model": model,
            "input": "What is the weather in Tokyo?",
            "instructions": "Be terse.",
        },
        "response": {
            "id": "chatcmpl-first-turn",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "toolu_weather",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                            }
                        ],
                    },
                }
            ],
        },
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([first_turn])),
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
    ):
        mock_acompletion.return_value = ModelResponse(
            id="id", created=0, model=model, object="chat.completion", choices=[]
        )
        await litellm.aresponses(
            model=model,
            previous_response_id="chatcmpl-first-turn",
            input=[{"type": "function_call_output", "call_id": "toolu_weather", "output": "47C"}],
            instructions="Answer in French.",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            api_key="sk-ant-fake",
        )

    anthropic_request: Final = _AnthropicRequest.model_validate(
        AnthropicConfig().transform_request(
            model="claude-sonnet-5-5",
            messages=mock_acompletion.call_args.kwargs["messages"],
            optional_params={},
            litellm_params={},
            headers={},
        )
    )

    assert [
        (message.role, [(block.type, block.id or block.tool_use_id or block.text) for block in message.content])
        for message in anthropic_request.messages
    ] == [
        ("user", [("text", "What is the weather in Tokyo?")]),
        ("assistant", [("tool_use", "toolu_weather")]),
        ("user", [("tool_result", "toolu_weather")]),
    ]
    assert [block.text for block in anthropic_request.system] == ["Answer in French."]
