import os
import sys
import pytest
import asyncio
from typing import Optional
from unittest.mock import patch, AsyncMock, MagicMock
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import ModelResponse


sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponseAPIUsage,
    IncompleteDetails,
)
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from base_responses_api import BaseResponsesAPITest
from openai.types.responses.function_tool import FunctionTool


class TestAnthropicResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        # litellm._turn_on_debug()
        return {
            "model": "anthropic/claude-sonnet-4-5",
        }

    async def test_basic_openai_responses_delete_endpoint(self, sync_mode=False):
        pytest.skip("DELETE responses is not supported for anthropic")

    async def test_basic_openai_responses_streaming_delete_endpoint(
        self, sync_mode=False
    ):
        pytest.skip("DELETE responses is not supported for anthropic")

    async def test_basic_openai_responses_get_endpoint(self, sync_mode=False):
        pytest.skip("GET responses is not supported for anthropic")

    async def test_basic_openai_responses_cancel_endpoint(self, sync_mode=False):
        pytest.skip("CANCEL responses is not supported for anthropic")

    async def test_cancel_responses_invalid_response_id(self, sync_mode=False):
        pytest.skip("CANCEL responses is not supported for anthropic")


def test_multiturn_tool_calls():
    # Test streaming response with tools for Anthropic
    litellm._turn_on_debug()
    shell_tool = dict(
        FunctionTool(
            type="function",
            name="shell",
            description="Runs a shell command, and returns its output.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "array", "items": {"type": "string"}},
                    "workdir": {
                        "type": "string",
                        "description": "The working directory for the command.",
                    },
                },
                "required": ["command"],
            },
            strict=True,
        )
    )

    # Step 1: Initial request with the tool
    response = litellm.responses(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "make a hello world html file"}
                ],
                "type": "message",
            }
        ],
        model="anthropic/claude-haiku-4-5-20251001",
        instructions="You are a helpful coding assistant.",
        tools=[shell_tool],
    )

    print("response=", response)

    # Step 2: Send the results of the tool call back to the model
    # Get the response ID and tool call ID from the response

    response_id = response.id
    tool_call_id = None
    for item in response.output:
        if hasattr(item, "type") and item.type == "function_call":
            tool_call_id = getattr(item, "call_id", None)
            if tool_call_id:
                break

    # Validate that we got a tool call with a valid call_id
    if not tool_call_id:
        raise AssertionError(
            f"Expected a function_call with a valid call_id in response.output, but got: {response.output}"
        )

    # Use await with asyncio.run for the async function
    follow_up_response = litellm.responses(
        model="anthropic/claude-haiku-4-5-20251001",
        previous_response_id=response_id,
        input=[
            {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": '{"output":"<html>\\n<head>\\n  <title>Hello Page</title>\\n</head>\\n<body>\\n  <h1>Hi</h1>\\n  <p>Welcome to this simple webpage!</p>\\n</body>\\n</html> > index.html\\n","metadata":{"exit_code":0,"duration_seconds":0}}',
            }
        ],
        tools=[shell_tool],
    )

    print("follow_up_response=", follow_up_response)


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

    async def fake_session_handler(previous_response_id, litellm_completion_request):
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
            # ensure acompletion called once with merged trace_id
            assert mock_acompletion.call_count == 1
            assert (
                mock_acompletion.call_args.kwargs["litellm_trace_id"] == "session-trace"
            )


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
