"""
Tests for the sandbox-based shell tool fallback.

When a provider lacks native shell / code-execution support (e.g. Mistral,
Cohere, Hugging Face, …), the Responses API translation layer converts the
``shell`` tool into a synthetic ``_litellm_shell`` function tool and the
handler executes commands in a sandboxed Docker container via
``SkillsSandboxExecutor``.

Covers:
  1. Mock: unsupported provider receives _litellm_shell function tool
  2. Mock: sandbox execution loop — model calls _litellm_shell, result is
     fed back, model produces final text response
  3. Mock: shell + function tools coexist for unsupported provider
  4. Mock: multiple sequential shell calls in a single conversation
  5. Mock: failed shell command propagates error to model
  6. Unit: request_has_litellm_shell_tool detection
  7. Unit: _extract_shell_tool_calls parsing
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)


SHELL_TOOL = {"type": "shell", "environment": {"type": "container_auto"}}

FUNCTION_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
    },
}


def _make_tool_call_response(
    tool_calls,
    model="mistral/mistral-large-latest",
    response_id="resp-shell-1",
):
    """Helper to build a ModelResponse with tool_calls."""
    return ModelResponse(
        id=response_id,
        created=1000,
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_text_response(
    text,
    model="mistral/mistral-large-latest",
    response_id="resp-shell-final",
):
    """Helper to build a final text ModelResponse."""
    return ModelResponse(
        id=response_id,
        created=1001,
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content=text),
            )
        ],
        usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )


# ---------------------------------------------------------------------------
# Unit tests — transformation layer
# ---------------------------------------------------------------------------


class TestSandboxShellTransformation:
    """Verify the transformation maps shell → _litellm_shell for various providers."""

    @pytest.mark.parametrize(
        "provider",
        ["mistral", "cohere", "huggingface", "ollama", "together_ai", None],
    )
    def test_shell_maps_to_litellm_shell_for_provider(self, provider):
        """Any non-native provider (or None) should get the synthetic function tool."""
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[SHELL_TOOL],
            custom_llm_provider=provider,
        )

        assert len(result_tools) == 1
        assert result_tools[0]["type"] == "function"
        fn = result_tools[0]["function"]
        assert fn["name"] == "_litellm_shell"
        assert fn["parameters"]["properties"]["command"]["type"] == "array"

    def test_shell_coexists_with_function_tools(self):
        """Shell and function tools should both appear in the output."""
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[FUNCTION_TOOL, SHELL_TOOL],
            custom_llm_provider="mistral",
        )

        assert len(result_tools) == 2
        assert result_tools[0]["function"]["name"] == "get_weather"
        assert result_tools[1]["function"]["name"] == "_litellm_shell"

    def test_native_providers_not_affected(self):
        """Anthropic/Bedrock/Vertex should still get their native mappings."""
        for provider, expected_key in [
            ("anthropic", "type"),
            ("bedrock", "type"),
            ("vertex_ai", "code_execution"),
        ]:
            result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
                tools=[SHELL_TOOL],
                custom_llm_provider=provider,
            )
            assert len(result_tools) == 1
            tool = result_tools[0]
            if expected_key == "type":
                assert tool.get("type") == "bash_20250124"
            else:
                assert expected_key in tool


# ---------------------------------------------------------------------------
# Unit tests — detection helpers
# ---------------------------------------------------------------------------


class TestShellToolDetection:

    def test_request_has_litellm_shell_tool_positive(self):
        shell_fn = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()
        assert LiteLLMCompletionResponsesConfig.request_has_litellm_shell_tool([shell_fn]) is True

    def test_request_has_litellm_shell_tool_negative(self):
        regular = {"type": "function", "function": {"name": "foo", "parameters": {}}}
        assert LiteLLMCompletionResponsesConfig.request_has_litellm_shell_tool([regular]) is False

    def test_request_has_litellm_shell_tool_empty(self):
        assert LiteLLMCompletionResponsesConfig.request_has_litellm_shell_tool([]) is False
        assert LiteLLMCompletionResponsesConfig.request_has_litellm_shell_tool(None) is False

    def test_extract_shell_tool_calls_basic(self):
        response = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["echo", "hi"]}',
                    ),
                )
            ]
        )
        calls = LiteLLMCompletionTransformationHandler._extract_shell_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["command"] == ["echo", "hi"]

    def test_extract_shell_tool_calls_ignores_other_functions(self):
        response = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="get_weather",
                        arguments='{"city": "NYC"}',
                    ),
                ),
                ChatCompletionMessageToolCall(
                    id="call_2",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["ls"]}',
                    ),
                ),
            ]
        )
        calls = LiteLLMCompletionTransformationHandler._extract_shell_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["id"] == "call_2"

    def test_extract_shell_tool_calls_string_command(self):
        """A string command (instead of array) should be wrapped in a list."""
        response = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": "whoami"}',
                    ),
                )
            ]
        )
        calls = LiteLLMCompletionTransformationHandler._extract_shell_tool_calls(response)
        assert calls[0]["command"] == ["whoami"]


# ---------------------------------------------------------------------------
# Mock tests — sandbox execution loop
# ---------------------------------------------------------------------------


class TestSandboxShellExecutionLoop:

    @pytest.mark.asyncio
    async def test_single_shell_call_then_final_response(self):
        """
        Model calls _litellm_shell once → sandbox executes → model gives text.
        """
        shell_call_response = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["echo", "hello"]}',
                    ),
                )
            ]
        )
        final_response = _make_text_response("The output was: hello")

        handler = LiteLLMCompletionTransformationHandler()
        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.litellm_completion_transformation.handler.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=final_response,
            ) as mock_acompletion,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={"success": True, "output": "hello\n", "error": "", "files": []},
            ) as mock_exec,
        ):
            result = await handler._run_shell_execution_loop(
                initial_response=shell_call_response,
                completion_args={
                    "model": "mistral/mistral-large-latest",
                    "messages": [{"role": "user", "content": "run echo hello"}],
                    "tools": [shell_fn_tool],
                },
            )

        mock_exec.assert_called_once_with(["echo", "hello"])
        mock_acompletion.assert_called_once()

        msgs = mock_acompletion.call_args.kwargs["messages"]
        assert msgs[-1]["role"] == "tool"
        assert msgs[-1]["tool_call_id"] == "call_1"
        assert "hello" in msgs[-1]["content"]

        assert result.id == "resp-shell-final"

    @pytest.mark.asyncio
    async def test_two_sequential_shell_calls(self):
        """
        Model calls _litellm_shell twice across two loop iterations.
        """
        first_shell = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["pwd"]}',
                    ),
                )
            ],
            response_id="resp-1",
        )
        second_shell = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_2",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["ls"]}',
                    ),
                )
            ],
            response_id="resp-2",
        )
        final = _make_text_response("Done. You are in /sandbox with 3 files.")

        handler = LiteLLMCompletionTransformationHandler()
        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.litellm_completion_transformation.handler.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=[second_shell, final],
            ) as mock_acompletion,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                side_effect=[
                    {"success": True, "output": "/sandbox\n", "error": "", "files": []},
                    {"success": True, "output": "a.txt\nb.txt\nc.txt\n", "error": "", "files": []},
                ],
            ) as mock_exec,
        ):
            result = await handler._run_shell_execution_loop(
                initial_response=first_shell,
                completion_args={
                    "model": "mistral/mistral-large-latest",
                    "messages": [{"role": "user", "content": "where am I and what files?"}],
                    "tools": [shell_fn_tool],
                },
            )

        assert mock_exec.call_count == 2
        assert mock_acompletion.call_count == 2
        assert result.id == "resp-shell-final"

    @pytest.mark.asyncio
    async def test_failed_command_error_propagated(self):
        """
        When a shell command fails, the error should be included in the tool
        result so the model can react.
        """
        shell_call = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["cat", "/nonexistent"]}',
                    ),
                )
            ]
        )
        final = _make_text_response("The file does not exist.")

        handler = LiteLLMCompletionTransformationHandler()
        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.litellm_completion_transformation.handler.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=final,
            ) as mock_acompletion,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={
                    "success": False,
                    "output": "",
                    "error": "cat: /nonexistent: No such file or directory",
                    "files": [],
                },
            ),
        ):
            result = await handler._run_shell_execution_loop(
                initial_response=shell_call,
                completion_args={
                    "model": "cohere/command-r-plus",
                    "messages": [{"role": "user", "content": "cat /nonexistent"}],
                    "tools": [shell_fn_tool],
                },
            )

        msgs = mock_acompletion.call_args.kwargs["messages"]
        tool_msg = msgs[-1]
        assert "STDERR" in tool_msg["content"]
        assert "No such file or directory" in tool_msg["content"]
        assert "non-zero" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_no_loop_when_model_does_not_call_shell(self):
        """If the model responds with text (no _litellm_shell call), loop is a no-op."""
        text_response = _make_text_response("I can't run commands directly.")

        handler = LiteLLMCompletionTransformationHandler()

        result = await handler._run_shell_execution_loop(
            initial_response=text_response,
            completion_args={
                "model": "mistral/mistral-large-latest",
                "messages": [{"role": "user", "content": "run ls"}],
                "tools": [],
            },
        )

        assert result is text_response

    @pytest.mark.asyncio
    async def test_parallel_shell_and_function_calls(self):
        """
        When model calls both _litellm_shell and a regular function in
        parallel, only the shell call is executed by the sandbox.
        """
        mixed_response = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_shell",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["date"]}',
                    ),
                ),
                ChatCompletionMessageToolCall(
                    id="call_weather",
                    type="function",
                    function=Function(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                ),
            ]
        )
        final = _make_text_response("Today is Feb 20. Weather in NYC is cold.")

        handler = LiteLLMCompletionTransformationHandler()
        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.litellm_completion_transformation.handler.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=final,
            ),
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={"success": True, "output": "Thu Feb 20 2026\n", "error": "", "files": []},
            ) as mock_exec,
        ):
            result = await handler._run_shell_execution_loop(
                initial_response=mixed_response,
                completion_args={
                    "model": "mistral/mistral-large-latest",
                    "messages": [{"role": "user", "content": "date and weather"}],
                    "tools": [shell_fn_tool, {"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
                },
            )

        mock_exec.assert_called_once_with(["date"])

    @pytest.mark.asyncio
    async def test_max_iterations_cap_async(self):
        """
        Loop should terminate after MAX_SHELL_ITERATIONS even if the model
        keeps calling _litellm_shell on every turn.
        """
        from litellm.responses.shell_tool_handler import MAX_SHELL_ITERATIONS

        always_shell = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_loop",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["echo", "again"]}',
                    ),
                )
            ]
        )

        handler = LiteLLMCompletionTransformationHandler()
        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.litellm_completion_transformation.handler.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=always_shell,
            ) as mock_acompletion,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={"success": True, "output": "again\n", "error": "", "files": []},
            ) as mock_exec,
        ):
            result = await handler._run_shell_execution_loop(
                initial_response=always_shell,
                completion_args={
                    "model": "mistral/mistral-large-latest",
                    "messages": [{"role": "user", "content": "loop forever"}],
                    "tools": [shell_fn_tool],
                },
            )

        assert mock_acompletion.call_count == MAX_SHELL_ITERATIONS
        assert mock_exec.call_count == MAX_SHELL_ITERATIONS
        assert result is always_shell


# ---------------------------------------------------------------------------
# Sync completion bridge shell execution loop
# ---------------------------------------------------------------------------


class TestSandboxShellExecutionLoopSync:
    """Verify the *synchronous* shell execution loop in the completion bridge."""

    def test_single_shell_call_sync(self):
        """Sync loop: model calls _litellm_shell once, then produces text."""
        shell_call = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["echo", "hello"]}',
                    ),
                )
            ]
        )
        final = _make_text_response("The output was hello.")

        handler = LiteLLMCompletionTransformationHandler()
        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.litellm_completion_transformation.handler.litellm.completion",
                return_value=final,
            ) as mock_completion,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={"success": True, "output": "hello\n", "error": "", "files": []},
            ) as mock_exec,
        ):
            result = handler._run_shell_execution_loop_sync(
                initial_response=shell_call,
                completion_args={
                    "model": "mistral/mistral-large-latest",
                    "messages": [{"role": "user", "content": "echo hello"}],
                    "tools": [shell_fn_tool],
                },
            )

        mock_exec.assert_called_once_with(["echo", "hello"])
        assert mock_completion.call_count == 1
        assert result.id == "resp-shell-final"

    def test_no_loop_when_no_shell_calls_sync(self):
        """Sync loop is a no-op when the model doesn't call _litellm_shell."""
        text_response = _make_text_response("No shell needed.")

        handler = LiteLLMCompletionTransformationHandler()

        result = handler._run_shell_execution_loop_sync(
            initial_response=text_response,
            completion_args={
                "model": "mistral/mistral-large-latest",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [],
            },
        )

        assert result is text_response

    def test_failed_command_sync(self):
        """Sync loop: failed command error is included in tool result."""
        shell_call = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["false"]}',
                    ),
                )
            ]
        )
        final = _make_text_response("The command failed.")

        handler = LiteLLMCompletionTransformationHandler()
        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.litellm_completion_transformation.handler.litellm.completion",
                return_value=final,
            ) as mock_completion,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={"success": False, "output": "", "error": "exit 1", "files": []},
            ),
        ):
            result = handler._run_shell_execution_loop_sync(
                initial_response=shell_call,
                completion_args={
                    "model": "cohere/command-r-plus",
                    "messages": [{"role": "user", "content": "run false"}],
                    "tools": [shell_fn_tool],
                },
            )

        msgs = mock_completion.call_args.kwargs["messages"]
        tool_msg = msgs[-1]
        assert "STDERR" in tool_msg["content"]
        assert "non-zero" in tool_msg["content"]
        assert result.id == "resp-shell-final"

    def test_max_iterations_cap_sync(self):
        """
        Sync loop should terminate after MAX_SHELL_ITERATIONS even if
        the model keeps calling _litellm_shell on every turn.
        """
        from litellm.responses.shell_tool_handler import MAX_SHELL_ITERATIONS

        always_shell = _make_tool_call_response(
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_loop",
                    type="function",
                    function=Function(
                        name="_litellm_shell",
                        arguments='{"command": ["echo", "again"]}',
                    ),
                )
            ]
        )

        handler = LiteLLMCompletionTransformationHandler()
        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.litellm_completion_transformation.handler.litellm.completion",
                return_value=always_shell,
            ) as mock_completion,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={"success": True, "output": "again\n", "error": "", "files": []},
            ) as mock_exec,
        ):
            result = handler._run_shell_execution_loop_sync(
                initial_response=always_shell,
                completion_args={
                    "model": "mistral/mistral-large-latest",
                    "messages": [{"role": "user", "content": "loop forever"}],
                    "tools": [shell_fn_tool],
                },
            )

        assert mock_completion.call_count == MAX_SHELL_ITERATIONS
        assert mock_exec.call_count == MAX_SHELL_ITERATIONS
        assert result is always_shell


# ---------------------------------------------------------------------------
# shell_tool_handler — utility functions for native Responses API path
# ---------------------------------------------------------------------------


class TestShellToolHandlerUtilities:
    """Tests for litellm.responses.shell_tool_handler utility functions."""

    def test_extract_shell_calls_from_responses_api_dict_output(self):
        """Extract _litellm_shell calls from ResponsesAPIResponse with dict output."""
        from litellm.responses.shell_tool_handler import (
            _extract_shell_calls_from_responses_api,
        )
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            **{
                "id": "resp-1",
                "created_at": 1000,
                "model": "xai/grok-3",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "fc_1",
                        "name": "_litellm_shell",
                        "arguments": '{"command": ["ls", "-la"]}',
                        "id": "out-1",
                        "status": "completed",
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )
        calls = _extract_shell_calls_from_responses_api(response)
        assert len(calls) == 1
        assert calls[0]["call_id"] == "fc_1"
        assert calls[0]["command"] == ["ls", "-la"]

    def test_extract_shell_calls_ignores_other_function_calls(self):
        """Only _litellm_shell calls should be extracted."""
        from litellm.responses.shell_tool_handler import (
            _extract_shell_calls_from_responses_api,
        )
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            **{
                "id": "resp-2",
                "created_at": 1000,
                "model": "xai/grok-3",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "fc_1",
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                        "id": "out-1",
                        "status": "completed",
                    },
                    {
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                ],
                "object": "response",
                "status": "completed",
            }
        )
        calls = _extract_shell_calls_from_responses_api(response)
        assert len(calls) == 0

    def test_extract_shell_calls_string_command(self):
        """A string command should be wrapped in a list."""
        from litellm.responses.shell_tool_handler import (
            _extract_shell_calls_from_responses_api,
        )
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            **{
                "id": "resp-3",
                "created_at": 1000,
                "model": "xai/grok-3",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "fc_1",
                        "name": "_litellm_shell",
                        "arguments": '{"command": "pwd"}',
                        "id": "out-1",
                        "status": "completed",
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )
        calls = _extract_shell_calls_from_responses_api(response)
        assert len(calls) == 1
        assert calls[0]["command"] == ["pwd"]

    def test_responses_api_has_shell_tool(self):
        from litellm.responses.shell_tool_handler import responses_api_has_shell_tool

        shell_fn = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()
        other_fn = {"type": "function", "function": {"name": "foo", "parameters": {}}}

        assert responses_api_has_shell_tool([shell_fn]) is True
        assert responses_api_has_shell_tool([other_fn]) is False
        assert responses_api_has_shell_tool([other_fn, shell_fn]) is True
        assert responses_api_has_shell_tool(None) is False
        assert responses_api_has_shell_tool([]) is False

    def test_build_follow_up_input(self):
        """Follow-up input should include assistant messages, function calls, and outputs."""
        from litellm.responses.shell_tool_handler import _build_follow_up_input
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            **{
                "id": "resp-1",
                "created_at": 1000,
                "model": "xai/grok-3",
                "output": [
                    {
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Running command..."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "fc_1",
                        "name": "_litellm_shell",
                        "arguments": '{"command": ["ls"]}',
                        "id": "out-1",
                        "status": "completed",
                    },
                ],
                "object": "response",
                "status": "completed",
            }
        )

        follow_up = _build_follow_up_input(
            response=response,
            shell_calls=[{"call_id": "fc_1", "command": ["ls"]}],
            shell_results=[{"call_id": "fc_1", "output": "file.txt\n"}],
        )

        assert len(follow_up) == 3
        assert follow_up[0]["type"] == "message"
        assert follow_up[0]["role"] == "assistant"
        assert follow_up[1]["type"] == "function_call"
        assert follow_up[1]["name"] == "_litellm_shell"
        assert follow_up[2]["type"] == "function_call_output"
        assert follow_up[2]["call_id"] == "fc_1"
        assert follow_up[2]["output"] == "file.txt\n"

    def test_build_follow_up_input_excludes_non_shell_function_calls(self):
        """Non-shell function calls must be excluded to avoid missing output errors."""
        from litellm.responses.shell_tool_handler import _build_follow_up_input
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            **{
                "id": "resp-mixed",
                "created_at": 1000,
                "model": "xai/grok-3",
                "output": [
                    {
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Let me check both."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "fc_shell",
                        "name": "_litellm_shell",
                        "arguments": '{"command": ["date"]}',
                        "id": "out-1",
                        "status": "completed",
                    },
                    {
                        "type": "function_call",
                        "call_id": "fc_weather",
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                        "id": "out-2",
                        "status": "completed",
                    },
                ],
                "object": "response",
                "status": "completed",
            }
        )

        follow_up = _build_follow_up_input(
            response=response,
            shell_calls=[{"call_id": "fc_shell", "command": ["date"]}],
            shell_results=[{"call_id": "fc_shell", "output": "Thu Feb 20\n"}],
        )

        types = [item["type"] for item in follow_up]
        assert types == ["message", "function_call", "function_call_output"]

        assert follow_up[1]["call_id"] == "fc_shell"
        assert follow_up[1]["name"] == "_litellm_shell"

        assert follow_up[2]["call_id"] == "fc_shell"
        assert follow_up[2]["output"] == "Thu Feb 20\n"

        call_ids_in_output = [
            item.get("call_id")
            for item in follow_up
            if item["type"] == "function_call"
        ]
        assert "fc_weather" not in call_ids_in_output


# ---------------------------------------------------------------------------
# Native Responses API shell execution loop (async)
# ---------------------------------------------------------------------------


class TestNativeResponsesAPIShellLoop:
    """Test the shell execution loop for the native Responses API path."""

    @pytest.mark.asyncio
    async def test_async_loop_single_shell_call(self):
        """
        Native path: model calls _litellm_shell once, handler executes
        via sandbox and re-invokes aresponses, getting a final text response.
        """
        from litellm.responses.shell_tool_handler import (
            run_shell_execution_loop_responses_api,
        )
        from litellm.types.llms.openai import ResponsesAPIResponse

        shell_response = ResponsesAPIResponse(
            **{
                "id": "resp-native-1",
                "created_at": 1000,
                "model": "xai/grok-3",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "fc_1",
                        "name": "_litellm_shell",
                        "arguments": '{"command": ["whoami"]}',
                        "id": "out-1",
                        "status": "completed",
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )

        final_response = ResponsesAPIResponse(
            **{
                "id": "resp-native-2",
                "created_at": 1001,
                "model": "xai/grok-3",
                "output": [
                    {
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "You are root."}],
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )

        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.main.aresponses",
                new_callable=AsyncMock,
                return_value=final_response,
            ) as mock_aresponses,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={"success": True, "output": "root\n", "error": "", "files": []},
            ) as mock_exec,
        ):
            result = await run_shell_execution_loop_responses_api(
                response=shell_response,
                model="xai/grok-3",
                tools=[shell_fn_tool],
            )

        mock_exec.assert_called_once_with(["whoami"])
        mock_aresponses.assert_called_once()
        call_kwargs = mock_aresponses.call_args
        assert call_kwargs.kwargs["previous_response_id"] == "resp-native-1"
        assert result.id == "resp-native-2"
        assert result.output[0]["content"][0]["text"] == "You are root."

    @pytest.mark.asyncio
    async def test_async_loop_no_op_when_no_shell_calls(self):
        """Loop is a no-op when there are no _litellm_shell calls in the output."""
        from litellm.responses.shell_tool_handler import (
            run_shell_execution_loop_responses_api,
        )
        from litellm.types.llms.openai import ResponsesAPIResponse

        text_response = ResponsesAPIResponse(
            **{
                "id": "resp-text",
                "created_at": 1000,
                "model": "xai/grok-3",
                "output": [
                    {
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello!"}],
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )

        result = await run_shell_execution_loop_responses_api(
            response=text_response,
            model="xai/grok-3",
            tools=[],
        )

        assert result is text_response

    @pytest.mark.asyncio
    async def test_async_loop_two_sequential_calls(self):
        """
        Native path: model calls _litellm_shell twice in sequence.
        """
        from litellm.responses.shell_tool_handler import (
            run_shell_execution_loop_responses_api,
        )
        from litellm.types.llms.openai import ResponsesAPIResponse

        first_response = ResponsesAPIResponse(
            **{
                "id": "resp-1",
                "created_at": 1000,
                "model": "volcengine/ep-xxx",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "fc_1",
                        "name": "_litellm_shell",
                        "arguments": '{"command": ["pwd"]}',
                        "id": "out-1",
                        "status": "completed",
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )

        second_response = ResponsesAPIResponse(
            **{
                "id": "resp-2",
                "created_at": 1001,
                "model": "volcengine/ep-xxx",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "fc_2",
                        "name": "_litellm_shell",
                        "arguments": '{"command": ["ls"]}',
                        "id": "out-2",
                        "status": "completed",
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )

        final_response = ResponsesAPIResponse(
            **{
                "id": "resp-3",
                "created_at": 1002,
                "model": "volcengine/ep-xxx",
                "output": [
                    {
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "You are in /sandbox with a.txt."}],
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )

        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.main.aresponses",
                new_callable=AsyncMock,
                side_effect=[second_response, final_response],
            ) as mock_aresponses,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                side_effect=[
                    {"success": True, "output": "/sandbox\n", "error": "", "files": []},
                    {"success": True, "output": "a.txt\n", "error": "", "files": []},
                ],
            ) as mock_exec,
        ):
            result = await run_shell_execution_loop_responses_api(
                response=first_response,
                model="volcengine/ep-xxx",
                tools=[shell_fn_tool],
            )

        assert mock_exec.call_count == 2
        assert mock_aresponses.call_count == 2
        assert result.id == "resp-3"

    @pytest.mark.asyncio
    async def test_async_loop_max_iterations_cap(self):
        """
        Native Responses API loop should terminate after MAX_SHELL_ITERATIONS
        even if the provider keeps returning _litellm_shell calls.
        """
        from litellm.responses.shell_tool_handler import (
            MAX_SHELL_ITERATIONS,
            run_shell_execution_loop_responses_api,
        )
        from litellm.types.llms.openai import ResponsesAPIResponse

        always_shell = ResponsesAPIResponse(
            **{
                "id": "resp-loop",
                "created_at": 1000,
                "model": "volcengine/ep-xxx",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "fc_loop",
                        "name": "_litellm_shell",
                        "arguments": '{"command": ["echo", "again"]}',
                        "id": "out-loop",
                        "status": "completed",
                    }
                ],
                "object": "response",
                "status": "completed",
            }
        )

        shell_fn_tool = LiteLLMCompletionResponsesConfig._get_litellm_shell_function_tool()

        with (
            patch(
                "litellm.responses.main.aresponses",
                new_callable=AsyncMock,
                return_value=always_shell,
            ) as mock_aresponses,
            patch(
                "litellm.llms.litellm_proxy.skills.sandbox_executor.SkillsSandboxExecutor.execute_shell_command",
                return_value={"success": True, "output": "again\n", "error": "", "files": []},
            ) as mock_exec,
        ):
            result = await run_shell_execution_loop_responses_api(
                response=always_shell,
                model="volcengine/ep-xxx",
                tools=[shell_fn_tool],
            )

        assert mock_aresponses.call_count == MAX_SHELL_ITERATIONS
        assert mock_exec.call_count == MAX_SHELL_ITERATIONS
        assert result.id == "resp-loop"
