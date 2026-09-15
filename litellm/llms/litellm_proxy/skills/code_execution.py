"""
Automatic Code Execution Handler for LiteLLM Skills

When `litellm_code_execution` tool is present, this handler automatically:
1. Makes the LLM call
2. Executes any code the model generates
3. Continues the conversation with results
4. Returns final response with generated files inline (base64)

This mimics Anthropic's behavior where code execution happens automatically.
Generated files are returned directly in the response - no separate storage needed.
"""

import base64
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Final, Protocol, TypedDict

from typing_extensions import ReadOnly

from litellm._logging import verbose_logger


class _ToolParameterSchema(TypedDict, total=False):
    type: ReadOnly[str]
    description: ReadOnly[str]


class _ToolArgumentSchema(TypedDict, total=False):
    type: ReadOnly[str]
    properties: ReadOnly[Mapping[str, _ToolParameterSchema]]
    required: ReadOnly[Sequence[str]]


class _OpenAIToolFunction(TypedDict, total=False):
    name: ReadOnly[str]
    description: ReadOnly[str]
    parameters: ReadOnly[_ToolArgumentSchema]


class _OpenAIToolSpec(TypedDict, total=False):
    type: ReadOnly[str]
    function: ReadOnly[_OpenAIToolFunction]


class _AnthropicToolSpec(TypedDict, total=False):
    name: ReadOnly[str]
    description: ReadOnly[str]
    input_schema: ReadOnly[_ToolArgumentSchema]


class _CodeExecutionArguments(TypedDict, total=False):
    code: ReadOnly[str]


class _GeneratedFile(TypedDict, total=False):
    name: ReadOnly[str]
    mime_type: ReadOnly[str]
    content_base64: ReadOnly[str]
    size: ReadOnly[int]


class _SandboxGeneratedFile(TypedDict):
    name: ReadOnly[str]
    mime_type: ReadOnly[str]
    content_base64: ReadOnly[str]


class _SandboxExecutionResult(TypedDict):
    success: ReadOnly[bool]
    output: ReadOnly[str]
    error: ReadOnly[str]
    files: ReadOnly[Sequence[_SandboxGeneratedFile]]


class _ExecutionResult(TypedDict, total=False):
    iteration: ReadOnly[int]
    success: ReadOnly[bool]
    output: ReadOnly[str]
    error: ReadOnly[str]
    files: ReadOnly[Sequence[str]]


class _ToolCallFunction(Protocol):
    name: str
    arguments: str


class _ToolCall(Protocol):
    id: str
    function: _ToolCallFunction


class _AssistantMessage(Protocol):
    content: str | None
    tool_calls: Sequence[_ToolCall] | None


class _ResponseChoice(Protocol):
    message: _AssistantMessage
    finish_reason: str | None


class _CompletionResponse(Protocol):
    choices: Sequence[_ResponseChoice]


class _CodeExecutionOutcome(TypedDict, total=False):
    response: ReadOnly[_CompletionResponse | None]
    files: ReadOnly[Sequence[_GeneratedFile]]
    execution_results: ReadOnly[Sequence[_ExecutionResult]]
    messages: ReadOnly[Sequence[dict[str, object]]]
    max_iterations_reached: ReadOnly[bool]


def _parse_code_execution_arguments(serialized_arguments: str) -> _CodeExecutionArguments:
    return json.loads(serialized_arguments)


class LiteLLMInternalTools(str, Enum):
    """
    Enum for internal LiteLLM tools that are injected into requests.

    These tools are handled automatically by LiteLLM hooks and are not
    passed to the underlying LLM provider directly.
    """

    CODE_EXECUTION = "litellm_code_execution"


def get_litellm_code_execution_tool() -> _OpenAIToolSpec:
    """
    Returns the litellm_code_execution tool definition in OpenAI format.

    This tool enables automatic code execution in a sandboxed environment
    when skills include executable Python code.
    """
    return {
        "type": "function",
        "function": {
            "name": LiteLLMInternalTools.CODE_EXECUTION.value,
            "description": "Execute Python code in a sandboxed environment. Use this to run code that generates files, processes data, or performs computations. Generated files will be returned directly.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
        },
    }


def get_litellm_code_execution_tool_anthropic() -> _AnthropicToolSpec:
    """
    Returns the litellm_code_execution tool definition in Anthropic/messages API format.

    This tool enables automatic code execution in a sandboxed environment
    when skills include executable Python code.
    """
    return {
        "name": LiteLLMInternalTools.CODE_EXECUTION.value,
        "description": "Execute Python code in a sandboxed environment. Use this to run code that generates files, processes data, or performs computations. Generated files will be returned directly.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute"}},
            "required": ["code"],
        },
    }


# Singleton tool definition for backwards compatibility
LITELLM_CODE_EXECUTION_TOOL: Final = get_litellm_code_execution_tool()


class CodeExecutionHandler:
    """
    Handles automatic code execution for LiteLLM skills.

    When enabled, this handler intercepts LLM responses with code execution
    tool calls, executes them in a sandbox, and continues the conversation
    automatically until completion.
    """

    def __init__(
        self,
        max_iterations: int | None = None,
        sandbox_timeout: int | None = None,
    ):
        from litellm.llms.litellm_proxy.skills.constants import (
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_SANDBOX_TIMEOUT,
        )

        self.max_iterations = max_iterations or DEFAULT_MAX_ITERATIONS
        self.sandbox_timeout = sandbox_timeout or DEFAULT_SANDBOX_TIMEOUT

    async def execute_with_code_execution(
        self,
        model: str,
        messages: list[dict[str, object]],
        tools: list[_OpenAIToolSpec],
        skill_files: dict[str, bytes],
        skill_id: str | None = None,
        **kwargs,
    ) -> _CodeExecutionOutcome:
        """
        Execute an LLM call with automatic code execution handling.

        This method:
        1. Makes the initial LLM call
        2. If model calls litellm_code_execution, executes the code
        3. Continues conversation with results
        4. Repeats until model stops calling tools
        5. Returns final response with generated files inline

        Args:
            model: Model to use
            messages: Initial messages
            tools: Tools including litellm_code_execution
            skill_files: Dict of skill files for execution
            skill_id: Optional skill ID for tracking
            **kwargs: Additional args for litellm.acompletion

        Returns:
            Dict with:
            - response: Final LLM response
            - files: List of generated files with content (base64)
            - execution_results: List of code execution results
        """
        import litellm
        from litellm.llms.litellm_proxy.skills.sandbox_executor import (
            SkillsSandboxExecutor,
        )

        current_messages: Final = list(messages)
        generated_files: Final[list[_GeneratedFile]] = []  # Files returned directly
        execution_results: Final[list[_ExecutionResult]] = []

        executor: Final = SkillsSandboxExecutor(timeout=self.sandbox_timeout)
        response: Any = None  # Initialize to avoid possibly unbound error

        for iteration in range(self.max_iterations):
            verbose_logger.debug("CodeExecutionHandler: Iteration %s/%s", iteration + 1, self.max_iterations)

            # Make LLM call
            response = await litellm.acompletion(
                model=model,
                messages=current_messages,
                tools=tools,
                **kwargs,
            )

            choice: _ResponseChoice = response.choices[0]
            assistant_message = choice.message
            stop_reason = choice.finish_reason

            # Build assistant message for conversation history
            assistant_msg_dict: dict[str, object] = {
                "role": "assistant",
                "content": assistant_message.content,
            }
            if assistant_message.tool_calls:
                assistant_msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ]
            current_messages.append(assistant_msg_dict)

            # Check if we're done (no tool calls or not tool_calls finish reason)
            if stop_reason != "tool_calls" or not assistant_message.tool_calls:
                verbose_logger.debug("CodeExecutionHandler: Completed after %s iterations", iteration + 1)
                return {
                    "response": response,
                    "files": generated_files,  # Files returned directly with base64 content
                    "execution_results": execution_results,
                    "messages": current_messages,
                }

            # Handle tool calls
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name

                if tool_name == LiteLLMInternalTools.CODE_EXECUTION.value:
                    # Execute code in sandbox
                    try:
                        args = _parse_code_execution_arguments(tool_call.function.arguments)
                        code = args.get("code", "")

                        verbose_logger.debug("CodeExecutionHandler: Executing code (%s chars)", len(code))

                        exec_result: _SandboxExecutionResult = executor.execute(
                            code=code,
                            skill_files=skill_files,
                        )

                        verbose_logger.debug("CodeExecutionHandler: Execution result: %s", exec_result)

                        sandbox_files: Sequence[_SandboxGeneratedFile] = exec_result["files"]

                        execution_results.append(
                            {
                                "iteration": iteration,
                                "success": exec_result["success"],
                                "output": exec_result["output"],
                                "error": exec_result["error"],
                                "files": [f["name"] for f in sandbox_files],
                            }
                        )

                        # Build tool result content
                        tool_result = exec_result["output"] or ""

                        # Collect generated files (returned directly, no storage)
                        if sandbox_files:
                            tool_result += "\n\nGenerated files:"
                            for f in sandbox_files:
                                file_content = base64.b64decode(f["content_base64"])
                                # Add to generated files list (returned in response)
                                generated_files.append(
                                    {
                                        "name": f["name"],
                                        "mime_type": f["mime_type"],
                                        "content_base64": f["content_base64"],
                                        "size": len(file_content),
                                    }
                                )
                                tool_result += f"\n- {f['name']} ({len(file_content)} bytes)"

                                verbose_logger.debug(
                                    "CodeExecutionHandler: Generated file %s (%s bytes)", f["name"], len(file_content)
                                )

                        if exec_result["error"]:
                            tool_result += f"\n\nError:\n{exec_result['error']}"

                    except Exception as e:
                        tool_result = f"Code execution failed: {e}"
                        execution_results.append(
                            {
                                "iteration": iteration,
                                "success": False,
                                "error": str(e),
                            }
                        )

                    # Add tool result to messages
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )
                else:
                    # Non-code-execution tool - pass through
                    # In a full implementation, this would call other tool handlers
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Tool '{tool_name}' not handled by code execution handler",
                        }
                    )

        # Max iterations reached
        verbose_logger.warning("CodeExecutionHandler: Max iterations (%s) reached", self.max_iterations)
        return {
            "response": response,
            "files": generated_files,
            "execution_results": execution_results,
            "messages": current_messages,
            "max_iterations_reached": True,
        }


def has_code_execution_tool(tools: list[_OpenAIToolSpec] | None) -> bool:
    """Check if litellm_code_execution tool is in the tools list."""
    if not tools:
        return False
    for tool in tools:
        func = tool.get("function", {})
        if func.get("name") == LiteLLMInternalTools.CODE_EXECUTION.value:
            return True
    return False


def add_code_execution_tool(tools: list[_OpenAIToolSpec] | None) -> list[_OpenAIToolSpec]:
    """Add litellm_code_execution tool if not already present."""
    tools = tools or []
    if not has_code_execution_tool(tools):
        tools.append(LITELLM_CODE_EXECUTION_TOOL)
    return tools


# Global handler instance
code_execution_handler: Final = CodeExecutionHandler()
