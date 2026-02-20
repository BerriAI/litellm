"""
Shell tool execution loop for the native Responses API path.

When a provider has a native ``ResponsesAPIConfig`` but no native shell /
code-execution support (e.g. VolcEngine, Perplexity, XAI), the transformation
layer injects a synthetic ``_litellm_shell`` function tool.  The provider's
model may call this tool, and the ``ResponsesAPIResponse`` will contain
``function_call`` output items with ``name="_litellm_shell"``.

This module intercepts those calls, executes them in a sandboxed Docker
container via ``SkillsSandboxExecutor``, and re-invokes the Responses API with
the results until the model produces a final non-tool-call response.
"""

import json
from typing import Any, Dict, Iterable, List, Optional

from litellm._logging import verbose_logger
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIResponse

_SHELL_TOOL_NAME = LiteLLMCompletionResponsesConfig.LITELLM_SHELL_TOOL_NAME
_MAX_SHELL_ITERATIONS = 10


def _extract_shell_calls_from_responses_api(
    response: ResponsesAPIResponse,
) -> List[Dict[str, Any]]:
    """
    Extract ``_litellm_shell`` function calls from a ``ResponsesAPIResponse``.

    Returns a list of ``{"call_id": str, "command": List[str]}``.
    """
    shell_calls: List[Dict[str, Any]] = []
    for item in response.output or []:
        item_dict = item if isinstance(item, dict) else item.model_dump()
        if item_dict.get("type") != "function_call":
            continue
        if item_dict.get("name") != _SHELL_TOOL_NAME:
            continue
        try:
            args = json.loads(item_dict.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        command = args.get("command")
        call_id = item_dict.get("call_id") or item_dict.get("id")
        if not call_id:
            continue
        if isinstance(command, list):
            shell_calls.append({"call_id": call_id, "command": command})
        elif isinstance(command, str):
            shell_calls.append({"call_id": call_id, "command": [command]})
    return shell_calls


def responses_api_has_shell_tool(
    tools: Optional[Iterable[Any]],
) -> bool:
    """Return True if the tools iterable includes ``_litellm_shell``."""
    if not tools:
        return False
    for tool in tools:
        t = tool if isinstance(tool, dict) else (tool.model_dump() if hasattr(tool, "model_dump") else {})
        if t.get("type") == "function":
            fn = t.get("function") or {}
            if fn.get("name") == _SHELL_TOOL_NAME:
                return True
    return False


def _build_follow_up_input(
    response: ResponsesAPIResponse,
    shell_calls: List[Dict[str, Any]],
    shell_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build Responses API follow-up input items from the model's output and
    sandbox execution results.

    This follows the same pattern as the MCP handler: include assistant
    messages, function call items, and function_call_output items.
    """
    follow_up_input: List[Dict[str, Any]] = []

    assistant_message_content: List[Any] = []
    function_calls: List[Dict[str, Any]] = []

    for item in response.output or []:
        item_dict = item if isinstance(item, dict) else item.model_dump()
        if item_dict.get("type") == "function_call":
            function_calls.append(item_dict)
        elif item_dict.get("type") == "message":
            content = item_dict.get("content", [])
            if isinstance(content, list):
                assistant_message_content.extend(content)
            else:
                assistant_message_content.append(content)

    if assistant_message_content:
        follow_up_input.append(
            {
                "type": "message",
                "role": "assistant",
                "content": assistant_message_content,
            }
        )

    for fc in function_calls:
        follow_up_input.append(fc)

    for sr in shell_results:
        follow_up_input.append(
            {
                "type": "function_call_output",
                "call_id": sr["call_id"],
                "output": sr["output"],
            }
        )

    return follow_up_input


def _execute_shell_calls(
    shell_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Execute shell commands via sandbox and return results."""
    from litellm.llms.litellm_proxy.skills.sandbox_executor import (
        SkillsSandboxExecutor,
    )

    executor = SkillsSandboxExecutor()
    results: List[Dict[str, Any]] = []
    for sc in shell_calls:
        result = executor.execute_shell_command(sc["command"])
        output_parts: List[str] = []
        if result.get("output"):
            output_parts.append(result["output"])
        if result.get("error"):
            output_parts.append(f"STDERR:\n{result['error']}")
        if not result.get("success"):
            output_parts.append("[command exited with non-zero status]")
        results.append(
            {
                "call_id": sc["call_id"],
                "output": "\n".join(output_parts) or "(no output)",
            }
        )
    return results


async def run_shell_execution_loop_responses_api(
    response: ResponsesAPIResponse,
    model: str,
    tools: Optional[Iterable[Any]],
    **call_params: Any,
) -> ResponsesAPIResponse:
    """
    Async shell execution loop for the native Responses API path.

    Detects ``_litellm_shell`` function calls in the response, executes them
    in the sandbox, and re-invokes ``litellm.aresponses()`` with the results
    until the model produces a final response without shell calls.
    """
    from litellm.responses.main import aresponses

    current_response = response
    shell_calls = _extract_shell_calls_from_responses_api(current_response)
    if not shell_calls:
        return current_response

    for _iteration in range(_MAX_SHELL_ITERATIONS):
        if not shell_calls:
            break

        shell_results = _execute_shell_calls(shell_calls)

        follow_up_input = _build_follow_up_input(
            response=current_response,
            shell_calls=shell_calls,
            shell_results=shell_results,
        )

        verbose_logger.debug(
            "shell_tool_handler: iteration %d, executed %d command(s)",
            _iteration + 1,
            len(shell_calls),
        )

        current_response = await aresponses(
            input=follow_up_input,
            model=model,
            tools=tools,
            previous_response_id=current_response.id,
            **call_params,
        )

        if not isinstance(current_response, ResponsesAPIResponse):
            break

        shell_calls = _extract_shell_calls_from_responses_api(current_response)

    if shell_calls:
        verbose_logger.warning(
            "shell_tool_handler: max shell iterations (%d) reached",
            _MAX_SHELL_ITERATIONS,
        )

    return current_response


def run_shell_execution_loop_responses_api_sync(
    response: ResponsesAPIResponse,
    model: str,
    tools: Optional[Iterable[Any]],
    **call_params: Any,
) -> ResponsesAPIResponse:
    """
    Sync shell execution loop for the native Responses API path.

    Same as :func:`run_shell_execution_loop_responses_api` but uses
    ``litellm.responses()`` (sync) for follow-up calls.
    """
    from litellm.responses.main import responses

    current_response = response
    shell_calls = _extract_shell_calls_from_responses_api(current_response)
    if not shell_calls:
        return current_response

    for _iteration in range(_MAX_SHELL_ITERATIONS):
        if not shell_calls:
            break

        shell_results = _execute_shell_calls(shell_calls)

        follow_up_input = _build_follow_up_input(
            response=current_response,
            shell_calls=shell_calls,
            shell_results=shell_results,
        )

        verbose_logger.debug(
            "shell_tool_handler: sync iteration %d, executed %d command(s)",
            _iteration + 1,
            len(shell_calls),
        )

        current_response = responses(
            input=follow_up_input,
            model=model,
            tools=tools,
            previous_response_id=current_response.id,
            **call_params,
        )

        if not isinstance(current_response, ResponsesAPIResponse):
            break

        shell_calls = _extract_shell_calls_from_responses_api(current_response)

    if shell_calls:
        verbose_logger.warning(
            "shell_tool_handler: max sync shell iterations (%d) reached",
            _MAX_SHELL_ITERATIONS,
        )

    return current_response
