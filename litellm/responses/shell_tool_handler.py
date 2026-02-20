"""
Shell tool execution helpers and execution loops.

Shared utilities for formatting sandbox results and executing shell commands
are used by both:

1. The **Chat Completion bridge** (``handler.py``) — providers without a native
   Responses API that go through ``litellm.completion()``.
2. The **native Responses API path** (this module's ``run_shell_execution_loop_*``
   functions) — providers with a native ``ResponsesAPIConfig`` but no native
   shell / code-execution support (e.g. VolcEngine, Perplexity, XAI).

In both cases a synthetic ``_litellm_shell`` function tool is injected, the
model's calls are intercepted, executed in a sandboxed Docker container via
``SkillsSandboxExecutor``, and the output is fed back until the model produces
a final non-tool-call response.
"""

import asyncio
import json
from typing import Any, Dict, Iterable, List, Optional

from litellm._logging import verbose_logger
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIResponse

_SHELL_TOOL_NAME = LiteLLMCompletionResponsesConfig.LITELLM_SHELL_TOOL_NAME
MAX_SHELL_ITERATIONS = 10


# ------------------------------------------------------------------
# Shared helpers (used by both handler.py and this module)
# ------------------------------------------------------------------


def format_shell_result(result: Dict[str, Any]) -> str:
    """
    Format a ``SkillsSandboxExecutor.execute_shell_command`` result dict
    into a human-readable string suitable for feeding back to the model.
    """
    output_parts: List[str] = []
    if result.get("output"):
        output_parts.append(result["output"])
    if result.get("error"):
        output_parts.append(f"STDERR:\n{result['error']}")
    if not result.get("success"):
        output_parts.append("[command exited with non-zero status]")
    return "\n".join(output_parts) or "(no output)"


def execute_shell_calls_for_completion(
    executor: Any,
    shell_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Execute shell calls and return Chat Completion tool-result messages.

    Each returned dict is ``{"role": "tool", "tool_call_id": ..., "content": ...}``.
    Designed for the Chat Completion bridge in ``handler.py``.
    """
    tool_messages: List[Dict[str, Any]] = []
    for sc in shell_calls:
        result = executor.execute_shell_command(sc["command"])
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": sc["id"],
                "content": format_shell_result(result),
            }
        )
    return tool_messages


# ------------------------------------------------------------------
# Responses API extraction / detection helpers
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Responses API follow-up helpers
# ------------------------------------------------------------------


def _build_follow_up_input(
    response: ResponsesAPIResponse,
    shell_calls: List[Dict[str, Any]],
    shell_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build Responses API follow-up input items from the model's output and
    sandbox execution results.

    Only shell function calls (``_litellm_shell``) and their outputs are
    included.  Non-shell function calls are excluded because we cannot
    provide outputs for them — including them without a corresponding
    ``function_call_output`` would cause an API error from the provider.
    """
    follow_up_input: List[Dict[str, Any]] = []

    assistant_message_content: List[Any] = []
    shell_call_ids = {sc["call_id"] for sc in shell_calls}

    for item in response.output or []:
        item_dict = item if isinstance(item, dict) else item.model_dump()
        if item_dict.get("type") == "function_call":
            call_id = item_dict.get("call_id") or item_dict.get("id")
            if call_id in shell_call_ids:
                follow_up_input.append(item_dict)
        elif item_dict.get("type") == "message":
            content = item_dict.get("content", [])
            if isinstance(content, list):
                assistant_message_content.extend(content)
            else:
                assistant_message_content.append(content)

    if assistant_message_content:
        follow_up_input.insert(
            0,
            {
                "type": "message",
                "role": "assistant",
                "content": assistant_message_content,
            },
        )

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
    executor: Any,
) -> List[Dict[str, Any]]:
    """Execute shell commands via sandbox and return Responses API results.

    Each returned dict is ``{"call_id": ..., "output": ...}``.

    Args:
        shell_calls: List of shell call dicts with ``call_id`` and ``command``.
        executor: A ``SkillsSandboxExecutor`` instance (reused across iterations).
    """
    results: List[Dict[str, Any]] = []
    for sc in shell_calls:
        result = executor.execute_shell_command(sc["command"])
        results.append(
            {
                "call_id": sc["call_id"],
                "output": format_shell_result(result),
            }
        )
    return results


async def _execute_shell_calls_async(
    shell_calls: List[Dict[str, Any]],
    executor: Any,
) -> List[Dict[str, Any]]:
    """Execute shell commands via sandbox in a thread pool (async-safe).

    Each returned dict is ``{"call_id": ..., "output": ...}``.

    Args:
        shell_calls: List of shell call dicts with ``call_id`` and ``command``.
        executor: A ``SkillsSandboxExecutor`` instance (reused across iterations).
    """
    loop = asyncio.get_running_loop()
    results: List[Dict[str, Any]] = []
    for sc in shell_calls:
        result = await loop.run_in_executor(
            None, executor.execute_shell_command, sc["command"]
        )
        results.append(
            {
                "call_id": sc["call_id"],
                "output": format_shell_result(result),
            }
        )
    return results


def _prepare_responses_follow_up(
    current_response: ResponsesAPIResponse,
    shell_calls: List[Dict[str, Any]],
    shell_results: List[Dict[str, Any]],
    iteration: int,
    log_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Build the follow-up input for the next Responses API call from
    pre-computed shell results.
    """
    follow_up_input = _build_follow_up_input(
        response=current_response,
        shell_calls=shell_calls,
        shell_results=shell_results,
    )

    verbose_logger.debug(
        "shell_tool_handler: %siteration %d, executed %d command(s)",
        log_prefix,
        iteration + 1,
        len(shell_calls),
    )

    return follow_up_input


# ------------------------------------------------------------------
# Responses API execution loops (async + sync)
# ------------------------------------------------------------------


async def run_shell_execution_loop_responses_api(
    response: ResponsesAPIResponse,
    model: str,
    tools: Optional[Iterable[Any]],
    **call_params: Any,
) -> ResponsesAPIResponse:
    """
    Async shell execution loop for the native Responses API path.

    Detects ``_litellm_shell`` function calls in the response, executes them
    in the sandbox (offloaded to a thread pool to avoid blocking the event
    loop), and re-invokes ``litellm.aresponses()`` with the results until
    the model produces a final response without shell calls.
    """
    from litellm.llms.litellm_proxy.skills.sandbox_executor import (
        SkillsSandboxExecutor,
    )
    from litellm.responses.main import aresponses

    current_response = response
    shell_calls = _extract_shell_calls_from_responses_api(current_response)
    if not shell_calls:
        return current_response

    executor = SkillsSandboxExecutor()

    for _iteration in range(MAX_SHELL_ITERATIONS):
        if not shell_calls:
            break

        shell_results = await _execute_shell_calls_async(shell_calls, executor)
        follow_up_input = _prepare_responses_follow_up(
            current_response, shell_calls, shell_results, _iteration, "",
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
            MAX_SHELL_ITERATIONS,
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
    from litellm.llms.litellm_proxy.skills.sandbox_executor import (
        SkillsSandboxExecutor,
    )
    from litellm.responses.main import responses

    current_response = response
    shell_calls = _extract_shell_calls_from_responses_api(current_response)
    if not shell_calls:
        return current_response

    executor = SkillsSandboxExecutor()

    for _iteration in range(MAX_SHELL_ITERATIONS):
        if not shell_calls:
            break

        shell_results = _execute_shell_calls(shell_calls, executor)
        follow_up_input = _prepare_responses_follow_up(
            current_response, shell_calls, shell_results, _iteration, "sync ",
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
            MAX_SHELL_ITERATIONS,
        )

    return current_response
