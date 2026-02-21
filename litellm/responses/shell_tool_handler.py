"""
Shell tool execution helpers.

Shared utilities for formatting sandbox results and executing shell commands.

Used by two paths:

1. **Chat Completion bridge** (``handler.py``) — providers without a native
   Responses API that go through ``litellm.completion()``.
2. **Responses API HTTP handler** (``main.py``) — providers with their own
   Responses API config but no native shell support (xAI, Perplexity, …).

A synthetic ``_litellm_shell`` function tool is injected, the model's calls are
intercepted, executed in a sandboxed Docker container via
``SkillsSandboxExecutor``, and the output is fed back until the model produces
a final non-tool-call response.
"""

import json
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.llms.litellm_proxy.skills.sandbox_executor import SkillsSandboxExecutor

_SHELL_TOOL_NAME = "_litellm_shell"
MAX_SHELL_ITERATIONS = 10


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


# ---------------------------------------------------------------------------
# Responses API helpers
# ---------------------------------------------------------------------------


def extract_shell_calls_from_response(response: Any) -> List[Dict[str, Any]]:
    """
    Extract ``_litellm_shell`` function-call output items from a
    ``ResponsesAPIResponse``.

    Returns a list of dicts with keys ``call_id``, ``command``.
    """
    output = getattr(response, "output", None) or []
    calls: List[Dict[str, Any]] = []
    for item in output:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "function_call":
            continue
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        if name != _SHELL_TOOL_NAME:
            continue
        call_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
        args_raw = item.get("arguments") if isinstance(item, dict) else getattr(item, "arguments", None)
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        command = args.get("command", [])
        if isinstance(command, str):
            command = [command]
        calls.append({"call_id": call_id, "command": command})
    return calls


def response_has_shell_tool(tools: Optional[List[Any]]) -> bool:
    """Return True if ``_litellm_shell`` is among the Responses API tools."""
    if not tools:
        return False
    for tool in tools:
        if isinstance(tool, dict) and tool.get("name") == _SHELL_TOOL_NAME:
            return True
    return False


async def run_responses_shell_execution_loop(
    response: Any,
    model: str,
    tools: Optional[List[Any]],
    previous_response_id: Optional[str],
    responses_kwargs: Dict[str, Any],
) -> Any:
    """
    Auto-execute ``_litellm_shell`` calls for the Responses API path.

    Mimics OpenAI's native shell tool behaviour: when the model returns a
    ``function_call`` for ``_litellm_shell``, this function executes the
    command in a sandbox, sends the ``function_call_output`` back, and
    repeats until the model returns a final response without shell calls.
    """
    shell_calls = extract_shell_calls_from_response(response)
    if not shell_calls:
        return response

    executor = SkillsSandboxExecutor()
    current_response = response

    for _iteration in range(MAX_SHELL_ITERATIONS):
        if not shell_calls:
            break

        function_call_outputs: List[Dict[str, Any]] = []
        for sc in shell_calls:
            verbose_logger.debug(
                "shell_tool_handler: executing command %s (call_id=%s)",
                sc["command"], sc["call_id"],
            )
            result = executor.execute_shell_command(sc["command"])
            function_call_outputs.append({
                "type": "function_call_output",
                "call_id": sc["call_id"],
                "output": format_shell_result(result),
            })

        current_response = await litellm.aresponses(
            model=model,
            previous_response_id=current_response.id,
            input=function_call_outputs,
            tools=tools,
            **responses_kwargs,
        )
        shell_calls = extract_shell_calls_from_response(current_response)

    if shell_calls:
        verbose_logger.warning(
            "shell_tool_handler: max iterations (%d) reached, returning last response",
            MAX_SHELL_ITERATIONS,
        )

    return current_response
