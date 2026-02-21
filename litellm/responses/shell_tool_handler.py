"""
Shell tool execution helpers.

Shared utilities for formatting sandbox results and executing shell commands,
used by the **Chat Completion bridge** (``handler.py``) — providers without a
native Responses API that go through ``litellm.completion()``.

A synthetic ``_litellm_shell`` function tool is injected, the model's calls are
intercepted, executed in a sandboxed Docker container via
``SkillsSandboxExecutor``, and the output is fed back until the model produces
a final non-tool-call response.
"""

from typing import Any, Dict, List

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
