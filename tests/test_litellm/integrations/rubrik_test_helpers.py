"""Shared helpers for Rubrik plugin tests."""

from typing import Any, Dict

from litellm.types.utils import GenericGuardrailAPIInputs


def make_tool_call_dict(tc_id: str, name: str, arguments: str = "{}") -> Dict[str, Any]:
    """Create a tool call dict matching the ChatCompletionMessageToolCall schema."""
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def make_inputs_with_tools(
    tool_calls: list, texts: list | None = None
) -> GenericGuardrailAPIInputs:
    """Create GenericGuardrailAPIInputs with tool_calls."""
    return GenericGuardrailAPIInputs(texts=texts or [], tool_calls=tool_calls)
