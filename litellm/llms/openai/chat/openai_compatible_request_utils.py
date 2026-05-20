"""
Shared OpenAI-compatible chat request normalization helpers.

Used by OpenAIConfig and OpenAIGPTConfig so Codex/Responses tool and JSON-mode
fixes stay in one place.
"""

from typing import Any, List, Optional, cast

from litellm.types.llms.openai import AllMessageValues
from litellm.utils import requires_json_keyword_for_json_object

_JSON_OBJECT_HINT = "Return valid JSON only."


def normalize_flat_function_tools(
    tools: Optional[List[Any]],
) -> Optional[List[Any]]:
    """
    Normalize Responses/Codex flat function tools to OpenAI chat.completions shape.
    """
    if tools is None:
        return None

    normalized_tools: List[Any] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            normalized_tools.append(tool)
            continue

        existing_function = tool.get("function")
        if isinstance(existing_function, dict):
            normalized_tools.append(tool)
            continue

        name = tool.get("name")
        if name is None:
            normalized_tools.append(tool)
            continue

        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object"}
        elif "type" not in parameters:
            parameters = {**parameters, "type": "object"}

        function_payload = {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": parameters,
            "strict": bool(tool.get("strict", False)),
        }
        normalized_tools.append(
            {
                **{
                    key: value
                    for key, value in tool.items()
                    if key not in {"name", "description", "parameters", "strict"}
                },
                "type": "function",
                "function": function_payload,
            }
        )

    return normalized_tools


def response_format_is_json_object(optional_params: dict) -> bool:
    response_format = optional_params.get("response_format")
    return (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_object"
    )


def messages_contain_json_keyword(messages: List[AllMessageValues]) -> bool:
    stack: list[Any] = list(messages)
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            if "json" in value.lower():
                return True
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
    return False


def _append_json_hint_to_system_message(
    message: AllMessageValues,
) -> AllMessageValues:
    copied = dict(message)
    content = copied.get("content")
    if isinstance(content, str):
        copied["content"] = f"{content.rstrip()}\n\n{_JSON_OBJECT_HINT}"
    elif isinstance(content, list):
        copied["content"] = [
            *content,
            {"type": "text", "text": _JSON_OBJECT_HINT},
        ]
    else:
        copied["content"] = _JSON_OBJECT_HINT
    return cast(AllMessageValues, copied)


def maybe_inject_json_keyword_hint_for_json_object(
    model: str,
    messages: List[AllMessageValues],
    optional_params: dict,
    custom_llm_provider: Optional[str] = None,
) -> List[AllMessageValues]:
    """
    Some OpenAI-compatible models require the prompt to include the word "json"
    when response_format={"type":"json_object"} is requested.
    """
    if not requires_json_keyword_for_json_object(
        model=model, custom_llm_provider=custom_llm_provider
    ):
        return messages
    if not response_format_is_json_object(optional_params):
        return messages
    if messages_contain_json_keyword(messages):
        return messages

    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") in {
            "system",
            "developer",
        }:
            updated = _append_json_hint_to_system_message(message)
            return [*messages[:index], updated, *messages[index + 1 :]]

    json_hint: AllMessageValues = {
        "role": "system",
        "content": _JSON_OBJECT_HINT,
    }
    return [json_hint, *messages]
