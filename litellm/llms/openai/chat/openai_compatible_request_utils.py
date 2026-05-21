"""
Shared OpenAI-compatible chat request normalization helpers.

Used by OpenAIConfig and OpenAIGPTConfig so Codex/Responses tool and JSON-mode
fixes stay in one place.
"""

from typing import Any, Dict, List, Optional, cast

import litellm
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import requires_json_keyword_for_json_object

_JSON_OBJECT_HINT = "Return valid JSON only."
_LOCAL_BACKUP_MODEL_COST: Optional[Dict[str, Any]] = None

# Codex/Responses built-in tools that are not valid Chat Completions tools.
_CODEX_UNSUPPORTED_CHAT_TOOL_TYPES = frozenset(
    {"shell", "computer_use_preview", "namespace"}
)


def _coerce_tool_parameters(parameters: Any) -> Dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object"}
    if "type" not in parameters:
        return {**parameters, "type": "object"}
    return parameters


def _custom_tool_to_function(tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert OpenAI Responses style custom tools into chat.completions function tools."""
    tool_name = tool.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None

    parameters = tool.get("input_schema")
    if parameters is None:
        parameters = tool.get("parameters")
    parameters = _coerce_tool_parameters(parameters)
    if parameters.get("type") == "object":
        parameters.setdefault("properties", {})

    function_payload: Dict[str, Any] = {
        "name": tool_name,
        "parameters": parameters,
        "strict": bool(tool.get("strict", False)),
    }
    description = tool.get("description")
    if isinstance(description, str) and description:
        function_payload["description"] = description

    return {"type": "function", "function": function_payload}


def normalize_flat_function_tools(
    tools: Optional[List[Any]],
) -> Optional[List[Any]]:
    """
    Normalize Responses/Codex tools to OpenAI chat.completions shape.

    - Drops Codex built-in tools (shell, computer_use_preview, namespace)
    - Converts Responses ``custom`` tools to ``function`` tools
    - Wraps flat ``function`` tools (name at top level) in a ``function`` object
    """
    if tools is None:
        return None

    normalized_tools: List[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized_tools.append(tool)
            continue

        tool_type = tool.get("type")
        if tool_type in _CODEX_UNSUPPORTED_CHAT_TOOL_TYPES:
            continue

        if tool_type == "custom":
            converted = _custom_tool_to_function(tool)
            if converted is not None:
                normalized_tools.append(converted)
            continue

        if tool_type != "function":
            normalized_tools.append(tool)
            continue

        existing_function = tool.get("function")
        if isinstance(existing_function, dict):
            normalized_tools.append(tool)
            continue

        name = tool.get("name")
        if name is None:
            continue

        parameters = _coerce_tool_parameters(tool.get("parameters"))
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


def response_format_requires_json_keyword_in_prompt(optional_params: dict) -> bool:
    """
    Return True when structured output may require the word "json" in messages.

    Some OpenAI-compatible providers (e.g. GLM) enforce this for json_object and
    also reject json_schema requests unless the prompt mentions json.
    """
    response_format = optional_params.get("response_format")
    if not isinstance(response_format, dict):
        return False
    return response_format.get("type") in ("json_object", "json_schema")


def _get_local_backup_model_cost() -> Dict[str, Any]:
    global _LOCAL_BACKUP_MODEL_COST
    if _LOCAL_BACKUP_MODEL_COST is None:
        from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

        _LOCAL_BACKUP_MODEL_COST = GetModelCostMap.load_local_model_cost_map()
    return _LOCAL_BACKUP_MODEL_COST


def _candidate_models_for_json_keyword_lookup(
    model: str,
    upstream_model: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()
    for name in (model, upstream_model):
        if not isinstance(name, str) or not name:
            continue
        for candidate in (name,):
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
            try:
                stripped_model, _, _, _ = litellm.get_llm_provider(
                    model=name, custom_llm_provider=custom_llm_provider
                )
                if stripped_model not in seen:
                    seen.add(stripped_model)
                    candidates.append(stripped_model)
            except Exception:
                bare_model = name.split("/")[-1]
                if bare_model not in seen:
                    seen.add(bare_model)
                    candidates.append(bare_model)
    return candidates


def _local_backup_requires_json_keyword(model_name: str) -> bool:
    backup_entry = _get_local_backup_model_cost().get(model_name) or {}
    return backup_entry.get("requires_json_keyword_for_json_object") is True


def _model_requires_json_keyword_for_json_object(
    model: str,
    custom_llm_provider: Optional[str] = None,
    upstream_model: Optional[str] = None,
) -> bool:
    for candidate in _candidate_models_for_json_keyword_lookup(
        model=model,
        upstream_model=upstream_model,
        custom_llm_provider=custom_llm_provider,
    ):
        if requires_json_keyword_for_json_object(
            model=candidate, custom_llm_provider=custom_llm_provider
        ):
            return True
        if _local_backup_requires_json_keyword(candidate):
            return True
    return False


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
    upstream_model: Optional[str] = None,
) -> List[AllMessageValues]:
    """
    Some OpenAI-compatible models require the prompt to include the word "json"
    when response_format={"type":"json_object"} is requested.
    """
    if not _model_requires_json_keyword_for_json_object(
        model=model,
        custom_llm_provider=custom_llm_provider,
        upstream_model=upstream_model,
    ):
        return messages
    if not response_format_requires_json_keyword_in_prompt(optional_params):
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
