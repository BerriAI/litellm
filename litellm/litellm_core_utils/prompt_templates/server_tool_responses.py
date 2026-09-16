from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.prompt_templates.factory import NormalizedToolCall
from litellm.litellm_core_utils.prompt_templates.server_tools import ServerToolRoute

_OBJECT: Final = TypeAdapter(dict[str, object])
_OBJECTS: Final = TypeAdapter(tuple[dict[str, object], ...])


def object_value(value: object) -> Mapping[str, object]:
    return (
        _OBJECT.validate_python(value)
        if isinstance(value, dict)
        else {  # mutable-ok: Native provider JSON containers.
        }
    )


def object_items(value: object) -> tuple[Mapping[str, object], ...]:
    return _OBJECTS.validate_python(value) if isinstance(value, (tuple, list)) else ()


def assistant_message(response: Mapping[str, object]) -> Mapping[str, object]:
    choices: Final = object_items(response.get("choices"))
    return (
        object_value(choices[0].get("message"))
        if choices
        else {  # mutable-ok: Native provider JSON containers.
        }
    )


def public_tool_response(
    response: Mapping[str, object], route: ServerToolRoute, server_names: frozenset[str], hide_text: bool = False
) -> Mapping[str, object]:
    if route != "acompletion":
        field: Final = "output" if route == "aresponses" else "content"
        return {  # mutable-ok: Native provider JSON containers.
            **response,
            field: [  # mutable-ok: Native provider JSON containers.
                item
                for item in object_items(response.get(field))
                if not (item.get("type") in ("tool_use", "function_call") and item.get("name") in server_names)
                and not (hide_text and item.get("type") in ("text", "message"))
            ],
        }
    choices: Final = object_items(response.get("choices"))
    message: Final = assistant_message(response)
    calls: Final = tuple(
        call
        for call in object_items(message.get("tool_calls"))
        if object_value(call.get("function")).get("name") not in server_names
    )
    return {  # mutable-ok: Native provider JSON containers.
        **response,
        "choices": [  # mutable-ok: Native provider JSON containers.
            {  # mutable-ok: Native provider JSON containers.
                **(
                    choices[0]
                    if choices
                    else {  # mutable-ok: Native provider JSON containers.
                    }
                ),
                "message": {  # mutable-ok: Native provider JSON containers.
                    **message,
                    "content": None if hide_text else message.get("content"),
                    "tool_calls": list(  # mutable-ok: Native provider JSON containers.
                        calls
                    )
                    if calls
                    else None,
                },
            }
        ],
    }


def combined_usage(usages: Sequence[Mapping[str, object]], depth: int = 0) -> Mapping[str, object]:
    if depth > 16:
        raise ValueError("Server tool usage nesting exceeds 16 levels")
    names: Final = frozenset(key for usage in usages for key in usage)

    def combined(name: str) -> object:
        values: Final = tuple(usage[name] for usage in usages if usage.get(name) is not None)
        if any(isinstance(value, dict) for value in values):
            return combined_usage(tuple(object_value(value) for value in values), depth + 1)
        numbers: Final = tuple(
            value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        return sum(numbers) if numbers else values[-1] if values else None

    return {  # mutable-ok: Native provider JSON containers.
        name: combined(name) for name in sorted(names)
    }


def combined_tool_response(responses: tuple[Mapping[str, object], ...], route: ServerToolRoute) -> Mapping[str, object]:
    if not responses:
        raise ValueError("No model response was received")
    last: Final = responses[-1]
    usage: Final = combined_usage(tuple(object_value(response.get("usage")) for response in responses))
    return MappingProxyType({**last, "usage": usage})


def response_messages(response: Mapping[str, object], route: ServerToolRoute) -> tuple[Mapping[str, object], ...]:
    if route == "aresponses":
        return object_items(response.get("output"))
    if route == "anthropic_messages":
        return (
            {  # mutable-ok: Native provider JSON containers.
                "role": "assistant",
                "content": response.get(
                    "content",
                    [  # mutable-ok: Native provider JSON containers.
                    ],
                ),
            },
        )
    return (assistant_message(response),)


def response_has_client_tools(
    response: Mapping[str, object], route: ServerToolRoute, server_names: frozenset[str]
) -> bool:
    public: Final = public_tool_response(response, route, server_names)
    if route == "acompletion":
        message: Final = assistant_message(public)
        return bool(message.get("tool_calls") or message.get("function_call"))
    field: Final = "output" if route == "aresponses" else "content"
    return any(
        item.get("type")
        in ("tool_use", "function_call", "custom_tool_call", "local_shell_call", "shell_call", "apply_patch_call")
        for item in object_items(public.get(field))
    )


def response_is_truncated(response: Mapping[str, object]) -> bool:
    return (
        response.get("status") == "incomplete"
        or response.get("stop_reason") == "max_tokens"
        or any(choice.get("finish_reason") == "length" for choice in object_items(response.get("choices")))
    )


def executable_server_calls(
    response: Mapping[str, object], route: ServerToolRoute, server_names: frozenset[str]
) -> tuple[NormalizedToolCall, ...]:
    items: Final = (
        object_items(assistant_message(response).get("tool_calls"))
        if route == "acompletion"
        else object_items(response.get("output" if route == "aresponses" else "content"))
    )
    definitions: Final = tuple(
        (item, object_value(item.get("function")) if route == "acompletion" else item) for item in items
    )
    calls: Final = tuple(
        (item, definition) for item, definition in definitions if definition.get("name") in server_names
    )
    if not calls:
        return ()
    choices: Final = object_items(response.get("choices"))
    completed: Final = (
        response.get("status") == "completed"
        if route == "aresponses"
        else response.get("stop_reason") == "tool_use"
        if route == "anthropic_messages"
        else bool(choices) and choices[0].get("finish_reason") == "tool_calls"
    )
    if not completed:
        if response_is_truncated(response):
            return ()
        raise ValueError("The model did not complete its memory tool calls")

    def normalize(item: Mapping[str, object], definition: Mapping[str, object]) -> NormalizedToolCall:
        identifier: Final = item.get("call_id", item.get("id"))
        name: Final = definition.get("name")
        raw: Final = definition.get("input" if route == "anthropic_messages" else "arguments")
        if not isinstance(identifier, str) or not identifier or not isinstance(name, str):
            raise ValueError("The model returned an invalid memory tool call")
        arguments: Final = _OBJECT.validate_json(raw) if isinstance(raw, str) else _OBJECT.validate_python(raw)
        return {  # mutable-ok: Native provider JSON containers.
            "id": identifier,
            "name": name,
            "arguments": arguments,
        }

    return tuple(normalize(item, definition) for item, definition in calls)
