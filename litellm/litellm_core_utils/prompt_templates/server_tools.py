import json
from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm.litellm_core_utils.prompt_templates.factory import NormalizedToolCall

ServerToolRoute: TypeAlias = Literal["acompletion", "aresponses", "anthropic_messages"]
_LIST: Final = TypeAdapter(tuple[object, ...])
_OBJECT: Final = TypeAdapter(dict[str, object])


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return _LIST.validate_python(value)
    if isinstance(value, str):
        return (
            {  # mutable-ok: Native provider JSON containers.
                "role": "user",
                "content": value,
            },
        )
    return ()


def append_server_instructions(
    data: Mapping[str, object], route: ServerToolRoute, instructions: str
) -> Mapping[str, object]:
    if route == "aresponses":
        previous: Final = data.get("instructions")
        return {  # mutable-ok: Provider wire format requires native JSON containers.
            **data,
            "instructions": f"{previous}\n\n{instructions}" if isinstance(previous, str) else instructions,
        }
    if route == "anthropic_messages":
        system: Final = data.get("system")
        blocks: Final = (
            _LIST.validate_python(system)
            if isinstance(system, list)
            else [  # mutable-ok: Provider wire format requires native JSON containers.
                {  # mutable-ok: Provider wire format requires native JSON containers.
                    "type": "text",
                    "text": system,
                }
            ]
            if isinstance(system, str)
            else [  # mutable-ok: Provider wire format requires native JSON containers.
            ]
        )
        return {  # mutable-ok: Provider wire format requires native JSON containers.
            **data,
            "system": [  # mutable-ok: Provider wire format requires native JSON containers.
                *blocks,
                {  # mutable-ok: Provider wire format requires native JSON containers.
                    "type": "text",
                    "text": instructions,
                },
            ],
        }
    messages: Final = _items(data.get("messages"))
    insertion: Final = next(
        (
            index
            for index, message in enumerate(messages)
            if not isinstance(message, dict)
            or _OBJECT.validate_python(message).get("role") not in ("system", "developer")
        ),
        len(messages),
    )
    return {  # mutable-ok: Provider wire format requires native JSON containers.
        **data,
        "messages": [  # mutable-ok: Provider wire format requires native JSON containers.
            *messages[:insertion],
            {  # mutable-ok: Provider wire format requires native JSON containers.
                "role": "system",
                "content": instructions,
            },
            *messages[insertion:],
        ],
    }


def inject_server_tools(
    data: Mapping[str, object], route: ServerToolRoute, functions: Sequence[Mapping[str, object]], instructions: str
) -> Mapping[str, object]:
    client_tools: Final = _items(data.get("tools"))
    tool_choice: Final = data.get("tool_choice")
    names: Final = frozenset(str(function["name"]) for function in functions)
    if any(_tool_name(tool) in names for tool in client_tools):
        raise ValueError("A client tool conflicts with a gateway memory tool name")
    tools: Final = tuple(
        {  # mutable-ok: Native provider JSON containers.
            "name": f["name"],
            "description": f["description"],
            "input_schema": f["parameters"],
        }
        if route == "anthropic_messages"
        else {  # mutable-ok: Native provider JSON containers.
            "type": "function",
            **f,
        }
        if route == "aresponses"
        else {  # mutable-ok: Native provider JSON containers.
            "type": "function",
            "function": f,
        }
        for f in functions
    )
    return append_server_instructions(
        {  # mutable-ok: Native provider JSON containers.
            **data,
            "tool_choice": tool_choice
            if tool_choice is not None
            else (
                {  # mutable-ok: Native provider tool-choice JSON.
                    "type": "auto"
                }
                if route == "anthropic_messages"
                else "auto"
            ),
            "tools": [  # mutable-ok: Native provider JSON containers.
                *client_tools,
                *tools,
            ],
        },
        route,
        instructions,
    )


def _tool_name(tool: object) -> object:
    if not isinstance(tool, dict):
        return None
    definition: Final = _OBJECT.validate_python(tool)
    function: Final = definition.get("function") or definition.get("custom")
    return _OBJECT.validate_python(function).get("name") if isinstance(function, dict) else definition.get("name")


def append_server_reference(data: Mapping[str, object], route: ServerToolRoute, reference: str) -> Mapping[str, object]:
    field: Final = "input" if route == "aresponses" else "messages"
    return {  # mutable-ok: Provider wire format requires native JSON containers.
        **data,
        field: [  # mutable-ok: Provider wire format requires native JSON containers.
            *_items(data.get(field)),
            {  # mutable-ok: Provider wire format requires native JSON containers.
                "role": "user",
                "content": reference,
            },
        ],
    }


def continue_server_tools(
    data: Mapping[str, object],
    route: ServerToolRoute,
    response: Mapping[str, object],
    calls: Sequence[NormalizedToolCall],
    results: Sequence[object],
) -> Mapping[str, object]:
    if len(calls) != len(results) or any(not call["id"] for call in calls):
        raise ValueError("Server tool results must match every tool call")
    if route == "aresponses":
        return {  # mutable-ok: Provider wire format requires native JSON containers.
            **data,
            "input": [  # mutable-ok: Provider wire format requires native JSON containers.
                *_items(data.get("input")),
                *_items(response.get("output")),
                *[  # mutable-ok: Provider wire format requires native JSON containers.
                    {  # mutable-ok: Provider wire format requires native JSON containers.
                        "type": "function_call_output",
                        "call_id": call["id"],
                        "output": json.dumps(result),
                    }
                    for call, result in zip(calls, results)
                ],
            ],
        }
    if route == "anthropic_messages":
        return {  # mutable-ok: Provider wire format requires native JSON containers.
            **data,
            "messages": [  # mutable-ok: Provider wire format requires native JSON containers.
                *_items(data.get("messages")),
                {  # mutable-ok: Provider wire format requires native JSON containers.
                    "role": "assistant",
                    "content": response.get(
                        "content",
                        [  # mutable-ok: Provider wire format requires native JSON containers.
                        ],
                    ),
                },
                {  # mutable-ok: Provider wire format requires native JSON containers.
                    "role": "user",
                    "content": [  # mutable-ok: Provider wire format requires native JSON containers.
                        {  # mutable-ok: Provider wire format requires native JSON containers.
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": json.dumps(result),
                        }
                        for call, result in zip(calls, results)
                    ],
                },
            ],
        }
    choices: Final = response.get("choices")
    choice_items: Final = _items(choices)
    first: Final = choice_items[0] if choice_items else None
    message: Final = _OBJECT.validate_python(first).get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise TypeError("Server tool response has no assistant message")
    return {  # mutable-ok: Provider wire format requires native JSON containers.
        **data,
        "messages": [  # mutable-ok: Provider wire format requires native JSON containers.
            *_items(data.get("messages")),
            _OBJECT.validate_python(message),
            *[  # mutable-ok: Provider wire format requires native JSON containers.
                {  # mutable-ok: Provider wire format requires native JSON containers.
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result),
                }
                for call, result in zip(calls, results)
            ],
        ],
    }
