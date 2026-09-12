import json
from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm.litellm_core_utils.prompt_templates.factory import NormalizedToolCall

ServerToolRoute: TypeAlias = Literal["acompletion", "aresponses", "anthropic_messages"]
_LIST: Final = TypeAdapter(list[object])
_OBJECT: Final = TypeAdapter(dict[str, object])


def _items(value: object) -> list[object]:
    if isinstance(value, list):
        return _LIST.validate_python(value)
    if isinstance(value, str):
        return [  # mutable-ok: Provider wire format requires native JSON containers.
            {  # mutable-ok: Provider wire format requires native JSON containers.
                "role": "user",
                "content": value,
            }
        ]
    return [  # mutable-ok: Provider wire format requires native JSON containers.
    ]


def append_server_instructions(
    data: Mapping[str, object], route: ServerToolRoute, instructions: str
) -> dict[str, object]:
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
    return {  # mutable-ok: Provider wire format requires native JSON containers.
        **data,
        "messages": [  # mutable-ok: Provider wire format requires native JSON containers.
            *_items(data.get("messages")),
            {  # mutable-ok: Provider wire format requires native JSON containers.
                "role": "system",
                "content": instructions,
            },
        ],
    }


def append_server_reference(data: Mapping[str, object], route: ServerToolRoute, reference: str) -> dict[str, object]:
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


def prepare_server_tools(
    data: Mapping[str, object], route: ServerToolRoute, functions: Sequence[Mapping[str, object]], instructions: str
) -> dict[str, object]:
    tools: Final = (
        [  # mutable-ok: Provider wire format requires native JSON containers.
            {  # mutable-ok: Provider wire format requires native JSON containers.
                "name": f["name"],
                "description": f["description"],
                "input_schema": f["parameters"],
            }
            for f in functions
        ]
        if route == "anthropic_messages"
        else [  # mutable-ok: Provider wire format requires native JSON containers.
            {  # mutable-ok: Provider wire format requires native JSON containers.
                "type": "function",
                **f,
            }
            for f in functions
        ]
        if route == "aresponses"
        else [  # mutable-ok: Provider wire format requires native JSON containers.
            {  # mutable-ok: Provider wire format requires native JSON containers.
                "type": "function",
                "function": f,
            }
            for f in functions
        ]
    )
    omitted: Final = frozenset(
        (
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "stream_options",
            "response_format",
            "text",
            "n",
            "stop",
            "stop_sequences",
            "background",
            "output_config",
            "idempotency_key",
            "litellm_call_id",
        )
    )
    output_field: Final = (
        "max_output_tokens"
        if route == "aresponses"
        else "max_completion_tokens"
        if "max_completion_tokens" in data
        else "max_tokens"
    )
    thinking: Final = data.get("thinking")
    thinking_budget: Final = (
        _OBJECT.validate_python(thinking).get("budget_tokens") if isinstance(thinking, dict) else None
    )
    minimum: Final = max(2048, thinking_budget + 2048) if isinstance(thinking_budget, int) else 2048
    limit: Final = data.get(output_field)
    header_fields: Final = {  # mutable-ok: The proxy accepts native provider header dictionaries.
        field: {  # mutable-ok: These headers are sent through HTTP JSON serialization.
            key: value
            for key, value in _OBJECT.validate_python(data[field]).items()
            if key.lower() not in ("idempotency-key", "x-request-id", "x-litellm-call-id")
        }
        for field in ("headers", "extra_headers")
        if isinstance(data.get(field), dict)
    }
    base: Final = {  # mutable-ok: Provider wire format requires native JSON containers.
        **{  # mutable-ok: Provider wire format requires native JSON containers.
            key: value for key, value in data.items() if key not in omitted
        },
        output_field: max(minimum, limit) if isinstance(limit, int) else minimum,
        **header_fields,
    }
    return append_server_instructions(
        {  # mutable-ok: Provider wire format requires native JSON containers.
            **base,
            "tools": tools,
            "stream": False,
            **(
                {  # mutable-ok: Provider wire format requires native JSON containers.
                    "store": False
                }
                if route == "aresponses"
                else {  # mutable-ok: Provider wire format requires native JSON containers.
                }
            ),
        },
        route,
        instructions,
    )


def continue_server_tools(
    data: Mapping[str, object],
    route: ServerToolRoute,
    response: Mapping[str, object],
    calls: Sequence[NormalizedToolCall],
    results: Sequence[object],
) -> dict[str, object]:
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
            message,
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
