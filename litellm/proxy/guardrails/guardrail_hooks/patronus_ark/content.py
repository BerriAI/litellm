import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, NoReturn, TypeAlias

from pydantic import JsonValue, TypeAdapter

from litellm.types.proxy.guardrails.guardrail_hooks.patronus_ark import Source

JSON_VALUE: Final = TypeAdapter[JsonValue](JsonValue)  # pyright: ignore[reportArgumentType]  # Pydantic's recursive JsonValue alias is not annotated as TypeForm
Visit: TypeAlias = Callable[[str, Source, bool], Awaitable[str]]


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError("Non-finite tool argument")


def _object_pairs(pairs: Sequence[tuple[str, JsonValue]]) -> Mapping[str, JsonValue]:
    result: Final = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate tool argument key")
    return result


async def _json_value(value: JsonValue, visit: Visit) -> JsonValue:
    if isinstance(value, str):
        return await visit(value, "tool_call", False)
    if isinstance(value, dict):
        pairs: Final = tuple(
            [(await visit(key, "tool_call", False), await _json_value(item, visit)) for key, item in value.items()]
        )
        rewritten: Final = dict(pairs)
        if len(rewritten) != len(pairs):
            raise ValueError("Redaction would create duplicate tool argument keys")
        return rewritten
    if isinstance(value, list):
        return [await _json_value(item, visit) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raw: Final = JSON_VALUE.dump_json(value).decode()
        rewritten_number: Final = await visit(raw, "tool_call", False)
        return value if raw == rewritten_number else rewritten_number
    return value


async def _function(value: JsonValue, visit: Visit) -> JsonValue:
    if not isinstance(value, dict):
        raise TypeError("Malformed tool function")
    arguments: Final = value.get("arguments")
    if not isinstance(arguments, str):
        raise TypeError("Tool arguments must be a JSON string")
    await visit(arguments, "tool_call", True)
    parsed: Final = JSON_VALUE.validate_python(
        json.loads(
            arguments,
            parse_constant=_invalid_constant,
            object_pairs_hook=_object_pairs,
        )
    )
    rewritten: Final = await _json_value(parsed, visit)
    return {**value, "arguments": JSON_VALUE.dump_json(rewritten).decode()}


async def _tool_call(value: JsonValue, visit: Visit) -> JsonValue:
    if not isinstance(value, dict) or value.get("type") != "function":
        raise ValueError("Unsupported tool call")
    return {**value, "function": await _function(value.get("function"), visit)}


async def _content(value: JsonValue, source: Source, visit: Visit) -> JsonValue:
    if isinstance(value, str):
        return await visit(value, source, False)
    if not isinstance(value, list):
        return value
    return [await _part(part, source, visit) for part in value]


async def _part(value: JsonValue, source: Source, visit: Visit) -> JsonValue:
    if not isinstance(value, dict) or value.get("type") != "text":
        return value
    text: Final = value.get("text")
    if not isinstance(text, str):
        raise TypeError("Malformed text part")
    return {**value, "text": await visit(text, source, False)}


async def rewrite_message(value: JsonValue, visit: Visit) -> JsonValue:
    if not isinstance(value, dict):
        raise TypeError("Malformed chat message")
    role: Final = value.get("role")
    if role not in ("user", "assistant", "system", "developer", "tool", "function"):
        raise ValueError("Unsupported chat role")
    source: Final[Source | None] = (
        "user_input" if role == "user" else "tool_result" if role in ("tool", "function") else None
    )
    content: Final = await _content(value.get("content"), source, visit) if source else value.get("content")
    calls: Final = value.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        raise ValueError("Malformed tool calls")
    rewritten_calls: Final = [await _tool_call(call, visit) for call in calls] if isinstance(calls, list) else None
    function: Final = value.get("function_call")
    rewritten_function: Final = await _function(function, visit) if function is not None else None
    return {
        **value,
        **({"content": content} if "content" in value else {}),
        **({"tool_calls": rewritten_calls} if calls is not None else {}),
        **({"function_call": rewritten_function} if function is not None else {}),
    }
