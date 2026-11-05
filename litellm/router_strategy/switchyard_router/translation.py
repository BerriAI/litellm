from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from switchyard.libsy import LlmResponse

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_SEQUENCE: Final = TypeAdapter(tuple[object, ...])


def mapping(value: object) -> Mapping[str, object]:
    return _MAPPING.validate_python(value)


def sequence(value: object) -> tuple[object, ...]:
    return _SEQUENCE.validate_python(value)


def text_content(value: object) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return ({"type": "text", "text": value},)
    blocks: Final = tuple(mapping(block) for block in sequence(value))
    if any(block.get("type") != "text" or not isinstance(block.get("text"), str) for block in blocks):
        raise ValueError("Switchyard routing supports text content only; media and audio are unsupported")
    return tuple({"type": "text", "text": block["text"]} for block in blocks)


def tool_call(value: object) -> Mapping[str, object]:
    call: Final = mapping(value)
    function: Final = mapping(call.get("function"))
    arguments: Final = function.get("arguments")
    if call.get("type") != "function" or not isinstance(arguments, str):
        raise ValueError("Switchyard requires function calls with JSON arguments")
    return {
        "type": "tool_call",
        "id": call["id"],
        "name": function["name"],
        "arguments": mapping(json.loads(arguments)),
    }


def message_to_switchyard(value: object) -> Mapping[str, object]:
    message: Final = mapping(value)
    if message.get("audio") is not None:
        raise ValueError("Switchyard routing does not support audio")
    role: Final = message.get("role")
    content: Final = text_content(message.get("content"))
    if role == "tool":
        return {
            "role": "tool",
            "content": (
                {
                    "type": "tool_result",
                    "tool_call_id": message["tool_call_id"],
                    "content": content,
                    "is_error": message.get("is_error"),
                },
            ),
        }
    if role not in ("user", "system", "developer", "assistant"):
        raise ValueError("Switchyard received an unsupported message role")
    calls: Final = tuple(tool_call(call) for call in sequence(message.get("tool_calls") or ()))
    return {"role": role, "content": (*content, *calls)}


def request_to_switchyard(
    messages: Sequence[Mapping[str, object]] | None, request_kwargs: Mapping[str, object]
) -> Mapping[str, object]:
    structured: Final = resolve_structured_messages(
        [dict(message) for message in messages] if messages is not None else None, dict(request_kwargs)
    )
    if not structured:
        raise ValueError("Switchyard requires structured messages")
    return {
        "model": "auto",
        "messages": tuple(message_to_switchyard(message) for message in structured),
        "stream": False,
    }


def judge_message(value: object) -> Mapping[str, object]:
    message: Final = mapping(value)
    blocks: Final = text_content(message.get("content"))
    return {"role": message["role"], "content": "".join(str(block["text"]) for block in blocks)}


def judge_request(request: Mapping[str, object], model: str) -> Mapping[str, object]:
    output: Final = mapping(request.get("output") or {})
    sampling: Final = mapping(request.get("sampling") or {})
    options: Final = {
        "max_tokens": output.get("max_output_tokens"),
        "response_format": output.get("response_format"),
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
    }
    return {
        "model": model,
        "stream": False,
        "messages": [
            judge_message(message)
            for message in (
                *sequence(request.get("instructions") or ()),
                *sequence(request.get("messages") or ()),
            )
        ],
        **{key: value for key, value in options.items() if value is not None},
    }


def judge_response(response: ModelResponse) -> LlmResponse.Agg:
    from switchyard.libsy import LlmResponse

    body: Final = mapping(response.model_dump())
    choices: Final = tuple(mapping(choice) for choice in sequence(body["choices"]))
    return LlmResponse.Agg(
        {
            "id": response.id,
            "model": response.model,
            "outputs": tuple(message_to_switchyard(choice["message"]) for choice in choices),
        }
    )
