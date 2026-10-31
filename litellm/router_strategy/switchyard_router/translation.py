from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, TypeAdapter

from litellm.types.utils import ModelResponse

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_SEQUENCE: Final = TypeAdapter(tuple[object, ...])
_CONTROLLED: Final = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "response_format",
        "reasoning_effort",
        "stream",
        "stream_options",
    }
)


def mapping(value: object) -> Mapping[str, object]:
    return _MAPPING.validate_python(value)


def sequence(value: object) -> tuple[object, ...]:
    return _SEQUENCE.validate_python(value)


def text_blocks(content: object) -> tuple[Mapping[str, object], ...]:
    if content is None:
        return ()
    if isinstance(content, str):
        return ({"type": "text", "text": content},)
    blocks: Final = tuple(mapping(block) for block in sequence(content))
    if any(block.get("type") != "text" or not isinstance(block.get("text"), str) for block in blocks):
        raise ValueError("Switchyard currently supports text and function-tool chat content only")
    return blocks


def normalized_tool_call(raw: object) -> Mapping[str, object]:
    call: Final = mapping(raw)
    function: Final = mapping(call.get("function"))
    arguments: Final = function.get("arguments")
    if call.get("type") != "function" or not isinstance(arguments, str):
        raise ValueError("Switchyard supports function tool calls with JSON arguments only")
    return {"type": "tool_call", "id": call["id"], "name": function["name"], "arguments": json.loads(arguments)}


def normalized_message(raw: object) -> Mapping[str, object]:
    message: Final = mapping(raw)
    role: Final = message.get("role")
    blocks: Final = text_blocks(message.get("content"))
    if role == "tool":
        return {
            "role": "tool",
            "content": [
                {"type": "tool_result", "tool_call_id": message["tool_call_id"], "content": blocks, "is_error": None}
            ],
        }
    if role not in ("system", "developer", "user", "assistant"):
        raise ValueError("Switchyard received an unsupported chat role")
    calls: Final = tuple(normalized_tool_call(call) for call in sequence(message.get("tool_calls") or ()))
    refusal: Final = message.get("refusal")
    refusals: Final = ({"type": "refusal", "text": refusal},) if isinstance(refusal, str) else ()
    return {"role": role, "content": (*blocks, *calls, *refusals)}


def normalized_tool(raw: object) -> Mapping[str, object]:
    tool: Final = mapping(raw)
    if tool.get("type") != "function":
        raise ValueError("Switchyard currently supports function tools only")
    function: Final = mapping(tool.get("function"))
    return {
        "name": function["name"],
        "description": function.get("description"),
        "parameters": function.get("parameters", {}),
        "strict": function.get("strict"),
    }


def normalized_tool_choice(raw: object) -> Mapping[str, object] | None:
    if raw is None:
        return None
    if raw in ("auto", "none", "required"):
        return {"type": raw}
    choice: Final = mapping(raw)
    if choice.get("type") != "function":
        raise ValueError("Switchyard currently supports function tool_choice only")
    return {"type": "tool", "data": {"name": mapping(choice.get("function"))["name"]}}


def normalize_request(request: Mapping[str, object]) -> Mapping[str, object]:
    messages: Final = tuple(normalized_message(message) for message in sequence(request.get("messages")))
    if not messages:
        raise ValueError("Switchyard requires at least one message")
    if request.get("n", 1) != 1:
        raise ValueError("Switchyard requires n=1")
    return {
        "model": "auto",
        "instructions": [],
        "messages": messages,
        "tools": tuple(normalized_tool(tool) for tool in sequence(request.get("tools") or ())),
        "tool_choice": normalized_tool_choice(request.get("tool_choice")),
        "sampling": {
            "temperature": request.get("temperature"),
            "top_p": request.get("top_p"),
            "top_k": request.get("top_k"),
        },
        "output": {
            "max_output_tokens": request.get("max_completion_tokens", request.get("max_tokens")),
            "response_format": request.get("response_format"),
        },
        "reasoning": {"effort": request.get("reasoning_effort"), "raw": None},
        "stream": False,
    }


def chat_message(raw: object, originals: Sequence[object]) -> Mapping[str, object]:
    message: Final = mapping(raw)
    for original in originals:
        if json.dumps(normalized_message(original), sort_keys=True) == json.dumps(message, sort_keys=True):
            return mapping(original)
    blocks: Final = tuple(mapping(block) for block in sequence(message.get("content", ())))
    text: Final = "".join(str(block["text"]) for block in blocks if block.get("type") == "text")
    calls: Final = tuple(
        {
            "id": block["id"],
            "type": "function",
            "function": {"name": block["name"], "arguments": json.dumps(block["arguments"])},
        }
        for block in blocks
        if block.get("type") == "tool_call"
    )
    if message.get("role") == "tool":
        if len(blocks) != 1 or blocks[0].get("type") != "tool_result":
            raise ValueError("Switchyard produced an unsupported tool result")
        result: Final = blocks[0]
        return {
            "role": "tool",
            "tool_call_id": result["tool_call_id"],
            "content": tuple(mapping(block) for block in sequence(result["content"])),
        }
    if any(block.get("type") not in ("text", "tool_call", "refusal") for block in blocks):
        raise ValueError("Switchyard produced unsupported normalized message content")
    refusals: Final = tuple(str(block["text"]) for block in blocks if block.get("type") == "refusal")
    return {
        "role": message["role"],
        "content": text or None,
        **({"tool_calls": calls} if calls else {}),
        **({"refusal": "".join(refusals)} if refusals else {}),
    }


def chat_tool_choice(raw: object) -> object:
    if raw is None:
        return None
    choice: Final = mapping(raw)
    if choice.get("type") in ("auto", "none", "required"):
        return choice["type"]
    if choice.get("type") == "tool":
        return {"type": "function", "function": mapping(choice["data"])}
    raise ValueError("Switchyard produced unsupported tool_choice")


def chat_request(
    normalized: Mapping[str, object], original: Mapping[str, object], *, model: str, stream: bool, judge: bool
) -> Mapping[str, object]:
    originals: Final = sequence(original.get("messages", ()))
    instructions: Final = sequence(normalized.get("instructions", ()))
    messages: Final = sequence(normalized.get("messages", ()))
    sampling: Final = mapping(normalized.get("sampling", {}))
    output: Final = mapping(normalized.get("output", {}))
    reasoning: Final = mapping(normalized.get("reasoning", {}))
    token_key: Final = "max_completion_tokens" if not judge and "max_completion_tokens" in original else "max_tokens"
    options: Final[Mapping[str, object]] = {
        **sampling,
        token_key: output.get("max_output_tokens"),
        "response_format": output.get("response_format"),
        "reasoning_effort": reasoning.get("effort"),
        "tool_choice": chat_tool_choice(normalized.get("tool_choice")),
    }
    tools: Final = tuple(
        {"type": "function", "function": {k: v for k, v in mapping(tool).items() if v is not None}}
        for tool in sequence(normalized.get("tools", ()))
    )
    passthrough: Final = {} if judge else {key: value for key, value in original.items() if key not in _CONTROLLED}
    return {
        **passthrough,
        **{key: value for key, value in options.items() if value is not None},
        "model": model,
        "messages": [chat_message(message, originals) for message in (*instructions, *messages)],
        "stream": stream,
        **({"tools": tools} if tools else {}),
        **({"stream_options": original["stream_options"]} if stream and "stream_options" in original else {}),
    }


def normalize_response(response: ModelResponse, response_id: str) -> Mapping[str, object]:
    body: Final = mapping(BaseModel.model_dump(response))
    choices: Final = tuple(mapping(choice) for choice in sequence(body.get("choices")))
    return {"id": response_id, "model": body.get("model"), "outputs": tuple(normalized_output(c) for c in choices)}


def normalized_output(choice: Mapping[str, object]) -> Mapping[str, object]:
    reasons: Final = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filter",
    }
    reason: Final = choice.get("finish_reason")
    return {
        **normalized_message(choice.get("message")),
        "stop_reason": reasons.get(str(reason), "unknown") if reason is not None else None,
    }
