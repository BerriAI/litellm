"""Translate text-only stored batch inputs for the local input-token counter."""

from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping


def _part_text(part: object) -> str:
    fields: Final = as_str_mapping(part)
    if fields is None or fields.keys() - {"type", "text"} or fields.get("type", "text") != "text":
        raise ValueError("Unsupported batch content part")
    text: Final = fields.get("text")
    if not isinstance(text, str):
        raise ValueError("Expected a text part")
    return text


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    parts: Final = as_object_list(content)
    if not parts:
        raise ValueError("Expected text parts")
    return "\n".join(_part_text(part) for part in parts)


def _chat_turn(turn: object, field: str) -> Mapping[str, object]:
    fields: Final = as_str_mapping(turn)
    if fields is None or fields.keys() - {"role", field}:
        raise ValueError("Unsupported batch turn")
    role: Final = fields.get("role", "user")
    return {"role": "assistant" if role == "model" else role, "content": _text(fields.get(field))}


def _chat(messages: object, system: object, *, vertex_parts: bool = False) -> Mapping[str, object]:
    turns: Final = as_object_list(messages)
    if not turns:
        raise ValueError("Expected chat messages")
    field: Final = "parts" if vertex_parts else "content"
    prefix: Final = () if system is None else ({"role": "system", "content": _text(system)},)
    return {"messages": [*prefix, *(_chat_turn(turn, field) for turn in turns)]}


def bedrock_batch_input(body: Mapping[str, object]) -> Mapping[str, object]:
    if body.keys() - {
        "messages",
        "system",
        "prompt",
        "inputText",
        "anthropic_version",
        "schemaVersion",
        "max_tokens",
        "max_completion_tokens",
        "max_gen_len",
        "temperature",
        "top_k",
        "top_p",
        "stop_sequences",
        "inferenceConfig",
        "service_tier",
        "dimensions",
        "normalize",
        "embeddingTypes",
    }:
        raise ValueError("Unsupported structured batch input")
    if "messages" in body:
        return _chat(body["messages"], body.get("system"))
    for field in ("prompt", "inputText"):
        if isinstance(body.get(field), str) and body[field]:
            return {"input": body[field]}
    raise ValueError("Unsupported Bedrock batch input")


def vertex_batch_input(body: Mapping[str, object]) -> Mapping[str, object]:
    if body.keys() - {
        "messages",
        "system",
        "anthropic_version",
        "max_tokens",
        "temperature",
        "top_k",
        "top_p",
        "stop_sequences",
        "contents",
        "content",
        "system_instruction",
        "systemInstruction",
        "labels",
        "generationConfig",
        "generation_config",
        "safetySettings",
        "safety_settings",
        "task_type",
        "output_dimensionality",
        "embed_content_config",
        "title",
    }:
        raise ValueError("Unsupported structured batch input")
    if "messages" in body:
        return _chat(body["messages"], body.get("system"))
    if "contents" in body:
        system: Final = body.get("systemInstruction", body.get("system_instruction"))
        system_fields: Final = as_str_mapping(system)
        if system is not None and system_fields is None:
            raise ValueError("Expected a Vertex system instruction")
        return _chat(body["contents"], system_fields["parts"] if system_fields is not None else None, vertex_parts=True)
    content: Final = as_str_mapping(body.get("content"))
    if content is not None:
        text: Final = _text(content["parts"])
        if "title" in body:
            return {"input": _text(body["title"]) + "\n" + text}
        return {"input": text}
    raise ValueError("Unsupported Vertex batch input")
