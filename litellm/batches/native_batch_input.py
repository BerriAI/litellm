"""Translate text-only stored batch inputs for the local input-token counter."""

from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping


def _part_text(part: object) -> str | None:
    fields: Final = as_str_mapping(part)
    if fields is None or fields.keys() - {"type", "text"} or fields.get("type", "text") != "text":
        return None
    text: Final = fields.get("text")
    return text if isinstance(text, str) else None


def _text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    parts: Final = as_object_list(content)
    if not parts:
        return None
    texts: Final = tuple(_part_text(part) for part in parts)
    if None in texts:
        return None
    return "\n".join(text for text in texts if text is not None)


def _chat_turn(turn: object, field: str) -> Mapping[str, object] | None:
    fields: Final = as_str_mapping(turn)
    if fields is None or fields.keys() - {"role", field}:
        return None
    role: Final = fields.get("role", "user")
    text: Final = _text(fields.get(field))
    return {"role": "assistant" if role == "model" else role, "content": text} if text is not None else None


def _chat(messages: object, system: object, *, vertex_parts: bool = False) -> Mapping[str, object] | None:
    turns: Final = as_object_list(messages)
    if not turns:
        return None
    field: Final = "parts" if vertex_parts else "content"
    system_text: Final = _text(system)
    if system is not None and system_text is None:
        return None
    prefix: Final = () if system_text is None else ({"role": "system", "content": system_text},)
    parsed_turns: Final = tuple(_chat_turn(turn, field) for turn in turns)
    if None in parsed_turns:
        return None
    return {"messages": [*prefix, *parsed_turns]}


def bedrock_batch_input(body: Mapping[str, object]) -> Mapping[str, object] | None:
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
        return None
    if "messages" in body:
        return _chat(body["messages"], body.get("system"))
    for field in ("prompt", "inputText"):
        if isinstance(body.get(field), str) and body[field]:
            return {"input": body[field]}
    return None


def vertex_batch_input(body: Mapping[str, object]) -> Mapping[str, object] | None:
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
        return None
    if "messages" in body:
        return _chat(body["messages"], body.get("system"))
    if "contents" in body:
        system: Final = body.get("systemInstruction", body.get("system_instruction"))
        system_fields: Final = as_str_mapping(system)
        if system is not None and system_fields is None:
            return None
        if system_fields is not None and "parts" not in system_fields:
            return None
        return _chat(
            body["contents"], system_fields.get("parts") if system_fields is not None else None, vertex_parts=True
        )
    content: Final = as_str_mapping(body.get("content"))
    if content is not None:
        text: Final = _text(content.get("parts"))
        if text is None:
            return None
        if "title" in body:
            title: Final = _text(body["title"])
            return {"input": title + "\n" + text} if title is not None else None
        return {"input": text}
    return None
