"""Pure helpers shared by harness configs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Final

SKILL_MANIFEST: Final = "SKILL.md"
_JSON_DECODER: Final = json.JSONDecoder()


def normalize_tool_name(native_name: str, mapping: Mapping[str, str]) -> str:
    """Normalized tool name (read, write, edit, bash, ...) or the native name if unmapped."""
    return mapping.get(native_name, native_name)


def native_tool_names(normalized: Sequence[str], mapping: Mapping[str, Sequence[str]]) -> list[str]:
    """Native names for normalized tool names, de-duplicated, order kept."""
    natives: list[str] = []
    for name in normalized:
        for native in mapping.get(name, (name,)):
            if native not in natives:
                natives.append(native)
    return natives


def last_json_object(text: str) -> str | None:
    """The last top-level `{...}` in text that parses as a JSON object, re-serialized."""
    last: str | None = None
    index = text.find("{")
    while index != -1:
        try:
            obj, end = _JSON_DECODER.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        if isinstance(obj, dict):
            last = json.dumps(obj)
        index = text.find("{", end)
    return last


def structured_output_instruction(schema: Mapping[str, Any]) -> str:
    return (
        "When you have finished, your final message must be a single JSON object that "
        "matches this JSON schema, with no other text before or after it:\n"
        f"{json.dumps(schema)}"
    )


def strict_json_schema(schema: Any) -> Any:
    """Make a JSON schema acceptable to OpenAI strict structured outputs.

    Every object gets `additionalProperties: false` and all of its properties required,
    recursively. Keywords strict mode rejects next to $ref are dropped.
    """
    if isinstance(schema, list):
        return [strict_json_schema(entry) for entry in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: strict_json_schema(value) for key, value in schema.items()}
    if "$ref" in result:
        return {"$ref": result["$ref"]}
    result.pop("default", None)
    properties = result.get("properties")
    if result.get("type") == "object" or isinstance(properties, dict):
        props = properties if isinstance(properties, dict) else {}
        result = {**result, "properties": props, "required": list(props.keys()), "additionalProperties": False}
    return result


def decode_json_line(line: bytes | str) -> dict[str, Any] | None:
    """One JSONL line as a dict, or None for blank / non-JSON / non-object lines."""
    text = line.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def stderr_tail_text(stderr_tail: Sequence[str]) -> str:
    return "\n".join(line for line in stderr_tail if line.strip())


def read_skill_files(skill_dir: str) -> list[tuple[str, bytes]]:
    """(relative path, bytes) for every file under a local skill folder."""
    root = os.path.realpath(os.fspath(skill_dir))
    if not os.path.isfile(os.path.join(root, SKILL_MANIFEST)):
        raise ValueError(f"skill folder {skill_dir!r} has no {SKILL_MANIFEST}")
    files: list[tuple[str, bytes]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            path = os.path.join(dirpath, filename)
            with open(path, "rb") as fh:
                files.append((os.path.relpath(path, root), fh.read()))
    return files
