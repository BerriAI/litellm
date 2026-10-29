"""Pure helpers shared by harness configs."""

from __future__ import annotations

import itertools
import json
import os
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter, ValidationError

if typing.TYPE_CHECKING:
    from litellm.harness.context import SessionContext

SKILL_MANIFEST: Final = "SKILL.md"
_JSON_DECODER: Final = json.JSONDecoder()
_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_RAW_DECODE_ADAPTER: Final = TypeAdapter(tuple[object, int])


def gateway_headers(ctx: SessionContext) -> dict[str, str]:  # mutable-ok: acompletion(extra_headers=) requires dict
    metadata_json: Final = json.dumps(dict(ctx.metadata), default=str) if ctx.metadata else None
    return {
        "x-litellm-tags": f"harness,{ctx.harness.value}",
        **({"x-litellm-spend-logs-metadata": metadata_json} if metadata_json is not None else {}),
    }


def normalize_tool_name(native_name: str, mapping: Mapping[str, str]) -> str:
    """Normalized tool name (read, write, edit, bash, ...) or the native name if unmapped."""
    return mapping.get(native_name, native_name)


def native_tool_names(normalized: Sequence[str], mapping: Mapping[str, Sequence[str]]) -> Sequence[str]:
    """Native names for normalized tool names, de-duplicated, order kept."""
    expanded: Final = itertools.chain.from_iterable(mapping.get(name, (name,)) for name in normalized)
    return list(dict.fromkeys(expanded))  # mutable-ok: public helper whose callers/tests compare against list literals


def last_json_object(text: str) -> str | None:
    """The last top-level `{...}` in text that parses as a JSON object, re-serialized."""
    last: str | None = None
    index = text.find("{")
    while index != -1:
        try:
            raw_decoded: object = _JSON_DECODER.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        obj, end = _RAW_DECODE_ADAPTER.validate_python(raw_decoded)
        if isinstance(obj, dict):
            last = json.dumps(obj)
        index = text.find("{", end)
    return last


def structured_output_instruction(schema: Mapping[str, object]) -> str:
    return (
        "When you have finished, your final message must be a single JSON object that "
        "matches this JSON schema, with no other text before or after it:\n"
        f"{json.dumps(schema)}"
    )


def build_instructions(ctx: SessionContext) -> str | None:
    """System instructions plus the structured-output instruction, for runtimes that take one prompt file."""
    schema_part: Final = (
        structured_output_instruction(ctx.output.model_json_schema()) if ctx.output is not None else None
    )
    sections: Final = tuple(section for section in (ctx.instructions, schema_part) if section)
    return "\n\n".join(sections) if sections else None


def turn_prompt(ctx: SessionContext, prompt: str) -> str:
    """Repeat the schema instruction in the user turn; system instructions alone are too weak."""
    if ctx.output is None:
        return prompt
    return f"{prompt}\n\n{structured_output_instruction(ctx.output.model_json_schema())}"


def decode_json_line(line: bytes | str) -> Mapping[str, object] | None:
    """One JSONL line as a dict, or None for blank / non-JSON / non-object lines."""
    text = line.strip()
    if not text:
        return None
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(text)
    except ValidationError:
        return None


def stderr_tail_text(stderr_tail: Sequence[str]) -> str:
    return "\n".join(line for line in stderr_tail if line.strip())


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _walk_files(root: str) -> Iterator[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        yield from (os.path.join(dirpath, filename) for filename in sorted(filenames))


def read_skill_files(skill_dir: str) -> tuple[tuple[str, bytes], ...]:
    """(relative path, bytes) for every file under a local skill folder."""
    root: Final = os.path.realpath(os.fspath(skill_dir))
    if not os.path.isfile(os.path.join(root, SKILL_MANIFEST)):
        raise ValueError(f"skill folder {skill_dir!r} has no {SKILL_MANIFEST}")
    return tuple((os.path.relpath(path, root), _read_bytes(path)) for path in _walk_files(root))
