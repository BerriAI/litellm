"""
Vertex AI loads a batch input file into BigQuery before it runs the rows, so every object key in a row becomes a
column name, and BigQuery rejects `$` in column names. A Gemini batch row's response schema therefore has every local
`$ref` inlined, and a Gemini row that still carries a key starting with `$` is refused before upload
"""

import json
from collections.abc import Iterator, Mapping
from functools import partial, reduce
from itertools import chain
from typing import Final, TypeAlias
from urllib.parse import unquote

from pydantic import JsonValue, TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping, iter_items
from litellm.litellm_core_utils.prompt_templates.common_utils import DEFS_MAX_INLINED_BYTES

_Pointer: TypeAlias = tuple[str, ...]
_RefChain: TypeAlias = tuple[_Pointer, ...]

_JSON_SCHEMA: Final = TypeAdapter(dict[str, JsonValue])
_DROPPED_KEYWORDS: Final = frozenset(
    {"$ref", "$defs", "definitions", "$schema", "$id", "$comment", "$anchor", "$dynamicAnchor", "$vocabulary"}
)
_NAMED_SCHEMA_MAPS: Final = frozenset(
    {"properties", "patternProperties", "dependentSchemas", "dependentRequired", "dependencies"}
)
_DATA_KEYWORDS: Final = frozenset({"enum", "const", "default", "examples"})


def _pointer(ref: JsonValue) -> _Pointer:
    fragment: Final = unquote(ref[1:]) if isinstance(ref, str) and ref.startswith("#") else None
    if fragment is None or (fragment and not fragment.startswith("/")):
        raise ValueError(
            f"response_format json_schema has $ref {ref!r}, which is not a JSON pointer into the schema itself, "
            "and a Vertex AI batch row can only carry a $ref it can inline"
        )
    return tuple(segment.replace("~1", "/").replace("~0", "~") for segment in fragment.split("/")[1:])


def _child(ref: JsonValue, node: JsonValue, key: str) -> JsonValue:
    if isinstance(node, dict) and key in node:
        return node[key]
    if isinstance(node, list) and key.isdecimal() and int(key) < len(node):
        return node[int(key)]
    raise ValueError(f"response_format json_schema has $ref {ref!r}, which points at nothing in the schema")


class _SchemaInliner:
    __slots__ = ("_budget_bytes", "_root")

    def __init__(self, root: JsonValue) -> None:
        self._root: Final = root
        self._budget_bytes: int = DEFS_MAX_INLINED_BYTES

    def inlined(self, node: JsonValue, ref_chain: _RefChain) -> JsonValue:
        if isinstance(node, list):
            return [self.inlined(item, ref_chain) for item in node]
        if isinstance(node, dict):
            return dict(self._inlined_items(node, ref_chain))
        return node

    def _inlined_items(self, node: Mapping[str, JsonValue], ref_chain: _RefChain) -> Iterator[tuple[str, JsonValue]]:
        target_items: Final = self._ref_target_items(node["$ref"], ref_chain) if "$ref" in node else iter(())
        return chain(
            target_items,
            (
                (key, self._inlined_keyword(key, value, ref_chain))
                for key, value in node.items()
                if key not in _DROPPED_KEYWORDS
            ),
        )

    def _ref_target_items(self, ref: JsonValue, ref_chain: _RefChain) -> Iterator[tuple[str, JsonValue]]:
        pointer: Final = _pointer(ref)
        if pointer in ref_chain:
            raise ValueError(
                f"response_format json_schema is recursive through $ref {ref!r}, and a Vertex AI batch row must "
                "inline every $ref, so send this row as a regular request instead"
            )
        target: Final = reduce(partial(_child, ref), pointer, self._root)
        if not isinstance(target, dict):
            raise TypeError(
                f"response_format json_schema has $ref {ref!r}, which points at something other than a schema"
            )
        self._spend(target)
        return self._inlined_items(target, (*ref_chain, pointer))

    def _spend(self, target: Mapping[str, JsonValue]) -> None:
        self._budget_bytes -= len(json.dumps(target, separators=(",", ":")))
        if self._budget_bytes < 0:
            raise ValueError(
                f"response_format json_schema grows past {DEFS_MAX_INLINED_BYTES:,} bytes once its $ref targets "
                "are inlined, which a Vertex AI batch row needs"
            )

    def _inlined_keyword(self, keyword: str, value: JsonValue, ref_chain: _RefChain) -> JsonValue:
        if keyword in _DATA_KEYWORDS:
            return value
        if keyword in _NAMED_SCHEMA_MAPS and isinstance(value, dict):
            return {name: self.inlined(schema, ref_chain) for name, schema in value.items()}
        return self.inlined(value, ref_chain)


def with_batch_loadable_response_json_schema(optional_params: Mapping[str, object]) -> Mapping[str, object]:
    schema: Final = optional_params.get("response_json_schema")
    if not isinstance(schema, dict):
        return optional_params
    root: Final = _JSON_SCHEMA.validate_python(schema)
    return {**optional_params, "response_json_schema": _SchemaInliner(root).inlined(root, ())}


def _dollar_key_paths(value: object, path: str) -> Iterator[str]:
    mapping: Final = as_str_mapping(value)
    if mapping is not None:
        return chain.from_iterable(
            _key_paths(f"{path}.{key}" if path else key, key, child) for key, child in mapping.items()
        )
    if type(value) in (list, tuple):
        return chain.from_iterable(
            _dollar_key_paths(item, f"{path}[{index}]") for index, item in enumerate(iter_items(value))
        )
    return iter(())


def _key_paths(path: str, key: str, child: object) -> Iterator[str]:
    return iter((path,)) if key.startswith("$") else _dollar_key_paths(child, path)


def bigquery_loadable_row(row: Mapping[str, object]) -> Mapping[str, object]:
    dollar_key_path: Final = next(_dollar_key_paths(row, ""), None)
    if dollar_key_path is not None:
        raise ValueError(
            f"the row has the key {dollar_key_path}, but Vertex AI batch input cannot carry an object key starting "
            "with '$', so send this row as a regular request instead"
        )
    return row
