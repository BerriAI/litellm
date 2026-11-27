from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping

from ..common_utils import VertexAIError

JSON_SCHEMA_CONFIG_COLUMN: Final = "generation_config"

_SchemaPath: TypeAlias = tuple[str, ...]
_JSON_ARRAY: Final = TypeAdapter(tuple[object, ...])
_ARRAY_ELEMENT: Final = "[]"
_SUBSCHEMA_KEYWORDS: Final = frozenset({"items", "additionalProperties"})
_SUBSCHEMA_LIST_KEYWORDS: Final = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_KEYWORDS_VERTEX_REFUSES_AS_NULL: Final = frozenset(
    {
        "type",
        "format",
        "pattern",
        "title",
        "description",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "properties",
        *_SUBSCHEMA_KEYWORDS,
    }
)


@dataclass(frozen=True, slots=True)
class _SchemaNode:
    path: _SchemaPath
    fields: Mapping[str, object]
    names_properties: bool


@dataclass(frozen=True, slots=True)
class _SchemaShape:
    node_paths: frozenset[_SchemaPath]
    field_paths: frozenset[_SchemaPath]
    null_refusing_field_paths: frozenset[_SchemaPath]


@dataclass(frozen=True, slots=True)
class _MergeClash:
    earlier_lineno: int
    nulled_lineno: int
    path: _SchemaPath


def _schema_nodes(value: object, path: _SchemaPath, names_properties: bool) -> Iterator[_SchemaNode]:
    fields: Final = as_str_mapping(value)
    if fields is None:
        return
    yield _SchemaNode(path=path, fields=fields, names_properties=names_properties)
    for key, child in fields.items():
        yield from _child_schema_nodes(key, child, (*path, key), names_properties)


def _child_schema_nodes(
    key: str, child: object, path: _SchemaPath, parent_names_properties: bool
) -> Iterator[_SchemaNode]:
    if parent_names_properties or key in _SUBSCHEMA_KEYWORDS:
        yield from _schema_nodes(child, path, names_properties=False)
    elif key == "properties":
        yield from _schema_nodes(child, path, names_properties=True)
    elif key in _SUBSCHEMA_LIST_KEYWORDS and isinstance(child, list):
        for element in _JSON_ARRAY.validate_python(child):
            yield from _schema_nodes(element, (*path, _ARRAY_ELEMENT), names_properties=False)


def _refuses_null(node: _SchemaNode, key: str, value: object) -> bool:
    if isinstance(value, list):
        return False
    return node.names_properties or key in _KEYWORDS_VERTEX_REFUSES_AS_NULL


def _field_paths(nodes: Iterable[_SchemaNode]) -> Iterator[tuple[_SchemaPath, bool]]:
    for node in nodes:
        for key, value in node.fields.items():
            yield (*node.path, key), _refuses_null(node, key, value)


def _schema_shape(schema: object) -> _SchemaShape:
    nodes: Final = tuple(_schema_nodes(schema, (), names_properties=False))
    fields: Final = tuple(_field_paths(nodes))
    return _SchemaShape(
        node_paths=frozenset(node.path for node in nodes),
        field_paths=frozenset(path for path, _ in fields),
        null_refusing_field_paths=frozenset(path for path, refuses_null in fields if refuses_null),
    )


def _null_vertex_refuses(target: _SchemaShape, source: _SchemaShape) -> _SchemaPath | None:
    missing: Final = source.null_refusing_field_paths - target.field_paths
    return min((path for path in missing if path[:-1] in target.node_paths), default=None)


def _merge_clash(earlier: _SchemaShape, earlier_lineno: int, shape: _SchemaShape, lineno: int) -> _MergeClash | None:
    nulled_in_earlier: Final = _null_vertex_refuses(earlier, shape)
    if nulled_in_earlier is not None:
        return _MergeClash(earlier_lineno=earlier_lineno, nulled_lineno=earlier_lineno, path=nulled_in_earlier)
    nulled_in_later: Final = _null_vertex_refuses(shape, earlier)
    if nulled_in_later is None:
        return None
    return _MergeClash(earlier_lineno=earlier_lineno, nulled_lineno=lineno, path=nulled_in_later)


def _response_json_schema_shape(vertex_row: Mapping[str, object]) -> _SchemaShape | None:
    request: Final = as_str_mapping(vertex_row.get("request"))
    schema_config: Final = None if request is None else as_str_mapping(request.get(JSON_SCHEMA_CONFIG_COLUMN))
    if schema_config is None or "response_json_schema" not in schema_config:
        return None
    return _schema_shape(schema_config["response_json_schema"])


def _shown_path(path: _SchemaPath) -> str:
    return ".".join(path).replace(f".{_ARRAY_ELEMENT}", _ARRAY_ELEMENT)


def _schema_shapes_checked(
    seen: Mapping[_SchemaShape, int], lineno: int, vertex_row: Mapping[str, object]
) -> Mapping[_SchemaShape, int]:
    shape: Final = _response_json_schema_shape(vertex_row)
    if shape is None or shape in seen:
        return seen
    clashes: Final = (_merge_clash(earlier, earlier_lineno, shape, lineno) for earlier, earlier_lineno in seen.items())
    clash: Final = next((found for found in clashes if found is not None), None)
    if clash is not None:
        raise VertexAIError(
            status_code=400,
            message=f"Batch input file line {lineno} sets a json_schema that cannot share a Vertex AI batch with the "
            f"one on line {clash.earlier_lineno}. Vertex AI loads every row of a batch file into one "
            f"BigQuery table, which would set {_shown_path(clash.path)} to null in the schema of line "
            f"{clash.nulled_lineno}, and Vertex AI refuses that null. Put rows with these schemas in separate batch "
            "files",
        )
    return MappingProxyType({**seen, shape: lineno})


def reject_json_schemas_the_batch_merge_breaks(
    numbered_rows: Iterable[tuple[int, Mapping[str, object]]],
) -> Iterator[tuple[int, Mapping[str, object]]]:
    seen: Mapping[_SchemaShape, int] = MappingProxyType({})  # rebind-ok: earlier rows' distinct schema shapes
    for lineno, row in numbered_rows:
        seen = _schema_shapes_checked(seen, lineno, row)
        yield lineno, row
