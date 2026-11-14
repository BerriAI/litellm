import re
from collections.abc import Callable, Iterator, Mapping
from graphlib import CycleError, TopologicalSorter
from itertools import chain, islice
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

_TOP_LEVEL_SCHEMA_COMBINATORS: Final = ("allOf", "anyOf", "oneOf")
_OPENAI_REJECTED_TOP_LEVEL_SCHEMA_KEYS: Final = ("enum", "const", "not")
_LOCAL_SCHEMA_REF_PREFIXES: Final = (("#/$defs/", "$defs"), ("#/definitions/", "definitions"))
_MAX_SCHEMA_FLATTEN_DEPTH: Final = 32
_EMPTY_SCHEMA: Final[Mapping[str, object]] = MappingProxyType({})


def _schema_properties(schema: Mapping[str, object]) -> Mapping[str, object]:
    properties: Final = schema.get("properties")
    return properties if isinstance(properties, dict) else _EMPTY_SCHEMA


def _schema_branches(schema: Mapping[str, object], combinator: str) -> tuple[object, ...]:
    branches: Final = schema.get(combinator)
    return tuple(branches) if isinstance(branches, list) else ()


def _schema_required_names(schema: Mapping[str, object]) -> frozenset[str]:
    required: Final = schema.get("required")
    if not isinstance(required, list):
        return frozenset()
    return frozenset(name for name in required if isinstance(name, str))


def _combinator_required_names(combinator: str, branches: tuple[Mapping[str, object], ...]) -> frozenset[str]:
    branch_names: Final = tuple(_schema_required_names(branch) for branch in branches)
    if not branch_names:
        return frozenset()
    if combinator == "allOf":
        return branch_names[0].union(*branch_names[1:])
    return branch_names[0].intersection(*branch_names[1:])


def _resolve_local_schema_ref(root: Mapping[str, object], ref: str) -> Mapping[str, object] | None:
    matched: Final = next(
        ((prefix, container) for prefix, container in _LOCAL_SCHEMA_REF_PREFIXES if ref.startswith(prefix)),
        None,
    )
    if matched is None:
        return None
    prefix, container = matched
    definitions: Final = root.get(container)
    if not isinstance(definitions, dict):
        return None
    target: Final = definitions.get(ref[len(prefix) :])
    return target if isinstance(target, dict) else None


def _mergeable_branch(
    root: Mapping[str, object],
    branch: object,
    seen_refs: frozenset[str],
    depth: int,
    expanded_refs: dict[str, Mapping[str, object] | None],  # mutable-ok: per-call memo bounding repeated $ref work
) -> Mapping[str, object] | None:
    if not isinstance(branch, dict) or depth > _MAX_SCHEMA_FLATTEN_DEPTH:
        return None
    ref: Final = branch.get("$ref")
    if not isinstance(ref, str):
        flattened: Final = _flatten_schema_against_root(branch, root, seen_refs, depth, expanded_refs)
        if any(combinator in flattened for combinator in _TOP_LEVEL_SCHEMA_COMBINATORS):
            return None
        return flattened
    if ref in expanded_refs:
        return expanded_refs[ref]
    if ref in seen_refs:
        return None
    target: Final = _resolve_local_schema_ref(root, ref)
    expanded: Final = (
        None
        if target is None
        else _mergeable_branch(root, target, seen_refs | frozenset((ref,)), depth + 1, expanded_refs)
    )
    expanded_refs[ref] = expanded
    return expanded


def _is_object_schema(schema: Mapping[str, object]) -> bool:
    return schema.get("type") == "object" or ("type" not in schema and "properties" in schema)


def _flatten_schema_against_root(
    schema: Mapping[str, object],
    root: Mapping[str, object],
    seen_refs: frozenset[str],
    depth: int,
    expanded_refs: dict[str, Mapping[str, object] | None],  # mutable-ok: per-call memo bounding repeated $ref work
) -> Mapping[str, object]:
    raw_branch_groups: Final = tuple(
        (
            combinator,
            tuple(
                _mergeable_branch(root, branch, seen_refs, depth + 1, expanded_refs)
                for branch in _schema_branches(schema, combinator)
            ),
        )
        for combinator in _TOP_LEVEL_SCHEMA_COMBINATORS
        if isinstance(schema.get(combinator), list)
    )
    dropped: Final = (
        *(combinator for combinator, _ in raw_branch_groups),
        *(key for key in _OPENAI_REJECTED_TOP_LEVEL_SCHEMA_KEYS if key in schema),
    )
    if not dropped:
        return schema

    if any(branch is None for _, group in raw_branch_groups for branch in group):
        return schema
    branch_groups: Final = tuple(
        (combinator, tuple(branch for branch in group if branch is not None)) for combinator, group in raw_branch_groups
    )
    branches: Final = tuple(branch for _, group in branch_groups for branch in group)
    is_object_schema: Final = _is_object_schema(schema) or (
        "type" not in schema and branches != () and all(_is_object_schema(branch) for branch in branches)
    )
    if not is_object_schema:
        return schema

    merged_properties: Final = {
        name: value for source in (*reversed(branches), schema) for name, value in _schema_properties(source).items()
    }
    required_names: Final = _schema_required_names(schema).union(
        *(_combinator_required_names(combinator, group) for combinator, group in branch_groups)
    )
    kept: Final = MappingProxyType({key: value for key, value in schema.items() if key not in dropped})
    required_update: Final = MappingProxyType({"required": sorted(required_names)}) if required_names else _EMPTY_SCHEMA
    return {
        **kept,
        "type": "object",
        "properties": merged_properties,
        **required_update,
    }


def flatten_top_level_schema_combinators(schema: Mapping[str, object]) -> Mapping[str, object]:
    """Merge top-level ``allOf``/``anyOf``/``oneOf`` branches into an object tool schema.

    OpenAI's function-calling validator rejects tool ``parameters`` carrying
    'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the top level (nested uses
    are accepted), while lenient backends such as the ChatGPT backend Codex
    talks to natively accept them, so an MCP tool declaring a top-level union
    400s through LiteLLM. Branch properties merge without clobbering (the
    top-level schema wins, then earlier branches); ``required`` becomes the
    top-level list plus the intersection of the branch lists for anyOf/oneOf
    or their union for allOf. Branches that are local ``$ref``s
    (``#/$defs/...`` or ``#/definitions/...``) are resolved first, each ref
    at most once per call, and branches that are themselves combinators are
    flattened recursively up to a fixed depth; a branch that cannot be fully
    merged (a boolean schema, an external or cyclic ``$ref``, a non-object
    union, or nesting past the depth cap) leaves the whole schema untouched so
    OpenAI's own validation still applies. Non-object schemas pass through
    unchanged and the input is never mutated.
    """
    return _flatten_schema_against_root(schema, schema, frozenset(), 0, {})


_SUBSCHEMA_KEYWORDS: Final = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SUBSCHEMA_LIST_KEYWORDS: Final = frozenset({"allOf", "anyOf", "items", "oneOf", "prefixItems"})
_SUBSCHEMA_MAP_KEYWORDS: Final = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)

_MAX_SCHEMA_NESTING: Final = 1024


def drop_non_python_regex_patterns(schema: Mapping[str, object]) -> Mapping[str, object]:
    """Drop every regex in a schema position that Python's ``re`` cannot compile.

    OpenAI validates tool ``parameters`` against the 2020-12 metaschema with
    ``jsonschema``'s format checker, which hands each ``pattern`` value and each
    ``patternProperties`` key to ``re.compile``, so a regex written for an
    ECMA-262 engine (Unicode property escapes such as ``\\p{Cc}``, as in Claude
    Code's ``Artifact`` tool) is refused with "'...' is not a 'regex'" by every
    model family on both the chat and Responses wires. Only schema positions are
    walked (properties, items, combinators, ``$defs`` and the other applicators),
    so a ``pattern`` key inside ``default``, ``examples``, ``const`` or vendor
    extensions is data and stays. Outside strict mode the keyword is only a
    hint, so dropping it costs the model a constraint and the caller nothing.
    Compilable regexes and everything else pass through, the input is never
    mutated, and the same object comes back when nothing was dropped. The walk
    is level-order rather than recursive, rebuilt deepest level first, and stops
    at more schema levels than a JSON parser admits, so a cyclic schema built in
    code cannot spin it.
    """
    return _schema_without_rejected_regex(schema, _is_not_python_regex)


def drop_lookaround_regex_patterns(schema: Mapping[str, object]) -> Mapping[str, object]:
    """Drop every regex in a schema position that uses a lookaround assertion.

    Some Bedrock Converse families compile tool schema regexes with an engine that
    has no lookahead or lookbehind and refuse the whole request over one. The ``(?=``,
    ``(?!``, ``(?<=`` and ``(?<!`` openers are matched textually, so an escaped literal
    that spells one is dropped too, trading a hint for a request that goes through.
    A ``patternProperties`` key dropped from an object closed by ``additionalProperties:
    false`` leaves its value schema as that object's ``additionalProperties``, so the
    names it allowed stay allowed; :func:`drop_non_python_regex_patterns` shares the walk.
    """
    return _schema_without_rejected_regex(schema, _uses_regex_lookaround)


def _schema_without_rejected_regex(
    schema: Mapping[str, object], rejected: Callable[[str], bool]
) -> Mapping[str, object]:
    rebuilt: dict[int, Mapping[str, object]] = {}  # mutable-ok: per-call memo of rewritten nodes, deepest level first
    for level in reversed(tuple(islice(_schema_levels(schema), _MAX_SCHEMA_NESTING))):
        rebuilt.update(
            (id(node), rewritten)
            for node in level
            if (rewritten := _node_without_rejected_regex(node, rebuilt, rejected)) is not node
        )
    return rebuilt.get(id(schema), schema)


def _schema_levels(schema: Mapping[str, object]) -> Iterator[tuple[Mapping[str, object], ...]]:
    frontier: tuple[Mapping[str, object], ...] = (schema,)  # rebind-ok: level-order cursor, one level a round
    while frontier:
        yield frontier
        frontier = tuple(child for node in frontier for child in _subschemas(node))


def schema_fragment_ref(ref: str) -> str:
    return f"#{ref}" if ref.startswith(("/$defs/", "/definitions/")) else ref


def schema_has_unresolvable_ref(schema: Mapping[str, object]) -> bool:
    definitions: Final = MappingProxyType(dict(_root_definitions(schema)))
    if not _refs_in(schema) <= definitions.keys():
        return True
    try:
        tuple(TopologicalSorter({pointer: _refs_in(body) for pointer, body in definitions.items()}).static_order())
    except CycleError:
        return True
    return False


def schema_union_property_count(schema: Mapping[str, object]) -> int:
    return sum(1 for _, node in _property_schemas(schema) if _is_union(node))


def schema_optional_property_count(schema: Mapping[str, object]) -> int:
    return sum(len(_optional_property_names(node)) for node in _schema_nodes(schema))


def schema_property_count(schema: Mapping[str, object]) -> int:
    return sum(len(properties) for node in _schema_nodes(schema) if (properties := _schema_at(node, "properties")))


def schema_grammar_branch_count(schema: Mapping[str, object]) -> int:
    return sum(_grammar_branches(node) for node in _schema_nodes(schema))


def schema_has_open_object(schema: Mapping[str, object]) -> bool:
    return any(_is_open_object(node) for node in _schema_nodes(schema))


def schema_has_lookaround_pattern(schema: Mapping[str, object]) -> bool:
    return any(
        isinstance(pattern := node.get("pattern"), str) and _uses_regex_lookaround(pattern)
        for node in _schema_nodes(schema)
    )


def schema_string_formats(schema: Mapping[str, object]) -> frozenset[str]:
    return frozenset(
        value for node in _schema_nodes(schema) if _may_be_string(node) and isinstance(value := node.get("format"), str)
    )


def _may_be_string(node: Mapping[str, object]) -> bool:
    node_type: Final = node.get("type")
    return node_type is None or node_type == "string" or "string" in _list_at(node, "type")


def is_object_schema(node: Mapping[str, object]) -> bool:
    node_type: Final = node.get("type")
    return node_type == "object" or (isinstance(node_type, list) and "object" in node_type)


def _schema_nodes(schema: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    return chain.from_iterable(islice(_schema_levels(schema), _MAX_SCHEMA_NESTING))


_SCHEMA_LIST: Final = TypeAdapter(tuple[object, ...])


def _list_at(container: Mapping[str, object], key: str) -> tuple[object, ...]:
    value: Final = container.get(key)
    return _SCHEMA_LIST.validate_python(value) if isinstance(value, list) else ()


def _named_schemas(container: Mapping[str, object] | None) -> Iterator[tuple[str, Mapping[str, object]]]:
    if container is None:
        return iter(())
    return ((key, value) for key in container if (value := _schema_at(container, key)) is not None)


def _root_definitions(schema: Mapping[str, object]) -> Iterator[tuple[str, Mapping[str, object]]]:
    for defs_key in ("$defs", "definitions"):
        yield from ((f"#/{defs_key}/{name}", body) for name, body in _named_schemas(_schema_at(schema, defs_key)))


def _refs_in(schema: Mapping[str, object]) -> frozenset[str]:
    return frozenset(
        schema_fragment_ref(ref) for node in _schema_nodes(schema) if isinstance(ref := node.get("$ref"), str)
    )


def _property_schemas(schema: Mapping[str, object]) -> Iterator[tuple[str, Mapping[str, object]]]:
    return chain.from_iterable(_named_schemas(_schema_at(node, "properties")) for node in _schema_nodes(schema))


def _is_union(node: Mapping[str, object]) -> bool:
    return isinstance(node.get("anyOf"), list) or isinstance(node.get("oneOf"), list) or len(_list_at(node, "type")) > 1


def _grammar_branches(node: Mapping[str, object]) -> int:
    properties: Final = _schema_at(node, "properties")
    if properties is None:
        return 0
    optional: Final = len(_optional_property_names(node))
    return (1 << optional) * (len(properties) - optional + 1)


def _optional_property_names(node: Mapping[str, object]) -> frozenset[str]:
    properties: Final = _schema_at(node, "properties")
    if properties is None:
        return frozenset()
    return frozenset(properties) - frozenset(name for name in _list_at(node, "required") if isinstance(name, str))


def _is_open_object(node: Mapping[str, object]) -> bool:
    additional_properties: Final = node.get("additionalProperties")
    if additional_properties is True or isinstance(additional_properties, dict) or "patternProperties" in node:
        return True
    return is_object_schema(node) and "properties" not in node and additional_properties is not False


def _subschemas(node: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    for key, value in node.items():
        if key in _SUBSCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            yield from (sub for sub in value.values() if isinstance(sub, dict))
        elif key in _SUBSCHEMA_LIST_KEYWORDS and isinstance(value, list):
            yield from (sub for sub in value if isinstance(sub, dict))
        elif key in _SUBSCHEMA_KEYWORDS and isinstance(value, dict):
            yield value


def _node_without_rejected_regex(
    node: Mapping[str, object],
    rebuilt: Mapping[int, Mapping[str, object]],
    rejected: Callable[[str], bool],
) -> Mapping[str, object]:
    kept: Final = {
        key: _keyword_value_rebuilt(key, value, rebuilt, rejected)
        for key, value in node.items()
        if key != "pattern" or not isinstance(value, str) or not rejected(value)
    }
    if len(kept) == len(node) and all(kept[key] is node[key] for key in kept):
        return node
    dropped_pattern_properties: Final = _dropped_pattern_properties(node, kept, rebuilt)
    if not dropped_pattern_properties or kept.get("additionalProperties") is not False:
        return kept
    return {**kept, "additionalProperties": _any_of(dropped_pattern_properties)}


def _dropped_pattern_properties(
    node: Mapping[str, object],
    kept: Mapping[str, object],
    rebuilt: Mapping[int, Mapping[str, object]],
) -> tuple[object, ...]:
    before: Final = _schema_at(node, "patternProperties")
    after: Final = _schema_at(kept, "patternProperties")
    if before is None or after is None:
        return ()
    return tuple(rebuilt.get(id(sub), sub) for name, sub in before.items() if name not in after)


def _schema_at(container: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value: Final = container.get(key)
    return value if isinstance(value, dict) else None


def _any_of(schemas: tuple[object, ...]) -> object:
    return schemas[0] if len(schemas) == 1 else {"anyOf": list(schemas)}


def _keyword_value_rebuilt(
    key: str,
    value: object,
    rebuilt: Mapping[int, Mapping[str, object]],
    rejected: Callable[[str], bool],
) -> object:
    if key in _SUBSCHEMA_MAP_KEYWORDS and isinstance(value, dict):
        kept: Final = {
            name: rebuilt.get(id(sub), sub)
            for name, sub in value.items()
            if key != "patternProperties" or not isinstance(name, str) or not rejected(name)
        }
        return value if len(kept) == len(value) and all(kept[name] is value[name] for name in kept) else kept
    if key in _SUBSCHEMA_LIST_KEYWORDS and isinstance(value, list):
        items: Final = [rebuilt.get(id(sub), sub) for sub in value]
        return value if all(new is old for new, old in zip(items, value, strict=True)) else items
    if key in _SUBSCHEMA_KEYWORDS and isinstance(value, dict):
        return rebuilt.get(id(value), value)
    return value


def _is_not_python_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except (re.error, RecursionError):
        return True
    return False


_REGEX_LOOKAROUND_RE: Final = re.compile(r"\(\?<?[=!]")


def _uses_regex_lookaround(pattern: str) -> bool:
    return _REGEX_LOOKAROUND_RE.search(pattern) is not None


def flatten_combinators_and_drop_non_python_regex_patterns(schema: Mapping[str, object]) -> Mapping[str, object]:
    return flatten_top_level_schema_combinators(drop_non_python_regex_patterns(schema))


def tool_with_sanitized_parameters(
    tool: Mapping[str, object],
    sanitize: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> Mapping[str, object]:
    """Run the tool's JSON schema through ``sanitize``: ``function.parameters`` on an
    OpenAI tool, ``input_schema`` on an Anthropic one. The same object comes back when
    nothing changed."""
    function: Final = _schema_at(tool, "function")
    if function is not None:
        parameters: Final = _schema_at(function, "parameters")
        if parameters is None:
            return tool
        sanitized_parameters: Final = sanitize(parameters)
        if sanitized_parameters is parameters:
            return tool
        return {**tool, "function": {**function, "parameters": sanitized_parameters}}
    input_schema: Final = _schema_at(tool, "input_schema")
    if input_schema is None:
        return tool
    sanitized_schema: Final = sanitize(input_schema)
    return tool if sanitized_schema is input_schema else {**tool, "input_schema": sanitized_schema}
