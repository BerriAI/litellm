"""Minimal ``schema.prisma`` parser used by the SQLModel parity test.

This is intentionally **not** a full Prisma parser. It targets only the
constructs that actually appear in ``litellm``'s ``schema.prisma`` (as of the
start of the Prisma -> SQLModel migration) and is exercised by the parity
test in ``tests/test_litellm/proxy/db/sqlmodel/``.

The parser produces a structured representation that is easy to compare
against the SQLAlchemy ``MetaData`` of the generated SQLModel classes:

* Top-level ``PrismaSchema`` with ``models`` (dict by model name) and
  ``enums`` (dict by enum name).
* Each ``PrismaModel`` carries its **scalar** fields, primary key,
  uniqueness constraints, and indexes.
* Relation fields (``Foo[]`` / ``Foo? @relation(...)``) are recorded
  separately in ``relations`` and are explicitly ignored by the column
  parity check -- relations are not columns.

The parser is pure-Python (no third-party deps) so it can run in any test
environment and serve as a building block for future code generators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PrismaField:
    """A single scalar (or scalar-array) column on a Prisma model."""

    name: str  # field name as written in schema.prisma
    column_name: str  # column name on disk (respects @map(...))
    base_type: str  # e.g. "String", "Int", "BigInt", "DateTime", "Json", "Bytes", "Float", "Boolean", or an enum name
    is_optional: bool  # True if `?`
    is_list: bool  # True if `[]`
    is_id: bool  # True if marked `@id`
    is_unique: bool  # True if marked `@unique`
    has_default: bool
    default_raw: Optional[str]  # raw text inside `@default(...)`
    has_updated_at: bool  # True if marked `@updatedAt`
    attributes: List[str] = field(default_factory=list)  # raw `@...` attributes


@dataclass
class PrismaRelation:
    """A relation field (``Foo[]`` or ``Foo? @relation(...)``) -- not a column."""

    name: str
    target_model: str
    is_optional: bool
    is_list: bool
    relation_attributes: List[str] = field(default_factory=list)


@dataclass
class PrismaIndex:
    """A ``@@index([...])`` declaration."""

    fields: Tuple[str, ...]
    map_name: Optional[str] = None


@dataclass
class PrismaUnique:
    """A ``@@unique([...])`` declaration."""

    fields: Tuple[str, ...]


@dataclass
class PrismaModel:
    """A Prisma ``model`` block, scalar columns + constraints only."""

    name: str
    table_name: str  # respects ``@@map("...")``; defaults to model name
    fields: List[PrismaField] = field(default_factory=list)
    relations: List[PrismaRelation] = field(default_factory=list)
    primary_key: Tuple[str, ...] = ()  # field names (not column names)
    uniques: List[PrismaUnique] = field(default_factory=list)
    indexes: List[PrismaIndex] = field(default_factory=list)
    raw_attributes: List[str] = field(default_factory=list)

    def field_by_name(self, name: str) -> Optional[PrismaField]:
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass
class PrismaEnum:
    name: str
    values: Tuple[str, ...]


@dataclass
class PrismaSchema:
    models: Dict[str, PrismaModel] = field(default_factory=dict)
    enums: Dict[str, PrismaEnum] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Built-in Prisma scalar types we know how to map.
_SCALAR_TYPES = {
    "String",
    "Int",
    "BigInt",
    "Float",
    "Decimal",
    "Boolean",
    "DateTime",
    "Json",
    "Bytes",
}


_MODEL_RE = re.compile(r"^\s*model\s+(\w+)\s*\{\s*$")
_ENUM_RE = re.compile(r"^\s*enum\s+(\w+)\s*\{\s*$")
_DATASOURCE_RE = re.compile(r"^\s*(datasource|generator)\s+\w+\s*\{\s*$")
_TABLE_ATTR_RE = re.compile(r"^\s*@@(\w+)\s*\((.*)\)\s*$")
_TABLE_MAP_RE = re.compile(r"^\s*@@map\s*\(\s*\"([^\"]+)\"\s*\)\s*$")


def _strip_comment(line: str) -> str:
    """Remove a trailing ``// ...`` comment, ignoring `//` inside quotes."""
    out: List[str] = []
    in_str = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
            out.append(ch)
            i += 1
            continue
        if not in_str and ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _split_top_level_commas(s: str) -> List[str]:
    """Split a parenthesized argument list on top-level commas only."""
    parts: List[str] = []
    depth = 0
    in_str = False
    buf: List[str] = []
    for ch in s:
        if ch == '"':
            in_str = not in_str
            buf.append(ch)
        elif in_str:
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_attributes(rest: str) -> List[str]:
    """Extract ``@foo(...)`` / ``@foo`` attribute substrings from a field tail."""
    attrs: List[str] = []
    i = 0
    while i < len(rest):
        if rest[i] == "@":
            j = i + 1
            while j < len(rest) and (rest[j].isalnum() or rest[j] in "._"):
                j += 1
            if j < len(rest) and rest[j] == "(":
                depth = 1
                k = j + 1
                in_str = False
                while k < len(rest) and depth > 0:
                    ch = rest[k]
                    if ch == '"' and rest[k - 1] != "\\":
                        in_str = not in_str
                    elif not in_str:
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                    k += 1
                attrs.append(rest[i:k])
                i = k
                continue
            attrs.append(rest[i:j])
            i = j
            continue
        i += 1
    return attrs


def _parse_default_value(attr: str) -> Optional[str]:
    m = re.match(r"^@default\((.*)\)$", attr)
    if not m:
        return None
    return m.group(1).strip()


def _parse_map_value(attr: str) -> Optional[str]:
    m = re.match(r"^@map\(\s*\"([^\"]+)\"\s*\)$", attr)
    if not m:
        return None
    return m.group(1)


def _parse_field_line(line: str) -> Optional[Any]:
    """Parse a single field line inside a model block.

    Returns either a ``PrismaField``, a ``PrismaRelation``, or ``None`` if the
    line is blank/comment-only.
    """
    stripped = _strip_comment(line).strip()
    if not stripped:
        return None
    if stripped.startswith("@@"):
        return None  # handled separately

    parts = stripped.split(None, 2)
    if len(parts) < 2:
        return None
    name = parts[0]
    type_token = parts[1]
    rest = parts[2] if len(parts) == 3 else ""

    is_list = type_token.endswith("[]")
    if is_list:
        base = type_token[:-2]
        is_optional = False
    elif type_token.endswith("?"):
        base = type_token[:-1]
        is_optional = True
    else:
        base = type_token
        is_optional = False

    attributes = _extract_attributes(rest)

    is_relation = base not in _SCALAR_TYPES and any(
        a.startswith("@relation") for a in attributes
    )
    is_relation = is_relation or (
        base not in _SCALAR_TYPES and is_list  # `Foo[]` back-reference
    )

    if is_relation:
        return PrismaRelation(
            name=name,
            target_model=base,
            is_optional=is_optional,
            is_list=is_list,
            relation_attributes=attributes,
        )

    column_name = name
    has_default = False
    default_raw: Optional[str] = None
    has_updated_at = False
    is_id = False
    is_unique = False

    for attr in attributes:
        if attr == "@id":
            is_id = True
        elif attr == "@unique":
            is_unique = True
        elif attr == "@updatedAt":
            has_updated_at = True
        elif attr.startswith("@default("):
            has_default = True
            default_raw = _parse_default_value(attr)
        elif attr.startswith("@map("):
            mapped = _parse_map_value(attr)
            if mapped is not None:
                column_name = mapped

    return PrismaField(
        name=name,
        column_name=column_name,
        base_type=base,
        is_optional=is_optional,
        is_list=is_list,
        is_id=is_id,
        is_unique=is_unique,
        has_default=has_default,
        default_raw=default_raw,
        has_updated_at=has_updated_at,
        attributes=attributes,
    )


def _parse_field_list(arg: str) -> Tuple[str, ...]:
    """Parse the field list inside ``@@id([...])`` / ``@@index([...])``.

    Field expressions like ``checked_at(sort: Desc)`` are reduced to the bare
    field name, which is what we need for parity (SQLAlchemy index objects
    don't capture sort direction in the simple comparison we do).
    """
    m = re.match(r"^\s*\[(.*)\]\s*(?:,\s*map\s*:\s*\"([^\"]+)\")?\s*$", arg)
    if not m:
        return ()
    inner = m.group(1)
    pieces = _split_top_level_commas(inner)
    out: List[str] = []
    for p in pieces:
        # strip ``(sort: Desc)`` etc.
        bare = re.sub(r"\(.*\)", "", p).strip()
        if bare:
            out.append(bare)
    return tuple(out)


def _parse_index_attr(arg: str) -> PrismaIndex:
    map_name = None
    m = re.search(r"map\s*:\s*\"([^\"]+)\"", arg)
    if m:
        map_name = m.group(1)
    fields = _parse_field_list(arg)
    return PrismaIndex(fields=fields, map_name=map_name)


def parse_schema(text: str) -> PrismaSchema:
    """Parse a ``schema.prisma`` source string."""
    schema = PrismaSchema()
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = _strip_comment(lines[i])
        m_model = _MODEL_RE.match(line)
        m_enum = _ENUM_RE.match(line)
        m_ds = _DATASOURCE_RE.match(line)
        if m_ds:
            i = _skip_block(lines, i)
            continue
        if m_enum:
            name = m_enum.group(1)
            values, i = _consume_enum(lines, i + 1)
            schema.enums[name] = PrismaEnum(name=name, values=values)
            continue
        if m_model:
            name = m_model.group(1)
            model, i = _consume_model(lines, i + 1, name)
            schema.models[name] = model
            continue
        i += 1
    return schema


def parse_schema_file(path: Path) -> PrismaSchema:
    return parse_schema(Path(path).read_text())


def _skip_block(lines: List[str], i: int) -> int:
    """Skip a balanced ``{ ... }`` block starting at ``lines[i]``."""
    depth = 0
    while i < len(lines):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return i


def _consume_enum(lines: List[str], i: int) -> Tuple[Tuple[str, ...], int]:
    values: List[str] = []
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if stripped == "}":
            return tuple(values), i + 1
        if stripped:
            # one identifier per line
            tok = stripped.split()[0]
            values.append(tok)
        i += 1
    return tuple(values), i


def _consume_model(
    lines: List[str], i: int, model_name: str
) -> Tuple[PrismaModel, int]:
    model = PrismaModel(name=model_name, table_name=model_name)
    while i < len(lines):
        raw = lines[i]
        stripped_no_comment = _strip_comment(raw).strip()
        if stripped_no_comment == "}":
            i += 1
            break
        if not stripped_no_comment:
            i += 1
            continue

        # @@map / @@id / @@unique / @@index / other table-level attrs
        m_map = _TABLE_MAP_RE.match(raw)
        if m_map:
            model.table_name = m_map.group(1)
            i += 1
            continue
        m_attr = _TABLE_ATTR_RE.match(raw)
        if m_attr:
            kind = m_attr.group(1)
            arg = m_attr.group(2).strip()
            model.raw_attributes.append(stripped_no_comment)
            if kind == "id":
                model.primary_key = _parse_field_list(arg)
            elif kind == "unique":
                model.uniques.append(PrismaUnique(fields=_parse_field_list(arg)))
            elif kind == "index":
                model.indexes.append(_parse_index_attr(arg))
            i += 1
            continue

        parsed = _parse_field_line(raw)
        if parsed is None:
            i += 1
            continue
        if isinstance(parsed, PrismaField):
            model.fields.append(parsed)
            if parsed.is_id and not model.primary_key:
                model.primary_key = (parsed.name,)
        elif isinstance(parsed, PrismaRelation):
            model.relations.append(parsed)
        i += 1
    return model, i


# ---------------------------------------------------------------------------
# Convenience helpers used by the parity test
# ---------------------------------------------------------------------------


def column_specs_for(model: PrismaModel) -> Dict[str, Dict[str, Any]]:
    """Return a normalized ``{column_name: spec}`` for parity comparison."""
    specs: Dict[str, Dict[str, Any]] = {}
    for f in model.fields:
        specs[f.column_name] = {
            "field_name": f.name,
            "base_type": f.base_type,
            "is_optional": f.is_optional,
            "is_list": f.is_list,
            "is_id": f.is_id,
            "is_unique": f.is_unique,
            "has_default": f.has_default,
            "has_updated_at": f.has_updated_at,
        }
    return specs
