"""Generator: emit SQLModel class definitions from ``schema.prisma``.

This is a developer tool, not runtime code. Re-run after schema changes
(or rely on the parity test to flag drift) and copy the output into
:mod:`litellm.proxy.db.sqlmodel.models`. The output is plain Python that
should be reviewed and committed by hand -- this generator is here for
correctness, not for automatic codegen at import time.

Usage::

    uv run python -m litellm.proxy.db.sqlmodel._generate \\
        --schema schema.prisma \\
        --out litellm/proxy/db/sqlmodel/models.py

The generated file is structurally equivalent to ``schema.prisma`` (every
model becomes a SQLModel class with one column per scalar field, plus
table-level constraints and indexes). It does **not** model relations -
those will be added by hand in subsequent migration phases as needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from litellm.proxy.db.sqlmodel.schema_parser import (
    PrismaEnum,
    PrismaField,
    PrismaModel,
    PrismaSchema,
    parse_schema_file,
)

# Map Prisma scalar -> (python annotation, SQLAlchemy column type expression).
# We deliberately use SQLAlchemy types (not SQLModel sugar) to match what the
# Prisma migrations have shipped historically: BigInt -> BigInteger,
# String -> Text (Prisma's default for `String` is unbounded text on Postgres),
# Json -> JSONB, etc.
_SCALAR_PY_TYPE = {
    "String": "str",
    "Int": "int",
    "BigInt": "int",
    "Float": "float",
    "Decimal": "Decimal",
    "Boolean": "bool",
    "DateTime": "datetime",
    "Json": "Any",
    "Bytes": "bytes",
}

_SCALAR_SA_TYPE = {
    "String": "Text()",
    "Int": "Integer()",
    "BigInt": "BigInteger()",
    "Float": "Double()",
    "Decimal": "Numeric()",
    "Boolean": "Boolean()",
    "DateTime": "DateTime(timezone=True)",
    "Json": "JSONB()",
    "Bytes": "LargeBinary()",
}

# Default expressions for typical Prisma defaults. Returned strings are
# Python source that produces a SQLAlchemy ``Column(... default=..., server_default=...)``
# argument value. We bias toward server defaults for ``now()``, scalar defaults
# for booleans/numbers, and Python factories for ``uuid()``/``cuid()`` (so the
# value is set at INSERT time, matching prisma-client-py behavior).


def _python_default_for(default_raw: str, base_type: str) -> Optional[str]:
    """Return Python ``default=`` argument source, or ``None``."""
    raw = default_raw.strip()
    if raw == "uuid()":
        return "default_factory=lambda: str(__import__('uuid').uuid4())"
    if raw == "cuid()":
        # cuid is roughly equivalent to uuid for our purposes; the only
        # current user is ``LiteLLM_CronJob.cronjob_id`` and downstream
        # consumers treat it as an opaque string.
        return "default_factory=lambda: str(__import__('uuid').uuid4())"
    if raw == "now()":
        return "default_factory=lambda: __import__('datetime').datetime.utcnow()"
    if raw == "true":
        return "default=True"
    if raw == "false":
        return "default=False"
    if raw == "[]":
        return "default_factory=list"
    if raw == '"{}"':
        return "default_factory=dict"
    if raw == '"[]"':
        return "default_factory=list"
    # Quoted string literal
    if raw.startswith('"') and raw.endswith('"'):
        return f"default={raw}"
    # Numeric literal
    try:
        float(raw)
        return f"default={raw}"
    except ValueError:
        pass
    # Enum reference (e.g. JobStatus value: INACTIVE)
    if (
        raw.replace("_", "").isalnum()
        and raw[:1].isalpha()
        and base_type not in _SCALAR_PY_TYPE
    ):
        # we can't reference the Python enum here without an import wrangle,
        # so fall back to a string default.
        return f'default="{raw}"'
    return None


def _server_default_for(default_raw: str, base_type: str) -> Optional[str]:
    """Optional ``server_default=`` to match the existing Postgres DDL."""
    raw = default_raw.strip()
    if raw == "now()":
        return "server_default=text('CURRENT_TIMESTAMP')"
    if base_type == "Json":
        if raw == '"{}"':
            return "server_default=text(\"'{}'\")"
        if raw == '"[]"':
            return "server_default=text(\"'[]'\")"
    return None


def _sa_type_for(field: PrismaField, schema: PrismaSchema) -> str:
    """SQLAlchemy column type expression for a Prisma field."""
    if field.is_list:
        inner = _SCALAR_SA_TYPE.get(field.base_type, "Text()")
        return f"ARRAY({inner})"
    if field.base_type in _SCALAR_SA_TYPE:
        return _SCALAR_SA_TYPE[field.base_type]
    if field.base_type in schema.enums:
        # Use a plain Text column; we already index/filter these as strings
        # everywhere in production and Prisma's enum type is mostly a
        # client-side affair. (Subsequent phases can introduce a real
        # ``sa.Enum`` if the call sites benefit from it.)
        return "Text()"
    return "Text()"


def _py_type_for(field: PrismaField, schema: PrismaSchema) -> str:
    if field.base_type in _SCALAR_PY_TYPE:
        py = _SCALAR_PY_TYPE[field.base_type]
    elif field.base_type in schema.enums:
        py = "str"
    else:
        py = "str"
    if field.is_list:
        py = f"List[{py}]"
    if field.is_optional:
        py = f"Optional[{py}]"
    return py


# Names SQLAlchemy's Declarative API reserves on a mapped class.
# When a Prisma column collides with one of these we emit the Python attribute
# with a trailing underscore but keep the on-disk column name unchanged via
# ``sa_column_kwargs={'name': '...'}``.
_RESERVED_PY_ATTRS: Set[str] = {"metadata", "registry"}


def _format_field(field: PrismaField, schema: PrismaSchema) -> str:
    """Render one ``Foo: <type> = Field(...)`` line for a SQLModel class."""
    py_type = _py_type_for(field, schema)
    sa_type = _sa_type_for(field, schema)

    field_kwargs: List[str] = [f"sa_type={sa_type}"]
    sa_column_kwargs: List[str] = []

    py_attr_name = field.name
    if field.name in _RESERVED_PY_ATTRS:
        py_attr_name = f"{field.name}_"

    if field.column_name != py_attr_name:
        sa_column_kwargs.append(f"'name': {field.column_name!r}")
    if field.is_id:
        field_kwargs.append("primary_key=True")
    if field.is_unique and not field.is_id:
        field_kwargs.append("unique=True")

    py_default: Optional[str] = None
    srv_default: Optional[str] = None
    if field.has_default and field.default_raw is not None:
        py_default = _python_default_for(field.default_raw, field.base_type)
        srv_default = _server_default_for(field.default_raw, field.base_type)

    if py_default is not None:
        field_kwargs.append(py_default)
    elif field.is_optional:
        field_kwargs.append("default=None")
    elif field.is_list:
        field_kwargs.append("default_factory=list")

    if srv_default is not None:
        # ``server_default`` lives on the SA column, not on the SQLModel Field.
        # _server_default_for returns ``server_default=text('...')``; rip the
        # value off and stuff it into sa_column_kwargs so SQLModel forwards it.
        value = srv_default.split("=", 1)[1]
        sa_column_kwargs.append(f"'server_default': {value}")

    if field.has_updated_at:
        sa_column_kwargs.append(
            "'onupdate': lambda: __import__('datetime').datetime.utcnow()"
        )

    if sa_column_kwargs:
        joined = ", ".join(sa_column_kwargs)
        field_kwargs.append(f"sa_column_kwargs={{{joined}}}")

    field_args = ", ".join(field_kwargs)
    return f"    {py_attr_name}: {py_type} = Field({field_args})"


def _format_index_args(model: PrismaModel) -> List[str]:
    args: List[str] = []
    composite_pk: Tuple[str, ...] = (
        model.primary_key if len(model.primary_key) > 1 else ()
    )
    if composite_pk:
        cols = ", ".join(repr(c) for c in composite_pk)
        args.append(f"PrimaryKeyConstraint({cols})")
    for u in model.uniques:
        cols = ", ".join(repr(c) for c in u.fields)
        args.append(f"UniqueConstraint({cols})")
    for idx in model.indexes:
        cols = ", ".join(repr(c) for c in idx.fields)
        if idx.map_name:
            args.append(f"Index({idx.map_name!r}, {cols})")
        else:
            # Default index name: <table>_<col>_<col>_idx (matches the
            # convention Prisma generates so existing DBs stay happy).
            default_name = f"{model.table_name}_" + "_".join(idx.fields) + "_idx"
            args.append(f"Index({default_name!r}, {cols})")
    return args


def _model_class_name(model: PrismaModel) -> str:
    """Map ``LiteLLM_FooTable`` -> ``LiteLLMFooTable`` (CamelCase, no underscores)."""
    parts = model.name.split("_")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _render_model_class(model: PrismaModel, schema: PrismaSchema) -> str:
    cls_name = _model_class_name(model)
    lines: List[str] = []
    lines.append(f"class {cls_name}(SQLModel, table=True):")
    lines.append(f"    __tablename__ = {model.table_name!r}")
    index_args = _format_index_args(model)
    if index_args:
        if len(index_args) == 1:
            lines.append(f"    __table_args__ = ({index_args[0]},)")
        else:
            lines.append("    __table_args__ = (")
            for arg in index_args:
                lines.append(f"        {arg},")
            lines.append("    )")
    lines.append("")
    for field in model.fields:
        lines.append(_format_field(field, schema))
    lines.append("")
    return "\n".join(lines)


def _render_enum(enum: PrismaEnum) -> str:
    lines = [f"class {enum.name}(str, Enum):"]
    for v in enum.values:
        lines.append(f"    {v} = {v!r}")
    lines.append("")
    return "\n".join(lines)


_DOCSTRING = '''"""SQLModel ORM definitions mirroring ``schema.prisma``.

THIS FILE IS GENERATED by ``litellm.proxy.db.sqlmodel._generate`` but is
CHECKED IN as ordinary Python source. Hand-edits are allowed -- the parity
test in ``tests/test_litellm/proxy/db/sqlmodel_orm/test_parity.py`` will
fail CI if structural drift from ``schema.prisma`` is introduced (in
either direction).

Re-generate with::

    uv run python -m litellm.proxy.db.sqlmodel._generate \\
        --schema schema.prisma \\
        --out litellm/proxy/db/sqlmodel/models.py

Phase 1 of the Prisma -> SQLModel migration only ships these definitions;
nothing in the runtime proxy currently imports them. See the package
README for the multi-phase plan.
"""'''


_FOOTER_TEMPLATE = """

ALL_MODELS: List[Type[SQLModel]] = [
{model_lines}
]
"""


# Map Prisma scalar -> (SA import name, sets has_jsonb, sets has_datetime, sets has_decimal, sets has_any)
_SA_IMPORT_FOR_BASE = {
    "String": "Text",
    "Int": "Integer",
    "BigInt": "BigInteger",
    "Float": "Double",
    "Decimal": "Numeric",
    "Boolean": "Boolean",
    "DateTime": "DateTime",
    "Json": "JSONB",
    "Bytes": "LargeBinary",
}


def _classify_field(
    field: PrismaField, schema: PrismaSchema, flags: Dict[str, bool]
) -> Optional[str]:
    """Return the SQLAlchemy import name needed for ``field`` and update ``flags``."""
    base = field.base_type
    if base in schema.enums:
        return "Text"
    sa_name = _SA_IMPORT_FOR_BASE.get(base, "Text")
    if base == "Json":
        flags["jsonb"] = True
        flags["any"] = True
    elif base == "DateTime":
        flags["datetime"] = True
    elif base == "Decimal":
        flags["decimal"] = True
    return sa_name


def _gather_features(schema: PrismaSchema) -> Tuple[Set[str], Dict[str, bool]]:
    """Walk the schema once and return (sqlalchemy import names, feature flags)."""
    sa_imports: Set[str] = set()
    flags: Dict[str, bool] = {
        "optional": False,
        "any": False,
        "datetime": False,
        "decimal": False,
        "enum_class": bool(schema.enums),
        "indexes": False,
        "uniques": False,
        "composite_pk": False,
        "text_default": False,
        "array": False,
        "jsonb": False,
    }
    for model in schema.models.values():
        if len(model.primary_key) > 1:
            flags["composite_pk"] = True
        if model.uniques:
            flags["uniques"] = True
        if model.indexes:
            flags["indexes"] = True
        for f in model.fields:
            if f.is_optional:
                flags["optional"] = True
            if f.is_list:
                flags["array"] = True
            sa = _classify_field(f, schema, flags)
            if sa:
                sa_imports.add(sa)
            if (
                f.has_default
                and f.default_raw is not None
                and _server_default_for(f.default_raw, f.base_type) is not None
            ):
                flags["text_default"] = True
    return sa_imports, flags


def _collect_used_symbols(schema: PrismaSchema) -> Set[str]:
    """Return a sentinel-encoded set describing imports needed by the output."""
    sa_imports, flags = _gather_features(schema)

    if flags["indexes"]:
        sa_imports.add("Index")
    if flags["composite_pk"]:
        sa_imports.add("PrimaryKeyConstraint")
    if flags["uniques"]:
        sa_imports.add("UniqueConstraint")
    if flags["text_default"]:
        sa_imports.add("text")

    pg_imports: List[str] = []
    if flags["array"]:
        pg_imports.append("ARRAY")
    if flags["jsonb"]:
        pg_imports.append("JSONB")

    typing_imports: List[str] = ["List", "Type"]
    if flags["any"]:
        typing_imports.append("Any")
    if flags["optional"]:
        typing_imports.append("Optional")

    stdlib_lines: List[str] = []
    if flags["datetime"]:
        stdlib_lines.append("from datetime import datetime")
    if flags["decimal"]:
        stdlib_lines.append("from decimal import Decimal")
    if flags["enum_class"]:
        stdlib_lines.append("from enum import Enum")

    used: Set[str] = set(sa_imports)
    used.update(f"_pg::{name}" for name in pg_imports)
    used.update(f"_typing::{name}" for name in sorted(set(typing_imports)))
    used.update(f"_stdlib::{line}" for line in stdlib_lines)
    return used


def _render_imports(schema: PrismaSchema) -> str:
    used = _collect_used_symbols(schema)
    sa = sorted(
        s
        for s in used
        if not s.startswith("_")
        and s
        in {
            "BigInteger",
            "Boolean",
            "DateTime",
            "Double",
            "Index",
            "Integer",
            "LargeBinary",
            "Numeric",
            "PrimaryKeyConstraint",
            "Text",
            "UniqueConstraint",
            "text",
        }
    )
    pg = sorted(s.split("::", 1)[1] for s in used if s.startswith("_pg::"))
    typing = sorted(s.split("::", 1)[1] for s in used if s.startswith("_typing::"))
    stdlib = sorted(s.split("::", 1)[1] for s in used if s.startswith("_stdlib::"))

    lines: List[str] = ["from __future__ import annotations", ""]
    lines.extend(stdlib)
    if stdlib:
        lines.append("")
    lines.append(f"from typing import {', '.join(typing)}")
    lines.append("")
    if sa:
        if len(sa) == 1:
            lines.append(f"from sqlalchemy import {sa[0]}")
        else:
            lines.append("from sqlalchemy import (")
            for s in sa:
                lines.append(f"    {s},")
            lines.append(")")
    if pg:
        lines.append(f"from sqlalchemy.dialects.postgresql import {', '.join(pg)}")
    lines.append("from sqlmodel import Field, SQLModel")
    return "\n".join(lines)


def _format_with_black(src: str) -> str:
    """Run Black over ``src`` so generator output matches the committed style.

    Black is already a hard CI requirement for this repo (see ``CLAUDE.md``),
    so we lean on it as the canonical formatter rather than carrying our own
    line-wrapping logic. Falls back to the unformatted source if Black is
    unavailable -- the parity test will catch the resulting drift.
    """
    try:
        import black  # type: ignore[import-not-found]
    except ImportError:
        return src
    mode = black.Mode(line_length=88)
    try:
        return black.format_str(src, mode=mode)
    except black.InvalidInput:
        return src


def render_module(schema: PrismaSchema) -> str:
    """Render the entire ``models.py`` source for the given schema."""
    out: List[str] = [_DOCSTRING, "", _render_imports(schema), ""]

    if schema.enums:
        for name in sorted(schema.enums):
            out.append(_render_enum(schema.enums[name]))

    for name in sorted(schema.models):
        out.append(_render_model_class(schema.models[name], schema))

    model_lines = ",\n".join(
        f"    {_model_class_name(schema.models[n])}" for n in sorted(schema.models)
    )
    out.append(_FOOTER_TEMPLATE.format(model_lines=model_lines))
    return _format_with_black("\n".join(out))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schema", type=Path, default=Path("schema.prisma"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    schema = parse_schema_file(args.schema)
    src = render_module(schema)
    args.out.write_text(src)
    sys.stdout.write(
        f"wrote {args.out} ({len(schema.models)} models, "
        f"{len(schema.enums)} enums)\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
