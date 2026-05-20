"""Parity test: SQLModel definitions must match ``schema.prisma``.

If this test fails, either:

* ``schema.prisma`` was changed and ``litellm/proxy/db/sqlmodel/models.py``
  was not regenerated, OR
* ``models.py`` was hand-edited in a way that no longer reflects the Prisma
  schema (which is still the source of truth during the migration).

Re-run the generator and commit the diff::

    uv run python -m litellm.proxy.db.sqlmodel._generate \\
        --schema schema.prisma \\
        --out litellm/proxy/db/sqlmodel/models.py

The test only enforces structural parity that matters for behavioural
equivalence at the database layer:

* every Prisma model has exactly one SQLModel class,
* every scalar Prisma field has a column with the same on-disk name and
  nullability,
* primary keys, ``@@unique`` and ``@@index`` clauses match,
* table names (``@@map``) match.

It deliberately does *not* check Python attribute names, type granularity
beyond the broad SQL category, default values, or relation back-refs --
those are implementation details of the SQLModel layer that may diverge
once we hand-tune for SQLAlchemy idioms in later phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Tuple

import pytest
from sqlalchemy import Index, PrimaryKeyConstraint, Table, UniqueConstraint

from litellm.proxy.db.sqlmodel.models import ALL_MODELS
from litellm.proxy.db.sqlmodel.schema_parser import (
    PrismaModel,
    PrismaSchema,
    parse_schema_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    while not (p / "schema.prisma").exists():
        if p.parent == p:
            raise RuntimeError("schema.prisma not found in any ancestor directory")
        p = p.parent
    return p


@pytest.fixture(scope="module")
def prisma_schema() -> PrismaSchema:
    return parse_schema_file(_find_repo_root() / "schema.prisma")


@pytest.fixture(scope="module")
def sqlmodel_tables() -> Dict[str, Table]:
    return {cls.__tablename__: cls.__table__ for cls in ALL_MODELS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_signatures(table: Table) -> Set[Tuple[str, ...]]:
    """Set of ``(col1, col2, ...)`` tuples from non-unique SQLAlchemy indexes."""
    sigs: Set[Tuple[str, ...]] = set()
    for ix in table.indexes:
        if ix.unique:
            continue
        sigs.add(tuple(c.name for c in ix.columns))
    return sigs


def _unique_signatures(table: Table) -> Set[Tuple[str, ...]]:
    sigs: Set[Tuple[str, ...]] = set()
    for cons in table.constraints:
        if isinstance(cons, UniqueConstraint):
            sigs.add(tuple(c.name for c in cons.columns))
    for col in table.columns:
        if col.unique and not col.primary_key:
            sigs.add((col.name,))
    return sigs


def _pk_signature(table: Table) -> Tuple[str, ...]:
    return tuple(c.name for c in table.primary_key.columns)


def _prisma_pk_columns(model: PrismaModel) -> Tuple[str, ...]:
    """Map field-name PK to column-name PK (respects ``@map``)."""
    cols: list[str] = []
    for fname in model.primary_key:
        f = model.field_by_name(fname)
        cols.append(f.column_name if f is not None else fname)
    return tuple(cols)


def _prisma_unique_signatures(model: PrismaModel) -> Set[Tuple[str, ...]]:
    sigs: Set[Tuple[str, ...]] = set()
    for u in model.uniques:
        sigs.add(tuple(_field_to_column(model, fn) for fn in u.fields))
    for f in model.fields:
        if f.is_unique and not f.is_id:
            sigs.add((f.column_name,))
    return sigs


def _prisma_index_signatures(model: PrismaModel) -> Set[Tuple[str, ...]]:
    sigs: Set[Tuple[str, ...]] = set()
    for idx in model.indexes:
        sigs.add(tuple(_field_to_column(model, fn) for fn in idx.fields))
    return sigs


def _field_to_column(model: PrismaModel, fname: str) -> str:
    f = model.field_by_name(fname)
    return f.column_name if f is not None else fname


# Prisma scalar -> coarse SQL category we expect on the generated column.
_EXPECTED_TYPE_CATEGORIES = {
    "String": {"text", "varchar"},
    "Int": {"integer"},
    "BigInt": {"biginteger", "bigint"},
    "Float": {"double", "double_precision", "float"},
    "Decimal": {"numeric", "decimal"},
    "Boolean": {"boolean"},
    "DateTime": {"datetime", "timestamp"},
    "Json": {"json", "jsonb"},
    "Bytes": {"largebinary", "bytea"},
}


def _column_type_category(col) -> str:
    return type(col.type).__name__.lower()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_one_sqlmodel_class_per_prisma_model(prisma_schema, sqlmodel_tables):
    prisma_table_names = {m.table_name for m in prisma_schema.models.values()}
    sqlmodel_table_names = set(sqlmodel_tables)
    missing_in_sqlmodel = prisma_table_names - sqlmodel_table_names
    extra_in_sqlmodel = sqlmodel_table_names - prisma_table_names
    assert not missing_in_sqlmodel, (
        f"Prisma tables with no SQLModel class: {sorted(missing_in_sqlmodel)}. "
        "Did you forget to regenerate models.py?"
    )
    assert not extra_in_sqlmodel, (
        f"SQLModel classes with no Prisma model: {sorted(extra_in_sqlmodel)}. "
        "Did you forget to update schema.prisma?"
    )


def test_columns_match_for_every_table(prisma_schema, sqlmodel_tables):
    failures: list[str] = []
    for prisma_model in prisma_schema.models.values():
        table = sqlmodel_tables[prisma_model.table_name]
        prisma_cols = {f.column_name: f for f in prisma_model.fields}
        sqlmodel_cols = {c.name: c for c in table.columns}

        missing = set(prisma_cols) - set(sqlmodel_cols)
        extra = set(sqlmodel_cols) - set(prisma_cols)
        if missing:
            failures.append(
                f"{prisma_model.table_name}: missing columns in SQLModel: {sorted(missing)}"
            )
        if extra:
            failures.append(
                f"{prisma_model.table_name}: unexpected columns in SQLModel: {sorted(extra)}"
            )
    assert not failures, "\n".join(failures)


def test_column_nullability_matches(prisma_schema, sqlmodel_tables):
    failures: list[str] = []
    for prisma_model in prisma_schema.models.values():
        table = sqlmodel_tables[prisma_model.table_name]
        sqlmodel_cols = {c.name: c for c in table.columns}
        for f in prisma_model.fields:
            col = sqlmodel_cols.get(f.column_name)
            if col is None:
                continue
            expected_nullable = f.is_optional
            if col.nullable != expected_nullable:
                failures.append(
                    f"{prisma_model.table_name}.{f.column_name}: "
                    f"prisma optional={f.is_optional} but SQLModel nullable={col.nullable}"
                )
    assert not failures, "\n".join(failures)


def test_column_types_in_expected_category(prisma_schema, sqlmodel_tables):
    """Coarse type check: e.g. ``BigInt`` -> a BigInteger-class type, not Integer.

    We deliberately do not enforce exact ``server_default`` or precision -- those
    are implementation details that can drift without behavioural impact, and
    they are guarded separately by the migration tests.
    """
    failures: list[str] = []
    for prisma_model in prisma_schema.models.values():
        table = sqlmodel_tables[prisma_model.table_name]
        sqlmodel_cols = {c.name: c for c in table.columns}
        for f in prisma_model.fields:
            col = sqlmodel_cols.get(f.column_name)
            if col is None:
                continue
            expected = _EXPECTED_TYPE_CATEGORIES.get(f.base_type)
            if expected is None:
                # enum reference or unknown scalar -> skip
                continue
            actual_kind = _column_type_category(col)
            ok = any(token in actual_kind for token in expected)
            # ARRAY columns wrap an inner type; check the item type instead.
            if not ok and "array" in actual_kind and f.is_list:
                inner = type(col.type.item_type).__name__.lower()
                ok = any(token in inner for token in expected)
            if not ok:
                failures.append(
                    f"{prisma_model.table_name}.{f.column_name}: "
                    f"prisma type={f.base_type}{'[]' if f.is_list else ''} "
                    f"but SQLModel column type is {actual_kind}"
                )
    assert not failures, "\n".join(failures)


def test_array_columns_match(prisma_schema, sqlmodel_tables):
    failures: list[str] = []
    for prisma_model in prisma_schema.models.values():
        table = sqlmodel_tables[prisma_model.table_name]
        sqlmodel_cols = {c.name: c for c in table.columns}
        for f in prisma_model.fields:
            col = sqlmodel_cols.get(f.column_name)
            if col is None:
                continue
            actual_is_array = "array" in type(col.type).__name__.lower()
            if f.is_list != actual_is_array:
                failures.append(
                    f"{prisma_model.table_name}.{f.column_name}: "
                    f"prisma is_list={f.is_list} but SQLModel ARRAY={actual_is_array}"
                )
    assert not failures, "\n".join(failures)


def test_primary_keys_match(prisma_schema, sqlmodel_tables):
    failures: list[str] = []
    for prisma_model in prisma_schema.models.values():
        table = sqlmodel_tables[prisma_model.table_name]
        prisma_pk = _prisma_pk_columns(prisma_model)
        sqlmodel_pk = _pk_signature(table)
        if set(prisma_pk) != set(sqlmodel_pk):
            failures.append(
                f"{prisma_model.table_name}: prisma PK={prisma_pk} but SQLModel PK={sqlmodel_pk}"
            )
    assert not failures, "\n".join(failures)


def test_unique_constraints_match(prisma_schema, sqlmodel_tables):
    failures: list[str] = []
    for prisma_model in prisma_schema.models.values():
        table = sqlmodel_tables[prisma_model.table_name]
        prisma_uniques = _prisma_unique_signatures(prisma_model)
        sqlmodel_uniques = _unique_signatures(table)
        # Set comparison ignores ordering of the unique-constraint columns,
        # which matches what Postgres treats as logically equivalent.
        prisma_norm = {tuple(sorted(s)) for s in prisma_uniques}
        sqlmodel_norm = {tuple(sorted(s)) for s in sqlmodel_uniques}
        missing = prisma_norm - sqlmodel_norm
        extra = sqlmodel_norm - prisma_norm
        if missing:
            failures.append(
                f"{prisma_model.table_name}: missing unique constraints in SQLModel: {sorted(missing)}"
            )
        if extra:
            failures.append(
                f"{prisma_model.table_name}: unexpected unique constraints in SQLModel: {sorted(extra)}"
            )
    assert not failures, "\n".join(failures)


def test_indexes_match(prisma_schema, sqlmodel_tables):
    failures: list[str] = []
    for prisma_model in prisma_schema.models.values():
        table = sqlmodel_tables[prisma_model.table_name]
        prisma_idx = _prisma_index_signatures(prisma_model)
        sqlmodel_idx = _index_signatures(table)
        # We compare ordered tuples here because index column order
        # affects which queries the index can serve.
        missing = prisma_idx - sqlmodel_idx
        extra = sqlmodel_idx - prisma_idx
        if missing:
            failures.append(
                f"{prisma_model.table_name}: missing indexes in SQLModel: {sorted(missing)}"
            )
        if extra:
            failures.append(
                f"{prisma_model.table_name}: unexpected indexes in SQLModel: {sorted(extra)}"
            )
    assert not failures, "\n".join(failures)


def test_generator_output_is_committed(tmp_path):
    """Re-run the generator and assert the result matches the checked-in file.

    This is the strongest guard: it catches any drift in either the schema
    or the generator (or hand-edits to ``models.py`` that don't roundtrip).
    """
    from litellm.proxy.db.sqlmodel import _generate

    schema = parse_schema_file(_find_repo_root() / "schema.prisma")
    expected = _generate.render_module(schema)
    actual = (
        _find_repo_root() / "litellm" / "proxy" / "db" / "sqlmodel" / "models.py"
    ).read_text()
    if expected != actual:
        # Surface a small diff so the failure message is actionable.
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                actual.splitlines(),
                expected.splitlines(),
                fromfile="models.py (committed)",
                tofile="models.py (regenerated)",
                lineterm="",
                n=3,
            )
        )
        pytest.fail(
            "litellm/proxy/db/sqlmodel/models.py is out of sync with "
            "schema.prisma. Run:\n"
            "  uv run python -m litellm.proxy.db.sqlmodel._generate "
            "--schema schema.prisma --out litellm/proxy/db/sqlmodel/models.py\n\n"
            f"Diff (truncated to first 60 lines):\n{chr(10).join(diff.splitlines()[:60])}"
        )
