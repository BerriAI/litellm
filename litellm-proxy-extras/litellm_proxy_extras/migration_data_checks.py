"""Read-only checks that the rows already in the database can take the pending schema
changes, run before `prisma migrate deploy` and before each request-log index build.

Postgres has a short list of schema statements that fail because of existing rows: a
btree index whose entry for some row would not fit in a third of a page (2704 bytes on
the default 8 KB page, counting every indexed column), UNIQUE (duplicate values), SET NOT NULL and ADD COLUMN NOT NULL with no default (nulls, any row at all),
FOREIGN KEY (orphans) and CHECK (rows failing the expression). Each pending statement
of one of those kinds becomes one count query derived from the statement text and the
column types in the catalog. A count above zero names the table, the column, the
statement it blocks and the SQL that clears it, and the migration is refused before
anything is applied. Every other statement a migration can contain cannot fail on
existing data and gets no query. A statement the derivation cannot read gets none
either, so the migration keeps today's behavior for it.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from litellm_proxy_extras._logging import logger

if TYPE_CHECKING:
    import psycopg
    from psycopg import sql

DEFAULT_BLOCK_SIZE: Final = 8192
INDEX_TUPLE_HEADER_BYTES: Final = 8
NULL_BITMAP_BYTES: Final = 8
VARLENA_HEADER_BYTES: Final = 4
SHORT_VARLENA_MAX_BYTES: Final = 126
_PAGE_HEADER_WITH_THREE_LINE_POINTERS_BYTES: Final = 40
_BTREE_PAGE_OPAQUE_BYTES: Final = 16
_HEAP_TID_BYTES: Final = 8
_ALIGNMENT: Final = MappingProxyType({"c": 1, "s": 2, "i": 4, "d": 8})
_OCTET_LENGTH_TYPES: Final = frozenset({"text", "varchar", "bpchar", "bytea", "name"})
MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR: Final = "LITELLM_MIGRATION_DATA_CHECK_TIMEOUT"
DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS: Final = 600.0
REFUSAL: Final = "Existing rows would make a pending schema change fail:"

Kind = Literal["too_long"]


@dataclass(frozen=True, slots=True)
class Check:
    kind: Kind
    table: str
    columns: tuple[str, ...]
    statement: str
    source: str
    index: str | None = None


@dataclass(frozen=True, slots=True)
class BlockedChange:
    check: Check
    rows: int
    detail: str


@dataclass(frozen=True, slots=True)
class CouldNotVerify:
    check: Check
    timeout_seconds: float


Finding = BlockedChange | CouldNotVerify


@dataclass(frozen=True, slots=True)
class ColumnType:
    """One column as `pg_attribute` and `pg_type` describe it: `length` is `attlen` (-1 for a
    variable-length value), `align` is `attalign` (c, s, i or d), `typmod` is `atttypmod`."""

    type_name: str
    length: int
    align: str
    typmod: int

    @property
    def is_variable_length(self) -> bool:
        return self.length == -1

    @property
    def can_overflow_btree(self) -> bool:
        """Whether one value of this column alone can push a btree entry past the limit: a
        variable-length type with no declared cap, or a cap of more bytes than the limit."""
        if not self.is_variable_length:
            return False
        return self.typmod == -1 or (self.typmod - VARLENA_HEADER_BYTES) * 4 > btree_max_entry_bytes(DEFAULT_BLOCK_SIZE)


ColumnTypeLookup = Callable[[str, str], ColumnType | None]


def migration_data_check_timeout_seconds() -> float:
    """Seconds one count query may run before the preflight gives up on verifying it."""
    raw: Final = os.getenv(MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number, using %s",
            MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR,
            raw,
            DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS,
        )
        return DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS


def checks_for_index(table: str, name: str, columns: tuple[str, ...]) -> tuple[Check, ...]:
    """The entry-size check a request-log index needs before `CREATE INDEX CONCURRENTLY`."""
    return (Check("too_long", table, columns, f'CREATE INDEX "{name}"', "request-log index", index=name),)


def _applied_migrations(connection: "psycopg.Connection[tuple[object, ...]]", schema: str) -> frozenset[str]:
    import psycopg
    from psycopg import sql

    try:
        rows: Final = connection.execute(
            sql.SQL("SELECT migration_name FROM {} WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL").format(
                sql.Identifier(schema, "_prisma_migrations")
            )
        ).fetchall()
    except psycopg.errors.UndefinedTable:
        return frozenset()
    return frozenset(str(row[0]) for row in rows)


def pending_migrations(
    connection: "psycopg.Connection[tuple[object, ...]]", schema: str, migrations_dir: Path
) -> tuple[tuple[str, str], ...]:
    """(name, sql) of every shipped migration the ledger has not finished, oldest first.
    Without a ledger every migration is pending, and the checks skip tables that do not exist yet."""
    applied: Final = _applied_migrations(connection, schema)
    folders: Final = tuple(folder for folder in sorted(migrations_dir.iterdir()) if folder.is_dir())
    return tuple(
        (folder.name, (folder / "migration.sql").read_text())
        for folder in folders
        if folder.name not in applied and (folder / "migration.sql").is_file()
    )


def _int(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def column_type_lookup(connection: "psycopg.Connection[tuple[object, ...]]", schema: str) -> ColumnTypeLookup:
    def lookup(table: str, column: str) -> ColumnType | None:
        row: Final = connection.execute(
            "SELECT t.typname, a.attlen, a.attalign, a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_type t ON t.oid = a.atttypid "
            "WHERE n.nspname = %s AND c.relname = %s AND a.attname = %s AND a.attnum > 0 AND NOT a.attisdropped",
            (schema, table, column),
        ).fetchone()
        if row is None:
            return None
        return ColumnType(str(row[0]), _int(row[1], -1), str(row[2]), _int(row[3], -1))

    return lookup


def _column_types(check: Check, column_type: ColumnTypeLookup) -> tuple[ColumnType, ...] | None:
    """The catalog types of the check's columns, None when one of them is not there yet."""
    types: Final = tuple(column_type(check.table, column) for column in check.columns)
    if any(column is None for column in types):
        return None
    return tuple(column for column in types if column is not None)


def _table_exists(connection: "psycopg.Connection[tuple[object, ...]]", schema: str, table: str) -> bool:
    row: Final = connection.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s", (schema, table)
    ).fetchone()
    return row is not None


def _index_is_valid(connection: "psycopg.Connection[tuple[object, ...]]", schema: str, index: str) -> bool:
    row: Final = connection.execute(
        "SELECT x.indisvalid FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid "
        "JOIN pg_namespace n ON n.oid = i.relnamespace WHERE n.nspname = %s AND i.relname = %s",
        (schema, index),
    ).fetchone()
    return row is not None and bool(row[0])


def btree_max_entry_bytes(block_size: int) -> int:
    """`BTMaxItemSize` from nbtree.h for btree version 4: a third of the page left after the page
    header, three line pointers and the btree opaque area, rounded down to 8, minus the heap TID a
    pivot tuple carries. 2704 on the default 8 KB page, the number in Postgres's error."""
    usable: Final = block_size - _PAGE_HEADER_WITH_THREE_LINE_POINTERS_BYTES - _BTREE_PAGE_OPAQUE_BYTES
    return (usable // 3) // 8 * 8 - _HEAP_TID_BYTES


def _value_bytes(column: ColumnType, identifier: "sql.Identifier") -> "sql.Composed":
    """The bytes a value takes in the index entry: its length, or its stored size when
    Postgres compressed it, since `index_form_tuple` keeps the compressed bytes."""
    from psycopg import sql

    length: Final = (
        sql.SQL("octet_length({})").format(identifier)
        if column.type_name in _OCTET_LENGTH_TYPES
        else sql.SQL("octet_length({}::text)").format(identifier)
    )
    return sql.SQL("least({}, pg_column_size({}))").format(length, identifier)


def _aligned(offset: "sql.Composable", align: str) -> "sql.Composed":
    from psycopg import sql

    unit: Final = _ALIGNMENT.get(align, 8)
    return sql.SQL("(({off} + {pad}) / {unit}) * {unit}").format(
        off=offset, pad=sql.Literal(unit - 1), unit=sql.Literal(unit)
    )


def _value_end(column: ColumnType, identifier: "sql.Identifier", offset: "sql.Composable") -> "sql.Composed":
    """Where this column's value ends when laid down from `offset` the way `index_form_tuple`
    does it: NULL takes no room, a fixed-length value is aligned to its type, a variable-length
    value of up to 126 bytes gets a 1-byte header and no alignment, a longer one a 4-byte header."""
    from psycopg import sql

    length: Final = _value_bytes(column, identifier)
    placed: Final = (
        sql.SQL("{} + {}").format(_aligned(offset, column.align), sql.Literal(column.length))
        if not column.is_variable_length
        else sql.SQL("CASE WHEN {len} <= {short} THEN {off} + 1 + {len} ELSE {aligned} + {hdr} + {len} END").format(
            len=length,
            short=sql.Literal(SHORT_VARLENA_MAX_BYTES),
            off=offset,
            aligned=_aligned(offset, column.align),
            hdr=sql.Literal(VARLENA_HEADER_BYTES),
        )
    )
    return sql.SQL("CASE WHEN {col} IS NULL THEN {off} ELSE {placed} END").format(
        col=identifier, off=offset, placed=placed
    )


def _entry_size_query(check: Check, schema: str, types: tuple[ColumnType, ...], limit_bytes: int) -> "sql.Composed":
    """count(*) and the largest entry over the rows whose btree entry would not fit, plus the
    longest value of each variable-length column: an 8-byte tuple header, 8 more for the null
    bitmap when a column is NULL, each value laid down in index order, the total rounded up to 8."""
    from psycopg import sql

    identifiers: Final = tuple(sql.Identifier(column) for column in check.columns)
    any_null: Final = sql.SQL(" OR ").join(sql.SQL("{} IS NULL").format(identifier) for identifier in identifiers)
    header: Final = sql.SQL("{} + CASE WHEN {} THEN {} ELSE 0 END").format(
        sql.Literal(INDEX_TUPLE_HEADER_BYTES), any_null, sql.Literal(NULL_BITMAP_BYTES)
    )
    innermost: Final = sql.SQL("SELECT {cols}, {hdr} AS off0 FROM {t}").format(
        cols=sql.SQL(", ").join(identifiers), hdr=header, t=sql.Identifier(schema, check.table)
    )

    def lay_down(inner: "sql.Composed", step: tuple[int, tuple[ColumnType, "sql.Identifier"]]) -> "sql.Composed":
        position, (column, identifier) = step
        return sql.SQL("SELECT *, {end} AS {next} FROM ({inner}) AS {alias}").format(
            end=_value_end(column, identifier, sql.Identifier(f"off{position}")),
            next=sql.Identifier(f"off{position + 1}"),
            inner=inner,
            alias=sql.Identifier(f"s{position}"),
        )

    laid_out: Final = reduce(lay_down, enumerate(zip(types, identifiers, strict=True)), innermost)
    entry: Final = _aligned(sql.Identifier(f"off{len(types)}"), "d")
    longest: Final = sql.SQL("").join(
        sql.SQL(", max({})").format(_value_bytes(column, identifier))
        for column, identifier in zip(types, identifiers, strict=True)
        if column.is_variable_length
    )
    return sql.SQL("SELECT count(*), max({entry}){longest} FROM ({rows}) AS entries WHERE {entry} > {limit}").format(
        entry=entry, longest=longest, rows=laid_out, limit=sql.Literal(limit_bytes)
    )


def _count_query(check: Check, schema: str, types: tuple[ColumnType, ...], limit_bytes: int) -> "sql.Composed":
    return _entry_size_query(check, schema, types, limit_bytes)


def _applies(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    check: Check,
    types: tuple[ColumnType, ...] | None,
) -> bool:
    """False when the check has nothing to count yet: the table or a column is not there
    (a pending migration adds it), the index is already valid, or no column can overflow."""
    if not _table_exists(connection, schema, check.table):
        return False
    if check.index is not None and _index_is_valid(connection, schema, check.index):
        return False
    return types is not None and any(column.can_overflow_btree for column in types)


def _too_long_detail(check: Check, types: tuple[ColumnType, ...], values: tuple[object, ...], limit_bytes: int) -> str:
    variable: Final = tuple(
        column for column, column_type in zip(check.columns, types, strict=True) if column_type.is_variable_length
    )
    longest: Final = ", ".join(
        f"{column} up to {length} bytes"
        for column, length in zip(variable, values[2:], strict=True)
        if length is not None
    )
    return f"with an index entry over {limit_bytes} bytes, largest {values[1]} bytes ({longest})"


def _run_check(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    check: Check,
    types: tuple[ColumnType, ...],
    limit_bytes: int,
    timeout_seconds: float,
) -> Finding | None:
    import psycopg

    try:
        row: Final = connection.execute(_count_query(check, schema, types, limit_bytes)).fetchone()
    except psycopg.errors.QueryCanceled:
        return CouldNotVerify(check, timeout_seconds)
    values: Final[tuple[object, ...]] = row if row is not None else ()
    rows: Final = _int(values[0], 0) if values else 0
    if rows == 0:
        return None
    return BlockedChange(check, rows, _too_long_detail(check, types, values, limit_bytes))


def find_blockers(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    checks: tuple[Check, ...],
    timeout_seconds: float = DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS,
) -> tuple[Finding, ...]:
    """Run every applicable check read-only, each under `timeout_seconds`. Never writes
    and never takes a table lock; a check that times out is a finding, not a pass."""
    from psycopg import sql

    previous_timeout: Final = connection.execute("SHOW statement_timeout").fetchone()
    previous_read_only: Final = connection.execute("SHOW default_transaction_read_only").fetchone()
    block_size: Final = connection.execute("SHOW block_size").fetchone()
    limit_bytes: Final = btree_max_entry_bytes(int(str(block_size[0])) if block_size else DEFAULT_BLOCK_SIZE)
    connection.execute("SET default_transaction_read_only = on")
    connection.execute(sql.SQL("SET statement_timeout = {}").format(sql.Literal(int(timeout_seconds * 1000))))
    column_type: Final = column_type_lookup(connection, schema)
    typed: Final = tuple((check, _column_types(check, column_type)) for check in checks)
    applicable: Final = tuple(
        (check, types if types is not None else ())
        for check, types in typed
        if _applies(connection, schema, check, types)
    )
    results: Final = tuple(
        _run_check(connection, schema, check, types, limit_bytes, timeout_seconds) for check, types in applicable
    )
    connection.execute(
        sql.SQL("SET statement_timeout = {}").format(sql.Literal(str(previous_timeout[0]) if previous_timeout else "0"))
    )
    connection.execute(
        sql.SQL("SET default_transaction_read_only = {}").format(
            sql.Literal(str(previous_read_only[0]) if previous_read_only else "off")
        )
    )
    return tuple(result for result in results if result is not None)


def _remedy(check: Check) -> str:
    joined: Final = ", ".join(check.columns)
    return f'Shorten or hash the long "{check.table}" ({joined}) values so every row\'s index entry fits.'


def _describe(finding: Finding) -> str:
    check: Final = finding.check
    where: Final = f'"{check.table}"' + (f".{', '.join(check.columns)}" if check.columns else "")
    match finding:
        case CouldNotVerify(timeout_seconds=seconds):
            return (
                f"  {where}: could not be checked within {seconds:g} s, refusing rather than guessing\n"
                f"    blocks: {check.statement} ({check.source})\n"
                f"    raise {MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR} to give the scan more time"
            )
        case BlockedChange(rows=rows, detail=detail):
            return (
                f"  {where}: {rows} row(s) {detail}\n"
                f"    blocks: {check.statement} ({check.source})\n"
                f"    fix:    {_remedy(check).replace(chr(10), chr(10) + ' ' * 8)}"
            )


def report(findings: tuple[Finding, ...]) -> str:
    """The operator-facing refusal: one block per finding, then what to do next."""
    lines: Final = "\n".join(_describe(finding) for finding in findings)
    return f"{REFUSAL}\n{lines}\nNo index was built. Fix the rows and the next index build picks it up."
