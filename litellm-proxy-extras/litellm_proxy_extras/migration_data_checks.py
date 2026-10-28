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
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import reduce
from itertools import accumulate
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

Kind = Literal["too_long", "duplicates", "nulls", "any_row", "orphans", "check_violations"]


@dataclass(frozen=True, slots=True)
class Check:
    kind: Kind
    table: str
    columns: tuple[str, ...]
    statement: str
    source: str
    index: str | None = None
    referenced_table: str | None = None
    referenced_columns: tuple[str, ...] = ()
    expression: str | None = None


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

_IDENT: Final = r'"?(\w+)"?'
_QUALIFIED: Final = r'(?:"?\w+"?\s*\.\s*)?"?(\w+)"?'
_CREATE_INDEX: Final = re.compile(
    rf"^\s*CREATE\s+(UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?{_IDENT}\s+ON\s+(?:ONLY\s+)?"
    rf"{_QUALIFIED}\s*(?:USING\s+(\w+)\s*)?\((.*?)\)",
    re.IGNORECASE | re.DOTALL,
)
_ALTER_TABLE: Final = re.compile(
    rf"^\s*ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?{_QUALIFIED}\s+(.*)$", re.IGNORECASE | re.DOTALL
)
_SET_NOT_NULL: Final = re.compile(rf"^ALTER\s+(?:COLUMN\s+)?{_IDENT}\s+SET\s+NOT\s+NULL$", re.IGNORECASE)
_ADD_COLUMN: Final = re.compile(
    rf"^ADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?(?!(?:CONSTRAINT|FOREIGN|UNIQUE|CHECK|PRIMARY|EXCLUDE)\b){_IDENT}\s+(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_CONSTRAINT: Final = rf"^ADD\s+(?:CONSTRAINT\s+{_IDENT}\s+)?"
_ADD_FOREIGN_KEY: Final = re.compile(
    rf"{_CONSTRAINT}FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+{_QUALIFIED}\s*\(([^)]*)\)(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_ADD_UNIQUE: Final = re.compile(rf"{_CONSTRAINT}UNIQUE\s*\(([^)]*)\)", re.IGNORECASE | re.DOTALL)
_ADD_CHECK: Final = re.compile(rf"{_CONSTRAINT}CHECK\s*\((.*)\)\s*(NOT\s+VALID)?\s*$", re.IGNORECASE | re.DOTALL)
_PLAIN_COLUMN: Final = re.compile(rf"^{_IDENT}(?:\s+(?:ASC|DESC|NULLS\s+(?:FIRST|LAST)))*$", re.IGNORECASE)
_COMMENT: Final = re.compile(r"--[^\n]*")
_NOT_NULL: Final = re.compile(r"\bNOT\s+NULL\b", re.IGNORECASE)
_HAS_DEFAULT: Final = re.compile(r"\bDEFAULT\b|\bGENERATED\b", re.IGNORECASE)
_UPDATE: Final = re.compile(rf"^\s*UPDATE\s+(?:ONLY\s+)?{_QUALIFIED}\s+SET\s+(.*)$", re.IGNORECASE | re.DOTALL)
_CLAUSE: Final = re.compile(r"\s(FROM|WHERE|RETURNING)\s", re.IGNORECASE)
_ASSIGNMENT: Final = re.compile(rf"^{_IDENT}\s*=\s*(.*)$", re.DOTALL)
_IS_NULL: Final = re.compile(rf"^{_IDENT}\s+IS\s+NULL$", re.IGNORECASE)
_STRING_LITERAL: Final = re.compile(r"'[^']*'")
_IDENTIFIER: Final = re.compile(r'"([^"]+)"|\b([A-Za-z_]\w*)\b')


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


def statements(migration_sql: str) -> tuple[str, ...]:
    """The statements of one migration file, comments stripped, empty ones dropped."""
    parts: Final = _COMMENT.sub("", migration_sql).split(";")
    return tuple(part.strip() for part in parts if part.strip())


def _mask_strings(text: str) -> str:
    """The text with the inside of every single quoted literal blanked, same length, so a scan
    for commas or keywords never matches inside a value."""
    return _STRING_LITERAL.sub(lambda match: f"'{'_' * (len(match.group()) - 2)}'", text)


def _paren_depth(text: str) -> tuple[int, ...]:
    return tuple(accumulate((1 if char == "(" else -1 if char == ")" else 0) for char in text))


def _split_top_level(text: str) -> tuple[str, ...]:
    """Split on the commas outside parentheses and string literals."""
    masked: Final = _mask_strings(text)
    cuts: Final = tuple(
        position
        for position, (char, level) in enumerate(zip(masked, _paren_depth(masked), strict=True))
        if char == "," and level == 0
    )
    starts: Final = (0, *(cut + 1 for cut in cuts))
    stops: Final = (*cuts, len(text))
    pieces: Final = tuple(text[start:stop].strip() for start, stop in zip(starts, stops, strict=True))
    return tuple(piece for piece in pieces if piece)


def _plain_columns(column_list: str) -> tuple[str, ...] | None:
    """The column names of an index or constraint column list, None when one entry is an
    expression the derivation cannot turn into a count."""
    matches: Final = tuple(_PLAIN_COLUMN.match(entry) for entry in _split_top_level(column_list))
    if any(match is None for match in matches):
        return None
    return tuple(match.group(1) for match in matches if match is not None)


def _index_checks(match: "re.Match[str]", statement_text: str, source: str) -> Iterator[Check]:
    unique, name, table, method, column_list = match.groups()
    columns: Final = _plain_columns(column_list)
    if columns is None:
        return
    label: Final = f'CREATE {"UNIQUE " if unique else ""}INDEX "{name}"'
    if method is None or method.lower() == "btree":
        yield Check("too_long", table, columns, label, source, index=name)
    if unique:
        yield Check("duplicates", table, columns, label, source, index=name)


def _alter_table_checks(table: str, action: str, source: str) -> Iterator[Check]:
    if (set_not_null := _SET_NOT_NULL.match(action)) is not None:
        yield Check("nulls", table, (set_not_null.group(1),), f'ALTER TABLE "{table}" {action}', source)
        return
    if (add_column := _ADD_COLUMN.match(action)) is not None:
        column, body = add_column.groups()
        if _NOT_NULL.search(body) is not None and _HAS_DEFAULT.search(body) is None:
            yield Check(
                "any_row", table, (column,), f'ALTER TABLE "{table}" ADD COLUMN "{column}" ... NOT NULL', source
            )
        return
    if (foreign_key := _ADD_FOREIGN_KEY.match(action)) is not None:
        constraint, columns, referenced_table, referenced_columns, tail = foreign_key.groups()
        own: Final = _plain_columns(columns)
        theirs: Final = _plain_columns(referenced_columns)
        if own is None or theirs is None or len(own) != len(theirs) or "NOT VALID" in tail.upper():
            return
        yield Check(
            "orphans",
            table,
            own,
            _constraint_label(table, constraint, "FOREIGN KEY"),
            source,
            referenced_table=referenced_table,
            referenced_columns=theirs,
        )
        return
    if (unique := _ADD_UNIQUE.match(action)) is not None:
        constraint, columns = unique.groups()
        unique_columns: Final = _plain_columns(columns)
        if unique_columns is not None:
            yield Check("duplicates", table, unique_columns, _constraint_label(table, constraint, "UNIQUE"), source)
        return
    if (check := _ADD_CHECK.match(action)) is not None:
        constraint, expression, not_valid = check.groups()
        if not_valid is None:
            yield Check(
                "check_violations",
                table,
                (),
                _constraint_label(table, constraint, "CHECK"),
                source,
                expression=expression,
            )


def _constraint_label(table: str, constraint: str | None, kind: str) -> str:
    named: Final = f'CONSTRAINT "{constraint}" ' if constraint else ""
    return f'ALTER TABLE "{table}" ADD {named}{kind}'


def checks_for_statement(statement_text: str, source: str) -> tuple[Check, ...]:
    """The count queries one SQL statement needs before it can be applied, empty for a
    statement that cannot fail on existing rows or that the derivation cannot read."""
    if (index := _CREATE_INDEX.match(statement_text)) is not None:
        return tuple(_index_checks(index, statement_text, source))
    if (alter := _ALTER_TABLE.match(statement_text)) is not None:
        table, actions = alter.groups()
        return tuple(
            check for action in _split_top_level(actions) for check in _alter_table_checks(table, action, source)
        )  # comprehension-ok: flattening the per-action checks of one statement
    return ()


@dataclass(frozen=True, slots=True)
class Update:
    """One UPDATE as the derivation reads it. `left_null` maps each assigned column to the
    predicate selecting the rows still NULL once the UPDATE ran, or to None when the shape
    (a FROM, a WHERE other than that column IS NULL, a DEFAULT) can leave NULLs it cannot
    count. It is None itself when the SET list cannot be read, so nothing is known about the table."""

    table: str
    left_null: MappingProxyType[str, str | None] | None


@dataclass(frozen=True, slots=True)
class Trail:
    """What the UPDATEs read so far did: `fills` the predicate counting the rows still NULL per
    (table, column), `written` every (table, column) one of them assigned, lower cased, with
    (table, "*") standing for a SET list that could not be read."""

    fills: MappingProxyType[tuple[str, str], str]
    written: frozenset[tuple[str, str]]


def _clause_start(text: str) -> int | None:
    """Offset of the first FROM, WHERE or RETURNING outside parentheses and string literals."""
    masked: Final = _mask_strings(text)
    depth: Final = _paren_depth(masked)
    return next((match.start() for match in _CLAUSE.finditer(masked) if depth[match.start()] == 0), None)


def _left_null_everywhere(value: str) -> str | None:
    return None if value.upper() == "DEFAULT" else f"({value}) IS NULL"


def _left_null_where_null(column: str, value: str) -> str | None:
    return None if value.upper() == "DEFAULT" else f'"{column}" IS NULL AND ({value}) IS NULL'


def parse_update(statement_text: str) -> Update | None:
    update: Final = _UPDATE.match(statement_text)
    if update is None:
        return None
    table, rest = update.groups()
    cut: Final = _clause_start(rest)
    matches: Final = tuple(_ASSIGNMENT.match(piece) for piece in _split_top_level(rest if cut is None else rest[:cut]))
    if any(match is None for match in matches):
        return Update(table, None)
    assigned: Final = MappingProxyType({m.group(1): m.group(2).strip() for m in matches if m is not None})
    clause: Final = "" if cut is None else rest[cut:].strip()
    if not clause or clause[:9].upper() == "RETURNING":
        return Update(table, MappingProxyType({col: _left_null_everywhere(value) for col, value in assigned.items()}))
    is_null: Final = _IS_NULL.match(clause[5:].strip()) if clause[:5].upper() == "WHERE" else None
    filled: Final = is_null.group(1) if is_null is not None and is_null.group(1) in assigned else None
    return Update(
        table,
        MappingProxyType(
            {
                column: _left_null_where_null(column, value) if column == filled else None
                for column, value in assigned.items()
            }
        ),
    )


def _survives(key: tuple[str, str], update: Update) -> bool:
    table, column = key
    return table != update.table or (update.left_null is not None and column not in update.left_null)


def _reads_a_written_column(table: str, predicate: str, written: frozenset[tuple[str, str]]) -> bool:
    names: Final = frozenset((quoted or bare).lower() for quoted, bare in _IDENTIFIER.findall(_mask_strings(predicate)))
    return (table, "*") in written or any((table, name) in written for name in names)


def _after(trail: Trail, update: Update | None) -> Trail:
    """The trail once `update` ran: what it leaves replaces the earlier fills of the columns it
    writes, a fill that reads a column an earlier UPDATE wrote is not counted (the count would
    read the old values), and an unreadable SET list drops every fill of its table."""
    if update is None:
        return trail
    columns: Final = MappingProxyType({}) if update.left_null is None else update.left_null
    kept: Final = {key: fill for key, fill in trail.fills.items() if _survives(key, update)}
    added: Final = {
        (update.table, column): fill
        for column, fill in columns.items()
        if fill is not None and not _reads_a_written_column(update.table, fill, trail.written)
    }
    written: Final = (
        frozenset({(update.table, "*")})
        if update.left_null is None
        else frozenset((update.table, column.lower()) for column in columns)
    )
    return Trail(MappingProxyType({**kept, **added}), trail.written | written)


def _counting_what_is_left(check: Check, trail: Trail) -> Check:
    if check.kind != "nulls" or (check.table, check.columns[0]) not in trail.fills:
        return check
    return replace(check, expression=trail.fills[(check.table, check.columns[0])])


def checks_for_migration(name: str, migration_sql: str) -> tuple[Check, ...]:
    """The checks of every statement in order. A `nulls` check counts the rows an earlier UPDATE
    of the same migration still leaves NULL, since `prisma migrate deploy` runs that UPDATE first."""
    ordered: Final = statements(migration_sql)
    trails: Final = tuple(
        accumulate((parse_update(text) for text in ordered), _after, initial=Trail(MappingProxyType({}), frozenset()))
    )
    return tuple(
        _counting_what_is_left(check, trails[position])
        for position, statement_text in enumerate(ordered)
        for check in checks_for_statement(statement_text, f"migration {name}")
    )  # comprehension-ok: flattening the per-statement checks of one migration


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
    from psycopg import sql

    if column.type_name in _OCTET_LENGTH_TYPES:
        return sql.SQL("octet_length({})").format(identifier)
    return sql.SQL("octet_length({}::text)").format(identifier)


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
    from psycopg import sql

    table: Final = sql.Identifier(schema, check.table)
    columns: Final = sql.SQL(", ").join(sql.Identifier(column) for column in check.columns)
    match check.kind:
        case "too_long":
            return _entry_size_query(check, schema, types, limit_bytes)
        case "duplicates":
            not_null: Final = sql.SQL(" AND ").join(
                sql.SQL("{} IS NOT NULL").format(sql.Identifier(column)) for column in check.columns
            )
            return sql.SQL(
                "SELECT count(*) FROM (SELECT 1 FROM {t} WHERE {nn} GROUP BY {cols} HAVING count(*) > 1) AS d"
            ).format(t=table, nn=not_null, cols=columns)
        case "nulls":
            left_null: Final = (
                sql.SQL(check.expression)  # pyright: ignore[reportArgumentType]  # copied from our own migration file
                if check.expression is not None
                else sql.SQL("{} IS NULL").format(sql.Identifier(check.columns[0]))
            )
            return sql.SQL("SELECT count(*) FROM {t} WHERE {left_null}").format(t=table, left_null=left_null)
        case "any_row":
            return sql.SQL("SELECT count(*) FROM {t}").format(t=table)
        case "orphans":
            pairs: Final = sql.SQL(" AND ").join(
                sql.SQL("parent.{} = child.{}").format(sql.Identifier(theirs), sql.Identifier(own))
                for own, theirs in zip(check.columns, check.referenced_columns, strict=True)
            )
            present: Final = sql.SQL(" AND ").join(
                sql.SQL("child.{} IS NOT NULL").format(sql.Identifier(column)) for column in check.columns
            )
            return sql.SQL(
                "SELECT count(*) FROM {t} AS child WHERE {present} AND NOT EXISTS (SELECT 1 FROM {ref} AS parent WHERE {pairs})"
            ).format(t=table, present=present, ref=sql.Identifier(schema, check.referenced_table or ""), pairs=pairs)
        case "check_violations":
            return sql.SQL("SELECT count(*) FROM {t} WHERE NOT ({expr})").format(
                t=table,
                expr=sql.SQL(check.expression or "true"),  # pyright: ignore[reportArgumentType]  # copied from our own migration file
            )


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
    if check.kind == "any_row":
        return types is None
    if types is None:
        return False
    if check.kind == "too_long":
        return any(column.can_overflow_btree for column in types)
    if check.kind == "orphans":
        return check.referenced_table is not None and _table_exists(connection, schema, check.referenced_table)
    return True


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
    except psycopg.errors.ProgrammingError:
        if check.kind != "nulls" or check.expression is None:
            raise
        return _run_check(connection, schema, replace(check, expression=None), types, limit_bytes, timeout_seconds)
    values: Final[tuple[object, ...]] = row if row is not None else ()
    rows: Final = _int(values[0], 0) if values else 0
    if rows == 0:
        return None
    match check.kind:
        case "too_long":
            return BlockedChange(check, rows, _too_long_detail(check, types, values, limit_bytes))
        case "duplicates":
            return BlockedChange(check, rows, "duplicated value groups")
        case "nulls":
            return BlockedChange(
                check, rows, "NULL" if check.expression is None else "still NULL once its UPDATE has run"
            )
        case "any_row":
            return BlockedChange(
                check, rows, "rows, and a NOT NULL column with no DEFAULT cannot be added to a table with rows"
            )
        case "orphans":
            return BlockedChange(check, rows, f'with no matching row in "{check.referenced_table}"')
        case "check_violations":
            return BlockedChange(check, rows, f"failing CHECK ({check.expression})")


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
    match check.kind:
        case "too_long":
            return f'Shorten or hash the long "{check.table}" ({joined}) values so every row\'s index entry fits.'
        case "duplicates":
            return f'Remove or merge the duplicate "{check.table}" ({joined}) values.'
        case "nulls":
            return f'Fill the NULL "{check.table}"."{joined}" values, or give the column a default first.'
        case "any_row":
            return "Give the new column a DEFAULT in the migration, or add it nullable and fill it afterwards."
        case "orphans":
            return f'Delete or repoint the "{check.table}" ({joined}) values with no "{check.referenced_table}" row.'
        case "check_violations":
            return f'Fix the "{check.table}" rows that fail the expression before adding the constraint.'


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
    return (
        f"{REFUSAL}\n{lines}\n"
        "No schema change was made. Fix the rows and start again, or run `litellm --check_migrations_against_existing_data` with the new image to check again."
    )


def pending_change_checks(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    migrations_dir: Path | None,
    indexes: tuple[tuple[str, str, tuple[str, ...]], ...],
) -> tuple[Check, ...]:
    """Every check the pending migrations and the request-log indexes need. `migrations_dir`
    is None when the schema is pushed rather than migrated, so only the indexes are checked."""
    migration_checks: Final = (
        tuple(
            check
            for name, migration_sql in pending_migrations(connection, schema, migrations_dir)
            for check in checks_for_migration(name, migration_sql)
        )  # comprehension-ok: flattening the per-migration checks
        if migrations_dir is not None
        else ()
    )
    index_checks: Final = tuple(
        check for table, name, columns in indexes for check in checks_for_index(table, name, columns)
    )  # comprehension-ok: flattening the per-index checks
    return migration_checks + index_checks
