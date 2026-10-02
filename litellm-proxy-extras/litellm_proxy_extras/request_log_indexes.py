"""The request-log indexes built after `prisma migrate deploy` instead of by a migration:
by the migration job, or by a serving proxy that ran the migrations itself (in the
background, once it serves).

A migration cannot build them: a plain `CREATE INDEX` blocks spend-log inserts for the
whole build, and `CREATE INDEX CONCURRENTLY` is refused on a partitioned parent
(db_scripts/partition_spend_logs.sql). `REQUEST_LOG_INDEXES` is the one list to extend;
names match what Prisma derives from the `@@index` declarations in schema.prisma, so an
index a database already has is recognized and never rebuilt.
"""

import hashlib
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm_proxy_extras._logging import logger
from litellm_proxy_extras.migration_lock import held_migration_lock

if TYPE_CHECKING:
    import psycopg
    from psycopg import sql


@dataclass(frozen=True, slots=True)
class RequestLogIndex:
    """One index the migration job owns: the table, the exact Prisma index name and the
    column list as it would be written after `ON <table>`."""

    table: str
    name: str
    definition: str

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(re.findall(r'"([^"]+)"', self.definition))

    def partition_index_name(self, partition: str) -> str:
        """The child index name for one partition, built the way Postgres names the
        children of a partitioned index, and kept within the 63 byte identifier limit."""
        name: Final = f"{partition}_{self.name.removeprefix(f'{self.table}_')}"
        if len(name.encode()) <= _IDENTIFIER_MAX_BYTES:
            return name
        digest: Final = hashlib.sha256(name.encode()).hexdigest()[:_DIGEST_LENGTH]
        budget: Final = _IDENTIFIER_MAX_BYTES - _DIGEST_LENGTH - 1
        kept: Final = next(name[:length] for length in range(len(name), 0, -1) if len(name[:length].encode()) <= budget)
        return f"{kept}_{digest}"


REQUEST_LOG_INDEXES: Final = (
    RequestLogIndex("LiteLLM_SpendLogs", "LiteLLM_SpendLogs_api_key_startTime_idx", '("api_key", "startTime")'),
    RequestLogIndex("LiteLLM_SpendLogs", "LiteLLM_SpendLogs_litellm_call_id_idx", '("litellm_call_id")'),
)

_IDENTIFIER_MAX_BYTES: Final = 63
_DDL_LOCK_TIMEOUT: Final = "200ms"
_DDL_LOCK_ATTEMPTS: Final = 10
_DDL_RETRY_BASE_SECONDS: Final = 0.25
_DDL_RETRY_MAX_SECONDS: Final = 8.0
_LOCK_HANDOVER_SECONDS: Final = 2.0
_DIGEST_LENGTH: Final = 8
_CREATE_INDEX_STATEMENT: Final = re.compile(
    r'^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"(?P<index>[^"]+)"\s+ON\b',
    re.IGNORECASE,
)
_TABLE_KIND_SQL: Final = "SELECT c.relkind = 'p' AS partitioned FROM pg_class c WHERE c.oid = to_regclass(%s)"
_CHILDREN_WITHOUT_THE_INDEX_SQL: Final = (
    "SELECT child.relname AS name, n.nspname AS schema, child.relkind = 'p' AS partitioned "
    "FROM pg_inherits i JOIN pg_class child ON child.oid = i.inhrelid "
    "JOIN pg_namespace n ON n.oid = child.relnamespace "
    "WHERE i.inhparent = to_regclass(%s) AND NOT EXISTS ("
    "SELECT 1 FROM pg_inherits attached JOIN pg_index x ON x.indexrelid = attached.inhrelid "
    "WHERE attached.inhparent = to_regclass(%s) AND x.indrelid = child.oid) "
    "ORDER BY child.relname"
)
_EQUIVALENT_INDEXES_SQL: Final = (
    "SELECT i.relname AS name, x.indisvalid AS valid "
    "FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_am am ON am.oid = i.relam "
    "WHERE x.indrelid = to_regclass(%s) AND i.relname <> %s AND am.amname = 'btree' AND NOT x.indisunique "
    "AND x.indexprs IS NULL AND x.indpred IS NULL AND x.indnkeyatts = x.indnatts "
    "AND NOT EXISTS (SELECT 1 FROM unnest(x.indoption::int2[]) o WHERE o <> 0) "
    "AND NOT EXISTS (SELECT 1 FROM unnest(x.indclass::oid[]) c JOIN pg_opclass oc ON oc.oid = c WHERE NOT oc.opcdefault) "
    "AND NOT EXISTS (SELECT 1 FROM unnest(x.indcollation::oid[]) WITH ORDINALITY c(coll, ord) "
    "JOIN unnest(x.indkey::int2[]) WITH ORDINALITY k(attnum, ord) ON k.ord = c.ord "
    "JOIN pg_attribute a ON a.attrelid = x.indrelid AND a.attnum = k.attnum "
    "WHERE c.coll <> 0 AND c.coll <> a.attcollation) "
    "AND (SELECT array_agg(a.attname::text ORDER BY k.ord) FROM unnest(x.indkey::int2[]) WITH ORDINALITY k(attnum, ord) "
    "JOIN pg_attribute a ON a.attrelid = x.indrelid AND a.attnum = k.attnum) = %s::text[] "
    "AND NOT EXISTS (SELECT 1 FROM pg_inherits WHERE inhrelid = x.indexrelid) "
    "ORDER BY x.indisvalid DESC, i.relname"
)
_INDEX_STATE_SQL: Final = (
    'SELECT x.indisvalid AS valid, t.relname AS "table" '
    "FROM pg_index x JOIN pg_class t ON t.oid = x.indrelid WHERE x.indexrelid = to_regclass(%s)"
)


@dataclass(frozen=True, slots=True)
class _Relation:
    name: str
    schema: str
    partitioned: bool


@dataclass(frozen=True, slots=True)
class _IndexState:
    valid: bool
    table: str


@dataclass(frozen=True, slots=True)
class _EquivalentIndex:
    name: str
    valid: bool


@dataclass(frozen=True, slots=True)
class _TableKind:
    partitioned: bool


def filter_request_log_index_diff(diff_sql: str, indexes: tuple[RequestLogIndex, ...] = REQUEST_LOG_INDEXES) -> str:
    """The `prisma migrate diff` script without the statements that create a migration-job-owned
    index, which the schema declares and the migrations deliberately do not build."""
    names: Final = frozenset(index.name for index in indexes)
    statements: Final = diff_sql.split(";")
    kept: Final = tuple(statement for statement in statements if not _creates_one_of(statement, names))
    return ";".join(kept) if any(part.strip() for part in kept) else ""


def _creates_one_of(statement: str, names: frozenset[str]) -> bool:
    match: Final = _CREATE_INDEX_STATEMENT.match(_without_comments(statement))
    return match is not None and match["index"] in names


def _without_comments(statement: str) -> str:
    return "\n".join(line for line in statement.splitlines() if not line.lstrip().startswith("--"))


def _connect(database_url: str) -> "psycopg.Connection[tuple[object, ...]]":
    import psycopg

    return psycopg.connect(database_url, connect_timeout=10, autocommit=True)


def ensure_request_log_indexes(
    database_url: str,
    schema: str,
    indexes: tuple[RequestLogIndex, ...] = REQUEST_LOG_INDEXES,
    connect: "Callable[[str], psycopg.Connection[tuple[object, ...]]]" = _connect,
) -> bool:
    """Build every listed index that is missing or invalid. Each build step runs under
    the migration coordinator lock, held per statement so a resolver booting on another
    replica gets in between partitions rather than waiting for the whole table. Any
    failure is logged and left for the next index build; the result says whether
    every index ended up valid. Never raises."""
    import psycopg

    try:
        with connect(database_url) as connection:
            connection.execute("SET statement_timeout = 0")
            results: Final = tuple(_ensure_index(connection, schema, index) for index in indexes)
    except psycopg.Error as exc:
        logger.warning("Could not build the request-log indexes, leaving them for the next index build: %s", exc)
        return False
    if not all(results):
        logger.warning("Some request-log indexes are not in place yet, leaving them for the next index build")
        return False
    logger.info("Request-log indexes are all in place")
    return True


def _under_migration_lock(connection: "psycopg.Connection[tuple[object, ...]]", step: Callable[[], bool]) -> bool:
    with held_migration_lock(connection) as held:
        if not held:
            logger.info(
                "Another process holds the migration lock, leaving the request-log indexes to the next index build"
            )
            return False
        return step()


def _with_bounded_lock(
    connection: "psycopg.Connection[tuple[object, ...]]", step: Callable[[], bool], what: str
) -> bool:
    """Run `step` under the migration lock with a short lock_timeout, so a DDL statement that has to wait for open
    transactions holds new writes back for at most that long; retry with capped exponential backoff, holding the
    migration lock per attempt only and releasing it while sleeping. False when another process holds the migration
    lock or every attempt timed out."""
    import psycopg
    from psycopg import sql

    for attempt in range(_DDL_LOCK_ATTEMPTS):
        if attempt:
            time.sleep(min(_DDL_RETRY_MAX_SECONDS, _DDL_RETRY_BASE_SECONDS * 2.0**attempt) * random.uniform(0.5, 1.0))
        connection.execute(sql.SQL("SET lock_timeout = {}").format(sql.Literal(_DDL_LOCK_TIMEOUT)))
        try:
            return _under_migration_lock(connection, step)
        except psycopg.errors.LockNotAvailable:
            logger.info("Waiting for open transactions before %s", what)
        finally:
            connection.execute("SET lock_timeout = 0")
    logger.warning(
        "Could not get the lock for %s without holding writes back, leaving it for the next index build", what
    )
    return False


def _ensure_index(connection: "psycopg.Connection[tuple[object, ...]]", schema: str, index: RequestLogIndex) -> bool:
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(_TableKind)) as cursor:
        table: Final = cursor.execute(_TABLE_KIND_SQL, (_regclass_name(connection, schema, index.table),)).fetchone()
    if table is None:
        logger.info("Table %s does not exist yet, skipping index %s", index.table, index.name)
        return True
    if table.partitioned:
        return build_index_on_partitioned_table(connection, schema, index)
    return _build_leaf_index(connection, schema, index.table, index.name, index)


def _regclass_name(connection: "psycopg.Connection[tuple[object, ...]]", schema: str, name: str) -> str:
    from psycopg import sql

    return sql.Identifier(schema, name).as_string(connection)


def _create_index_statement(
    connection: "psycopg.Connection[tuple[object, ...]]", prefix: "sql.Composed", definition: str
) -> bytes:
    return (prefix.as_string(connection) + definition).encode()


def _index_state(connection: "psycopg.Connection[tuple[object, ...]]", schema: str, index: str) -> "_IndexState | None":
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(_IndexState)) as cursor:
        return cursor.execute(_INDEX_STATE_SQL, (_regclass_name(connection, schema, index),)).fetchone()


def _equivalent_indexes(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    table: str,
    name: str,
    index: RequestLogIndex,
) -> tuple[_EquivalentIndex, ...]:
    """The indexes on `table` other than `name` with the same definition: default btree
    over the same columns in the same order, no expression, predicate, DESC or custom
    opclass or collation, and not attached under a partitioned index. Valid ones first."""
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(_EquivalentIndex)) as cursor:
        return tuple(
            cursor.execute(
                _EQUIVALENT_INDEXES_SQL, (_regclass_name(connection, schema, table), name, list(index.columns))
            ).fetchall()
        )


def _adopt_equivalent_index(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    table: str,
    name: str,
    index: RequestLogIndex,
) -> bool:
    """Rename a valid index of the same definition under another name (an operator's
    hand-built copy, say) to the name this code expects, instead of building a second
    one. RENAME on an index is a catalog change that lets writes through."""
    from psycopg import sql

    equivalent: Final = next(
        (found for found in _equivalent_indexes(connection, schema, table, name, index) if found.valid), None
    )
    if equivalent is None:
        return False
    logger.info(
        "Renaming the equivalent index %s on %s to %s instead of building a second one", equivalent.name, table, name
    )
    connection.execute(
        sql.SQL("ALTER INDEX {} RENAME TO {}").format(sql.Identifier(schema, equivalent.name), sql.Identifier(name))
    )
    return True


def _report_second_copies(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    table: str,
    name: str,
    index: RequestLogIndex,
    concurrently: bool,
) -> None:
    """Log every other index of the same definition with the statement that removes it.
    Dropping is the operator's call: a second copy costs writes and disk, never results."""
    from psycopg import sql

    drop: Final = "DROP INDEX CONCURRENTLY" if concurrently else "DROP INDEX"
    for copy in _equivalent_indexes(connection, schema, table, name, index):
        logger.warning(
            "Index %s on %s is a second copy of %s and only costs writes and disk; remove it with: %s %s",
            copy.name,
            table,
            name,
            drop,
            sql.Identifier(schema, copy.name).as_string(connection),
        )


def _children_without_the_index(
    connection: "psycopg.Connection[tuple[object, ...]]", schema: str, table: str, index: str
) -> tuple[_Relation, ...]:
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(_Relation)) as cursor:
        return tuple(
            cursor.execute(
                _CHILDREN_WITHOUT_THE_INDEX_SQL,
                (_regclass_name(connection, schema, table), _regclass_name(connection, schema, index)),
            ).fetchall()
        )


def _build_leaf_index(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    table: str,
    name: str,
    index: RequestLogIndex,
) -> bool:
    """Build one plain table's or partition's index with CONCURRENTLY so writes keep
    flowing. The catalog is read under the migration lock, so a replica that saw an
    invalid index before the lock finds the valid one another replica just built and
    leaves it. An invalid index left by an interrupted build is dropped and rebuilt; a
    valid index of the same definition under another name is renamed rather than
    duplicated; an index of that name on another table is a collision this code will
    not touch."""
    from psycopg import sql

    def build() -> bool:
        existing: Final = _index_state(connection, schema, name)
        if existing is not None and existing.table != table:
            logger.warning(
                "Index %s already exists on %s rather than %s, leaving it alone", name, existing.table, table
            )
            return False
        if existing is not None and existing.valid:
            return True
        if existing is not None:
            logger.info("Dropping the invalid index %s left by an interrupted build on %s", name, table)
            connection.execute(sql.SQL("DROP INDEX CONCURRENTLY {}").format(sql.Identifier(schema, name)))
        elif _adopt_equivalent_index(connection, schema, table, name, index):
            return True
        logger.info("Building index %s on %s concurrently", name, table)
        prefix: Final = sql.SQL("CREATE INDEX CONCURRENTLY IF NOT EXISTS {} ON {} ").format(
            sql.Identifier(name), sql.Identifier(schema, table)
        )
        connection.execute(_create_index_statement(connection, prefix, index.definition))
        built: Final = _index_state(connection, schema, name)
        return built is not None and built.valid

    current: Final = _index_state(connection, schema, name)
    if current is None or not current.valid or current.table != table:
        if not _under_migration_lock(connection, build):
            return False
        time.sleep(_LOCK_HANDOVER_SECONDS)
    _report_second_copies(connection, schema, table, name, index, concurrently=True)
    return True


def build_index_on_partitioned_table(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    index: RequestLogIndex,
    table: "str | None" = None,
    name: "str | None" = None,
) -> bool:
    """Build the index the way Postgres allows on a partitioned parent: a metadata-only
    parent index ON ONLY the parent, one CONCURRENTLY build per partition, and ATTACH
    PARTITION for each child. Partitions that are themselves partitioned get the same
    treatment one level down. Every step checks the catalog before acting, so an
    interrupted run resumes where it stopped and a second run finds nothing to do; a
    parent or child index of the same definition under another name is renamed and
    used rather than duplicated. The connection must be in autocommit mode. True when
    the parent index ends up valid."""

    parent_table: Final = index.table if table is None else table
    parent_index: Final = index.name if name is None else name
    existing: Final = _index_state(connection, schema, parent_index)
    if existing is not None and existing.table != parent_table:
        logger.warning(
            "Index %s already exists on %s rather than %s, leaving it alone", parent_index, existing.table, parent_table
        )
        return False
    if existing is None and not _with_bounded_lock(
        connection,
        lambda: (
            _adopt_equivalent_index(connection, schema, parent_table, parent_index, index)
            or _create_parent_index(connection, schema, parent_index, parent_table, index)
        ),
        f"creating the parent index {parent_index}",
    ):
        return False
    children: Final = _children_without_the_index(connection, schema, parent_table, parent_index)
    if not all(_attach_child_index(connection, schema, parent_index, child, index) for child in children):
        return False
    final: Final = _index_state(connection, schema, parent_index)
    if final is None or not final.valid:
        return False
    _report_second_copies(connection, schema, parent_table, parent_index, index, concurrently=False)
    return True


def _create_parent_index(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    name: str,
    table: str,
    index: RequestLogIndex,
) -> bool:
    """Create the metadata-only parent index. The caller bounds Postgres's SHARE lock wait on the parent."""
    from psycopg import sql

    prefix: Final = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON ONLY {} ").format(
        sql.Identifier(name), sql.Identifier(schema, table)
    )
    statement: Final = _create_index_statement(connection, prefix, index.definition)
    connection.execute(statement)
    return True


def _attach_child_index(
    connection: "psycopg.Connection[tuple[object, ...]]",
    schema: str,
    parent_index: str,
    child: _Relation,
    index: RequestLogIndex,
) -> bool:
    from psycopg import sql

    child_index: Final = index.partition_index_name(child.name)
    built: Final = (
        build_index_on_partitioned_table(connection, child.schema, index, child.name, child_index)
        if child.partitioned
        else _build_leaf_index(connection, child.schema, child.name, child_index, index)
    )
    if not built:
        return False

    def attach() -> bool:
        connection.execute(
            sql.SQL("ALTER INDEX {} ATTACH PARTITION {}").format(
                sql.Identifier(schema, parent_index), sql.Identifier(child.schema, child_index)
            )
        )
        logger.info("Attached index %s on partition %s to %s", child_index, child.name, parent_index)
        return True

    return _with_bounded_lock(connection, attach, f"attaching {child_index}")
