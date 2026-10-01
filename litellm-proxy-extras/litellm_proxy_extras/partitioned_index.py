import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm_proxy_extras._logging import logger

if TYPE_CHECKING:
    import psycopg
    from typing_extensions import LiteralString

PARTITIONED_CONCURRENT_INDEX_MARKER: Final = "cannot create index on partitioned table"
PARTITIONED_INDEX_LOCK_KEY: Final = int.from_bytes(b"llm_pidx", "big")
_IDENTIFIER_MAX_LENGTH: Final = 63
_COMMENT_RE: Final = re.compile(r"--[^\n]*")
_CONCURRENT_INDEX_RE: Final = re.compile(
    r'^CREATE\s+INDEX\s+CONCURRENTLY\s+(?:IF\s+NOT\s+EXISTS\s+)?"?(?P<index>[^"\s]+)"?\s+ON\s+(?:ONLY\s+)?'
    r'"?(?P<table>[^"\s(]+)"?\s*(?P<definition>(?:USING\s+\w+\s*)?\(.*)$',
    re.IGNORECASE | re.DOTALL,
)
_UNATTACHED_PARTITIONS_SQL: Final = (
    "SELECT n.nspname, c.relname, c.relkind FROM pg_inherits i "
    "JOIN pg_class c ON c.oid = i.inhrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE i.inhparent = to_regclass(%s) AND NOT EXISTS ("
    "SELECT 1 FROM pg_inherits ii JOIN pg_index x ON x.indexrelid = ii.inhrelid "
    "WHERE ii.inhparent = to_regclass(%s) AND x.indrelid = c.oid) "
    "ORDER BY n.nspname, c.relname"
)
_INDEX_IS_VALID_SQL: Final = "SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass(%s)"


@dataclass(frozen=True, slots=True)
class ConcurrentIndexMigration:
    """The single `CREATE INDEX CONCURRENTLY` statement a migration file ships."""

    index: str
    table: str
    definition: str

    def partition_index_name(self, partition: str) -> str:
        name: Final = f"{partition}_{self.index.removeprefix(f'{self.table}_')}"
        if len(name) <= _IDENTIFIER_MAX_LENGTH:
            return name
        digest: Final = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[: _IDENTIFIER_MAX_LENGTH - len(digest) - 1]}_{digest}"


def parse_concurrent_index_migration(script: str) -> "ConcurrentIndexMigration | None":
    statements: Final = tuple(part.strip() for part in _COMMENT_RE.sub("", script).split(";") if part.strip())
    if len(statements) != 1:
        return None
    match: Final = _CONCURRENT_INDEX_RE.match(statements[0])
    if match is None:
        return None
    return ConcurrentIndexMigration(match["index"], match["table"], match["definition"].strip())


def _regclass_name(connection: "psycopg.Connection[tuple[object, ...]]", schema: str, name: str) -> str:
    from psycopg import sql

    return sql.Identifier(schema, name).as_string(connection)


def _statement(
    connection: "psycopg.Connection[tuple[object, ...]]",
    template: "LiteralString",
    index: str,
    table: tuple[str, str],
    definition: str,
) -> bytes:
    """Render a CREATE INDEX statement with quoted identifiers; the migration's
    own index definition is appended verbatim, the way Prisma would have run it."""
    from psycopg import sql

    prefix: Final = sql.SQL(template).format(sql.Identifier(index), sql.Identifier(*table)).as_string(connection)
    return (prefix + definition).encode()


def _index_is_valid(connection: "psycopg.Connection[tuple[object, ...]]", schema: str, index: str) -> bool:
    row: Final = connection.execute(_INDEX_IS_VALID_SQL, (_regclass_name(connection, schema, index),)).fetchone()
    return row is not None and row[0] is True


def build_index_on_partitioned_table(
    connection: "psycopg.Connection[tuple[object, ...]]", schema: str, migration: ConcurrentIndexMigration
) -> bool:
    """Build the migration's index the way Postgres allows on a partitioned
    parent: a non-concurrent index on ONLY the parent, then one CONCURRENTLY
    build per leaf partition attached up the partition tree. Every statement
    is idempotent, so an interrupted run resumes where it stopped. The
    connection must be in autocommit mode. True when the parent index ends
    up valid."""
    _build_partitioned_index(connection, (schema, migration.table), (schema, migration.index), migration)
    return _index_is_valid(connection, schema, migration.index)


def _build_partitioned_index(
    connection: "psycopg.Connection[tuple[object, ...]]",
    table: tuple[str, str],
    index: tuple[str, str],
    migration: ConcurrentIndexMigration,
) -> None:
    from psycopg import sql

    connection.execute(
        _statement(connection, "CREATE INDEX IF NOT EXISTS {} ON ONLY {} ", index[1], table, migration.definition)
    )
    partitions = connection.execute(
        _UNATTACHED_PARTITIONS_SQL,
        (_regclass_name(connection, *table), _regclass_name(connection, *index)),
    ).fetchall()
    for nspname, relname, relkind in partitions:
        partition = (str(nspname), str(relname))
        child = (partition[0], migration.partition_index_name(partition[1]))
        if relkind == "p":
            _build_partitioned_index(connection, partition, child, migration)
        else:
            connection.execute(
                _statement(
                    connection,
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS {} ON {} ",
                    child[1],
                    partition,
                    migration.definition,
                )
            )
            if not _index_is_valid(connection, *child):
                connection.execute(sql.SQL("REINDEX INDEX CONCURRENTLY {}").format(sql.Identifier(*child)))
        connection.execute(
            sql.SQL("ALTER INDEX {} ATTACH PARTITION {}").format(sql.Identifier(*index), sql.Identifier(*child))
        )
        logger.info("Built index %s on partition %s of %s", child[1], partition[1], table[1])
