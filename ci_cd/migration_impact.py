#!/usr/bin/env python3
"""Report what a version bump does to the proxy's Postgres schema.

A release does not say what it does to the database. The migrations ship inside the
`litellm-proxy-extras` package (`litellm_proxy_extras/migrations/<timestamp>_<name>/migration.sql`)
and the first pod that boots a new version applies them with `prisma migrate deploy`, while the
generated release notes list commit titles, where a column addition reads as a feature ("add
lifecycle timestamps"). An operator upgrading a multi-replica deployment therefore cannot see,
from the release, which of those migrations the pods still running the previous version survive.

This script diffs the migration set between two refs, classifies every statement by what it does
to a pod still running the base version, and renders the result as markdown (for the release body)
or JSON. Severities, worst first:

- `breaking`: the old pods cannot recover on their own. A column they select is dropped, renamed
  or changed to a type their client cannot read, or a NOT NULL now rejects the rows they write
- `write-reject`: a new constraint the old pods never enforced. Rows they write that violate it
  are rejected; whether any do depends on the data
- `prepared-plan`: the result type of a query the old pods have prepared changes (a whole-row
  `SELECT <alias>.*` gains a column, or a column widens), so the plans cached on their pooled
  connections are rejected with `cached plan must not change result type` until those
  connections are recreated
- `lock`: the migration takes a blocking lock, or rewrites rows, on a table already serving traffic
- `info`: no effect on the old pods (new tables, changes to them, columns they do not know)

The evidence for what the old pods read and write is the base version itself: `schema.prisma` at
`--base` lists every column the old Prisma client selects and inserts, and the source tree at
`--base` holds the raw `SELECT <alias>.*` queries. Nothing is hardcoded, so the report keeps
working as the schema and those queries move.

    python3 ci_cd/migration_impact.py --base v1.93.0 --head v1.99.0

Standard library only, so it runs on a bare checkout with no `uv sync`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

MIGRATIONS_DIR: Final = "litellm-proxy-extras/litellm_proxy_extras/migrations"
# git pathspec wildcards cross directory boundaries, so these reach every file under each tree.
SOURCE_GLOBS: Final = ("litellm/*.py", "enterprise/*.py")
TABLE_MARKER: Final = 'FROM "'
# `*` inside a view is expanded when the view is created, so it holds no per-connection plan.
VIEW_MARKER: Final = re.compile(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?VIEW\b", re.IGNORECASE)

BREAKING: Final = "breaking"
WRITE_REJECT: Final = "write-reject"
PREPARED_PLAN: Final = "prepared-plan"
LOCK: Final = "lock"
INFO: Final = "info"
SEVERITY_ORDER: Final = (BREAKING, WRITE_REJECT, PREPARED_PLAN, LOCK, INFO)
SEVERITY_LABEL: Final = {
    BREAKING: "BREAKING",
    WRITE_REJECT: "WRITE-REJECT",
    PREPARED_PLAN: "PREPARED-PLAN",
    LOCK: "LOCK",
    INFO: "INFO",
}
SCHEMA_PATHS: Final = ("schema.prisma", "litellm/proxy/schema.prisma")
PRISMA_SCALARS: Final = frozenset(
    {"String", "Int", "BigInt", "Float", "Decimal", "Boolean", "DateTime", "Json", "Bytes"}
)
# Postgres types a column of each Prisma type can become and still be read by the old client.
# Anything else is rated breaking: the plan is rejected once, and the retry cannot map the value.
WIDENING: Final = {
    "Int": frozenset({"INT", "INT4", "INTEGER", "BIGINT", "INT8", "NUMERIC", "DECIMAL", "DOUBLE", "REAL", "FLOAT"}),
    "BigInt": frozenset({"BIGINT", "INT8", "NUMERIC", "DECIMAL"}),
    "Float": frozenset({"DOUBLE", "REAL", "FLOAT", "NUMERIC", "DECIMAL"}),
    "Decimal": frozenset({"NUMERIC", "DECIMAL"}),
    "String": frozenset({"TEXT", "VARCHAR", "CHARACTER", "CHAR", "CITEXT"}),
    "DateTime": frozenset({"TIMESTAMP", "TIMESTAMPTZ"}),
    "Json": frozenset({"JSON", "JSONB"}),
    "Boolean": frozenset({"BOOLEAN", "BOOL"}),
}

# Words that can follow a table name where an alias would sit, so they are never read as one.
ALIAS_STOPWORDS: Final = frozenset(
    {
        "as",
        "cross",
        "except",
        "fetch",
        "for",
        "full",
        "group",
        "having",
        "inner",
        "join",
        "left",
        "limit",
        "intersect",
        "natural",
        "offset",
        "on",
        "order",
        "outer",
        "returning",
        "right",
        "set",
        "tablesample",
        "union",
        "using",
        "values",
        "where",
        "window",
    }
)

_LINE_COMMENT: Final = re.compile(r"--[^\n]*")
_BLOCK_COMMENT: Final = re.compile(r"/\*.*?\*/", re.DOTALL)
_DOLLAR_QUOTED: Final = re.compile(r"\$(?P<tag>[A-Za-z_]*)\$(?P<body>.*?)\$(?P=tag)\$", re.DOTALL)

_CREATE_TABLE: Final = re.compile(r'\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"?(?P<table>\w+)"?', re.IGNORECASE)
_ALTER_TABLE: Final = re.compile(r'\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?"?(?P<table>\w+)"?', re.IGNORECASE)
_DROP_TABLE: Final = re.compile(r'\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?"?(?P<table>\w+)"?', re.IGNORECASE)
_CREATE_INDEX: Final = re.compile(
    r'\bCREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"?\w+"?\s+ON\s+(?:ONLY\s+)?"?(?P<table>\w+)"?',
    re.IGNORECASE,
)
# The `SET` is what separates a backfill from the `ON UPDATE CASCADE` of a foreign key.
_UPDATE: Final = re.compile(r'\bUPDATE\s+(?:ONLY\s+)?"?(?P<table>\w+)"?\s+SET\b', re.IGNORECASE)
_INSERT: Final = re.compile(r'\bINSERT\s+INTO\s+"?(?P<table>\w+)"?', re.IGNORECASE)

_DROP_COLUMN: Final = re.compile(r"\bDROP\s+COLUMN\b", re.IGNORECASE)
_ADD_COLUMN: Final = re.compile(r"\bADD\s+COLUMN\b", re.IGNORECASE)
# One ALTER TABLE carries many comma-separated clauses; split only at a clause keyword so the commas
# inside `NUMERIC(10,2)` or `ARRAY[]::TEXT[]` stay put.
_CLAUSE_SPLIT: Final = re.compile(r",\s*(?=(?:ADD|ALTER|DROP)\s+(?:COLUMN|CONSTRAINT)\b)", re.IGNORECASE)
# Column and table renames only; `ALTER INDEX ... RENAME TO` and `RENAME CONSTRAINT` change nothing a pod reads.
_RENAME: Final = re.compile(r"\bRENAME\s+(?:COLUMN\b|TO\b)", re.IGNORECASE)
_ALTER_TYPE: Final = re.compile(r"\bALTER\s+(?:COLUMN\s+)?\S+\s+(?:SET\s+DATA\s+)?TYPE\b", re.IGNORECASE)
_SET_NOT_NULL: Final = re.compile(r"\bSET\s+NOT\s+NULL\b", re.IGNORECASE)
# `DROP NOT NULL` relaxes a column and `IS NOT NULL` is a predicate; neither adds a constraint.
_NOT_NULL: Final = re.compile(r"(?<!DROP )(?<!IS )\bNOT\s+NULL\b", re.IGNORECASE)
_DEFAULT: Final = re.compile(r"\bDEFAULT\b", re.IGNORECASE)
_CONCURRENTLY: Final = re.compile(r"\bCONCURRENTLY\b", re.IGNORECASE)
_ADD_CONSTRAINT: Final = re.compile(r"\bADD\s+CONSTRAINT\b", re.IGNORECASE)
_NOT_VALID: Final = re.compile(r"\bNOT\s+VALID\b", re.IGNORECASE)
_UNIQUE_INDEX: Final = re.compile(r"\bCREATE\s+UNIQUE\s+INDEX\b", re.IGNORECASE)

_DROP_COLUMN_NAME: Final = re.compile(r'\bDROP\s+COLUMN\s+(?:IF\s+EXISTS\s+)?"?(?P<column>\w+)"?', re.IGNORECASE)
_RENAME_COLUMN: Final = re.compile(r'\bRENAME\s+COLUMN\s+"?(?P<column>\w+)"?\s+TO\b', re.IGNORECASE)
_RENAME_TABLE: Final = re.compile(r"\bRENAME\s+TO\b", re.IGNORECASE)
_SET_NOT_NULL_COLUMN: Final = re.compile(
    r'\bALTER\s+(?:COLUMN\s+)?"?(?P<column>\w+)"?\s+SET\s+NOT\s+NULL', re.IGNORECASE
)
_TYPE_CHANGE: Final = re.compile(
    r'\bALTER\s+(?:COLUMN\s+)?"?(?P<column>\w+)"?\s+(?:SET\s+DATA\s+)?TYPE\s+(?P<type>\w+)', re.IGNORECASE
)
_CONSTRAINT_COLUMNS: Final = re.compile(
    r"\b(?:FOREIGN\s+KEY|UNIQUE|PRIMARY\s+KEY)\s*\((?P<columns>[^)]*)\)", re.IGNORECASE
)
_CHECK_BODY: Final = re.compile(r"\bCHECK\s*\((?P<body>.*)\)", re.IGNORECASE | re.DOTALL)
_INDEX_COLUMNS: Final = re.compile(
    r'\bON\s+(?:ONLY\s+)?"?\w+"?\s*(?:USING\s+\w+\s*)?\((?P<columns>[^)]*)\)', re.IGNORECASE
)
_DROP_INDEX_NAME: Final = re.compile(
    r'\bDROP\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+EXISTS\s+)?"?(?P<index>\w+)"?', re.IGNORECASE
)
_QUOTED_IDENTIFIER: Final = re.compile(r'"(?P<name>\w+)"')
_BARE_WORD: Final = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\b")
_ADDED_COLUMN_NAME: Final = re.compile(r'\bADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?"?(?P<column>\w+)"?', re.IGNORECASE)
_INDEX_NAME: Final = re.compile(
    r'\bCREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"?(?P<index>\w+)"?', re.IGNORECASE
)

_MODEL_BLOCK: Final = re.compile(r"^\s*model\s+(?P<model>\w+)\s*\{(?P<body>.*?)^\s*\}", re.MULTILINE | re.DOTALL)
_ENUM_NAME: Final = re.compile(r"^\s*enum\s+(?P<name>\w+)", re.MULTILINE)
_PRISMA_FIELD: Final = re.compile(
    r"^\s*(?P<name>\w+)\s+(?P<type>\w+)(?P<optional>\?)?(?P<array>\[\])?(?P<rest>[^\n]*)$", re.MULTILINE
)
_FIELD_MAP: Final = re.compile(r'@map\("(?P<name>[^"]+)"\)')
_TABLE_MAP: Final = re.compile(r'@@map\("(?P<name>[^"]+)"\)')

_SQL_LITERAL: Final = re.compile(r"(?P<quote>\"{3}|'{3})(?P<body>.*?)(?P=quote)", re.DOTALL)
_FROM_JOIN: Final = re.compile(
    r'\b(?:FROM|JOIN)\s+"(?P<table>\w+)"(?:\s+AS)?(?:\s+(?P<alias>[A-Za-z_]\w*))?', re.IGNORECASE
)
_STAR_ALIAS: Final = re.compile(r"\b(?P<alias>[A-Za-z_]\w*)\.\*")
_BARE_STAR: Final = re.compile(r"\bSELECT\s+\*", re.IGNORECASE)
# Release tags an operator could actually be running: 1.99.0, v1.99.0, 1.99.0.post1, v1.99.0-stable.
_STABLE_TAG: Final = re.compile(r"^v?\d+\.\d+\.\d+(?:\.post\d+|-stable(?:\.patch\.\d+)?)?$")


@dataclass(frozen=True, slots=True)
class StarRead:
    """A query that reads a whole table row, i.e. one whose result type follows the schema."""

    table: str
    alias: str
    joined: tuple[str, ...]
    location: str


@dataclass(frozen=True, slots=True)
class Column:
    """One column as the base version's Prisma client knows it."""

    required: bool
    prisma_type: str


Schema = Mapping[str, Mapping[str, Column]]


@dataclass(frozen=True, slots=True)
class UpgradeContext:
    """Everything a statement is rated against: what this upgrade creates, and what the old pods use."""

    fresh_tables: frozenset[str]
    star_reads: tuple[StarRead, ...]
    schema: Schema | None
    dropped_indexes: frozenset[str] = frozenset()
    # (table, column) added by this upgrade with a DEFAULT: the old pods write that default on every insert.
    defaulted_columns: frozenset[tuple[str, str]] = frozenset()
    # Unique indexes that exist at the base, by name: what a dropped-and-recreated key used to cover.
    base_unique_indexes: Mapping[str, tuple[str, tuple[str, ...]]] = field(default_factory=lambda: MappingProxyType({}))

    def column(self, table: str, column: str) -> Column | None:
        return None if self.schema is None else self.schema.get(table, {}).get(column)

    def knows_table(self, table: str) -> bool:
        return self.schema is None or table in self.schema

    def knows_column(self, table: str, column: str) -> bool:
        """Whether the old client selects or writes this column. Without a schema, assume it does."""
        return self.schema is None or self.column(table, column) is not None

    def writes(self, table: str, column: str) -> bool:
        """Whether rows the old pods insert carry a value here: a column they know, or a default they get."""
        return self.knows_column(table, column) or (table, column) in self.defaulted_columns

    def replaced_key(self, table: str, columns: Sequence[str]) -> bool:
        """Whether a dropped unique index on this table covered a subset of `columns`: the new key then rejects
        nothing the old key did not already reject."""
        for name in self.dropped_indexes:
            found: Final = self.base_unique_indexes.get(name)
            if found is not None and found[0] == table and set(found[1]) <= set(columns):
                return True
        return False


@dataclass(frozen=True, slots=True)
class Finding:
    severity: str
    migration: str
    table: str
    statement: str
    effect: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "migration": self.migration,
            "table": self.table,
            "statement": self.statement,
            "effect": self.effect,
        }


@dataclass(frozen=True, slots=True)
class Report:
    base: str
    head: str
    migrations: tuple[str, ...]
    findings: tuple[Finding, ...]
    star_reads: tuple[StarRead, ...]
    schema_tables: int | None = None

    @property
    def worst(self) -> str | None:
        for severity in SEVERITY_ORDER:
            if any(finding.severity == severity for finding in self.findings):
                return severity
        return None


class GitError(RuntimeError):
    """A git command the report depends on could not be run."""


def _git(repo: Path, *args: str) -> str:
    result: Final = subprocess.run(("git", "-C", str(repo), *args), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _git_or_none(repo: Path, *args: str) -> str | None:
    try:
        return _git(repo, *args)
    except GitError:
        return None


def normalize(statement: str) -> str:
    return " ".join(statement.split()).strip()


def split_statements(sql: str) -> tuple[str, ...]:
    """Split a migration into single-line statements, descending into `DO $$ ... $$` bodies."""
    cleaned: Final = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))
    bodies: Final = tuple(match.group("body") for match in _DOLLAR_QUOTED.finditer(cleaned))
    outer: Final = _DOLLAR_QUOTED.sub(" ", cleaned)
    top_level: Final = tuple(normalize(part) for part in outer.split(";"))
    nested: Final = tuple(statement for body in bodies for statement in split_statements(body))
    return tuple(statement for statement in top_level + nested if statement)


def statement_table(statement: str) -> str:
    for pattern in (_CREATE_TABLE, _ALTER_TABLE, _DROP_TABLE, _CREATE_INDEX, _UPDATE, _INSERT):
        match: Final = pattern.search(statement)
        if match is not None:
            return match.group("table")
    return ""


def migration_names(repo: Path, ref: str) -> tuple[str, ...]:
    """The migration directories at `ref`, skipping `migration_lock.toml` and any stray file."""
    listing: Final = _git(repo, "ls-tree", ref, f"{MIGRATIONS_DIR}/")
    # <mode> <type> <object>\t<path>
    entries: Final = tuple(line.partition("\t") for line in listing.splitlines())
    return tuple(sorted(Path(path).name for meta, _, path in entries if meta.split()[1:2] == ["tree"]))


def read_migration(repo: Path, ref: str, name: str) -> str | None:
    """The SQL of one migration, or None when the directory has no `migration.sql` (Prisma skips it too)."""
    return _git_or_none(repo, "show", f"{ref}:{MIGRATIONS_DIR}/{name}/migration.sql")


def tables_created_before(repo: Path, ref: str) -> frozenset[str]:
    """Every table some migration at `ref` creates, so a later `CREATE TABLE IF NOT EXISTS` re-declaration
    of an existing table is not mistaken for a new one."""
    # Fixed string: `git grep -E` is POSIX ERE, where `\s` is not portable.
    listing: Final = _git_or_none(
        repo, "grep", "-h", "-i", "--fixed-strings", "CREATE TABLE", ref, "--", f"{MIGRATIONS_DIR}/"
    )
    return frozenset(match.group("table") for match in _CREATE_TABLE.finditer(listing or ""))


def _model_columns(body: str, enums: frozenset[str]) -> Mapping[str, Column]:
    declared: Final = tuple(_PRISMA_FIELD.finditer(body))
    return MappingProxyType(
        {
            (_FIELD_MAP.search(one.group("rest")) or one).group("name"): Column(
                required=one.group("optional") is None, prisma_type=one.group("type")
            )
            for one in declared
            if one.group("type") in PRISMA_SCALARS or one.group("type") in enums
        }
    )


def parse_prisma_schema(text: str) -> Schema:
    """The tables and columns a Prisma client generated from this schema selects and writes.

    Relation fields are not columns and are skipped; `@map` / `@@map` give the real names.
    """
    enums: Final = frozenset(_ENUM_NAME.findall(text))
    return MappingProxyType(
        {
            (_TABLE_MAP.search(block.group("body")) or block).group(
                "name" if _TABLE_MAP.search(block.group("body")) else "model"
            ): _model_columns(block.group("body"), enums)
            for block in _MODEL_BLOCK.finditer(text)
        }
    )


def load_base_schema(repo: Path, ref: str) -> Schema | None:
    texts: Final = (_git_or_none(repo, "show", f"{ref}:{path}") for path in SCHEMA_PATHS)
    return next((parsed for text in texts if text is not None and (parsed := parse_prisma_schema(text))), None)


def _identifiers(fragment: str) -> tuple[str, ...]:
    """Every name in the fragment, quoted or bare. Bare words include keywords and literals, so callers keep
    only the ones that are columns of the table at hand."""
    quoted: Final = tuple(match.group("name") for match in _QUOTED_IDENTIFIER.finditer(fragment))
    unquoted: Final = _QUOTED_IDENTIFIER.sub(" ", re.sub(r"'[^']*'", " ", fragment))
    bare: Final = tuple(match.group("name") for match in _BARE_WORD.finditer(unquoted))
    return tuple(dict.fromkeys(quoted + bare))


def _defaulted_columns(statements: Iterable[str]) -> frozenset[tuple[str, str]]:
    """(table, column) pairs these statements add with a DEFAULT."""
    return frozenset(
        (statement_table(statement), added.group("column"))
        for statement in statements
        for clause in _CLAUSE_SPLIT.split(statement)
        if (added := _ADDED_COLUMN_NAME.search(clause)) is not None and _DEFAULT.search(clause) is not None
    )


def unique_indexes_at(repo: Path, ref: str) -> Mapping[str, tuple[str, tuple[str, ...]]]:
    """Every unique index some migration at `ref` creates, by name: its table and columns.

    Read statement by statement, not line by line, because Prisma wraps long index definitions.
    """
    listing: Final = _git_or_none(
        repo, "grep", "-l", "-i", "--fixed-strings", "CREATE UNIQUE INDEX", ref, "--", f"{MIGRATIONS_DIR}/"
    )
    paths: Final = tuple(line.split(":", 1)[1] if ":" in line else line for line in (listing or "").splitlines())
    sources: Final = tuple(text for path in paths if (text := _git_or_none(repo, "show", f"{ref}:{path}")) is not None)
    return MappingProxyType(
        {
            named.group("index"): (statement_table(statement), _index_columns(statement))
            for source in sources
            for statement in split_statements(source)
            if _UNIQUE_INDEX.search(statement) is not None and (named := _INDEX_NAME.search(statement)) is not None
        }
    )


def _constraint_columns(statement: str) -> tuple[str, ...]:
    """Names a new constraint mentions: the listed columns, and every name inside a CHECK expression."""
    listed: Final = tuple(
        name for match in _CONSTRAINT_COLUMNS.finditer(statement) for name in _identifiers(match.group("columns"))
    )
    check: Final = _CHECK_BODY.search(statement)
    inside: Final = _identifiers(check.group("body")) if check is not None else ()
    return tuple(dict.fromkeys(listed + inside))


def _index_columns(statement: str) -> tuple[str, ...]:
    match: Final = _INDEX_COLUMNS.search(statement)
    return _identifiers(match.group("columns")) if match is not None else ()


def _alias_map(body: str) -> tuple[Mapping[str, str], tuple[str, ...]]:
    matches: Final = tuple(_FROM_JOIN.finditer(body))
    tables: Final = tuple(dict.fromkeys(match.group("table") for match in matches))
    aliases: Final = MappingProxyType(
        {
            match.group("alias"): match.group("table")
            for match in matches
            if match.group("alias") is not None and match.group("alias").lower() not in ALIAS_STOPWORDS
        }
    )
    return aliases, tables


def _star_reads_in_literal(body: str, location: str) -> tuple[StarRead, ...]:
    aliases, tables = _alias_map(body)
    starred: Final = tuple(
        (aliases[match.group("alias")], match.group("alias"))
        for match in _STAR_ALIAS.finditer(body)
        if match.group("alias") in aliases
    )
    bare: Final = ((tables[0], "*"),) if not starred and _BARE_STAR.search(body) and len(tables) == 1 else ()
    return tuple(
        StarRead(table=table, alias=alias, joined=tuple(other for other in tables if other != table), location=location)
        for table, alias in dict.fromkeys(starred + bare)
    )


def star_reads_in_source(text: str, path: str) -> tuple[StarRead, ...]:
    """Find whole-row reads (`SELECT v.*`, `SELECT *`) in the SQL literals of one source file."""
    return tuple(
        read
        for literal in _SQL_LITERAL.finditer(text)
        if TABLE_MARKER in literal.group("body")
        and "SELECT" in literal.group("body").upper()
        and VIEW_MARKER.search(literal.group("body")) is None
        # `-- Added comma to separate b.* columns` is a comment, not a read.
        for read in _star_reads_in_literal(
            _LINE_COMMENT.sub(" ", literal.group("body")), f"{path}:{text.count(chr(10), 0, literal.start()) + 1}"
        )
    )


def discover_star_reads(repo: Path, ref: str) -> tuple[StarRead, ...]:
    listing: Final = _git_or_none(repo, "grep", "-l", "--fixed-strings", TABLE_MARKER, ref, "--", *SOURCE_GLOBS)
    paths: Final = tuple(line.split(":", 1)[1] if ":" in line else line for line in (listing or "").splitlines())
    return tuple(
        read
        for path in paths
        if (text := _git_or_none(repo, "show", f"{ref}:{path}")) is not None
        for read in star_reads_in_source(text, path)
    )


def _shape(read: StarRead) -> str:
    return "SELECT *" if read.alias == "*" else f"SELECT {read.alias}.*"


def _star_read_for(table: str, star_reads: Sequence[StarRead]) -> StarRead | None:
    matches: Final = [read for read in star_reads if read.table == table]
    return max(matches, key=lambda read: len(read.joined)) if matches else None


def _joined_read_for(table: str, star_reads: Sequence[StarRead]) -> StarRead | None:
    return next((read for read in star_reads if table in read.joined), None)


@dataclass(frozen=True, slots=True)
class _Site:
    """One statement under rating, with what the rules need to know about it."""

    migration: str
    statement: str
    table: str
    context: UpgradeContext
    star: StarRead | None

    @property
    def where(self) -> str:
        return f" (`{self.star.location}`)" if self.star is not None else ""

    def rate(self, severity: str, effect: str) -> Finding:
        return Finding(
            severity=severity, migration=self.migration, table=self.table, statement=self.statement, effect=effect
        )

    def knows(self, column: str) -> bool:
        return self.context.knows_column(self.table, column)


def _rule_new_table(site: _Site) -> Finding | None:
    if _CREATE_TABLE.search(site.statement) is not None:
        return site.rate(INFO, "New table. The previous version does not read or write it.")
    if site.table and site.table in site.context.fresh_tables:
        return site.rate(INFO, "Table is created by this same upgrade, so no old pod reads or writes it yet.")
    return None


def _rule_drop(site: _Site) -> Finding | None:
    if _DROP_TABLE.search(site.statement) is not None:
        if site.context.knows_table(site.table):
            return site.rate(
                BREAKING, "Drops a table the previous version still queries. Old pods keep failing until they are gone."
            )
        return site.rate(INFO, "Drops a table the previous version does not know.")
    dropped: Final = _DROP_COLUMN_NAME.search(site.statement)
    if dropped is None:
        return None
    column: Final = dropped.group("column")
    if site.knows(column):
        return site.rate(
            BREAKING,
            f"Drops `{column}`, which the previous version's client selects on every read of this table{site.where}. "
            "Old pods keep failing until they are gone; reconnecting does not help.",
        )
    return site.rate(INFO, f"Drops `{column}`, a column the previous version does not know.")


def _rule_rename(site: _Site) -> Finding | None:
    renamed: Final = _RENAME_COLUMN.search(site.statement)
    if renamed is not None:
        old_name: Final = renamed.group("column")
        if site.knows(old_name):
            return site.rate(
                BREAKING,
                f"Renames `{old_name}`, which the previous version still selects and writes by that name{site.where}. "
                "Old pods cannot recover on their own.",
            )
        return site.rate(INFO, f"Renames `{old_name}`, a column the previous version does not know.")
    if _ALTER_TABLE.search(site.statement) is None or _RENAME_TABLE.search(site.statement) is None:
        return None
    if site.context.knows_table(site.table):
        return site.rate(BREAKING, "Renames a table the previous version still queries by its old name.")
    return site.rate(INFO, "Renames a table the previous version does not know.")


def _rule_not_null(site: _Site) -> Finding | None:
    required: Final = [match.group("column") for match in _SET_NOT_NULL_COLUMN.finditer(site.statement)]
    if required:
        unguarded: Final = [
            column
            for column in required
            if (known := site.context.column(site.table, column)) is None or not known.required
        ]
        if unguarded or site.context.schema is None:
            return site.rate(
                BREAKING,
                f"Requires `{'`, `'.join(unguarded or required)}`, which the previous version may write as NULL. "
                "Those writes are now rejected.",
            )
        return site.rate(INFO, "The previous version already always writes this column, so nothing changes for it.")
    if _ADD_COLUMN.search(site.statement) is not None and _adds_not_null_without_default(site.statement):
        return site.rate(BREAKING, "Adds a NOT NULL column with no default, so inserts from the previous version fail.")
    return None


def _rule_type_change(site: _Site) -> Finding | None:
    changes: Final = list(_TYPE_CHANGE.finditer(site.statement))
    if not changes:
        return None
    narrowing: Final = [
        f"`{match.group('column')}` to {match.group('type').upper()}"
        for match in changes
        if (known := site.context.column(site.table, match.group("column"))) is None
        or match.group("type").upper() not in WIDENING.get(known.prisma_type, frozenset())
    ]
    if narrowing:
        return site.rate(
            BREAKING,
            f"Changes {', '.join(narrowing)}, which the previous version's client cannot be shown to read. Its "
            "prepared plans are rejected once, and the retry may still fail to map the value.",
        )
    return site.rate(
        PREPARED_PLAN,
        "Widens a column's type. Every prepared query on the previous version that returns it is rejected with "
        "`cached plan must not change result type` until its connection is recreated, then reads it fine.",
    )


def _rule_constraint(site: _Site) -> Finding | None:
    if _ADD_CONSTRAINT.search(site.statement) is not None:
        if site.context.schema is None:
            return site.rate(
                WRITE_REJECT,
                "Adds a rule the previous version never enforced. Rows it writes that violate the rule are now "
                "rejected; whether any do depends on the data.",
            )
        checked: Final = [
            column for column in _constraint_columns(site.statement) if site.context.writes(site.table, column)
        ]
        if checked:
            return site.rate(
                WRITE_REJECT,
                f"Adds a rule on `{'`, `'.join(checked)}` that the previous version never enforced. Rows it writes "
                "that violate the rule are now rejected; whether any do depends on the data.",
            )
        if _NOT_VALID.search(site.statement) is not None:
            return site.rate(
                INFO, "Constraint on columns the previous version does not write, with no validation scan."
            )
        return site.rate(LOCK, "Validates the constraint against every existing row while holding the table lock.")
    if _UNIQUE_INDEX.search(site.statement) is None:
        return None
    covered: Final = _index_columns(site.statement)
    if not covered or not all(site.context.writes(site.table, column) for column in covered):
        return None
    if site.context.replaced_key(site.table, covered):
        return None  # the key it replaces already rejected everything this one does
    return site.rate(
        WRITE_REJECT,
        f"Makes `{'`, `'.join(covered)}` unique, which the previous version never enforced. Duplicate rows it "
        "writes are now rejected; whether any are depends on the data.",
    )


def _rule_whole_row_read(site: _Site) -> Finding | None:
    if _ADD_COLUMN.search(site.statement) is None or site.star is None:
        return None
    return site.rate(
        PREPARED_PLAN,
        f"Read whole-row by `{_shape(site.star)}`{site.where}, so the prepared plans cached on the previous "
        "version's pooled connections are rejected with `cached plan must not change result type` until those "
        "connections are recreated.",
    )


def _rule_lock(site: _Site) -> Finding | None:
    if _CREATE_INDEX.search(site.statement) is not None and _CONCURRENTLY.search(site.statement) is None:
        joined: Final = _joined_read_for(site.table, site.context.star_reads)
        read_note: Final = (
            f" The table is joined by `{_shape(joined)}` (`{joined.location}`), so that read waits too."
            if joined is not None
            else ""
        )
        return site.rate(
            LOCK,
            "Blocks writes to the table (reads continue) until the index is built; the wait scales with the "
            f"table's size.{read_note}",
        )
    if _UPDATE.search(site.statement) is not None:
        return site.rate(LOCK, "Rewrites existing rows, so it holds row locks for as long as the backfill runs.")
    return None


def _rule_rest(site: _Site) -> Finding:
    if _ADD_COLUMN.search(site.statement) is not None:
        return site.rate(INFO, "Adds a nullable column to a table the previous version does not read whole-row.")
    if _INSERT.search(site.statement) is not None:
        return site.rate(INFO, "Inserts rows. The previous version is not affected.")
    return site.rate(INFO, "No effect on the previous version.")


# Worst-first: the first rule that has something to say about a statement decides its rating.
RULES: Final = (
    _rule_new_table,
    _rule_drop,
    _rule_rename,
    _rule_not_null,
    _rule_type_change,
    _rule_constraint,
    _rule_whole_row_read,
    _rule_lock,
)


def classify(migration: str, statement: str, context: UpgradeContext) -> Finding:
    """Rate one statement by what it does to a pod still running the base version."""
    table: Final = statement_table(statement)
    site: Final = _Site(migration, statement, table, context, _star_read_for(table, context.star_reads))
    for rule in RULES:
        found: Final = rule(site)
        if found is not None:
            return found
    return _rule_rest(site)


def _adds_not_null_without_default(statement: str) -> bool:
    """True when any `ADD COLUMN` clause of the statement is NOT NULL and has no DEFAULT of its own."""
    return any(
        _NOT_NULL.search(clause) is not None and _DEFAULT.search(clause) is None
        for clause in _CLAUSE_SPLIT.split(statement)
        if _ADD_COLUMN.search(clause) is not None
    )


def _context_for(
    statements: Sequence[str],
    fresh_tables: frozenset[str],
    star_reads: tuple[StarRead, ...],
    schema: Schema | None,
    defaulted: frozenset[tuple[str, str]],
    base_unique: Mapping[str, tuple[str, tuple[str, ...]]],
) -> UpgradeContext:
    dropped: Final = frozenset(
        match.group("index") for statement in statements if (match := _DROP_INDEX_NAME.search(statement))
    )
    return UpgradeContext(fresh_tables, star_reads, schema, dropped, defaulted, base_unique)


def build_report(repo: Path, base: str, head: str) -> Report:
    base_names: Final = frozenset(migration_names(repo, base))
    new_names: Final = tuple(name for name in migration_names(repo, head) if name not in base_names)
    # The plans that break are the ones the old pods hold, so the queries that matter are the base version's.
    star_reads: Final = discover_star_reads(repo, base)

    statements_by_migration: Final = MappingProxyType(
        {name: split_statements(sql) for name in new_names if (sql := read_migration(repo, head, name)) is not None}
    )
    created_here: Final = frozenset(
        match.group("table")
        for statements in statements_by_migration.values()
        for statement in statements
        if (match := _CREATE_TABLE.search(statement)) is not None
    )
    # A re-declared `CREATE TABLE IF NOT EXISTS` of a table the old pods already use is not a new table.
    fresh_tables: Final = created_here - tables_created_before(repo, base)
    schema: Final = load_base_schema(repo, base)
    defaulted: Final = _defaulted_columns(s for statements in statements_by_migration.values() for s in statements)
    base_unique: Final = unique_indexes_at(repo, base)

    findings: Final = tuple(
        classify(name, statement, _context_for(statements, fresh_tables, star_reads, schema, defaulted, base_unique))
        for name, statements in statements_by_migration.items()
        for statement in statements
    )
    ranked: Final = sorted(findings, key=lambda finding: (SEVERITY_ORDER.index(finding.severity), finding.migration))
    return Report(
        base=base,
        head=head,
        migrations=new_names,
        findings=tuple(ranked),
        star_reads=star_reads,
        schema_tables=None if schema is None else len(schema),
    )


PROCEDURE: Final = MappingProxyType(
    {
        BREAKING: (
            "**Do not run the old and new versions side by side.** Take the previous version out of service before "
            "the new version applies these migrations, or split the change across two releases."
        ),
        WRITE_REJECT: (
            "Writes from the previous version that break the new rule are rejected until those pods are gone. Confirm "
            "the previous version already satisfies it on your data; if you cannot, do not overlap the two versions."
        ),
        PREPARED_PLAN: (
            "Expect the queries cited above to fail on pods still running the previous version until their pooled "
            "connections are recreated. LiteLLM recreates them on the first such failure as of `1.98.0`; on older "
            "versions, or to avoid the failures entirely, set `general_settings.database_disable_prepared_statements: "
            "true` before upgrading, or drain the old pods before the new ones migrate."
        ),
        LOCK: "Apply these migrations in a maintenance window, or during a traffic trough, and watch for lock waits.",
        INFO: "Safe to apply during a rolling upgrade.",
    }
)


def _cited_reads(report: Report) -> tuple[StarRead, ...]:
    """The whole-row reads that actually bear on this release, one per table."""
    tables: Final = {finding.table for finding in report.findings if finding.severity != INFO}
    representatives: Final = (_star_read_for(table, report.star_reads) for table in sorted(tables))
    return tuple(read for read in representatives if read is not None)


def _evidence_line(report: Report) -> str:
    schema_note: Final = (
        f"no `schema.prisma` at `{report.base}`, so drops, renames and constraints are rated as if the previous "
        "version used every column"
        if report.schema_tables is None
        else f"columns the previous version uses: `schema.prisma` at `{report.base}` ({report.schema_tables} tables)"
    )
    cited: Final = _cited_reads(report)
    shapes: Final = ", ".join(sorted({f"`{_shape(read)}` on `{read.table}` (`{read.location}`)" for read in cited}))
    reads_note: Final = (f"whole-row reads: {shapes}",) if cited else ()
    return f"<sub>Evidence: {'; '.join((schema_note, *reads_note))}.</sub>"


def render_markdown(report: Report) -> str:
    if not report.migrations:
        return f"## Database Schema Changes\n\nNo database schema changes since `{report.base}`.\n"

    count: Final = len(report.migrations)
    notable: Final = tuple(finding for finding in report.findings if finding.severity != INFO)
    rows: Final = tuple(
        f"| **{SEVERITY_LABEL[finding.severity]}** | `{finding.migration}` | `{_clip(finding.statement)}` | "
        f"{finding.effect} |"
        for finding in notable[:MAX_ROWS]
    )
    overflow: Final = (
        (
            f"| | | | ...and {len(notable) - MAX_ROWS} more, worst first. Run "
            f"`python3 ci_cd/migration_impact.py --base {report.base} --head {report.head}` for the full list. |",
        )
        if len(notable) > MAX_ROWS
        else ()
    )
    worst: Final = report.worst
    quiet: Final = (
        ("| **INFO** | - | - | Nothing in this release affects a pod running the previous version. |",)
        if worst is None or worst == INFO
        else ()
    )
    lines: Final = (
        "## Database Schema Changes",
        "",
        f"This release adds **{count} migration{'' if count == 1 else 's'}** to `litellm-proxy-extras`, applied by "
        f"the first pod that boots (`prisma migrate deploy`). Rated against a pod still running `{report.base}`:",
        "",
        "| | Migration | Statement | Effect on pods still on the previous version |",
        "|:--:|:--|:--|:--|",
        *rows,
        *overflow,
        *quiet,
        "",
        PROCEDURE[worst or INFO],
        "",
        _evidence_line(report),
        "",
    )
    return "\n".join(lines)


_DDL_START: Final = re.compile(
    r"\b(?:ALTER\s+TABLE|CREATE\s+(?:UNIQUE\s+)?INDEX|CREATE\s+TABLE|DROP\s+TABLE|DROP\s+INDEX|UPDATE|INSERT\s+INTO)\b",
    re.IGNORECASE,
)
MAX_ROWS: Final = 20


def _clip(statement: str, limit: int = 90) -> str:
    """Render a statement for the table: the DDL itself, clipped, without its `DO $$` preamble."""
    match: Final = _DDL_START.search(statement)
    ddl: Final = statement[match.start() :] if match is not None else statement
    clipped: Final = ddl if len(ddl) <= limit else f"{ddl[: limit - 1]}…"
    return clipped.replace("|", "\\|")


def render_json(report: Report) -> str:
    payload: Final = {
        "base": report.base,
        "head": report.head,
        "worst_severity": report.worst,
        "schema_tables": report.schema_tables,
        "migrations": list(report.migrations),
        "findings": [finding.as_dict() for finding in report.findings],
        "star_reads": [
            {"table": read.table, "alias": read.alias, "joined": list(read.joined), "location": read.location}
            for read in report.star_reads
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def default_base(repo: Path, head: str) -> str:
    """The newest stable release reachable from `head`, which is what an operator upgrades from.

    Pre-release tags are skipped: a release cut on top of `v1.102.0-rc.2` still has to tell the
    operator what changed since the last version they could have been running.
    """
    # Version order, not date order: a backport release branch reaches only its own line's tags,
    # and two tags cut in the same second sort arbitrarily by date.
    listing: Final = _git_or_none(repo, "tag", "--merged", head, "--sort=-v:refname")
    head_sha: Final = (_git_or_none(repo, "rev-parse", head) or "").strip()
    for line in (listing or "").splitlines():
        tag = line.strip()
        if not _STABLE_TAG.match(tag):
            continue
        tagged = (_git_or_none(repo, "rev-list", "-n", "1", tag) or "").strip()
        if tagged and tagged != head_sha:
            return tag
    described: Final = _git_or_none(repo, "describe", "--tags", "--abbrev=0", f"{head}^")
    if described is None or not described.strip():
        raise GitError(f"found no release tag before {head}; pass --base explicitly (the checkout may lack tags)")
    return described.strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser: Final = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="ref of the version being upgraded FROM (default: the tag before --head)")
    parser.add_argument("--head", default="HEAD", help="ref of the version being upgraded TO (default: HEAD)")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]), help="path to the repository")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", help="write the report here instead of stdout")
    parser.add_argument(
        "--fail-on",
        choices=(BREAKING, WRITE_REJECT, PREPARED_PLAN, LOCK),
        help="exit 1 when a finding at this severity or worse is present",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args: Final = parse_args(argv)
    repo: Final = Path(args.repo)
    try:
        base: Final = args.base or default_base(repo, args.head)
        report: Final = build_report(repo, base, args.head)
    except GitError as error:
        sys.stderr.write(f"migration impact report skipped: {error}\n")
        return 2

    rendered: Final = render_markdown(report) if args.format == "markdown" else render_json(report)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    worst: Final = report.worst
    if args.fail_on and worst is not None and SEVERITY_ORDER.index(worst) <= SEVERITY_ORDER.index(args.fail_on):
        sys.stderr.write(f"migration impact: found {worst} changes between {base} and {args.head}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
