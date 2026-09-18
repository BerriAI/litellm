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

- `breaking`       — the old pods cannot recover on their own: a column they read is gone, renamed
                     or retyped, or a constraint now rejects the rows they write
- `prepared-plan`  — the change alters the result type of a query the old pods have prepared
                     (a `SELECT <alias>.*` gaining a column, or a column changing type), so the
                     plans cached on their pooled connections are rejected with
                     `cached plan must not change result type` until those connections are
                     recreated
- `lock`           — the migration takes a blocking lock (or rewrites rows) on a table that is
                     already serving traffic
- `info`           — no effect on the old pods (new tables, and changes to them)

The tables that are read through `SELECT <alias>.*` are discovered from the source tree at `--base`
(the code the old pods are running) rather than hardcoded, so the report keeps working as those
queries move.

    python3 ci_cd/migration_impact.py --base v1.93.0 --head v1.99.0

Standard library only, so it runs on a bare checkout with no `uv sync`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MIGRATIONS_DIR: Final = "litellm-proxy-extras/litellm_proxy_extras/migrations"
# git pathspec wildcards cross directory boundaries, so these reach every file under each tree.
SOURCE_GLOBS: Final = ("litellm/*.py", "enterprise/*.py")
TABLE_MARKER: Final = 'FROM "'
# `*` inside a view is expanded when the view is created, so it holds no per-connection plan.
VIEW_MARKER: Final = re.compile(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?VIEW\b", re.IGNORECASE)

BREAKING: Final = "breaking"
PREPARED_PLAN: Final = "prepared-plan"
LOCK: Final = "lock"
INFO: Final = "info"
SEVERITY_ORDER: Final = (BREAKING, PREPARED_PLAN, LOCK, INFO)
SEVERITY_ICON: Final = {BREAKING: "🚨", PREPARED_PLAN: "⚠️", LOCK: "🔒", INFO: "ℹ️"}

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


def split_statements(sql: str) -> list[str]:
    """Split a migration into single-line statements, descending into `DO $$ ... $$` bodies."""
    cleaned: Final = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))
    bodies: Final[list[str]] = []

    def stash(match: re.Match[str]) -> str:
        bodies.append(match.group("body"))
        return " "

    outer: Final = _DOLLAR_QUOTED.sub(stash, cleaned)
    statements: Final = [normalize(part) for part in outer.split(";")]
    for body in bodies:
        statements.extend(split_statements(body))
    return [statement for statement in statements if statement]


def statement_table(statement: str) -> str:
    for pattern in (_CREATE_TABLE, _ALTER_TABLE, _DROP_TABLE, _CREATE_INDEX, _UPDATE, _INSERT):
        match: Final = pattern.search(statement)
        if match is not None:
            return match.group("table")
    return ""


def migration_names(repo: Path, ref: str) -> tuple[str, ...]:
    """The migration directories at `ref`, skipping `migration_lock.toml` and any stray file."""
    listing: Final = _git(repo, "ls-tree", ref, f"{MIGRATIONS_DIR}/")
    names: Final[list[str]] = []
    for line in listing.splitlines():
        # <mode> <type> <object>\t<path>
        meta, _, path = line.partition("\t")
        if meta.split()[1:2] == ["tree"]:
            names.append(Path(path).name)
    return tuple(sorted(names))


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


def _alias_map(body: str) -> tuple[dict[str, str], tuple[str, ...]]:
    aliases: Final[dict[str, str]] = {}
    tables: Final[list[str]] = []
    for match in _FROM_JOIN.finditer(body):
        table: Final = match.group("table")
        if table not in tables:
            tables.append(table)
        alias: Final = match.group("alias")
        if alias is not None and alias.lower() not in ALIAS_STOPWORDS:
            aliases[alias] = table
    return aliases, tuple(tables)


def star_reads_in_source(text: str, path: str) -> list[StarRead]:
    """Find whole-row reads (`SELECT v.*`, `SELECT *`) in the SQL literals of one source file."""
    found: Final[list[StarRead]] = []
    for literal in _SQL_LITERAL.finditer(text):
        raw: Final = literal.group("body")
        if TABLE_MARKER not in raw or "SELECT" not in raw.upper():
            continue
        if VIEW_MARKER.search(raw) is not None:
            continue
        # `-- Added comma to separate b.* columns` is a comment, not a read.
        body: Final = _LINE_COMMENT.sub(" ", raw)
        aliases, tables = _alias_map(body)
        line: Final = text.count("\n", 0, literal.start()) + 1
        location: Final = f"{path}:{line}"
        starred: Final[list[tuple[str, str]]] = [
            (aliases[match.group("alias")], match.group("alias"))
            for match in _STAR_ALIAS.finditer(body)
            if match.group("alias") in aliases
        ]
        if not starred and _BARE_STAR.search(body) is not None and len(tables) == 1:
            starred.append((tables[0], "*"))
        for table, alias in dict.fromkeys(starred):
            found.append(
                StarRead(
                    table=table,
                    alias=alias,
                    joined=tuple(other for other in tables if other != table),
                    location=location,
                )
            )
    return found


def discover_star_reads(repo: Path, ref: str) -> tuple[StarRead, ...]:
    listing: Final = _git_or_none(repo, "grep", "-l", "--fixed-strings", TABLE_MARKER, ref, "--", *SOURCE_GLOBS)
    if listing is None:
        return ()
    found: Final[list[StarRead]] = []
    for line in listing.splitlines():
        path: Final = line.split(":", 1)[1] if ":" in line else line
        text: Final = _git_or_none(repo, "show", f"{ref}:{path}")
        if text is not None:
            found.extend(star_reads_in_source(text, path))
    return tuple(found)


def _shape(read: StarRead) -> str:
    return "SELECT *" if read.alias == "*" else f"SELECT {read.alias}.*"


def _star_read_for(table: str, star_reads: Sequence[StarRead]) -> StarRead | None:
    matches: Final = [read for read in star_reads if read.table == table]
    return max(matches, key=lambda read: len(read.joined)) if matches else None


def _joined_read_for(table: str, star_reads: Sequence[StarRead]) -> StarRead | None:
    return next((read for read in star_reads if table in read.joined), None)


def classify(
    migration: str,
    statement: str,
    fresh_tables: frozenset[str],
    star_reads: Sequence[StarRead],
) -> Finding:
    """Rate one statement by what it does to a pod still running the base version."""
    table: Final = statement_table(statement)

    def finding(severity: str, effect: str) -> Finding:
        return Finding(severity=severity, migration=migration, table=table, statement=statement, effect=effect)

    if _CREATE_TABLE.search(statement) is not None:
        return finding(INFO, "New table. The previous version does not read or write it.")
    if table and table in fresh_tables:
        return finding(INFO, "Table is created by this same upgrade, so no old pod reads or writes it yet.")

    star: Final = _star_read_for(table, star_reads)
    where: Final = f" (`{star.location}`)" if star is not None else ""

    if _DROP_COLUMN.search(statement) is not None or _DROP_TABLE.search(statement) is not None:
        return finding(
            BREAKING,
            f"Removes something the previous version still reads{where}. Old pods keep failing until they are gone; "
            "reconnecting does not help.",
        )
    if _ALTER_TABLE.search(statement) is not None and _RENAME.search(statement) is not None:
        return finding(
            BREAKING,
            f"Renames something the previous version still addresses by its old name{where}. Old pods cannot "
            "recover on their own.",
        )
    if _SET_NOT_NULL.search(statement) is not None:
        return finding(BREAKING, "Rows the previous version writes without this column are now rejected.")
    if _ADD_COLUMN.search(statement) is not None and _adds_not_null_without_default(statement):
        return finding(BREAKING, "Adds a NOT NULL column with no default, so inserts from the previous version fail.")
    if _ALTER_TYPE.search(statement) is not None:
        return finding(
            PREPARED_PLAN,
            "Changes a column's type, so every prepared query on the previous version that returns it is rejected "
            "with `cached plan must not change result type` until its connection is recreated. If the new type "
            "is one the old client cannot map, it keeps failing after that.",
        )
    if _ADD_COLUMN.search(statement) is not None and star is not None:
        return finding(
            PREPARED_PLAN,
            f"Read whole-row by `{_shape(star)}`{where}, so the prepared plans cached on the previous version's "
            "pooled connections are rejected with `cached plan must not change result type` until those "
            "connections are recreated.",
        )
    if _CREATE_INDEX.search(statement) is not None and _CONCURRENTLY.search(statement) is None:
        joined: Final = _joined_read_for(table, star_reads)
        read_note: Final = (
            f" The table is joined by `{_shape(joined)}` (`{joined.location}`), so that read waits too."
            if joined is not None
            else ""
        )
        return finding(
            LOCK,
            "Blocks writes to the table (reads continue) until the index is built; the wait scales with the "
            f"table's size.{read_note}",
        )
    if _ADD_CONSTRAINT.search(statement) is not None and _NOT_VALID.search(statement) is None:
        return finding(LOCK, "Validates the constraint against every existing row while holding the table lock.")
    if _UPDATE.search(statement) is not None:
        return finding(LOCK, "Rewrites existing rows, so it holds row locks for as long as the backfill runs.")
    if _ADD_COLUMN.search(statement) is not None:
        return finding(INFO, "Adds a nullable column to a table the previous version does not read whole-row.")
    if _INSERT.search(statement) is not None:
        return finding(INFO, "Inserts rows. The previous version is not affected.")
    return finding(INFO, "No effect on the previous version.")


def _adds_not_null_without_default(statement: str) -> bool:
    """True when any `ADD COLUMN` clause of the statement is NOT NULL and has no DEFAULT of its own."""
    return any(
        _NOT_NULL.search(clause) is not None and _DEFAULT.search(clause) is None
        for clause in _CLAUSE_SPLIT.split(statement)
        if _ADD_COLUMN.search(clause) is not None
    )


def build_report(repo: Path, base: str, head: str) -> Report:
    base_names: Final = set(migration_names(repo, base))
    head_names: Final = migration_names(repo, head)
    new_names: Final = tuple(name for name in head_names if name not in base_names)
    # The plans that break are the ones the old pods hold, so the queries that matter are the base version's.
    star_reads: Final = discover_star_reads(repo, base)

    sql_by_migration: Final = {name: sql for name in new_names if (sql := read_migration(repo, head, name)) is not None}
    created_here: Final = frozenset(
        match.group("table")
        for sql in sql_by_migration.values()
        for statement in split_statements(sql)
        if (match := _CREATE_TABLE.search(statement)) is not None
    )
    # A re-declared `CREATE TABLE IF NOT EXISTS` of a table the old pods already use is not a new table.
    fresh_tables: Final = created_here - tables_created_before(repo, base)

    findings: Final[list[Finding]] = []
    for name, sql in sql_by_migration.items():
        for statement in split_statements(sql):
            findings.append(classify(name, statement, fresh_tables, star_reads))

    ranked: Final = sorted(findings, key=lambda finding: (SEVERITY_ORDER.index(finding.severity), finding.migration))
    return Report(base=base, head=head, migrations=new_names, findings=tuple(ranked), star_reads=star_reads)


PROCEDURE: Final = {
    BREAKING: (
        "**Do not run the old and new versions side by side.** Take the previous version out of service before the "
        "new version applies these migrations, or split the change across two releases."
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


def _cited_reads(report: Report) -> tuple[StarRead, ...]:
    """The whole-row reads that actually bear on this release, one per table."""
    tables: Final = {finding.table for finding in report.findings if finding.severity != INFO}
    representatives: Final = (_star_read_for(table, report.star_reads) for table in sorted(tables))
    return tuple(read for read in representatives if read is not None)


def render_markdown(report: Report) -> str:
    lines: Final[list[str]] = ["## Database Schema Changes", ""]
    if not report.migrations:
        lines.append(f"No database schema changes since `{report.base}`.")
        return "\n".join(lines) + "\n"

    count: Final = len(report.migrations)
    plural: Final = "" if count == 1 else "s"
    lines.extend(
        [
            f"This release adds **{count} migration{plural}** to `litellm-proxy-extras`, applied by the first pod "
            f"that boots (`prisma migrate deploy`). Rated against a pod still running `{report.base}`:",
            "",
            "| | Migration | Statement | Effect on pods still on the previous version |",
            "|:--:|:--|:--|:--|",
        ]
    )
    notable: Final = [finding for finding in report.findings if finding.severity != INFO]
    for finding in notable[:MAX_ROWS]:
        lines.append(
            f"| {SEVERITY_ICON[finding.severity]} | `{finding.migration}` | `{_clip(finding.statement)}` | "
            f"{finding.effect} |"
        )
    if len(notable) > MAX_ROWS:
        lines.append(
            f"| | | | …and {len(notable) - MAX_ROWS} more, worst first. Run "
            f"`python3 ci_cd/migration_impact.py --base {report.base} --head {report.head}` for the full list. |"
        )

    worst: Final = report.worst
    if worst is None or worst == INFO:
        lines.append("| ℹ️ | — | — | Nothing in this release affects a pod running the previous version. |")
    lines.extend(["", PROCEDURE[worst or INFO], ""])
    cited: Final = _cited_reads(report)
    if cited:
        shapes: Final = ", ".join(sorted({f"`{_shape(read)}` on `{read.table}` (`{read.location}`)" for read in cited}))
        lines.append(f"<sub>Whole-row reads considered: {shapes}.</sub>")
    lines.append("")
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
        choices=(BREAKING, PREPARED_PLAN, LOCK),
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
