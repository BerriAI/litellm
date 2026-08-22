#!/usr/bin/env python3
"""Ban row-rewriting DML from Prisma migrations.

Migrations run synchronously at proxy boot, before the process serves traffic, so
anything whose cost scales with existing table size turns into downtime. A single
`UPDATE` with no batching over a spend-log-sized table is minutes of unavailability
plus a doubled heap that plain autovacuum will not give back.

Flagged, per statement, by its leading keyword:

  UPDATE      rewrites every matching row, and `WHERE` does not bound the scan
  DELETE      same scan, and the dead tuples outlive the migration
  MERGE       both of the above in one statement
  INSERT      only when it draws rows from a `SELECT`; `INSERT ... VALUES` is bounded
              by the literal row list and passes, scalar subqueries in that list
              included
  WITH        a CTE-led statement containing any of the above

Referential actions (`ON DELETE CASCADE`, `ON UPDATE CASCADE`) are schema, never a
statement's leading keyword, so they pass.

Statements inside dollar-quoted bodies are scanned too. `DO $$ ... $$` is this
repo's idiom for conditional DDL, so a body is where an `UPDATE` would otherwise
hide. The SQL an `EXECUTE` runs is scanned the same way, since a rewrite reads the
same to Postgres whether it is spelled out or handed over as a string.

Line numbers always count against the whole migration file, however deeply the
statement is nested, so a reported line points at the statement and the markers
below line up with the statements they exempt.

Add a column and let the application populate it, or run the rewrite as an opt-in
batched job outside boot. When a rewrite is genuinely bounded and must ship inside
the migration, put `-- data-migration-ok: <reason>` on the statement, naming what
bounds it. The reason is required.

`GRANDFATHERED` freezes the violations that predate this check. Prisma records a
checksum for every applied migration and this repo treats applied files as
immutable, so those two cannot take an inline marker. The set is closed; a new
migration belongs nowhere in it.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"

GRANDFATHERED = frozenset(
    {
        "20260817000000_shadow_eval_multi_key",
        "20260818224500_add_shadow_eval_stopped_by",
    }
)

MARKER = re.compile(r"--[ \t]*data-migration-ok:[ \t]*(\S.*?)[ \t]*$", re.MULTILINE)
DOLLAR_TAG = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
FIRST_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STATEMENT = re.compile(r"[^;]+")

REWRITES_ROWS = frozenset({"UPDATE", "DELETE", "MERGE"})

STATEMENT_KEYWORDS = REWRITES_ROWS | frozenset(
    {
        "INSERT",
        "SELECT",
        "WITH",
        "ALTER",
        "CREATE",
        "DROP",
        "TRUNCATE",
        "COMMENT",
        "GRANT",
        "REVOKE",
        "COPY",
        "SET",
        "PERFORM",
        "RAISE",
        "RETURN",
        "EXECUTE",
        "CALL",
        "REINDEX",
        "REFRESH",
        "VACUUM",
        "ANALYZE",
    }
)

GUIDANCE = """
Migrations apply at proxy boot, before it serves traffic, so a statement whose cost
scales with table size is downtime. Add the column and let the application backfill
it, or move the rewrite to a batched job outside boot.

If the rewrite is genuinely bounded and has to ship in the migration, mark the
statement with the bound spelled out:

    -- data-migration-ok: <what bounds this>
    UPDATE ...
"""


@dataclass(frozen=True, slots=True)
class Violation:
    migration: str
    line: int
    keyword: str

    def render(self) -> str:
        location = f"{MIGRATIONS_DIR.relative_to(REPO_ROOT)}/{self.migration}/migration.sql"
        return f"{location}:{self.line}: {self.keyword} rewrites existing rows at boot"


def blank(text: str) -> str:
    return "".join(character if character == "\n" else " " for character in text)


def mask(sql: str) -> tuple[str, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Blank comments and quoted text, keeping offsets, and locate the spans that can still
    hold SQL: dollar-quoted bodies, and the single-quoted literals `EXECUTE` runs."""
    chunks: list[str] = []
    bodies: list[tuple[int, int]] = []
    literals: list[tuple[int, int]] = []
    index = 0
    length = len(sql)

    while index < length:
        pair = sql[index : index + 2]

        if pair == "--":
            stop = sql.find("\n", index)
            stop = length if stop == -1 else stop
            chunks.append(blank(sql[index:stop]))
            index = stop
            continue

        if pair == "/*":
            stop = skip_block_comment(sql, index)
            chunks.append(blank(sql[index:stop]))
            index = stop
            continue

        character = sql[index]

        if character in "'\"":
            stop = skip_quoted(sql, index, character)
            if character == "'":
                closed = sql[stop - 1 : stop] == character
                literals.append((index + 1, max(index + 1, stop - 1 if closed else stop)))
            chunks.append(blank(sql[index:stop]))
            index = stop
            continue

        if character == "$":
            tag = DOLLAR_TAG.match(sql, index)
            if tag is not None:
                closing = sql.find(tag.group(), tag.end())
                body_end = length if closing == -1 else closing
                stop = length if closing == -1 else closing + len(tag.group())
                bodies.append((tag.end(), body_end))
                chunks.append(blank(sql[index:stop]))
                index = stop
                continue

        chunks.append(character)
        index += 1

    return "".join(chunks), tuple(bodies), tuple(literals)


def skip_block_comment(sql: str, start: int) -> int:
    depth = 1
    index = start + 2
    while index < len(sql) and depth > 0:
        pair = sql[index : index + 2]
        if pair == "/*":
            depth += 1
            index += 2
        elif pair == "*/":
            depth -= 1
            index += 2
        else:
            index += 1
    return index


def skip_quoted(sql: str, start: int, quote: str) -> int:
    """One quoted run, up to and including its closing quote. A doubled quote needs no
    special case: closing on the first and reopening on the second masks the same span."""
    stop = sql.find(quote, start + 1)
    return len(sql) if stop == -1 else stop + 1


def strip_parens(statement: str) -> str:
    """Blank parenthesised groups in place, so an `IF EXISTS (SELECT ...)` guard does not
    stand in for the statement it guards."""
    chunks: list[str] = []
    depth = 0

    for character in statement:
        if character == "(":
            depth += 1
            chunks.append(" ")
        elif character == ")":
            depth = max(depth - 1, 0)
            chunks.append(" ")
        elif depth > 0 and character != "\n":
            chunks.append(" ")
        else:
            chunks.append(character)

    return "".join(chunks)


def leading_keyword(statement: str) -> re.Match[str] | None:
    """The statement's own keyword, looking past PL/pgSQL block syntax such as
    `BEGIN`, `IF ... THEN` and `END`."""
    return next(
        (word for word in FIRST_WORD.finditer(statement) if word.group().upper() in STATEMENT_KEYWORDS),
        None,
    )


def offending_keyword(statement: str) -> str | None:
    word = leading_keyword(strip_parens(statement))
    if word is None:
        return None

    keyword = word.group().upper()

    if keyword in REWRITES_ROWS:
        return keyword

    if keyword == "INSERT":
        return "INSERT ... SELECT" if draws_rows_from_a_select(statement) else None

    if keyword == "WITH":
        nested = next((name for name in sorted(REWRITES_ROWS) if contains(statement, name)), None)
        if nested is not None:
            return f"WITH ... {nested}"
        if contains(statement, "INSERT") and draws_rows_from_a_select(statement):
            return "WITH ... INSERT ... SELECT"

    return None


def draws_rows_from_a_select(statement: str) -> bool:
    """Whether an `INSERT` takes its rows from a query rather than a literal list. A
    top-level `VALUES` bounds the insert to the rows written out there, so the scalar
    subqueries and helper CTEs that sit in parentheses around it do not make it a
    rewrite."""
    return contains(statement, "SELECT") and not contains(strip_parens(statement), "VALUES")


def leads_with(statement: str, keyword: str) -> bool:
    word = leading_keyword(strip_parens(statement))
    return word is not None and word.group().upper() == keyword


def contains(statement: str, keyword: str) -> bool:
    return re.search(rf"\b{keyword}\b", statement, re.IGNORECASE) is not None


def exempt_lines(sql: str) -> frozenset[int]:
    return frozenset(sql.count("\n", 0, match.start()) + 1 for match in MARKER.finditer(sql))


def scan(sql: str, migration: str, exempt: frozenset[int]) -> Iterator[Violation]:
    yield from scan_region(sql, sql, migration, exempt, 0)


def scan_region(
    document: str, region: str, migration: str, exempt: frozenset[int], offset: int
) -> Iterator[Violation]:
    """Violations in one region of `document`, whose text begins at `offset`. Lines are
    always counted against the whole document, so a statement nested in a dollar-quoted
    body reports its real file line and lines up with the markers read from that file."""
    masked, bodies, literals = mask(region)

    for match in STATEMENT.finditer(masked):
        if leads_with(match.group(), "EXECUTE"):
            for start, end in literals:
                if match.start() <= start and end <= match.end():
                    yield from scan_region(document, region[start:end], migration, exempt, offset + start)
            continue
        keyword = offending_keyword(match.group())
        if keyword is None:
            continue
        first = line_of(document, offset + keyword_start(match))
        last = line_of(document, offset + match.end())
        if any(line in exempt for line in range(first - 1, last + 1)):
            continue
        yield Violation(migration, first, keyword)

    for start, end in bodies:
        yield from scan_region(document, region[start:end], migration, exempt, offset + start)


def keyword_start(statement: re.Match[str]) -> int:
    word = leading_keyword(strip_parens(statement.group()))
    return statement.start() + (0 if word is None else word.start())


def line_of(sql: str, offset: int) -> int:
    return sql.count("\n", 0, offset) + 1


def scan_migration(directory: Path) -> tuple[Violation, ...]:
    sql = (directory / "migration.sql").read_text(encoding="utf-8")
    return tuple(scan(sql, directory.name, exempt_lines(sql)))


def stale_grandfathers(found: Mapping[str, tuple[Violation, ...]]) -> tuple[str, ...]:
    clean = (name for name in GRANDFATHERED & found.keys() if not found[name])
    missing = GRANDFATHERED - found.keys()
    return tuple(sorted((*clean, *missing)))


def main() -> int:
    if not MIGRATIONS_DIR.is_dir():
        print(f"migrations directory not found: {MIGRATIONS_DIR}", file=sys.stderr)
        return 2

    directories = tuple(sorted(path for path in MIGRATIONS_DIR.iterdir() if (path / "migration.sql").is_file()))
    found = {directory.name: scan_migration(directory) for directory in directories}
    violations = tuple(
        violation for name, results in found.items() if name not in GRANDFATHERED for violation in results
    )

    for violation in violations:
        print(violation.render())

    stale = stale_grandfathers(found)
    for name in stale:
        print(f"{name}: listed in GRANDFATHERED but no longer violates; remove it from the set")

    if violations:
        print(f"\n{len(violations)} data-rewriting statement(s) in migrations.")
        print(GUIDANCE)

    if violations or stale:
        return 1

    print(f"No data-rewriting statements in {len(directories)} migrations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
