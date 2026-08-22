#!/usr/bin/env python3
"""Ban row-rewriting DML from Prisma migrations.

Migrations run synchronously at proxy boot, before the process serves traffic, so
anything whose cost scales with existing table size turns into downtime. A single
`UPDATE` with no batching over a spend-log-sized table is minutes of unavailability
plus a doubled heap that plain autovacuum will not give back.

What is banned is the row-rewriting DML behind that, not everything whose cost
scales that way. A non-concurrent `CREATE INDEX`, an `ALTER COLUMN ... TYPE` that is
not binary coercible, and a volatile `DEFAULT` on a new column all read the whole
table and all pass. That is deliberate: a rule wide enough to reach them fires on
most ordinary migrations, and a marker everyone adds by reflex stops carrying
information. The outage this was written for was a backfill.

Flagged, per statement, by its leading keyword:

  UPDATE      rewrites every matching row, and `WHERE` does not bound the scan
  DELETE      same scan, and the dead tuples outlive the migration
  MERGE       both of the above in one statement
  INSERT      only when its rows come from a query rather than a literal `VALUES`
              list. The query counts wherever it sits, since Postgres takes it
              parenthesised, and `TABLE t` is one as much as a `SELECT` is. An
              insert bounded by a leading `VALUES` passes, scalar subqueries in that
              list included, while a `VALUES` reached through a subquery or joined
              to a query by a set operation bounds nothing
  WITH        a CTE-led statement containing any of the above

Referential actions (`ON DELETE CASCADE`, `ON UPDATE CASCADE`) are schema, never a
statement's leading keyword, so they pass.

A statement wrapped in `EXPLAIN` is judged on the statement itself, because the
`ANALYZE` form runs it rather than only planning it, and a rewrite left under one
rewrites the table on the way to printing its timings. Explaining a rewrite without
`ANALYZE` is flagged too: nothing here needs the plan of a statement it is being
told not to run at boot, and a marker is a cheap answer if one ever does.

Statements inside dollar-quoted bodies are scanned too. `DO $$ ... $$` is this
repo's idiom for conditional DDL, so a body is where an `UPDATE` would otherwise
hide. The SQL an `EXECUTE` runs is scanned the same way, since a rewrite reads the
same to Postgres whether it is spelled out or handed over as a string, and so is a
literal parked in a variable some `EXECUTE` in the same body then runs by name,
however it got there: an assignment with `:=`, the bare `=` PL/pgSQL takes as the
same operator, or a query returning it through `INTO`. So is the body of a `DO`
written in single quotes rather than dollar quotes. A literal nothing runs is text, however much it reads like a statement, so
an error message naming a `DELETE` the application handles stays a message.

Each literal is read on its own, so a keyword built by concatenating fragments that
do not contain it (`'UPD' || 'ATE ...'`) is not caught. Every fragment is scanned,
so a concatenation is caught wherever the keyword survives whole in one of them,
which covers `'UPDATE ' || quote_ident(t)` and the rest of the readable shapes. The
gap needs a keyword deliberately split down the middle, and this check is a guard
against a rewrite reaching a boot unnoticed, not a defence against someone hiding
one on purpose.

Line numbers always count against the whole migration file, however deeply the
statement is nested, so a reported line points at the statement and the markers
below line up with the statements they exempt.

Add a column and let the application populate it, or run the rewrite as an opt-in
batched job outside boot. When a rewrite is genuinely bounded and must ship inside
the migration, put `-- data-migration-ok: <reason>` on the statement or on the line
above it, naming what bounds it. The reason is required. A marker sharing a line
with the statement it follows exempts that statement alone, so the next statement
down is still checked rather than picking the marker up as its own. A marker on an
`EXECUTE` or on the assignment feeding one covers the single-quoted SQL that
statement hands off, so it goes where the migration reads rather than inside the
string. A dollar-quoted payload is not a string to this check but a region read like
any other body, so a rewrite inside one takes its marker on the rewrite itself. That
placement is deliberate rather than an oversight: a marker covering a whole body
would let one written for a `DO` block silence a rewrite added to that block later.

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
RUN_BY_NAME = re.compile(r"\bEXECUTE\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
INTO_TARGETS = re.compile(
    r"\bINTO\s+(?:STRICT\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)",
    re.IGNORECASE,
)
ASSIGNS = re.compile(r":=|(?<![<>!:=])=(?!=)")
PRECEDING_WORD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)[^A-Za-z0-9_]*$")
EXPLAIN_OPTIONS = re.compile(r"\bEXPLAIN\b(?:\s+(?:ANALYZE|ANALYSE|VERBOSE)\b)+", re.IGNORECASE)

REWRITES_ROWS = frozenset({"UPDATE", "DELETE", "MERGE"})

JOINS_QUERIES = ("UNION", "INTERSECT", "EXCEPT")

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
        "DO",
        "CALL",
        "REINDEX",
        "REFRESH",
        "VACUUM",
        "ANALYZE",
    }
)

GUARDS_A_CONDITION = frozenset({"IF", "ELSIF", "ELSEIF", "CASE", "WHEN", "WHILE", "EXIT", "ASSERT"})

OPENS_A_BLOCK = frozenset({"BEGIN", "THEN", "ELSE", "LOOP"})

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


@dataclass(frozen=True, slots=True)
class Markers:
    lines: frozenset[int]
    standalone: frozenset[int]

    def exempt(self, first: int, last: int) -> bool:
        """Whether a statement spanning `first` to `last` carries a marker. A marker alone on
        its line speaks for the statement below it, which is how one written above a rewrite
        exempts it. A marker sharing its line with the statement it follows speaks for that
        statement only, so the next statement down does not inherit the exemption."""
        return any(line in self.lines for line in range(first, last + 1)) or first - 1 in self.standalone


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


def strip_explain(statement: str) -> str:
    """Blank an `EXPLAIN` written with bare options, since the `ANALYZE` among them would
    otherwise stand in for the keyword of the statement being explained. That statement is
    the one worth reading: `EXPLAIN ANALYZE` runs it rather than only planning it, so a
    rewrite underneath rewrites the table for real. The parenthesised option list needs
    nothing here, already being blanked as a group."""
    return EXPLAIN_OPTIONS.sub(lambda match: blank(match.group()), statement)


def leading_keyword(statement: str) -> re.Match[str] | None:
    """The statement's own keyword, looking past what wraps it: a parenthesised guard,
    PL/pgSQL block syntax such as `BEGIN`, `IF ... THEN` and `END`, and an `EXPLAIN`.
    Offsets survive both strips, so the match still points into `statement` itself."""
    return next(
        (
            word
            for word in FIRST_WORD.finditer(strip_explain(strip_parens(statement)))
            if word.group().upper() in STATEMENT_KEYWORDS
        ),
        None,
    )


def offending_keyword(statement: str) -> str | None:
    word = leading_keyword(statement)
    if word is None:
        return None

    keyword = word.group().upper()

    if keyword in REWRITES_ROWS:
        return keyword

    if keyword == "INSERT":
        source = row_source_keyword(statement)
        return None if source is None else f"INSERT ... {source}"

    if keyword == "WITH":
        nested = next((name for name in sorted(REWRITES_ROWS) if contains(statement, name)), None)
        if nested is not None:
            return f"WITH ... {nested}"
        if contains(statement, "INSERT"):
            source = row_source_keyword(statement)
            if source is not None:
                return f"WITH ... INSERT ... {source}"

    return None


def row_source_keyword(statement: str) -> str | None:
    """Which keyword supplies an `INSERT` its rows, or `None` when a literal `VALUES` list
    does. A query outside every parenthesis is the row source outright. Failing that, a
    `VALUES` outside every parenthesis is itself the row source, so the scalar subqueries
    and helper CTEs nested within that list do not make the insert a rewrite, though only
    while no set operation sits beside it at that same level: one that does joins the list
    to a second query term, and that term is the row source however deeply it is
    parenthesised. Failing both, the rows come from a parenthesised query, which Postgres
    accepts and which reading only the unparenthesised text would let through:
    `INSERT INTO "t" ("a") (SELECT ...)` copies a whole table."""
    outer = strip_parens(statement)
    joined = row_source_in(outer)
    if joined is not None:
        return joined
    if contains(outer, "VALUES") and not any(contains(outer, word) for word in JOINS_QUERIES):
        return None
    return row_source_in(statement)


def row_source_in(text: str) -> str | None:
    return next((word for word in ("SELECT", "TABLE") if contains(text, word)), None)


def hands_off_sql(statement: str, executed: frozenset[str]) -> bool:
    """Whether a statement gives the server a string literal to run as SQL. `EXECUTE` runs one
    outright, and so does `DO`, whose body is a string wherever it is not dollar-quoted. An
    assignment parks one in a variable, which counts only when something further down runs
    that variable by name, since a string the migration never executes is text."""
    if leads_with(statement, "EXECUTE") or leads_with(statement, "DO"):
        return True
    return bool(assigned_names(statement) & executed)


def assigned_names(statement: str) -> frozenset[str]:
    """The candidate variable names a statement writes to, taken as every word ahead of the
    assignment operator. A declaration carries its type and sometimes a leading `DECLARE`
    alongside the name, and none of that is worth parsing when the only question is which
    name is executed. PL/pgSQL spells that operator `:=` and takes a bare `=` as the same
    thing, so both count, the second only where `assigns_rather_than_compares` reads it as
    an assignment. Every operator in the statement is read rather than only the first, since
    a comparison earlier on the line would otherwise claim the one slot and hide the
    assignment after it: `IF n = 1 THEN stmt = '...'` writes `stmt` at its second `=`. A
    query assigns through the target list after its `INTO` instead, which is how a rewrite
    reaches a variable with neither operator appearing at all."""
    names: set[str] = set()

    for operator in ASSIGNS.finditer(statement):
        reached = reached_words(statement[: operator.start()])
        if operator.group() == ":=" or assigns_rather_than_compares(reached):
            names.update(word.lower() for word in reached)

    for targets in INTO_TARGETS.finditer(statement):
        if names_a_table(statement[: targets.start()]):
            continue
        names.update(word.group().lower() for word in FIRST_WORD.finditer(targets.group(1)))

    return frozenset(names)


def names_a_table(before: str) -> bool:
    """Whether the `INTO` this text runs up to introduces a table rather than a query's
    target list. `INSERT INTO` is the one that does, and reading its table as somewhere a
    string was parked would have an insert scanned for the SQL its own literals spell out.
    An `INSERT` that really does assign reaches its `INTO` through a `RETURNING` list, so
    the word immediately before is what separates the two."""
    word = PRECEDING_WORD.search(before)
    return word is not None and word.group(1).upper() == "INSERT"


def reached_words(head: str) -> tuple[str, ...]:
    """The words an assignment operator is reached through, which is everything since the last
    word to open a block. A `THEN` ends the condition its `IF` began, so nothing ahead of it
    describes what follows, and neither the name being written nor the keywords that would
    mark a comparison ever sit further back than that."""
    words = [word.group().upper() for word in FIRST_WORD.finditer(head)]
    opened = max((index for index, word in enumerate(words) if word in OPENS_A_BLOCK), default=-1)
    return tuple(words[opened + 1 :])


def assigns_rather_than_compares(reached: tuple[str, ...]) -> bool:
    """Whether a bare `=` reached through these words writes a variable or tests one. They are
    all that tells the two apart: an assignment is reached with a name and perhaps a type,
    while a comparison is reached either through a statement carrying its own keyword or
    through a word that guards a condition."""
    words = set(reached)
    return not (words & STATEMENT_KEYWORDS) and not (words & GUARDS_A_CONDITION)


def executed_names(masked: str) -> frozenset[str]:
    """The variables handed to an `EXECUTE` by name. Reading these off the masked text keeps
    an `EXECUTE` written inside a comment or a string from counting. Masking blanks a literal
    in place rather than removing it, so `EXECUTE '...'` can leave the word after it looking
    like the name being run. Reaching that word means crossing no semicolon, which leaves only
    the syntax `INTO`, `USING` and `END`, and no assignment ever writes one of those, so the
    stray name has nothing to match."""
    return frozenset(match.group(1).lower() for match in RUN_BY_NAME.finditer(masked))


def leads_with(statement: str, keyword: str) -> bool:
    word = leading_keyword(statement)
    return word is not None and word.group().upper() == keyword


def contains(statement: str, keyword: str) -> bool:
    return re.search(rf"\b{keyword}\b", statement, re.IGNORECASE) is not None


def read_markers(sql: str) -> Markers:
    lines: set[int] = set()
    standalone: set[int] = set()

    for match in MARKER.finditer(sql):
        line = sql.count("\n", 0, match.start()) + 1
        lines.add(line)
        if not sql[sql.rfind("\n", 0, match.start()) + 1 : match.start()].strip():
            standalone.add(line)

    return Markers(frozenset(lines), frozenset(standalone))


def scan(sql: str, migration: str, markers: Markers) -> Iterator[Violation]:
    yield from scan_region(sql, sql, migration, markers, 0)


def scan_region(
    document: str, region: str, migration: str, markers: Markers, offset: int
) -> Iterator[Violation]:
    """Violations in one region of `document`, whose text begins at `offset`. Lines are
    always counted against the whole document, so a statement nested in a dollar-quoted
    body reports its real file line and lines up with the markers read from that file."""
    masked, bodies, literals = mask(region)
    executed = executed_names(masked)

    for match in STATEMENT.finditer(masked):
        first = line_of(document, offset + keyword_start(match))
        last = line_of(document, offset + match.end())
        exempt = markers.exempt(first, last)

        if hands_off_sql(match.group(), executed) and not exempt:
            for start, end in literals:
                if match.start() <= start and end <= match.end():
                    yield from scan_region(document, region[start:end], migration, markers, offset + start)

        keyword = offending_keyword(match.group())
        if keyword is None or exempt:
            continue
        yield Violation(migration, first, keyword)

    for start, end in bodies:
        yield from scan_region(document, region[start:end], migration, markers, offset + start)


def keyword_start(statement: re.Match[str]) -> int:
    word = leading_keyword(statement.group())
    return statement.start() + (0 if word is None else word.start())


def line_of(sql: str, offset: int) -> int:
    return sql.count("\n", 0, offset) + 1


def scan_migration(directory: Path) -> tuple[Violation, ...]:
    sql = (directory / "migration.sql").read_text(encoding="utf-8")
    return tuple(scan(sql, directory.name, read_markers(sql)))


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
