from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from pydantic import TypeAdapter

WRITER_ROLE: Final = "litellm_writer"
READER_ROLE: Final = "litellm_reader"
ROLES: Final = (READER_ROLE, WRITER_ROLE)
DATABASE_NAME: Final = "circle_test"
OBSERVED_FILE: Final = "routing-observed.json"
DIFF_FILE: Final = "routing-diff.txt"
EITHER_ROLE_FILE: Final = Path(__file__).resolve().parents[1] / "routing" / "either_role.json"

RoleSet = frozenset[str]
RoutingMap = Mapping[str, frozenset[str]]
Snapshot = Mapping[tuple[str, str], int]

_PLACEHOLDERS: Final = re.compile(r"\$\d+(?:\s*,\s*\$\d+)*")
_QUERIES: Final = TypeAdapter(dict[str, tuple[str, ...]])
_OBSERVED: Final = TypeAdapter(dict[str, object])


def normalize(query: str) -> str:
    return _PLACEHOLDERS.sub("$n", " ".join(query.split()))


@dataclass(frozen=True, slots=True)
class Observation:
    queries: RoutingMap
    tests: Mapping[str, RoutingMap]
    calls: Mapping[str, int]
    dealloc: int


@dataclass(frozen=True, slots=True)
class Mismatch:
    test: str | None
    query: str
    base: tuple[str, ...]
    head: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Report:
    mismatches: tuple[Mismatch, ...]
    only_base: tuple[str, ...]
    only_head: tuple[str, ...]
    calls: Mapping[str, Mapping[str, int]]
    dealloc: Mapping[str, int]
    either_role: tuple[str, ...] = ()

    def failures(self) -> tuple[str, ...]:
        mismatch_failures: Final = tuple(
            f"{mismatch.test if mismatch.test is not None else 'global'}: {mismatch.query}: "
            f"base [{', '.join(mismatch.base)}] head [{', '.join(mismatch.head)}]"
            for mismatch in self.mismatches
        )
        side_failures: Final = tuple(
            failure
            for side in ("base", "head")
            for failure in (
                *(
                    (f"{side}: pg_stat_statements evicted {self.dealloc[side]} entries (dealloc > 0)",)
                    if self.dealloc[side] > 0
                    else ()
                ),
                *(f"{side}: no {role} calls observed" for role in ROLES if self.calls[side].get(role, 0) == 0),
            )
        )
        return (*mismatch_failures, *side_failures)


def _sorted_map(value: RoutingMap) -> RoutingMap:
    return MappingProxyType(dict(sorted(value.items())))


def compare(base: Observation, head: Observation, either_role: frozenset[str] = frozenset()) -> Report:
    mismatches: Final = (
        *(
            Mismatch(
                None,
                query,
                tuple(sorted(base_roles)),
                tuple(sorted(head.queries[query])),
            )
            for query, base_roles in base.queries.items()
            if query in head.queries and head.queries[query] != base_roles and query not in either_role
        ),
        *(
            Mismatch(
                test,
                query,
                tuple(sorted(base_roles)),
                tuple(sorted(head.tests[test][query])),
            )
            for test, queries in base.tests.items()
            if test in head.tests
            for query, base_roles in queries.items()
            if query in head.tests[test] and head.tests[test][query] != base_roles and query not in either_role
        ),
    )
    varying: Final = frozenset(
        query
        for query in either_role
        if (query in base.queries and query in head.queries and head.queries[query] != base.queries[query])
        or any(
            query in base.tests[test]
            and query in head.tests[test]
            and head.tests[test][query] != base.tests[test][query]
            for test in frozenset(base.tests) & frozenset(head.tests)
        )
    )
    return Report(
        mismatches,
        tuple(sorted(query for query in base.queries if query not in head.queries)),
        tuple(sorted(query for query in head.queries if query not in base.queries)),
        MappingProxyType({"base": base.calls, "head": head.calls}),
        MappingProxyType({"base": base.dealloc, "head": head.dealloc}),
        tuple(sorted(varying)),
    )


def render(report: Report) -> str:
    failures: Final = report.failures()
    lines: Final = (
        "== failures ==",
        *(failures or ("none",)),
        "",
        "== either role ==",
        *(report.either_role or ("none",)),
        "",
        "== queries only in base ==",
        *(report.only_base or ("none",)),
        "",
        "== queries only in head ==",
        *(report.only_head or ("none",)),
        "",
        "== calls ==",
        *(
            line
            for side in ("base", "head")
            for line in (
                *(f"{side} {role}: {report.calls[side].get(role, 0)}" for role in ROLES),
                f"{side} dealloc: {report.dealloc[side]}",
            )
        ),
    )
    return "\n".join(lines) + "\n"


def _roles(document: Mapping[str, tuple[str, ...]]) -> RoutingMap:
    return _sorted_map({query: frozenset(roles) for query, roles in document.items()})


def _tests(document: Mapping[str, Mapping[str, tuple[str, ...]]]) -> Mapping[str, RoutingMap]:
    return MappingProxyType({node: _roles(queries) for node, queries in document.items()})


def load_observation(path: Path) -> Observation:
    document: Final = _OBSERVED.validate_python(json.loads(path.read_text()))
    queries: Final = _QUERIES.validate_python(document.get("queries", {}))
    tests: Final = TypeAdapter(dict[str, dict[str, tuple[str, ...]]]).validate_python(document.get("tests", {}))
    calls: Final = TypeAdapter(dict[str, int]).validate_python(document.get("calls", {}))
    dealloc: Final = TypeAdapter(int).validate_python(document.get("dealloc", 0))
    return Observation(_roles(queries), _tests(tests), MappingProxyType(calls), dealloc)


def load_either_role(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    document: Final = TypeAdapter(dict[str, str]).validate_python(json.loads(path.read_text()))
    return frozenset(document)


def _serializable(queries: RoutingMap, tests: Mapping[str, RoutingMap]) -> dict[str, object]:
    return {
        "queries": {query: sorted(roles) for query, roles in queries.items()},
        "tests": {node: {query: sorted(roles) for query, roles in mapping.items()} for node, mapping in tests.items()},
    }


def dump_observation(observation: Observation) -> str:
    document: Final = _serializable(observation.queries, observation.tests)
    return (
        json.dumps(
            {**document, "calls": dict(observation.calls), "dealloc": observation.dealloc},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _maintenance_url() -> str:
    parsed: Final = urlsplit(os.environ["DATABASE_URL"])
    return urlunsplit(parsed._replace(path="/postgres"))


def snapshot(connection: psycopg.Connection[object]) -> Mapping[tuple[str, str], int]:
    rows: Final = connection.execute(
        """
        SELECT r.rolname, s.query, s.calls
        FROM pg_stat_statements s
        JOIN pg_roles r ON r.oid = s.userid
        WHERE s.dbid = (SELECT oid FROM pg_database WHERE datname = %s)
          AND r.rolname = ANY(%s)
        """,
        (DATABASE_NAME, list(ROLES)),
    ).fetchall()
    return MappingProxyType(
        {
            key: sum(calls for _, _, calls in grouped)
            for key, grouped in itertools.groupby(
                sorted((str(role), normalize(str(query)), int(calls)) for role, query, calls in rows),
                key=lambda row: (row[0], row[1]),
            )
        }
    )


def delta(before: Snapshot, after: Snapshot) -> RoutingMap:
    pairs: Final = {key: after.get(key, 0) - before.get(key, 0) for key in frozenset(before) | frozenset(after)}
    queries: Final = frozenset(query for (_, query), change in pairs.items() if change > 0)
    return MappingProxyType(
        {query: frozenset(role for role in ROLES if pairs.get((role, query), 0) > 0) for query in sorted(queries)}
    )


def role_calls(before: Snapshot, after: Snapshot) -> Mapping[str, int]:
    return MappingProxyType(
        {
            role: sum(
                max(after.get((role, query), 0) - before.get((role, query), 0), 0)
                for query in frozenset(q for _, q in before) | frozenset(q for _, q in after)
            )
            for role in ROLES
        }
    )


def dealloc(connection: psycopg.Connection[object]) -> int:
    return int(connection.execute("SELECT dealloc FROM pg_stat_statements_info").fetchone()[0])


class RoutingPlugin:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self._session_start: Snapshot | None = None
        self._tests: tuple[tuple[str, RoutingMap], ...] = ()

    def _snapshot(self) -> Snapshot:
        with psycopg.connect(_maintenance_url(), autocommit=True) as connection:
            return snapshot(connection)

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        if hasattr(self.config, "workerinput"):
            return
        self._session_start = self._snapshot()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
        if self.config.getoption("numprocesses", default=None) or hasattr(self.config, "workerinput"):
            yield
            return
        before: Final = self._snapshot()
        yield
        after: Final = self._snapshot()
        self._tests = (*self._tests, (item.nodeid, delta(before, after)))

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if hasattr(self.config, "workerinput"):
            return
        end: Final = self._snapshot()
        with psycopg.connect(_maintenance_url(), autocommit=True) as connection:
            evictions: Final = dealloc(connection)
        start: Final = self._session_start or {}
        tests: Final = MappingProxyType({node: mapping for node, mapping in self._tests})
        destination: Final = Path(os.environ["INTEGRATION_RESULTS_DIR"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / OBSERVED_FILE).write_text(
            dump_observation(
                Observation(
                    delta(start, end),
                    tests,
                    role_calls(start, end),
                    evictions,
                )
            )
        )


def main(argv: tuple[str, ...] | list[str]) -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check",))
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("head_dir", type=Path)
    parser.add_argument("--either-role", type=Path, default=EITHER_ROLE_FILE)
    parser.add_argument("--diff", type=Path, default=None)
    options: Final = parser.parse_args(argv)
    base_path: Final = options.base_dir / OBSERVED_FILE
    head_path: Final = options.head_dir / OBSERVED_FILE
    for path in (base_path, head_path):
        if not path.exists():
            sys.stderr.write(f"observed routing file missing: {path}\n")
    if not base_path.exists() or not head_path.exists():
        return 1
    report: Final = compare(
        load_observation(base_path),
        load_observation(head_path),
        load_either_role(options.either_role),
    )
    diff: Final = render(report)
    (options.diff or options.head_dir.parent / DIFF_FILE).write_text(diff)
    sys.stdout.write(diff)
    return 1 if report.failures() else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
