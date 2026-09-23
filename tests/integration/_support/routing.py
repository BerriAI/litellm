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

import psycopg
import pytest
from pydantic import TypeAdapter

WRITER_ROLE: Final = "litellm_writer"
READER_ROLE: Final = "litellm_reader"
ROLES: Final = (READER_ROLE, WRITER_ROLE)
DATABASE_NAME: Final = "circle_test"
OBSERVED_FILE: Final = "routing-observed.json"
RECORDED_FILE: Final = "routing-recorded.json"
DIFF_FILE: Final = "routing-diff.txt"
EXPECTED_DIRECTORY: Final = Path(__file__).resolve().parents[1] / "routing"

RoleSet = frozenset[str]
RoutingMap = Mapping[str, frozenset[str]]
Snapshot = Mapping[tuple[str, str], int]

_PLACEHOLDERS: Final = re.compile(r"\$\d+(?:\s*,\s*\$\d+)*")
_QUERIES: Final = TypeAdapter(dict[str, tuple[str, ...]])
_OBSERVED: Final = TypeAdapter(dict[str, object])


def normalize(query: str) -> str:
    return _PLACEHOLDERS.sub("$n", " ".join(query.split()))


@dataclass(frozen=True, slots=True)
class Expectation:
    queries: RoutingMap
    tests: Mapping[str, RoutingMap]


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
    expected: tuple[str, ...]
    observed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Report:
    mismatches: tuple[Mismatch, ...]
    only_expected: tuple[str, ...]
    only_observed: tuple[str, ...]
    calls: Mapping[str, int]
    dealloc: int

    def failures(self) -> tuple[str, ...]:
        mismatch_failures: Final = tuple(
            f"{mismatch.test if mismatch.test is not None else 'global'}: {mismatch.query}: "
            f"expected [{', '.join(mismatch.expected)}] observed [{', '.join(mismatch.observed)}]"
            for mismatch in self.mismatches
        )
        dealloc_failures: Final = (
            (f"pg_stat_statements evicted {self.dealloc} entries (dealloc > 0)",) if self.dealloc > 0 else ()
        )
        liveness_failures: Final = tuple(f"no {role} calls observed" for role in ROLES if self.calls.get(role, 0) == 0)
        return (*mismatch_failures, *dealloc_failures, *liveness_failures)


def _sorted_map(value: RoutingMap) -> RoutingMap:
    return MappingProxyType(dict(sorted(value.items())))


def compare(expected: Expectation, observed: Observation) -> Report:
    mismatches: Final = (
        *(
            Mismatch(
                None,
                query,
                tuple(sorted(expected_roles)),
                tuple(sorted(observed.queries[query])),
            )
            for query, expected_roles in expected.queries.items()
            if query in observed.queries and observed.queries[query] != expected_roles
        ),
        *(
            Mismatch(
                test,
                query,
                tuple(sorted(expected_roles)),
                tuple(sorted(observed.tests[test][query])),
            )
            for test, queries in expected.tests.items()
            if test in observed.tests
            for query, expected_roles in queries.items()
            if query in observed.tests[test] and observed.tests[test][query] != expected_roles
        ),
    )
    return Report(
        mismatches,
        tuple(sorted(query for query in expected.queries if query not in observed.queries)),
        tuple(sorted(query for query in observed.queries if query not in expected.queries)),
        observed.calls,
        observed.dealloc,
    )


def render(report: Report) -> str:
    failures: Final = report.failures()
    lines: Final = (
        "== failures ==",
        *(failures or ("none",)),
        "",
        "== queries only in expected ==",
        *(report.only_expected or ("none",)),
        "",
        "== queries only in observed ==",
        *(report.only_observed or ("none",)),
        "",
        "== calls ==",
        *(f"{role}: {report.calls.get(role, 0)}" for role in ROLES),
        f"dealloc: {report.dealloc}",
    )
    return "\n".join(lines) + "\n"


def _roles(document: Mapping[str, tuple[str, ...]]) -> RoutingMap:
    return _sorted_map({query: frozenset(roles) for query, roles in document.items()})


def _tests(document: Mapping[str, Mapping[str, tuple[str, ...]]]) -> Mapping[str, RoutingMap]:
    return MappingProxyType({node: _roles(queries) for node, queries in document.items()})


def load_expectation(path: Path) -> Expectation:
    document: Final = _OBSERVED.validate_python(json.loads(path.read_text()))
    queries: Final = _QUERIES.validate_python(document.get("queries", {}))
    tests: Final = TypeAdapter(dict[str, dict[str, tuple[str, ...]]]).validate_python(document.get("tests", {}))
    return Expectation(_roles(queries), _tests(tests))


def load_observation(path: Path) -> Observation:
    document: Final = _OBSERVED.validate_python(json.loads(path.read_text()))
    queries: Final = _QUERIES.validate_python(document.get("queries", {}))
    tests: Final = TypeAdapter(dict[str, dict[str, tuple[str, ...]]]).validate_python(document.get("tests", {}))
    calls: Final = TypeAdapter(dict[str, int]).validate_python(document.get("calls", {}))
    dealloc: Final = TypeAdapter(int).validate_python(document.get("dealloc", 0))
    return Observation(_roles(queries), _tests(tests), MappingProxyType(calls), dealloc)


def _serializable(queries: RoutingMap, tests: Mapping[str, RoutingMap]) -> dict[str, object]:
    return {
        "queries": {query: sorted(roles) for query, roles in queries.items()},
        "tests": {node: {query: sorted(roles) for query, roles in mapping.items()} for node, mapping in tests.items()},
    }


def dump_expectation(observation: Observation) -> str:
    return json.dumps(_serializable(observation.queries, observation.tests), indent=2, sort_keys=True) + "\n"


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
    from urllib.parse import urlsplit, urlunsplit

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
    commands: Final = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "check"):
        subcommand: Final = commands.add_parser(name)
        subcommand.add_argument("group")
        subcommand.add_argument("results_dir", type=Path)
    check_command: Final = commands.choices["check"]
    check_command.add_argument("--expected", type=Path, default=None)
    options: Final = parser.parse_args(argv)
    observed_path: Final = options.results_dir / OBSERVED_FILE
    if options.command == "record":
        observation: Final = load_observation(observed_path)
        recorded: Final = options.results_dir / RECORDED_FILE
        recorded.write_text(dump_expectation(observation))
        sys.stdout.write(f"recorded routing expectation at {recorded}\n")
        return 0
    expected_path: Final = options.expected or EXPECTED_DIRECTORY / f"{options.group}.json"
    if not expected_path.exists():
        sys.stderr.write(f"expected routing file missing: {expected_path}\n")
        return 1
    report: Final = compare(load_expectation(expected_path), load_observation(observed_path))
    diff: Final = render(report)
    (options.results_dir / DIFF_FILE).write_text(diff)
    sys.stdout.write(diff)
    return 1 if report.failures() else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
