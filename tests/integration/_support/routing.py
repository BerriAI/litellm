from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import threading
import uuid
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from pydantic import TypeAdapter
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.rust_bridge import catalog, observation
from litellm.rust_bridge.catalog import Feature, RouteContext, RouteRule, Rules, Target
from litellm.rust_bridge.configuration import Decision, Rollout, decision

WRITER_ROLE: Final = "litellm_writer"
READER_ROLE: Final = "litellm_reader"
ROLES: Final = (READER_ROLE, WRITER_ROLE)
DATABASE_NAME: Final = "circle_test"
WORKER_DATABASE_PREFIX: Final = "integration_worker"
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


def strict_current_policy(rules: Rules | None = None) -> Rules:
    source: Final = catalog.RULES if rules is None else rules
    return tuple(
        replace(
            rule,
            rollout=(Rollout.PYTHON_ONLY if decision(rule.rollout) is Decision.PYTHON else Rollout.RUST_REQUIRED),
        )
        if rule.rollout in (Rollout.RUST_OPT_IN, Rollout.RUST_OPT_OUT)
        else rule
        for rule in source
    )


def _rule_document(rule: RouteRule) -> dict[str, object]:
    def values(value: Collection[str | Feature | Target] | None) -> tuple[str, ...] | None:
        return (
            None
            if value is None
            else tuple(sorted(item.value if isinstance(item, (Feature, Target)) else item for item in value))
        )

    return {
        "route": rule.route.value,
        "rollout": rule.rollout.value,
        "providers": values(rule.providers),
        "models": values(rule.models),
        "targets": values(rule.targets),
        "features": values(rule.features),
    }


def _policy_document(rules: Rules) -> tuple[dict[str, object], ...]:
    return tuple(_rule_document(rule) for rule in rules)


class _TraceObserver:
    def __init__(self, rules: Rules, trace_dir: Path) -> None:
        self._rules: Final = rules
        self._trace_dir: Final = trace_dir
        self._lock: Final = threading.Lock()

    def record(self, event: observation.RouteObservation) -> None:
        observation.record_event(event)

    def write(self, trace_id: str, events: tuple[observation.RouteObservation, ...]) -> Path:
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        destination: Final = self._trace_dir / f"trace-{trace_id}.json"
        document: Final = {
            "trace_id": trace_id,
            "worker_pid": os.getpid(),
            "policy": _policy_document(self._rules),
            "events": tuple(_event_document(event) for event in events),
        }
        with self._lock:
            destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return destination


def _context_document(context: RouteContext | None) -> dict[str, object] | None:
    if context is None:
        return None
    return {
        "route": context.route.value,
        "provider": context.provider,
        "model": context.model,
        "target": None if context.target is None else context.target.value,
        "features": tuple(sorted(feature.value for feature in context.features)),
    }


def _event_document(event: observation.RouteObservation) -> dict[str, object]:
    return {
        "event": event.event,
        "trace_id": event.trace_id,
        "attempt_id": event.attempt_id,
        "parent_attempt_id": event.parent_attempt_id,
        "depth": event.depth,
        "route": None if event.route is None else event.route.value,
        "context": _context_document(event.context),
        "decision": None if event.decision is None else event.decision.value,
        "outcome": event.outcome,
        "reason": event.reason,
        "engine": event.engine,
    }


class _RoutingTraceMiddleware:
    def __init__(self, app: ASGIApp, observer: _TraceObserver) -> None:
        self._app: Final = app
        self._observer: Final = observer

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        trace_id: Final = uuid.uuid4().hex

        async def traced_send(message: Message) -> None:
            if message.get("type") == "http.response.start" and not any(
                event.event == "request" for event in observation.current_events()
            ):
                events: Final = observation.current_events()
                outcome: Final = (
                    "completed"
                    if any(event.event == "terminal" and event.outcome == "success" for event in events)
                    else "failure"
                    if any(event.event == "terminal" and event.outcome == "failure" for event in events)
                    else "no_dispatch"
                )
                observation.observe_request_outcome(trace_id, outcome)
                self._observer.write(trace_id, observation.current_events())
                headers: Final = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-integration-routing-id"
                ]
                response: Final = {
                    **message,
                    "headers": [*headers, (b"x-integration-routing-id", trace_id.encode())],
                }
                await send(response)
                return
            await send(message)

        observer_token: Final = observation.set_observer(self._observer)
        try:
            with observation.request_scope(trace_id), observation.trace_scope():
                try:
                    await self._app(scope, receive, traced_send)
                finally:
                    if not any(event.event == "request" for event in observation.current_events()):
                        events: Final = observation.current_events()
                        outcome: Final = (
                            "failure"
                            if any(event.event == "terminal" and event.outcome == "failure" for event in events)
                            else "no_dispatch"
                        )
                        observation.observe_request_outcome(trace_id, outcome)
                        self._observer.write(trace_id, observation.current_events())
        finally:
            observation.reset_observer(observer_token)


def install_rule_aware_harness() -> None:
    trace_dir: Final = Path(os.environ["LITELLM_INTEGRATION_ROUTING_TRACE_DIR"])
    rules: Final = strict_current_policy()
    policy_path: Final = trace_dir / f"policy-{os.getpid()}.json"
    trace_dir.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(_policy_document(rules), indent=2, sort_keys=True) + "\n")
    setattr(catalog, "RULES", rules)
    observer: Final = _TraceObserver(rules, trace_dir)
    from litellm.proxy.proxy_server import app

    middleware_stack: Final = app.middleware_stack
    if middleware_stack is None:
        app.add_middleware(_RoutingTraceMiddleware, observer=observer)
    else:
        app.middleware_stack = _RoutingTraceMiddleware(middleware_stack, observer)


def trace_path(trace_dir: Path, trace_id: str) -> Path:
    return trace_dir / f"trace-{trace_id}.json"


def rule_aware_overrides(directory: Path, *, rust: bool) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "LITELLM_RUST": "true" if rust else "false",
            "LITELLM_INTEGRATION_ROUTING_TRACE_DIR": str(directory / "routing-traces"),
            "LITELLM_WORKER_STARTUP_HOOKS": "integration._support.routing:install_rule_aware_harness",
        }
    )


def assert_routing_matches_policy(headers: Mapping[str, str], trace_location: Path) -> None:
    trace_id: Final = headers.get("x-integration-routing-id")
    assert trace_id is not None, dict(headers)
    assert trace_location.name == f"trace-{trace_id}.json", (trace_location, trace_id)
    assert trace_location.exists(), f"routing trace missing: {trace_location}"
    document: Final = TypeAdapter(dict[str, object]).validate_python(json.loads(trace_location.read_text()))
    assert document.get("trace_id") == trace_id, document
    events: Final = TypeAdapter(list[dict[str, object]]).validate_python(document.get("events", []))
    assert all(event.get("trace_id") == trace_id for event in events), events
    rules: Final = _rules_from_document(document.get("policy", []))
    requests: Final = tuple(event for event in events if event.get("event") == "request")
    assert len(requests) == 1, requests
    assert requests[0].get("outcome") in ("completed", "no_dispatch"), requests
    assert not any(
        event.get("event") == "early" and event.get("reason") in ("projection_failure", "detector_failure")
        for event in events
    ), events
    terminals: Final = tuple(
        event
        for event in events
        if event.get("event") == "terminal"
        and event.get("outcome") in ("success", "failure")
        and event.get("parent_attempt_id") is None
    )
    successes: Final = tuple(event for event in terminals if event.get("outcome") == "success")
    if successes:
        assert requests[0].get("outcome") == "completed", requests
        assert len(successes) == 1, successes
        terminal: Final = successes[0]
        context_document: Final = TypeAdapter(dict[str, object]).validate_python(terminal["context"])
        context: Final = RouteContext(
            route=catalog.Route(str(context_document["route"])),
            provider=None if context_document.get("provider") is None else str(context_document["provider"]),
            model=None if context_document.get("model") is None else str(context_document["model"]),
            target=(
                None if context_document.get("target") is None else catalog.Target(str(context_document["target"]))
            ),
            features=frozenset(
                Feature(feature)
                for feature in TypeAdapter(list[str]).validate_python(context_document.get("features", []))
            ),
        )
        expected: Final = catalog.decision(context, rules)
        assert expected in (Decision.PYTHON, Decision.RUST_REQUIRED), (terminal, expected)
        assert terminal.get("decision") == expected.value, (terminal, expected)
        marker: Final = headers.get("x-litellm-rust") == "true"
        if expected is Decision.RUST_REQUIRED:
            assert marker, headers
            assert terminal.get("engine") == "rust", terminal
        else:
            assert not marker, headers
            assert terminal.get("engine") == "python", terminal
        return

    assert not terminals, terminals
    no_candidates: Final = tuple(
        event
        for event in events
        if event.get("event") == "early"
        and event.get("reason") == "no_candidate"
        and event.get("parent_attempt_id") is None
    )
    assert len(no_candidates) == 1, events
    assert requests[0].get("outcome") == "no_dispatch", requests
    early: Final = no_candidates[0]
    assert early.get("route") is not None, early
    route: Final = catalog.Route(str(early["route"]))
    assert not catalog.rust_might_handle(route, rules), (early, rules)
    assert headers.get("x-litellm-rust") != "true", headers
    return


def _rules_from_document(value: object) -> Rules:
    documents: Final = TypeAdapter(list[dict[str, object]]).validate_python(value)

    return tuple(
        RouteRule(
            route=catalog.Route(str(document["route"])),
            rollout=Rollout(TypeAdapter(int).validate_python(document["rollout"])),
            providers=None
            if document.get("providers") is None
            else frozenset(TypeAdapter(list[str]).validate_python(document["providers"])),
            models=None
            if document.get("models") is None
            else frozenset(TypeAdapter(list[str]).validate_python(document["models"])),
            targets=None
            if document.get("targets") is None
            else frozenset(Target(str(value)) for value in TypeAdapter(list[str]).validate_python(document["targets"])),
            features=None
            if document.get("features") is None
            else frozenset(
                Feature(str(value)) for value in TypeAdapter(list[str]).validate_python(document["features"])
            ),
        )
        for document in documents
    )


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
        WHERE s.dbid IN (SELECT oid FROM pg_database WHERE datname = %s OR starts_with(datname, %s))
          AND r.rolname = ANY(%s)
        """,
        (DATABASE_NAME, f"{WORKER_DATABASE_PREFIX}_", list(ROLES)),
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
