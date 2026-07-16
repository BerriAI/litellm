"""Structured e2e result lines for Loki / Grafana status history.

Pytest progress lines are a bad dashboard source: they only expose file basenames,
break under quiet modes, and force status-history rows to explode with suite growth.

Each finished test emits one logfmt line:

    E2E_RESULT package=logging file=test_langfuse_e2e.py outcome=failed
        duration_ms=1234 node_id=logging/test_langfuse_e2e.py::TestX::test_y
        covers=logging.langfuse.team.success

Grafana package status-history queries max(fail) by package over E2E_RESULT lines.
Drill-down uses node_id / covers in Explore, not status-history cardinality.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

Outcome = Literal["passed", "failed", "error", "skipped", "xfailed", "xpassed"]


@dataclass(frozen=True, slots=True)
class E2EResult:
    package: str
    file: str
    outcome: Outcome
    duration_ms: int
    node_id: str
    covers: tuple[str, ...]


@runtime_checkable
class _MarkerArgs(Protocol):
    args: Sequence[object]


@runtime_checkable
class _ItemWithCovers(Protocol):
    def iter_markers(self, name: str) -> Iterable[object]: ...


def package_from_nodeid(nodeid: str) -> str:
    """Top-level suite package under tests/e2e/, or 'root' for top-level files."""
    path_part = nodeid.split("::", 1)[0].replace("\\", "/")
    parts = tuple(p for p in path_part.split("/") if p and p != ".")
    if len(parts) <= 1:
        return "root"
    return parts[0]


def file_from_nodeid(nodeid: str) -> str:
    path_part = nodeid.split("::", 1)[0].replace("\\", "/")
    return Path(path_part).name


def covers_from_item(item: object) -> tuple[str, ...]:
    """Read @pytest.mark.covers cell ids from a pytest Item."""
    if not isinstance(item, _ItemWithCovers):
        return ()
    return tuple(
        dict.fromkeys(
            arg
            for marker in item.iter_markers(name="covers")
            if isinstance(marker, _MarkerArgs)
            for arg in marker.args
            if isinstance(arg, str) and arg
        )
    )


def outcome_from_report(when: str, failed: bool, skipped: bool, passed: bool) -> Outcome | None:
    """Map pytest TestReport fields to a terminal outcome. None if not final."""
    if when == "setup" and skipped:
        return "skipped"
    if when == "setup" and failed:
        return "error"
    if when != "call":
        return None
    if skipped:
        return "skipped"
    if failed:
        return "failed"
    if passed:
        return "passed"
    return "failed"


def _logfmt_escape(value: str) -> str:
    if value == "":
        return '""'
    needs_quote = any(ch.isspace() or ch in "\"=\\" for ch in value)
    if not needs_quote:
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_e2e_result_line(result: E2EResult) -> str:
    covers = ",".join(result.covers)
    fields = (
        ("package", result.package),
        ("file", result.file),
        ("outcome", result.outcome),
        ("duration_ms", str(result.duration_ms)),
        ("node_id", result.node_id),
        ("covers", covers),
    )
    body = " ".join(f"{key}={_logfmt_escape(value)}" for key, value in fields)
    return f"E2E_RESULT {body}"


def result_from_pytest(
    *,
    nodeid: str,
    when: str,
    failed: bool,
    skipped: bool,
    passed: bool,
    duration_seconds: float,
    covers: tuple[str, ...] = (),
) -> E2EResult | None:
    outcome = outcome_from_report(when=when, failed=failed, skipped=skipped, passed=passed)
    if outcome is None:
        return None
    duration_ms = max(0, int(round(duration_seconds * 1000)))
    return E2EResult(
        package=package_from_nodeid(nodeid),
        file=file_from_nodeid(nodeid),
        outcome=outcome,
        duration_ms=duration_ms,
        node_id=nodeid,
        covers=covers,
    )
