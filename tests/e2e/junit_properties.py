"""Custom per-test signals for the standard JUnit reporter.

The e2e suite ships results to Loki/Grafana from a standard pytest JUnit report
(`--junitxml=e2e-report.xml`), not a bespoke log line. JUnit already records
outcome, duration, and node id for every `<testcase>`; the only signals it cannot
derive on its own are the normalized suite package and the coverage-registry cell
ids a test covers. Those ride along as JUnit `<property>` entries via each item's
`user_properties`, attached in `conftest.py::pytest_collection_modifyitems`.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest


def package_from_nodeid(nodeid: str) -> str:
    """Top-level suite package under tests/e2e/, or 'root' for top-level files.

    Pytest nodeids are relative to the invocation cwd. Repo-root runs look like
    `tests/e2e/logging/...`; suite-cwd runs look like `logging/...`. Strip the
    `tests/e2e` prefix so package is the suite dir either way.
    """
    path_part = nodeid.split("::", 1)[0].replace("\\", "/")
    raw = tuple(p for p in path_part.split("/") if p and p != ".")
    parts = raw[2:] if len(raw) >= 3 and raw[0] == "tests" and raw[1] == "e2e" else raw
    if len(parts) <= 1:
        return "root"
    return parts[0]


def dedupe_covers(marker_args: Iterable[tuple[object, ...]]) -> tuple[str, ...]:
    """Flatten @pytest.mark.covers arg lists into unique, order-preserving cell
    ids, dropping anything that is not a non-empty string."""
    return tuple(dict.fromkeys(arg for args in marker_args for arg in args if isinstance(arg, str) and arg))


def covers_from_item(item: pytest.Item) -> tuple[str, ...]:
    """Read @pytest.mark.covers cell ids off a pytest Item, order-preserving."""
    return dedupe_covers(marker.args for marker in item.iter_markers(name="covers"))


def result_properties(item: pytest.Item) -> tuple[tuple[str, str], ...]:
    """The custom signals a standard reporter cannot derive: the normalized suite
    package and the comma-joined coverage-registry cell ids this test covers."""
    return (
        ("package", package_from_nodeid(item.nodeid)),
        ("covers", ",".join(covers_from_item(item))),
    )


def attach_result_properties(item: pytest.Item) -> None:
    """Attach result_properties to an item's user_properties, idempotently: a
    second call is a no-op, so a collection that runs the hook more than once
    never emits duplicate <property> entries."""
    if any(name == "package" for name, _ in item.user_properties):
        return
    item.user_properties.extend(result_properties(item))
