"""Custom per-test signals for the standard JUnit reporter.

The e2e suite ships results to Loki/Grafana from a standard pytest JUnit report
(`--junitxml=e2e-report.xml`), not a bespoke log line. JUnit already records
outcome, duration, and node id for every `<testcase>`; the signals it cannot
derive on its own are the normalized suite package, the coverage-registry cell
ids a test covers, and where the test's source lives. Those ride along as JUnit
`<property>` entries via each item's `user_properties`, attached in
`conftest.py::pytest_collection_modifyitems`.

`source` is a property rather than the `file=` / `line=` attributes pytest used
to write, because the `xunit2` family this suite runs on drops those, and
switching families would change the XML for every consumer of it -- the
Buildkite Test Engine upload and the Loki pipeline included.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

# Hardcoded because the runner image copies tests/e2e/ to /app/e2e, so nothing
# at runtime names this suite's place in the repo. test_junit_properties.py
# fails from a checkout if it moves.
SUITE_ROOT = "tests/e2e"


def suite_parts(path_part: str) -> tuple[str, ...]:
    """Path components of a suite file relative to tests/e2e, however it ran.

    Pytest paths are rootdir-relative, and rootdir moves with the invocation: a
    repo-root run gives `tests/e2e/logging/test_x.py`, a suite-cwd run (the
    runner image) gives `logging/test_x.py`. Both collapse to the same tuple.
    """
    raw = tuple(p for p in path_part.replace("\\", "/").split("/") if p and p != ".")
    return raw[2:] if len(raw) >= 3 and raw[0] == "tests" and raw[1] == "e2e" else raw


def package_from_nodeid(nodeid: str) -> str:
    """Top-level suite package under tests/e2e/, or 'root' for top-level files."""
    parts = suite_parts(nodeid.split("::", 1)[0])
    if len(parts) <= 1:
        return "root"
    return parts[0]


def source_from_location(path: str, lineno: int | None) -> str:
    """Repo-relative `path:line` for a test, or '' when nothing is linkable.

    `pytest.Item.location` gives a rootdir-relative path and a ZERO-based line.
    The path is re-rooted at SUITE_ROOT so consumers need not know how pytest was
    started, and the line is emitted ONE-based to match editors, tracebacks and
    code hosts. A decorated test anchors at its first decorator, which is where
    pytest reports it.

    Empty rather than a guess for anything unlinkable: no line, a path reaching
    upward, or a path carrying a colon, which is both how an absolute Windows
    path arrives and a character `path:line` has no way to represent.
    """
    if lineno is None:
        return ""
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or ":" in normalized or ".." in normalized.split("/"):
        return ""
    parts = suite_parts(normalized)
    if not parts:
        return ""
    return f"{'/'.join((SUITE_ROOT, *parts))}:{lineno + 1}"


def source_from_item(item: pytest.Item) -> str:
    """Read the repo-relative `path:line` off a pytest Item's reported location."""
    path, lineno, _ = item.location
    return source_from_location(path, lineno)


def dedupe_covers(marker_args: Iterable[tuple[object, ...]]) -> tuple[str, ...]:
    """Flatten @pytest.mark.covers arg lists into unique, order-preserving cell
    ids, dropping anything that is not a non-empty string."""
    return tuple(dict.fromkeys(arg for args in marker_args for arg in args if isinstance(arg, str) and arg))


def covers_from_item(item: pytest.Item) -> tuple[str, ...]:
    """Read @pytest.mark.covers cell ids off a pytest Item, order-preserving."""
    return dedupe_covers(marker.args for marker in item.iter_markers(name="covers"))


def result_properties(item: pytest.Item) -> tuple[tuple[str, str], ...]:
    """The custom signals a standard reporter cannot derive: the normalized suite
    package, the comma-joined coverage-registry cell ids this test covers, and the
    repo-relative `path:line` its source sits at."""
    return (
        ("package", package_from_nodeid(item.nodeid)),
        ("covers", ",".join(covers_from_item(item))),
        ("source", source_from_item(item)),
    )


def attach_result_properties(item: pytest.Item) -> None:
    """Attach result_properties to an item's user_properties, idempotently: a
    second call is a no-op, so a collection that runs the hook more than once
    never emits duplicate <property> entries."""
    if any(name == "package" for name, _ in item.user_properties):
        return
    item.user_properties.extend(result_properties(item))
