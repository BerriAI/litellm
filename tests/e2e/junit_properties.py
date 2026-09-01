"""Custom per-test signals for the standard JUnit reporter.

The e2e suite ships results to Loki/Grafana from a standard pytest JUnit report
(`--junitxml=e2e-report.xml`), not a bespoke log line. JUnit already records
outcome, duration, and node id for every `<testcase>`; the signals it cannot
derive on its own are the normalized suite package, the coverage-registry cell
ids a test covers, and where the test's source lives. Those ride along as JUnit
`<property>` entries via each item's `user_properties`, attached in
`conftest.py::pytest_collection_modifyitems`.

`source` is here because of a reporter limitation rather than a missing pytest
fact. Pytest knows every test's file and line, and its `xunit1` report family
wrote them as `file=` / `line=` attributes on `<testcase>`. The default `xunit2`
family -- pytest's since 6.0, and this suite's, since pytest.ini names no family
-- drops both. Switching families to get them back would change the document
shape for every consumer of the same XML, the Buildkite Test Engine upload and
the Loki pipeline included; a property is additive, so nothing that reads the
report today sees a difference.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

# This module's own directory, relative to the repo root. Hardcoded because it
# cannot be discovered at runtime: the e2e runner image copies tests/e2e/ to
# /app/e2e and runs pytest from there, so no ancestor of this file names the
# suite's place in the litellm tree. Moving tests/e2e/ means editing this line,
# and test_junit_properties.py fails from a checkout until you do.
SUITE_ROOT = "tests/e2e"


def suite_parts(path_part: str) -> tuple[str, ...]:
    """Path components of a suite file, relative to tests/e2e, either way it ran.

    Pytest reports paths relative to its rootdir, which moves with the
    invocation: a repo-root run gives `tests/e2e/logging/test_x.py`, a suite-cwd
    run (what the runner image does) gives `logging/test_x.py`. Strip the
    `tests/e2e` prefix when present so both collapse to the same components.
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

    `pytest.Item.location` supplies a rootdir-relative path and a ZERO-based
    line number, and neither travels as-is. The path is re-rooted at SUITE_ROOT
    so consumers never have to know how pytest was started, and the line is
    emitted ONE-based, matching editors, tracebacks, and code hosts (GitHub's
    `#L41` is the file's 41st line). A decorated test anchors at its first
    decorator, which is where pytest reports it and which puts the marks and the
    `def` on screen together.

    Returns '' rather than a guess when pytest reports no line, or when the path
    escapes the suite root (absolute, or reaching upward): a test that renders
    without a link is a smaller failure than one that links somewhere wrong.
    """
    if lineno is None:
        return ""
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or ".." in normalized.split("/"):
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
