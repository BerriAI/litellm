"""Harness coverage for the custom JUnit properties.

No proxy and no ``e2e`` marker. Pins the two normalizations that have to agree
about where a suite file lives -- ``package_from_nodeid`` (strip the suite root)
and ``source_from_location`` (re-root at it) -- across both ways the suite is
launched, plus the one-based line offset and the refusal to emit a path that
escapes the suite. The consumers of these properties are the Loki/Grafana
rollups and, for ``source``, the status page's per-test links to GitHub.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from junit_properties import (
    SUITE_ROOT,
    attach_result_properties,
    dedupe_covers,
    package_from_nodeid,
    result_properties,
    source_from_location,
    suite_parts,
)


class FakeMarker:
    def __init__(self, name: str, *args: object) -> None:
        self.name = name
        self.args = args


class FakeItem:
    """The three attributes junit_properties reads off a pytest Item."""

    def __init__(
        self, nodeid: str, location: tuple[str, int | None, str], markers: tuple[FakeMarker, ...] = ()
    ) -> None:
        self.nodeid = nodeid
        self.location = location
        self.user_properties: list[tuple[str, object]] = []
        self._markers = markers

    def iter_markers(self, name: str):
        return (marker for marker in self._markers if marker.name == name)


def as_item(fake: FakeItem) -> pytest.Item:
    """FakeItem reports the three things junit_properties reads off a collected
    test, and a real pytest.Item cannot be built without a session, so the stand-in
    is handed over structurally."""
    return fake  # pyright: ignore[reportReturnType]  # structural stand-in for a collected Item


def repo_root() -> Path | None:
    """The litellm checkout above this file, or None when there isn't one."""
    return next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)


class TestSuiteParts:
    @pytest.mark.parametrize(
        "path",
        ["logging/test_x.py", "tests/e2e/logging/test_x.py", "./logging/test_x.py", "tests\\e2e\\logging\\test_x.py"],
    )
    def test_both_invocation_shapes_collapse_to_the_same_components(self, path: str) -> None:
        """A repo-root run and a suite-cwd run report the same file differently;
        every downstream signal has to see one spelling."""
        assert suite_parts(path) == ("logging", "test_x.py")

    def test_top_level_suite_file_keeps_its_single_component(self) -> None:
        assert suite_parts("tests/e2e/test_fixture_mode.py") == ("test_fixture_mode.py",)


class TestPackageFromNodeid:
    @pytest.mark.parametrize(
        ("nodeid", "expected"),
        [
            ("logging/test_x.py::TestFoo::test_bar", "logging"),
            ("tests/e2e/logging/test_x.py::TestFoo::test_bar", "logging"),
            ("quota_management/spend_tracking/test_x.py::test_bar", "quota_management"),
            ("test_fixture_mode.py::TestParseFixtureMode::test_known_values_normalize", "root"),
            ("tests/e2e/test_fixture_mode.py::test_bar", "root"),
        ],
    )
    def test_package_is_the_first_dir_under_the_suite_root(self, nodeid: str, expected: str) -> None:
        assert package_from_nodeid(nodeid) == expected


class TestSourceFromLocation:
    @pytest.mark.parametrize("path", ["a2a/test_a2a_agent_e2e.py", "tests/e2e/a2a/test_a2a_agent_e2e.py"])
    def test_path_is_repo_relative_however_pytest_was_started(self, path: str) -> None:
        assert source_from_location(path, 40) == "tests/e2e/a2a/test_a2a_agent_e2e.py:41"

    def test_line_is_emitted_one_based(self) -> None:
        """pytest.Item.location counts from 0; editors, tracebacks and GitHub's
        #L anchor all count from 1, and an off-by-one lands on the decorator."""
        assert source_from_location("a2a/test_x.py", 0) == "tests/e2e/a2a/test_x.py:1"

    def test_top_level_suite_file_sits_directly_under_the_suite_root(self) -> None:
        assert source_from_location("test_fixture_mode.py", 39) == "tests/e2e/test_fixture_mode.py:40"

    @pytest.mark.parametrize(
        ("path", "lineno"),
        [
            ("a2a/test_x.py", None),
            ("/app/e2e/a2a/test_x.py", 40),
            ("C:\\app\\e2e\\a2a\\test_x.py", 40),
            ("../conftest.py", 40),
            ("", 40),
        ],
    )
    def test_nothing_linkable_yields_empty_rather_than_a_guess(self, path: str, lineno: int | None) -> None:
        """A colon is rejected on two counts: it is how a Windows absolute path
        arrives, and `path:line` cannot represent one in the path half."""
        assert source_from_location(path, lineno) == ""


class TestResultProperties:
    def test_every_test_carries_package_covers_and_source(self) -> None:
        item = FakeItem(
            "logging/test_x.py::TestFoo::test_bar",
            ("logging/test_x.py", 40, "TestFoo.test_bar"),
            (FakeMarker("covers", "LOG-1", "LOG-2"),),
        )
        assert result_properties(as_item(item)) == (
            ("package", "logging"),
            ("covers", "LOG-1,LOG-2"),
            ("source", "tests/e2e/logging/test_x.py:41"),
        )

    def test_attach_is_idempotent(self) -> None:
        """Collection can run the hook more than once; a second pass must not
        double the <property> entries in the report."""
        item = FakeItem("logging/test_x.py::test_bar", ("logging/test_x.py", 40, "test_bar"))
        attach_result_properties(as_item(item))
        attach_result_properties(as_item(item))
        assert [name for name, _ in item.user_properties] == ["package", "covers", "source"]


class TestSuiteRoot:
    def test_suite_root_names_this_file_s_real_home(self) -> None:
        """SUITE_ROOT is hardcoded because the runner image has no repo to read it
        from. Where there IS a checkout, prove the constant still points at us --
        otherwise a moved tests/e2e/ ships links that 404."""
        root = repo_root()
        if root is None:
            pytest.skip("no checkout above this file (the runner image copies tests/e2e/ to /app/e2e)")
        assert (root / SUITE_ROOT / Path(__file__).name).resolve() == Path(__file__).resolve()


class TestDedupeCovers:
    def test_ids_are_unique_order_preserving_and_non_empty_strings(self) -> None:
        assert dedupe_covers([("A", "B"), ("B", ""), ("C", 7)]) == ("A", "B", "C")
