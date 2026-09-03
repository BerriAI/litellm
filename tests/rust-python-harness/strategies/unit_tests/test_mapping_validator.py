from __future__ import annotations

import pytest

from .mapping_validator import TestMapping as Mapping, validate_mapping


def test_matches_names_and_explicit_annotations() -> None:
    report = validate_mapping(
        ("tests/test_api.py::test_decode", "tests/test_api.py::test_error"),
        ("api::test_decode", "api::preserves_error"),
        (Mapping(python="tests/test_api.py::test_error", rust="api::preserves_error"),),
    )
    assert report.problems == ()
    assert {(pair.python, pair.rust) for pair in report.pairs} == {
        ("tests/test_api.py::test_decode", "api::test_decode"),
        ("tests/test_api.py::test_error", "api::preserves_error"),
    }


@pytest.mark.parametrize(
    ("python", "rust", "message"),
    (
        (("test_decode",), (), "missing Rust counterpart"),
        ((), ("test_decode",), "missing Python counterpart"),
        (("test_decode",), ("one::test_decode", "two::test_decode"), "ambiguous Rust counterparts"),
        (("one::test_decode", "two::test_decode"), ("test_decode",), "ambiguous Python counterparts"),
    ),
)
def test_reports_missing_and_ambiguous_counterparts(
    python: tuple[str, ...], rust: tuple[str, ...], message: str
) -> None:
    assert any(message in problem for problem in validate_mapping(python, rust).problems)


def test_rejects_stale_annotations_even_when_names_match() -> None:
    report = validate_mapping(("test_decode",), ("test_decode",), (Mapping(python="test_decode", rust="removed"),))
    assert "missing Rust counterpart: removed" in report.problems
