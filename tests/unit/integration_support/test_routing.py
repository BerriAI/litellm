from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

from tests.integration._support.routing import (
    DIFF_FILE,
    OBSERVED_FILE,
    RECORDED_FILE,
    Expectation,
    Mismatch,
    Observation,
    compare,
    dump_expectation,
    dump_observation,
    load_expectation,
    load_observation,
    main,
    normalize,
)

TOKEN_QUERY: Final = 'UPDATE "LiteLLM_VerificationToken" SET token = $n WHERE token = $n'
NODE_ID: Final = "tests/integration/management/test_keys.py::test_generate"


def _routing(entries: dict[str, tuple[str, ...]]) -> MappingProxyType[str, frozenset[str]]:
    return MappingProxyType({query: frozenset(roles) for query, roles in entries.items()})


def _expectation(
    queries: dict[str, tuple[str, ...]],
    tests: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> Expectation:
    return Expectation(
        _routing(queries),
        MappingProxyType({node: _routing(mapping) for node, mapping in (tests or {}).items()}),
    )


def _observation(
    queries: dict[str, tuple[str, ...]],
    tests: dict[str, dict[str, tuple[str, ...]]] | None = None,
    calls: dict[str, int] | None = None,
    dealloc: int = 0,
) -> Observation:
    return Observation(
        _routing(queries),
        MappingProxyType({node: _routing(mapping) for node, mapping in (tests or {}).items()}),
        MappingProxyType(calls if calls is not None else {"litellm_reader": 3, "litellm_writer": 7}),
        dealloc,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SELECT  a\n FROM t", "SELECT a FROM t"),
        ("SELECT * FROM t WHERE id IN ($1, $2, $3)", "SELECT * FROM t WHERE id IN ($n)"),
        ("SELECT * FROM t WHERE id IN ($1,$2)", "SELECT * FROM t WHERE id IN ($n)"),
        ("SELECT * FROM t WHERE id IN ($4)", "SELECT * FROM t WHERE id IN ($n)"),
        (
            "INSERT INTO t VALUES ($1, $2) ON CONFLICT ($3, $4, $5) DO NOTHING",
            "INSERT INTO t VALUES ($n) ON CONFLICT ($n) DO NOTHING",
        ),
    ],
)
def test_normalize_collapses_whitespace_and_placeholders(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_compare_reports_global_role_mismatch() -> None:
    expected: Final = _expectation({TOKEN_QUERY: ("litellm_reader",), "SELECT 1": ("litellm_writer",)})
    observed: Final = _observation({TOKEN_QUERY: ("litellm_writer",), "SELECT 1": ("litellm_writer",)})
    report: Final = compare(expected, observed)
    assert report.mismatches == (Mismatch(None, TOKEN_QUERY, ("litellm_reader",), ("litellm_writer",)),)
    assert report.failures() == (f"global: {TOKEN_QUERY}: expected [litellm_reader] observed [litellm_writer]",)


def test_compare_allows_observed_roles_to_shrink() -> None:
    expected: Final = _expectation({TOKEN_QUERY: ("litellm_reader", "litellm_writer")})
    observed: Final = _observation({TOKEN_QUERY: ("litellm_reader",)})
    report: Final = compare(expected, observed)
    assert report.mismatches == ()
    assert report.failures() == ()


def test_compare_reports_per_test_mismatch_with_nodeid() -> None:
    expected: Final = _expectation(
        {},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    observed: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_writer",)}},
    )
    report: Final = compare(expected, observed)
    assert report.mismatches == (Mismatch(NODE_ID, TOKEN_QUERY, ("litellm_reader",), ("litellm_writer",)),)
    assert report.failures() == (f"{NODE_ID}: {TOKEN_QUERY}: expected [litellm_reader] observed [litellm_writer]",)


def test_compare_per_test_allows_roles_in_global_expectation() -> None:
    expected: Final = _expectation(
        {TOKEN_QUERY: ("litellm_reader", "litellm_writer")},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    observed: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader", "litellm_writer")},
        {NODE_ID: {TOKEN_QUERY: ("litellm_writer",)}},
    )
    report: Final = compare(expected, observed)
    assert report.mismatches == ()
    assert report.failures() == ()


def test_compare_one_sided_queries_are_listed_not_failed() -> None:
    expected: Final = _expectation({"SELECT a": ("litellm_reader",), "SELECT gone": ("litellm_writer",)})
    observed: Final = _observation({"SELECT a": ("litellm_reader",), "SELECT new": ("litellm_writer",)})
    report: Final = compare(expected, observed)
    assert report.only_expected == ("SELECT gone",)
    assert report.only_observed == ("SELECT new",)
    assert report.mismatches == ()
    assert report.failures() == ()


def test_failures_flags_dealloc_evictions() -> None:
    report: Final = compare(_expectation({}), _observation({}, dealloc=1))
    assert report.failures() == ("pg_stat_statements evicted 1 entries (dealloc > 0)",)


def test_failures_flags_silent_reader() -> None:
    report: Final = compare(_expectation({}), _observation({}, calls={"litellm_reader": 0, "litellm_writer": 5}))
    assert report.failures() == ("no litellm_reader calls observed",)


def test_failures_flags_silent_writer() -> None:
    report: Final = compare(_expectation({}), _observation({}, calls={"litellm_reader": 5, "litellm_writer": 0}))
    assert report.failures() == ("no litellm_writer calls observed",)


def test_failures_counts_missing_role_as_silent() -> None:
    report: Final = compare(_expectation({}), _observation({}, calls={"litellm_writer": 5}))
    assert report.failures() == ("no litellm_reader calls observed",)


def test_compare_skips_per_test_mismatches_for_xdist_shape() -> None:
    expected: Final = _expectation(
        {TOKEN_QUERY: ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    observed: Final = _observation({TOKEN_QUERY: ("litellm_reader",)})
    assert observed.tests == {}
    report: Final = compare(expected, observed)
    assert report.mismatches == ()
    assert report.failures() == ()


def test_load_expectation_round_trips_through_dump(tmp_path: Path) -> None:
    observation: Final = _observation(
        {TOKEN_QUERY: ("litellm_writer",), "SELECT a": ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_writer",)}},
    )
    path: Final = tmp_path / "expected.json"
    path.write_text(dump_expectation(observation))
    loaded: Final = load_expectation(path)
    assert loaded.queries == observation.queries
    assert loaded.tests == observation.tests
    assert dump_expectation(observation) == dump_expectation(observation)
    assert json.loads(dump_expectation(observation))["queries"] == {
        TOKEN_QUERY: ["litellm_writer"],
        "SELECT a": ["litellm_reader"],
    }
    assert "calls" not in json.loads(dump_expectation(observation))


def _write_observed(results: Path, observation: Observation) -> None:
    (results / OBSERVED_FILE).write_text(dump_observation(observation))


def test_main_record_writes_dump_expectation_bytes(tmp_path: Path) -> None:
    observation: Final = _observation({TOKEN_QUERY: ("litellm_writer",)})
    _write_observed(tmp_path, observation)
    assert main(["record", "database", str(tmp_path)]) == 0
    assert (tmp_path / RECORDED_FILE).read_text() == dump_expectation(observation)


def test_main_check_returns_zero_for_matching_routes(tmp_path: Path) -> None:
    observation: Final = _observation({TOKEN_QUERY: ("litellm_reader",)})
    _write_observed(tmp_path, observation)
    expected: Final = tmp_path / "expected.json"
    expected.write_text(dump_expectation(observation))
    assert main(["check", "database", str(tmp_path), "--expected", str(expected)]) == 0
    diff: Final = (tmp_path / DIFF_FILE).read_text()
    assert "== failures ==\nnone\n" in diff


def test_main_check_returns_one_and_writes_exact_diff(tmp_path: Path) -> None:
    _write_observed(
        tmp_path,
        _observation(
            {TOKEN_QUERY: ("litellm_writer",)},
            calls={"litellm_reader": 0, "litellm_writer": 5},
            dealloc=2,
        ),
    )
    expected: Final = tmp_path / "expected.json"
    expected.write_text(
        json.dumps(
            {
                "queries": {TOKEN_QUERY: ["litellm_reader"], "SELECT absent": ["litellm_writer"]},
                "tests": {},
            }
        )
    )
    assert main(["check", "database", str(tmp_path), "--expected", str(expected)]) == 1
    assert (tmp_path / DIFF_FILE).read_text() == (
        "== failures ==\n"
        f"global: {TOKEN_QUERY}: expected [litellm_reader] observed [litellm_writer]\n"
        "pg_stat_statements evicted 2 entries (dealloc > 0)\n"
        "no litellm_reader calls observed\n"
        "\n"
        "== queries only in expected ==\n"
        "SELECT absent\n"
        "\n"
        "== queries only in observed ==\n"
        "none\n"
        "\n"
        "== calls ==\n"
        "litellm_reader: 0\n"
        "litellm_writer: 5\n"
        "dealloc: 2\n"
    )


def test_main_check_missing_expected_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_observed(tmp_path, _observation({}))
    assert main(["check", "database", str(tmp_path), "--expected", str(tmp_path / "nope.json")]) == 1
    assert "expected routing file missing" in capsys.readouterr().err


def test_load_observation_reads_calls_and_dealloc(tmp_path: Path) -> None:
    observation: Final = _observation({TOKEN_QUERY: ("litellm_reader",)}, dealloc=0)
    path: Final = tmp_path / OBSERVED_FILE
    path.write_text(dump_observation(observation))
    loaded: Final = load_observation(path)
    assert loaded == observation
