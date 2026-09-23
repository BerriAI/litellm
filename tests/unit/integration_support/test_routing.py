from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

from tests.integration._support.routing import (
    DIFF_FILE,
    OBSERVED_FILE,
    READER_ROLE,
    WRITER_ROLE,
    Mismatch,
    Observation,
    compare,
    delta,
    dump_observation,
    load_either_role,
    load_observation,
    main,
    normalize,
    render,
    role_calls,
)

TOKEN_QUERY: Final = 'UPDATE "LiteLLM_VerificationToken" SET token = $n WHERE token = $n'
NODE_ID: Final = "tests/integration/management/test_keys.py::test_generate"


def _routing(entries: dict[str, tuple[str, ...]]) -> MappingProxyType[str, frozenset[str]]:
    return MappingProxyType({query: frozenset(roles) for query, roles in entries.items()})


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
    base: Final = _observation({TOKEN_QUERY: ("litellm_reader",), "SELECT 1": ("litellm_writer",)})
    head: Final = _observation({TOKEN_QUERY: ("litellm_writer",), "SELECT 1": ("litellm_writer",)})
    report: Final = compare(base, head)
    assert report.mismatches == (Mismatch(None, TOKEN_QUERY, ("litellm_reader",), ("litellm_writer",)),)
    assert report.failures() == (f"global: {TOKEN_QUERY}: base [litellm_reader] head [litellm_writer]",)


def test_compare_reports_global_shrink_mismatch() -> None:
    base: Final = _observation({TOKEN_QUERY: ("litellm_reader", "litellm_writer")})
    head: Final = _observation({TOKEN_QUERY: ("litellm_writer",)})
    report: Final = compare(base, head)
    assert report.mismatches == (
        Mismatch(None, TOKEN_QUERY, ("litellm_reader", "litellm_writer"), ("litellm_writer",)),
    )
    assert report.failures() == (f"global: {TOKEN_QUERY}: base [litellm_reader, litellm_writer] head [litellm_writer]",)


def test_compare_reports_per_test_mismatch_with_nodeid() -> None:
    base: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    head: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_writer",)}},
    )
    report: Final = compare(base, head)
    assert report.mismatches == (Mismatch(NODE_ID, TOKEN_QUERY, ("litellm_reader",), ("litellm_writer",)),)
    assert report.failures() == (f"{NODE_ID}: {TOKEN_QUERY}: base [litellm_reader] head [litellm_writer]",)


def test_compare_per_test_mismatch_ignores_global_observation() -> None:
    base: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader", "litellm_writer")},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    head: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader", "litellm_writer")},
        {NODE_ID: {TOKEN_QUERY: ("litellm_writer",)}},
    )
    report: Final = compare(base, head)
    assert report.mismatches == (Mismatch(NODE_ID, TOKEN_QUERY, ("litellm_reader",), ("litellm_writer",)),)
    assert report.failures() == (f"{NODE_ID}: {TOKEN_QUERY}: base [litellm_reader] head [litellm_writer]",)


def test_compare_reports_per_test_shrink_mismatch() -> None:
    base: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader", "litellm_writer")},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader", "litellm_writer")}},
    )
    head: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader", "litellm_writer")},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    report: Final = compare(base, head)
    assert report.mismatches == (
        Mismatch(NODE_ID, TOKEN_QUERY, ("litellm_reader", "litellm_writer"), ("litellm_reader",)),
    )
    assert report.failures() == (
        f"{NODE_ID}: {TOKEN_QUERY}: base [litellm_reader, litellm_writer] head [litellm_reader]",
    )


def test_compare_reports_global_gain_mismatch() -> None:
    base: Final = _observation({TOKEN_QUERY: ("litellm_reader",)})
    head: Final = _observation({TOKEN_QUERY: ("litellm_reader", "litellm_writer")})
    report: Final = compare(base, head)
    assert report.mismatches == (
        Mismatch(None, TOKEN_QUERY, ("litellm_reader",), ("litellm_reader", "litellm_writer")),
    )
    assert report.failures() == (f"global: {TOKEN_QUERY}: base [litellm_reader] head [litellm_reader, litellm_writer]",)


def test_compare_reports_per_test_gain_mismatch() -> None:
    base: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    head: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader", "litellm_writer")}},
    )
    report: Final = compare(base, head)
    assert report.mismatches == (
        Mismatch(NODE_ID, TOKEN_QUERY, ("litellm_reader",), ("litellm_reader", "litellm_writer")),
    )
    assert report.failures() == (
        f"{NODE_ID}: {TOKEN_QUERY}: base [litellm_reader] head [litellm_reader, litellm_writer]",
    )


def test_compare_either_role_suppresses_and_reports_variance() -> None:
    base: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader", "litellm_writer"), "SELECT quiet": ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    head: Final = _observation(
        {TOKEN_QUERY: ("litellm_writer",), "SELECT quiet": ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_writer",)}},
    )
    report: Final = compare(base, head, either_role=frozenset({TOKEN_QUERY, "SELECT quiet"}))
    assert report.mismatches == ()
    assert report.failures() == ()
    assert report.either_role == (TOKEN_QUERY,)
    assert "== either role ==\n" + TOKEN_QUERY + "\n" in render(report)


def test_compare_either_role_matches_exact_keys_only() -> None:
    base: Final = _observation(
        {
            "SELECT $n": ("litellm_reader",),
            "SELECT $n FROM x": ("litellm_reader",),
            "SELECT $n FROM x WHERE y = $n": ("litellm_reader",),
        }
    )
    head: Final = _observation(
        {
            "SELECT $n": ("litellm_writer",),
            "SELECT $n FROM x": ("litellm_writer",),
            "SELECT $n FROM x WHERE y = $n": ("litellm_writer",),
        }
    )
    report: Final = compare(base, head, either_role=frozenset({"SELECT $n FROM x"}))
    assert frozenset(mismatch.query for mismatch in report.mismatches) == frozenset(
        {"SELECT $n", "SELECT $n FROM x WHERE y = $n"}
    )
    other: Final = compare(base, head, either_role=frozenset({"SELECT $n"}))
    assert frozenset(mismatch.query for mismatch in other.mismatches) == frozenset(
        {"SELECT $n FROM x", "SELECT $n FROM x WHERE y = $n"}
    )


def test_compare_one_sided_queries_are_listed_not_failed() -> None:
    base: Final = _observation({"SELECT a": ("litellm_reader",), "SELECT gone": ("litellm_writer",)})
    head: Final = _observation({"SELECT a": ("litellm_reader",), "SELECT new": ("litellm_writer",)})
    report: Final = compare(base, head)
    assert report.only_base == ("SELECT gone",)
    assert report.only_head == ("SELECT new",)
    assert report.mismatches == ()
    assert report.failures() == ()


def test_failures_flags_dealloc_evictions_on_base() -> None:
    report: Final = compare(_observation({}, dealloc=1), _observation({}))
    assert report.failures() == ("base: pg_stat_statements evicted 1 entries (dealloc > 0)",)


def test_failures_flags_dealloc_evictions_on_head() -> None:
    report: Final = compare(_observation({}), _observation({}, dealloc=1))
    assert report.failures() == ("head: pg_stat_statements evicted 1 entries (dealloc > 0)",)


def test_failures_flags_silent_reader_on_base() -> None:
    report: Final = compare(
        _observation({}, calls={"litellm_reader": 0, "litellm_writer": 5}),
        _observation({}),
    )
    assert report.failures() == ("base: no litellm_reader calls observed",)


def test_failures_flags_silent_reader_on_head() -> None:
    report: Final = compare(
        _observation({}),
        _observation({}, calls={"litellm_reader": 0, "litellm_writer": 5}),
    )
    assert report.failures() == ("head: no litellm_reader calls observed",)


def test_failures_flags_silent_writer() -> None:
    report: Final = compare(
        _observation({}, calls={"litellm_reader": 5, "litellm_writer": 0}),
        _observation({}),
    )
    assert report.failures() == ("base: no litellm_writer calls observed",)


def test_failures_counts_missing_role_as_silent() -> None:
    report: Final = compare(_observation({}), _observation({}, calls={"litellm_writer": 5}))
    assert report.failures() == ("head: no litellm_reader calls observed",)


def test_compare_skips_per_test_mismatches_for_xdist_shape() -> None:
    base: Final = _observation(
        {TOKEN_QUERY: ("litellm_reader",)},
        {NODE_ID: {TOKEN_QUERY: ("litellm_reader",)}},
    )
    head: Final = _observation({TOKEN_QUERY: ("litellm_reader",)})
    assert head.tests == {}
    report: Final = compare(base, head)
    assert report.mismatches == ()
    assert report.failures() == ()


class _WriterFirst(frozenset[str]):
    def __iter__(self) -> Iterator[str]:
        return iter((WRITER_ROLE, READER_ROLE))


def test_dump_observation_sorts_role_lists_and_round_trips(tmp_path: Path) -> None:
    queries: Final = [f"SELECT {index}" for index in range(4)]
    observation: Final = Observation(
        MappingProxyType({query: _WriterFirst({WRITER_ROLE, READER_ROLE}) for query in queries}),
        MappingProxyType(
            {NODE_ID: MappingProxyType({query: _WriterFirst({WRITER_ROLE, READER_ROLE}) for query in queries})}
        ),
        MappingProxyType({READER_ROLE: 1, WRITER_ROLE: 2}),
        0,
    )
    expected: Final = (
        json.dumps(
            {
                "queries": {query: ["litellm_reader", "litellm_writer"] for query in queries},
                "tests": {NODE_ID: {query: ["litellm_reader", "litellm_writer"] for query in queries}},
                "calls": {"litellm_reader": 1, "litellm_writer": 2},
                "dealloc": 0,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    dumped: Final = dump_observation(observation)
    assert dumped == expected
    path: Final = tmp_path / OBSERVED_FILE
    path.write_text(dumped)
    loaded: Final = load_observation(path)
    assert loaded.queries == _routing({query: (WRITER_ROLE, READER_ROLE) for query in queries})
    assert loaded.tests == {NODE_ID: loaded.queries}


def _write_observed(results: Path, observation: Observation) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / OBSERVED_FILE).write_text(dump_observation(observation))


def test_main_check_returns_zero_for_matching_routes(tmp_path: Path) -> None:
    base_dir: Final = tmp_path / "base"
    head_dir: Final = tmp_path / "head"
    observation: Final = _observation({TOKEN_QUERY: ("litellm_reader",)})
    _write_observed(base_dir, observation)
    _write_observed(head_dir, observation)
    assert main(["check", str(base_dir), str(head_dir)]) == 0
    diff: Final = (tmp_path / DIFF_FILE).read_text()
    assert "== failures ==\nnone\n" in diff


def test_main_check_returns_one_and_writes_exact_diff(tmp_path: Path) -> None:
    base_dir: Final = tmp_path / "parity" / "base"
    head_dir: Final = tmp_path / "parity" / "head"
    _write_observed(
        base_dir,
        _observation({TOKEN_QUERY: ("litellm_reader",), "SELECT absent": ("litellm_writer",)}),
    )
    _write_observed(
        head_dir,
        _observation(
            {TOKEN_QUERY: ("litellm_writer",)},
            calls={"litellm_reader": 0, "litellm_writer": 5},
            dealloc=2,
        ),
    )
    assert main(["check", str(base_dir), str(head_dir)]) == 1
    assert (head_dir.parent / DIFF_FILE).read_text() == (
        "== failures ==\n"
        f"global: {TOKEN_QUERY}: base [litellm_reader] head [litellm_writer]\n"
        "head: pg_stat_statements evicted 2 entries (dealloc > 0)\n"
        "head: no litellm_reader calls observed\n"
        "\n"
        "== either role ==\n"
        "none\n"
        "\n"
        "== queries only in base ==\n"
        "SELECT absent\n"
        "\n"
        "== queries only in head ==\n"
        "none\n"
        "\n"
        "== calls ==\n"
        "base litellm_reader: 3\n"
        "base litellm_writer: 7\n"
        "base dealloc: 0\n"
        "head litellm_reader: 0\n"
        "head litellm_writer: 5\n"
        "head dealloc: 2\n"
    )


def test_main_check_missing_observed_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base_dir: Final = tmp_path / "base"
    head_dir: Final = tmp_path / "head"
    _write_observed(base_dir, _observation({}))
    head_dir.mkdir()
    assert main(["check", str(base_dir), str(head_dir)]) == 1
    assert "observed routing file missing" in capsys.readouterr().err


def test_main_check_either_role_suppresses_shrink(tmp_path: Path) -> None:
    base_dir: Final = tmp_path / "base"
    head_dir: Final = tmp_path / "head"
    _write_observed(base_dir, _observation({TOKEN_QUERY: ("litellm_reader", "litellm_writer")}))
    _write_observed(head_dir, _observation({TOKEN_QUERY: ("litellm_writer",)}))
    argv: Final = ["check", str(base_dir), str(head_dir)]
    allowlist: Final = tmp_path / "either.json"
    allowlist.write_text(json.dumps({TOKEN_QUERY: "timer probe may use either pool"}))
    assert main([*argv, "--either-role", str(allowlist)]) == 0
    assert "== either role ==\n" + TOKEN_QUERY + "\n" in (tmp_path / DIFF_FILE).read_text()
    assert main(argv) == 1


def test_delta_maps_positive_increases_per_role() -> None:
    before: Final = MappingProxyType(
        {
            ("litellm_reader", "SELECT both"): 1,
            ("litellm_writer", "SELECT both"): 2,
            ("litellm_reader", "SELECT reader"): 3,
            ("litellm_writer", "SELECT gone"): 4,
            ("litellm_reader", "SELECT same"): 5,
        }
    )
    after: Final = MappingProxyType(
        {
            ("litellm_reader", "SELECT both"): 2,
            ("litellm_writer", "SELECT both"): 5,
            ("litellm_reader", "SELECT reader"): 6,
            ("litellm_reader", "SELECT same"): 5,
            ("litellm_writer", "SELECT writer"): 7,
        }
    )
    assert delta(before, after) == {
        "SELECT both": frozenset({"litellm_reader", "litellm_writer"}),
        "SELECT reader": frozenset({"litellm_reader"}),
        "SELECT writer": frozenset({"litellm_writer"}),
    }


def test_role_calls_sums_positive_increases_per_role() -> None:
    before: Final = MappingProxyType(
        {
            ("litellm_reader", "SELECT a"): 10,
            ("litellm_reader", "SELECT b"): 4,
            ("litellm_writer", "SELECT a"): 1,
        }
    )
    after: Final = MappingProxyType(
        {
            ("litellm_reader", "SELECT a"): 11,
            ("litellm_reader", "SELECT b"): 2,
            ("litellm_writer", "SELECT a"): 1,
            ("litellm_writer", "SELECT c"): 6,
        }
    )
    assert role_calls(before, after) == {"litellm_reader": 1, "litellm_writer": 6}


def test_load_either_role_missing_path_returns_empty(tmp_path: Path) -> None:
    assert load_either_role(tmp_path / "absent.json") == frozenset()


def test_load_either_role_reads_query_keys(tmp_path: Path) -> None:
    path: Final = tmp_path / "either.json"
    path.write_text(json.dumps({"SELECT $n": "probe", "SELECT now()": "clock"}))
    assert load_either_role(path) == frozenset({"SELECT $n", "SELECT now()"})


def test_load_observation_reads_calls_and_dealloc(tmp_path: Path) -> None:
    observation: Final = _observation({TOKEN_QUERY: ("litellm_reader",)}, dealloc=0)
    path: Final = tmp_path / OBSERVED_FILE
    path.write_text(dump_observation(observation))
    loaded: Final = load_observation(path)
    assert loaded == observation
