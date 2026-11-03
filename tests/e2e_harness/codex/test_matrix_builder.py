from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter
from codex.compat_results import CellResult, Failed, KnownGap, NotTested, Outcome, Passed, append_result
from codex.matrix_builder import (
    REGRESSION_EXIT_CODE,
    Matrix,
    MatrixRow,
    aggregate,
    build_matrix,
    find_regressions,
    main,
    setup_columns,
)
from codex.setups import SETUPS

CLAUDE_CODE_SAMPLE: Final = (
    Path(__file__).resolve().parents[2] / "e2e" / "claude_code" / "sample_compatibility-matrix.json"
)
GENERATED_AT: Final = datetime(2026, 1, 2, tzinfo=timezone.utc)
JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
JSON_OBJECTS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


def _result(nodeid: str, outcome: Outcome, *, row: str = "vision", setup: str = "openai") -> CellResult:
    return CellResult(nodeid=nodeid, row=row, row_name=row.capitalize(), setup=setup, outcome=outcome)


def _matrix(cells: MappingProxyType[tuple[str, str], Outcome]) -> Matrix:
    rows: Final = sorted(frozenset(row for row, _ in cells))
    return Matrix(
        schema_version="1",
        generated_at=GENERATED_AT.isoformat(),
        litellm_version="1.0.0",
        codex_version="0.1.0",
        providers=tuple(sorted(frozenset(column for _, column in cells))),
        features=tuple(
            MatrixRow(
                id=row,
                name=row.capitalize(),
                providers=MappingProxyType(
                    {column: outcome for (cell_row, column), outcome in cells.items() if cell_row == row}
                ),
            )
            for row in rows
        ),
    )


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    (
        ((Passed(), Failed(error="a"), Failed(error="b")), Failed(error="a; b")),
        ((Passed(), KnownGap(reason="no search route")), KnownGap(reason="no search route")),
        ((NotTested(reason="no credentials"), Passed()), Passed()),
        ((NotTested(reason="no credentials"),), NotTested(reason="no credentials")),
        ((), NotTested()),
    ),
)
def test_a_cell_is_as_bad_as_its_worst_test(outcomes: tuple[Outcome, ...], expected: Outcome) -> None:
    assert aggregate(outcomes) == expected


@pytest.mark.parametrize(
    ("attempts", "expected"),
    (
        ((Failed(error="502"), Passed()), Passed()),
        ((Passed(), Failed(error="timed out")), Failed(error="timed out")),
    ),
)
def test_the_last_attempt_of_a_test_decides_its_cell(attempts: tuple[Outcome, ...], expected: Outcome) -> None:
    matrix: Final = build_matrix(
        tuple(_result("test_vision_e2e.py::TestCodexVision::test_cell[openai]", outcome) for outcome in attempts),
        columns=("openai", "vertex_ai"),
        litellm_version="1.0.0",
        codex_version="0.1.0",
        generated_at=GENERATED_AT,
    )

    assert matrix.features[0].providers == {"openai": expected, "vertex_ai": NotTested()}


def test_every_setup_is_a_column_and_an_unknown_setup_is_kept() -> None:
    declared: Final = tuple(setup.id for setup in SETUPS)

    columns: Final = setup_columns((_result("a", Passed(), setup="zz_new_setup"), _result("b", Passed())))

    assert columns == (*declared, "zz_new_setup")


def test_regressions_are_cells_that_were_green_and_are_not_anymore() -> None:
    old: Final = _matrix(
        MappingProxyType(
            {
                ("vision", "openai"): Passed(),
                ("vision", "vertex_ai"): Passed(),
                ("vision", "azure_openai"): Passed(),
                ("vision", "bedrock_runtime"): Passed(),
                ("web_search", "openai"): Failed(error="old"),
                ("web_search", "vertex_ai"): NotTested(),
                ("web_search", "azure_openai"): KnownGap(reason="no route"),
                ("removed_row", "openai"): Passed(),
            }
        )
    )
    new: Final = _matrix(
        MappingProxyType(
            {
                ("vision", "openai"): Failed(error="no image"),
                ("vision", "vertex_ai"): KnownGap(reason="drops images"),
                ("vision", "azure_openai"): NotTested(reason="no credentials"),
                ("vision", "bedrock_runtime"): Passed(),
                ("web_search", "openai"): Failed(error="new"),
                ("web_search", "vertex_ai"): Failed(error="new"),
                ("web_search", "azure_openai"): Passed(),
                ("new_row", "openai"): Failed(error="new row"),
            }
        )
    )

    regressions: Final = find_regressions(old, new)

    assert frozenset((r.feature_id, r.provider, r.new_status, r.error) for r in regressions) == frozenset(
        (
            ("vision", "openai", "fail", "no image"),
            ("vision", "vertex_ai", "known_gap", "drops images"),
            ("vision", "azure_openai", "not_tested", "no credentials"),
        )
    )


def test_the_matrix_has_the_claude_code_matrix_shape() -> None:
    sample: Final = JSON_OBJECT.validate_json(CLAUDE_CODE_SAMPLE.read_bytes())
    matrix: Final = build_matrix(
        (_result("a", Passed()), _result("b", Failed(error="boom"), setup="vertex_ai")),
        columns=("openai", "vertex_ai"),
        litellm_version="1.0.0",
        codex_version="0.1.0",
        generated_at=GENERATED_AT,
    )

    built: Final = JSON_OBJECT.validate_json(matrix.model_dump_json(exclude_none=True))
    sample_row: Final = JSON_OBJECTS.validate_python(sample["features"])[0]
    built_row: Final = JSON_OBJECTS.validate_python(built["features"])[0]

    assert frozenset(built) == (frozenset(sample) - {"claude_code_version"}) | {"codex_version"}
    assert frozenset(built_row) == frozenset(sample_row)
    assert built_row["providers"] == {"openai": {"status": "pass"}, "vertex_ai": {"status": "fail", "error": "boom"}}


def test_cli_builds_the_matrix_and_fails_on_a_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    green_results: Final = tmp_path / "green.jsonl"
    red_results: Final = tmp_path / "red.jsonl"
    append_result(green_results, _result("a", Passed()))
    append_result(red_results, _result("a", Failed(error="no image")))
    green: Final = tmp_path / "green.json"
    red: Final = tmp_path / "red.json"

    built: Final = tuple(
        main(("build", "--results", str(results), "--out", str(out), "--litellm-version", "1.2.3"))
        for results, out in ((green_results, green), (red_results, red))
    )
    _ = capsys.readouterr()
    no_baseline: Final = main(("regressions", "--old", str(tmp_path / "missing.json"), "--new", str(red)))
    steady: Final = main(("regressions", "--old", str(green), "--new", str(green)))
    _ = capsys.readouterr()
    regressed: Final = main(("regressions", "--old", str(green), "--new", str(red)))

    assert built == (0, 0)
    assert Matrix.model_validate_json(red.read_bytes()).litellm_version == "1.2.3"
    assert (no_baseline, steady, regressed) == (0, 0, REGRESSION_EXIT_CODE)
    assert JSON_OBJECTS.validate_json(capsys.readouterr().out) == (
        {
            "feature_id": "vision",
            "feature_name": "Vision",
            "provider": "openai",
            "old_status": "pass",
            "new_status": "fail",
            "error": "no image",
        },
    )
