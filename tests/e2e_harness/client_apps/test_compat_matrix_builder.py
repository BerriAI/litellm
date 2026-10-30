from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from client_apps.compat_results import Failed, NotApplicable, NotTested, Outcome, Passed, ResultRow, write_rows
from client_apps.matrix_builder import (
    REGRESSION_EXIT_CODE,
    REGRESSIONS,
    KnownGap,
    Manifest,
    ManifestFeature,
    Matrix,
    Regression,
    aggregate,
    build_matrix,
    find_regressions,
    load_manifest,
    main,
    pinned_client_version,
)

MANIFEST: Final = Manifest(
    schema_version="1",
    client="codex",
    package="@openai/codex",
    providers=("openai", "vertex_ai"),
    features=(ManifestFeature(id="vision", name="Vision"), ManifestFeature(id="web_search", name="Web search")),
)
GENERATED_AT: Final = datetime(2026, 1, 2, tzinfo=timezone.utc)


def _row(model: str | None, outcome: Outcome, feature_id: str = "vision", provider: str = "openai") -> ResultRow:
    return ResultRow(feature_id=feature_id, provider=provider, model=model, outcome=outcome)


def _matrix(rows: tuple[ResultRow, ...]) -> Matrix:
    return build_matrix(MANIFEST, rows, litellm_version="1.0.0", client_version="0.1.0", generated_at=GENERATED_AT)


@pytest.mark.parametrize(
    ("rows", "expected"),
    (
        ((_row("a", Passed()), _row("b", Passed())), Passed()),
        ((_row("a", Passed()), _row("b", Failed(error="no image"))), Failed(error="b: no image")),
        (
            (_row("a", Failed(error="x")), _row(None, Failed(error="setup broke"))),
            Failed(error="a: x; setup broke"),
        ),
        ((_row("a", Passed()), _row("b", NotTested())), Passed()),
        ((_row(None, NotApplicable(reason="no apply_patch")),), NotApplicable(reason="no apply_patch")),
        ((), NotTested()),
    ),
)
def test_a_cell_is_green_only_when_no_model_failed(rows: tuple[ResultRow, ...], expected: Outcome) -> None:
    assert aggregate(rows) == expected


def test_build_matrix_fills_every_manifest_cell_and_drops_unknown_rows() -> None:
    matrix: Final = _matrix(
        (
            _row("a", Passed()),
            _row("a", Failed(error="404"), feature_id="web_search", provider="vertex_ai"),
            _row("a", Passed(), feature_id="retired_feature"),
            _row("a", Passed(), provider="retired_provider"),
        )
    )

    assert [(feature.id, dict(feature.providers)) for feature in matrix.features] == [
        ("vision", {"openai": Passed(), "vertex_ai": NotTested()}),
        ("web_search", {"openai": NotTested(), "vertex_ai": Failed(error="a: 404")}),
    ]
    assert matrix.generated_at == GENERATED_AT.isoformat()


def test_find_regressions_flags_only_green_to_red() -> None:
    old: Final = _matrix(
        (
            _row("a", Passed()),
            _row("a", Failed(error="old"), provider="vertex_ai"),
            _row("a", Passed(), feature_id="web_search"),
        )
    )
    new: Final = _matrix(
        (
            _row("a", Failed(error="429")),
            _row("a", Failed(error="still red"), provider="vertex_ai"),
            _row("a", Failed(error="new cell"), feature_id="web_search", provider="vertex_ai"),
            _row("a", Passed(), feature_id="web_search"),
        )
    )

    assert find_regressions(old, new) == (
        Regression(feature_id="vision", feature_name="Vision", provider="openai", error="a: 429"),
    )


def test_matrix_round_trips_through_json() -> None:
    matrix: Final = _matrix((_row("a", Passed()), _row(None, NotApplicable(reason="n/a"), provider="vertex_ai")))

    assert Matrix.model_validate_json(matrix.model_dump_json()) == matrix


def test_cli_build_then_regressions_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest: Final = tmp_path / "manifest.yaml"
    _ = manifest.write_text(json.dumps(MANIFEST.model_dump(mode="json")))
    green_rows: Final = tmp_path / "green.json"
    red_rows: Final = tmp_path / "red.json"
    write_rows(green_rows, (_row("a", Passed()),))
    write_rows(red_rows, (_row("a", Failed(error="400 from proxy")),))
    green: Final = tmp_path / "green-matrix.json"
    red: Final = tmp_path / "red-matrix.json"
    build: Final = ("build", "--manifest", str(manifest), "--litellm-version", "9.9.9")

    assert main((*build, "--results", str(green_rows), "--out", str(green))) == 0
    assert main((*build, "--results", str(red_rows), "--out", str(red))) == 0
    built: Final = Matrix.model_validate_json(red.read_bytes())
    assert (built.litellm_version, built.client_version) == ("9.9.9", pinned_client_version("@openai/codex"))
    _ = capsys.readouterr()

    assert main(("regressions", "--old", str(green), "--new", str(red))) == REGRESSION_EXIT_CODE
    regressions: Final = REGRESSIONS.validate_json(capsys.readouterr().out)
    assert [regression.error for regression in regressions] == ["a: 400 from proxy"]
    assert main(("regressions", "--old", str(red), "--new", str(green))) == 0
    assert main(("regressions", "--old", str(tmp_path / "missing.json"), "--new", str(red))) == 0


def test_shipped_manifests_validate_and_pin_their_client() -> None:
    manifests: Final = tuple(Path(__file__).resolve().parents[2].glob("e2e/client_apps/*/manifest.yaml"))

    assert manifests
    for path in manifests:
        manifest = load_manifest(path)
        assert len({feature.id for feature in manifest.features}) == len(manifest.features)
        assert pinned_client_version(manifest.package)



def _manifest_yaml(known_gaps: str) -> str:
    return f"""schema_version: "1"
client: codex
package: "@openai/codex"
providers: [openai, vertex_ai]
features:
  - {{id: vision, name: Vision}}
  - {{id: web_search, name: Web search}}
known_gaps: {known_gaps}
"""


def test_known_gaps_are_read_from_the_manifest_yaml(tmp_path: Path) -> None:
    path: Final = tmp_path / "manifest.yaml"
    _ = path.write_text(_manifest_yaml("[{feature: web_search, provider: vertex_ai, reason: drops search tools}]"))

    manifest: Final = load_manifest(path)

    assert manifest.known_gap("web_search", "vertex_ai") == KnownGap(
        feature="web_search", provider="vertex_ai", reason="drops search tools"
    )
    assert manifest.known_gap("web_search", "openai") is None
    assert manifest.known_gap("vision", "vertex_ai") is None
    assert MANIFEST.known_gaps == ()


@pytest.mark.parametrize(
    ("known_gaps", "problem"),
    (
        ("[{feature: web_serch, provider: openai, reason: r}]", "unknown feature 'web_serch'"),
        ("[{feature: vision, provider: azure, reason: r}]", "unknown provider 'azure'"),
        (
            "[{feature: vision, provider: openai, reason: a}, {feature: vision, provider: openai, reason: b}]",
            "known_gaps lists a cell more than once: vision/openai",
        ),
    ),
)
def test_a_known_gap_must_name_a_single_manifest_cell(tmp_path: Path, known_gaps: str, problem: str) -> None:
    path: Final = tmp_path / "manifest.yaml"
    _ = path.write_text(_manifest_yaml(known_gaps))

    with pytest.raises(ValidationError, match=problem):
        _ = load_manifest(path)


def test_a_known_gap_needs_a_reason(tmp_path: Path) -> None:
    path: Final = tmp_path / "manifest.yaml"
    _ = path.write_text(_manifest_yaml("[{feature: vision, provider: openai, reason: ''}]"))

    with pytest.raises(ValidationError) as raised:
        _ = load_manifest(path)

    assert [error["loc"] for error in raised.value.errors()] == [("known_gaps", 0, "reason")]
