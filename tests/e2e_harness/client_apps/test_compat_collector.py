from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

import pytest

from client_apps.compat_collector import (
    CLIENT_APPS_DIR,
    NO_RESULT_ERROR,
    CellAddress,
    cell_address,
    clear_shards,
    outcome_rows,
    publish,
)
from client_apps.compat_results import Failed, NotApplicable, Outcome, Passed, ResultRow, read_rows, shard_dir
from client_apps.matrix_builder import build_matrix, load_manifest

pytest_plugins = ("pytester",)

MANIFEST: Final = """schema_version: "1"
client: fakeclient
package: fake-client
providers: [openai, vertex_ai]
features:
  - {id: vision, name: Vision}
"""
ADDRESS: Final = CellAddress(client="fakeclient", feature_id="vision", provider="openai")
RESULTS_ENV: Final = "FAKECLIENT_COMPAT_RESULTS_PATH"


def _clients_dir(root: Path) -> Path:
    client: Final = root / "fakeclient"
    client.mkdir(parents=True)
    _ = (client / "manifest.yaml").write_text(MANIFEST)
    return root


def _report(
    when: Literal["setup", "call", "teardown"],
    outcome: Literal["passed", "failed", "skipped"],
    longrepr: str | None = None,
) -> pytest.TestReport:
    return pytest.TestReport(
        nodeid="fakeclient/vision/test_openai.py::test_vision_openai",
        location=("fakeclient/vision/test_openai.py", 1, "test_vision_openai"),
        keywords={},
        outcome=outcome,
        longrepr=longrepr,
        when=when,
    )


def _row(model: str | None, outcome: Passed | Failed | NotApplicable) -> ResultRow:
    return ResultRow(feature_id="vision", provider="openai", model=model, outcome=outcome)


@pytest.mark.parametrize(
    ("relative", "expected"),
    (
        ("fakeclient/vision/test_openai.py", ADDRESS),
        ("fakeclient/vision/test_vertex_ai.py", CellAddress("fakeclient", "vision", "vertex_ai")),
        ("fakeclient/vision/test_azure.py", None),
        ("fakeclient/web_search/test_openai.py", None),
        ("fakeclient/vision/helpers.py", None),
        ("fakeclient/test_openai.py", None),
        ("otherclient/vision/test_openai.py", None),
    ),
)
def test_cell_address_only_maps_manifest_cells(tmp_path: Path, relative: str, expected: CellAddress | None) -> None:
    clients: Final = _clients_dir(tmp_path)

    assert cell_address(clients / relative, clients) == expected


def test_cell_address_ignores_files_outside_the_clients_dir(tmp_path: Path) -> None:
    clients: Final = _clients_dir(tmp_path / "clients")

    assert cell_address(tmp_path / "fakeclient" / "vision" / "test_openai.py", clients) is None


def _call(when: Literal["setup", "call", "teardown"], error: BaseException | None = None) -> pytest.CallInfo[None]:
    def run() -> None:
        if error is not None:
            raise error

    return pytest.CallInfo[None].from_call(run, when)


def test_a_cell_that_passed_without_reporting_is_recorded_as_failed() -> None:
    rows: Final = outcome_rows(ADDRESS, (), _call("call"), _report("call", "passed"))

    assert rows == (_row(None, Failed(error=NO_RESULT_ERROR)),)


def test_a_cell_that_crashed_records_the_assertion_line() -> None:
    longrepr: Final = "def test():\n>   assert x\nE   AssertionError: proxy answered 500\n\nfile.py:3: AssertionError"
    call: Final = _call("call", AssertionError("proxy answered 500"))

    assert outcome_rows(ADDRESS, (), call, _report("call", "failed", longrepr)) == (
        _row(None, Failed(error="AssertionError: proxy answered 500")),
    )


def test_a_setup_failure_is_a_failed_cell() -> None:
    call: Final = _call("setup", RuntimeError("could not register gpt-6.1-sol"))

    rows: Final = outcome_rows(ADDRESS, (), call, _report("setup", "failed", "could not register gpt-6.1-sol"))

    assert rows == (_row(None, Failed(error="could not register gpt-6.1-sol")),)


@pytest.mark.parametrize(
    ("recorded", "call", "report"),
    (
        (
            (_row("m", Failed(error="no file_change")),),
            _call("call", AssertionError("m: no file_change")),
            _report("call", "failed", "Failed: m: no file_change"),
        ),
        ((_row("m", Passed()),), _call("call"), _report("call", "passed")),
        ((_row(None, NotApplicable(reason="no apply_patch")),), _call("call"), _report("call", "passed")),
        ((), _call("teardown", RuntimeError("delete_model warned")), _report("teardown", "failed", "delete_model")),
        ((), _call("setup"), _report("setup", "passed")),
        ((), _call("setup", pytest.skip.Exception("no codex binary")), _report("setup", "skipped")),
    ),
)
def test_a_cell_that_reported_its_own_outcome_gets_no_extra_row(
    recorded: tuple[ResultRow, ...], call: pytest.CallInfo[None], report: pytest.TestReport
) -> None:
    assert outcome_rows(ADDRESS, recorded, call, report) == ()


def test_workers_write_shards_and_the_controller_merges_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results: Final = tmp_path / "out" / "fakeclient.json"
    monkeypatch.setenv(RESULTS_ENV, str(results))
    first: Final = _row("a", Passed())
    second: Final = _row("b", Failed(error="boom"))

    publish("fakeclient", (second,), "gw1")
    publish("fakeclient", (first,), "gw0")
    publish("fakeclient", (), "gw2")

    assert sorted(path.name for path in shard_dir(results).iterdir()) == ["gw0.json", "gw1.json"]
    assert not results.exists()

    publish("fakeclient", (), None)

    assert read_rows(results) == (first, second)


def test_clear_shards_drops_a_previous_runs_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results: Final = tmp_path / "fakeclient.json"
    monkeypatch.setenv(RESULTS_ENV, str(results))
    publish("fakeclient", (_row("stale", Passed()),), "gw0")

    clear_shards("fakeclient")
    publish("fakeclient", (_row("fresh", Passed()),), None)

    assert [row.model for row in read_rows(results)] == ["fresh"]


def test_a_run_with_no_cells_leaves_the_results_file_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results: Final = tmp_path / "fakeclient.json"
    monkeypatch.setenv(RESULTS_ENV, str(results))
    _ = results.write_text("[]")

    publish("fakeclient", (), None)

    assert results.read_text() == "[]"


GAP_MANIFEST: Final = """schema_version: "1"
client: fakeclient
package: fake-client
providers: [openai, vertex_ai]
features:
  - {id: vision, name: Vision}
  - {id: web_search, name: Web search}
  - {id: structured_output, name: Structured output}
known_gaps:
  - {feature: web_search, provider: openai, reason: the proxy has no search route}
  - {feature: web_search, provider: vertex_ai, reason: the proxy drops search tools}
  - {feature: vision, provider: vertex_ai, reason: the relay forwards an unknown field}
  - {feature: structured_output, provider: openai, strict: false, reason: the model sometimes loops}
  - {feature: structured_output, provider: vertex_ai, strict: false, reason: the model sometimes stalls}
"""
CELL_BODIES: Final = (
    (
        "web_search",
        "openai",
        'compat_result.failed("m", "no completed web_search item")\n'
        '    pytest.fail("m: no completed web_search item", pytrace=False)',
    ),
    ("web_search", "vertex_ai", 'compat_result.passed("m")'),
    ("vision", "openai", 'compat_result.passed("m")'),
    ("vision", "vertex_ai", 'raise RuntimeError("proxy unreachable")'),
    ("structured_output", "openai", 'compat_result.passed("m")'),
    (
        "structured_output",
        "vertex_ai",
        'compat_result.failed("m", "codex exec timed out")\n    pytest.fail("m: codex exec timed out", pytrace=False)',
    ),
)


def _client_with_cells(pytester: pytest.Pytester, manifest: str) -> Path:
    _ = pytester.makeconftest((CLIENT_APPS_DIR / "conftest.py").read_text())
    client: Final = pytester.path / "fakeclient"
    client.mkdir()
    _ = (client / "manifest.yaml").write_text(manifest)
    for feature_id, provider, body in CELL_BODIES:
        cell = client / feature_id / f"test_{provider}.py"
        cell.parent.mkdir(exist_ok=True)
        _ = cell.write_text(f"import pytest\n\n\ndef test_cell(compat_result):\n    {body}\n")
    return client


def _run_cells(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, manifest: str
) -> tuple[pytest.RunResult, Path]:
    results: Final = pytester.path / "fakeclient-results.json"
    monkeypatch.setenv(RESULTS_ENV, str(results))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    client: Final = _client_with_cells(pytester, manifest)
    return pytester.runpytest_inprocess("--import-mode=importlib", "-p", "no:cacheprovider", str(client)), results


def test_known_gap_cells_xfail_while_the_matrix_records_what_happened(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, results = _run_cells(pytester, monkeypatch, GAP_MANIFEST)
    manifest: Final = load_manifest(pytester.path / "fakeclient" / "manifest.yaml")
    matrix: Final = build_matrix(
        manifest,
        read_rows(results),
        litellm_version="1.0.0",
        client_version="0.1.0",
        generated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    cells: Final[dict[tuple[str, str], Outcome]] = {
        (feature.id, provider): outcome for feature in matrix.features for provider, outcome in feature.providers.items()
    }

    run.assert_outcomes(passed=1, failed=1, xfailed=3, xpassed=1)
    assert "the proxy drops search tools" in run.stdout.str()
    assert cells == {
        ("vision", "openai"): Passed(),
        ("vision", "vertex_ai"): Failed(error="RuntimeError: proxy unreachable"),
        ("web_search", "openai"): Failed(error="m: no completed web_search item"),
        ("web_search", "vertex_ai"): Passed(),
        ("structured_output", "openai"): Passed(),
        ("structured_output", "vertex_ai"): Failed(error="m: codex exec timed out"),
    }


def test_a_known_gap_naming_an_unknown_cell_stops_the_run(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    typo: Final = GAP_MANIFEST.replace("provider: openai, reason: the proxy has no search route", "provider: opneai, reason: r")

    run, results = _run_cells(pytester, monkeypatch, typo)

    assert run.ret != pytest.ExitCode.OK
    assert "unknown provider 'opneai'" in run.stdout.str() + run.stderr.str()
    assert not results.exists()
