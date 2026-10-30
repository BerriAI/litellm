from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import pytest

from client_apps.compat_collector import NO_RESULT_ERROR, CellAddress, cell_address, clear_shards, outcome_rows, publish
from client_apps.compat_results import Failed, NotApplicable, Passed, ResultRow, read_rows, shard_dir

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
    when: Literal["setup", "call", "teardown"], outcome: Literal["passed", "failed"], longrepr: str | None = None
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


def test_a_cell_that_passed_without_reporting_is_recorded_as_failed() -> None:
    assert outcome_rows(ADDRESS, (), _report("call", "passed")) == (_row(None, Failed(error=NO_RESULT_ERROR)),)


def test_a_cell_that_crashed_records_the_assertion_line() -> None:
    longrepr: Final = "def test():\n>   assert x\nE   AssertionError: proxy answered 500\n\nfile.py:3: AssertionError"

    assert outcome_rows(ADDRESS, (), _report("call", "failed", longrepr)) == (
        _row(None, Failed(error="AssertionError: proxy answered 500")),
    )


def test_a_setup_failure_is_a_failed_cell() -> None:
    rows: Final = outcome_rows(ADDRESS, (), _report("setup", "failed", "could not register gpt-6.1-sol"))

    assert rows == (_row(None, Failed(error="could not register gpt-6.1-sol")),)


@pytest.mark.parametrize(
    ("recorded", "report"),
    (
        ((_row("m", Failed(error="no file_change")),), _report("call", "failed", "Failed: m: no file_change")),
        ((_row("m", Passed()),), _report("call", "passed")),
        ((_row(None, NotApplicable(reason="no apply_patch")),), _report("call", "passed")),
        ((), _report("teardown", "failed", "delete_model warned")),
        ((), _report("setup", "passed")),
    ),
)
def test_a_cell_that_reported_its_own_outcome_gets_no_extra_row(
    recorded: tuple[ResultRow, ...], report: pytest.TestReport
) -> None:
    assert outcome_rows(ADDRESS, recorded, report) == ()


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
