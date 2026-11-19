from __future__ import annotations

import asyncio
import sys
import webbrowser
from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import pytest

from litellm.lens.cli import main
from litellm.lens.core import RunStore, load_config, run_suite
from litellm.lens.demo import create_demo


def _invoke(monkeypatch: pytest.MonkeyPatch, arguments: Sequence[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["lens", *arguments])
    main()


@pytest.fixture
def local_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)
    tasks, config = create_demo(tmp_path / "suite", count=12, delay=0)
    with pytest.raises(SystemExit) as completed:
        _invoke(monkeypatch, ("run", str(tasks), "--config", str(config), "--quiet"))
    assert completed.value.code == 1
    return RunStore.list_runs(Path(".lens/runs"))[0].run_id


def test_should_show_commands_without_creating_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _invoke(monkeypatch, ("help",))
    output: Final = capsys.readouterr().out
    assert "/run tasks.csv --config evals.py" in output
    assert "/resume" in output
    assert not (tmp_path / ".lens").exists()


def test_should_compare_and_inspect_saved_failures_and_repeated_calls(
    local_run: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    _invoke(monkeypatch, ("compare", local_run))
    comparison: Final = capsys.readouterr().out
    assert "model-a" in comparison
    assert "model-b" in comparison
    assert "PASS" in comparison and "FAIL" in comparison and "ERROR" in comparison
    _invoke(
        monkeypatch,
        ("inspect", local_run, "--case", "ticket-0001", "--target", "model-b"),
    )
    failure: Final = capsys.readouterr().out
    assert "FAIL" in failure
    assert "billing" in failure and "technical" in failure
    assert "Repeated lookup_ticket" in failure
    assert "×2" in failure
    _invoke(
        monkeypatch,
        ("inspect", local_run, "--case", "ticket-0011", "--target", "model-b"),
    )
    error: Final = capsys.readouterr().out
    assert "ERROR" in error
    assert "Fixture model endpoint timed out" in error
    assert "Fixture lookup timed out" in error
    with pytest.raises(SystemExit) as missing:
        _invoke(monkeypatch, ("inspect", local_run, "--case", "missing-case"))
    assert missing.value.code == 1
    assert "No saved attempt matches" in capsys.readouterr().out


def test_should_show_unknown_custom_harness_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    tasks, config = create_demo(tmp_path / "suite", count=1, delay=0)
    suite: Final = replace(load_config(tasks, config), telemetry_complete=frozenset())
    with closing(RunStore.create(Path(".lens/runs"), suite)) as store:
        asyncio.run(run_suite(store, suite))
        run_id: Final = store.run_id
    _invoke(monkeypatch, ("compare", run_id))
    assert "unknown" in capsys.readouterr().out
    _invoke(monkeypatch, ("inspect", run_id, "--case", "ticket-0001", "--target", "model-a"))
    output: Final = capsys.readouterr().out
    assert "Agent unknown" in output
    assert "Model calls unknown" in output


def test_should_publish_metadata_and_open_the_saved_case_without_rerunning(
    local_run: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with closing(RunStore.open(Path(".lens/runs"), local_run)) as store:
        before: Final = store.rows()
        directory: Final = store.directory
        prompt: Final = store.manifest.tasks[0].prompt
    opened: Final[SimpleQueue[str]] = SimpleQueue()

    def open_browser(url: str) -> bool:
        opened.put(url)
        return True

    monkeypatch.setattr(webbrowser, "open", open_browser)
    capsys.readouterr()
    _invoke(monkeypatch, ("publish", local_run))
    output: Final = capsys.readouterr().out
    assert "No upload or model calls" in output
    metadata: Final = (directory / "handoff.metadata.json").read_text(encoding="utf-8")
    assert prompt not in metadata
    assert "Fixture ticket found" not in metadata
    _invoke(monkeypatch, ("open", local_run, "--case", "ticket-0001", "--target", "model-b"))
    url: Final = opened.get_nowait()
    assert url.startswith((directory / "handoff.metadata.html").as_uri() + "#case-")
    with closing(RunStore.open(Path(".lens/runs"), local_run)) as reopened:
        assert reopened.rows() == before


def test_should_report_missing_runs_and_unknown_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as missing:
        _invoke(monkeypatch, ("compare",))
    assert missing.value.code == 1
    assert "No local runs yet" in capsys.readouterr().out
    with pytest.raises(SystemExit) as unknown:
        _invoke(monkeypatch, ("unknown-command",))
    assert unknown.value.code == 2
    assert not (tmp_path / ".lens").exists()


def test_should_reject_invalid_options_without_starting_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as invalid:
        _invoke(monkeypatch, ("demo", "--concurrency", "0"))
    assert invalid.value.code == 2
    assert not (tmp_path / ".lens").exists()
