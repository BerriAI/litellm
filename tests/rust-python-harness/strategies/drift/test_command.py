from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Final

import pytest
from click.testing import CliRunner

from ...cli import build_cli
from ...cli.catalog import load_catalog
from .command import drift_command
from .runner import DriftReport


def _drift_git(repo_root: Path, *args: str) -> str:
    result: Final = subprocess.run(("git", "-C", str(repo_root), *args), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _drift_manifest(checkpoint_sha: str | None = None) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "checkpoint_sha": checkpoint_sha,
            "candidate_paths": ["litellm/", "litellm-rust/"],
            "mappings": [
                {
                    "id": "messages",
                    "python_paths": ["litellm/messages/"],
                    "rust_paths": ["litellm-rust/crates/inference-messages/"],
                }
            ],
        }
    )


def _drift_repo(tmp_path: Path) -> tuple[Path, str]:
    repo_root: Final = tmp_path / "repo"
    repo_root.mkdir()
    _drift_git(repo_root, "init", "--quiet", "--initial-branch=main")
    _drift_git(repo_root, "config", "user.name", "Harness Test")
    _drift_git(repo_root, "config", "user.email", "harness@example.invalid")
    source: Final = repo_root / "litellm/messages/dispatch.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 'initial'\n", encoding="utf-8")
    manifest: Final = repo_root / "tests/rust-python-harness/strategies/drift/drift.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(_drift_manifest(), encoding="utf-8")
    _drift_git(repo_root, "add", ".")
    _drift_git(repo_root, "commit", "--quiet", "-m", "initial")
    return repo_root, _drift_git(repo_root, "rev-parse", "HEAD")


def test_drift_command_reports_unreviewed_checkpoint_from_default_manifest(tmp_path: Path) -> None:
    repo_root, initial_sha = _drift_repo(tmp_path)
    result: Final = CliRunner().invoke(drift_command(repo_root))

    assert result.exit_code == 0, result.output
    report: Final = DriftReport.model_validate_json(result.output)
    assert report.status == "unreviewed"
    assert report.target_sha == initial_sha
    assert report.checkpoint_sha is None


@pytest.mark.parametrize("absolute_path", (False, True))
def test_drift_command_resolves_custom_manifest_paths(tmp_path: Path, absolute_path: bool) -> None:
    repo_root, initial_sha = _drift_repo(tmp_path)
    manifest: Final = repo_root / "custom.json"
    manifest.write_text(_drift_manifest(initial_sha), encoding="utf-8")
    manifest_arg: Final = str(manifest if absolute_path else manifest.relative_to(repo_root))
    result: Final = CliRunner().invoke(drift_command(repo_root), ("--manifest", manifest_arg))

    assert result.exit_code == 0, result.output
    report: Final = DriftReport.model_validate_json(result.output)
    assert report.status == "no_changes"
    assert report.checkpoint_sha == initial_sha


def test_drift_command_compares_requested_revisions_without_claiming_parity(tmp_path: Path) -> None:
    repo_root, initial_sha = _drift_repo(tmp_path)
    source: Final = repo_root / "litellm/messages/dispatch.py"
    source.write_text("VALUE = 'second'\n", encoding="utf-8")
    _drift_git(repo_root, "add", ".")
    _drift_git(repo_root, "commit", "--quiet", "-m", "second")
    second_sha: Final = _drift_git(repo_root, "rev-parse", "HEAD")
    source.write_text("VALUE = 'third'\n", encoding="utf-8")
    _drift_git(repo_root, "add", ".")
    _drift_git(repo_root, "commit", "--quiet", "-m", "third")
    result: Final = CliRunner().invoke(drift_command(repo_root), ("--since", initial_sha, "--target", second_sha))

    assert result.exit_code == 0, result.output
    report: Final = DriftReport.model_validate_json(result.output)
    assert report.status == "unreviewed"
    assert report.checkpoint_sha is None
    assert report.comparison_sha == initial_sha
    assert report.target_sha == second_sha
    assert tuple(change.path for change in report.changes) == ("litellm/messages/dispatch.py",)
    assert "+VALUE = 'second'" in report.changes[0].patch
    assert "third" not in report.changes[0].patch


def test_drift_command_writes_report_to_file(tmp_path: Path) -> None:
    repo_root, initial_sha = _drift_repo(tmp_path)
    output: Final = tmp_path / "report.json"
    result: Final = CliRunner().invoke(drift_command(repo_root), ("--output", str(output)))

    assert result.exit_code == 0, result.output
    assert result.output == ""
    report: Final = DriftReport.model_validate_json(output.read_text(encoding="utf-8"))
    assert report.status == "unreviewed"
    assert report.target_sha == initial_sha


def test_drift_command_surfaces_inventory_errors(tmp_path: Path) -> None:
    repo_root, _ = _drift_repo(tmp_path)
    result: Final = CliRunner().invoke(drift_command(repo_root), ("--manifest", "missing.json"))

    assert result.exit_code != 0
    assert "Error" in result.output
    assert "missing.json" in result.output


def test_drift_command_reports_output_write_errors(tmp_path: Path) -> None:
    repo_root, _ = _drift_repo(tmp_path)
    output: Final = tmp_path / "missing/report.json"
    result: Final = CliRunner().invoke(drift_command(repo_root), ("--output", str(output)))

    assert result.exit_code != 0
    assert "Cannot write drift report" in result.output
    assert str(output) in result.output


@pytest.mark.parametrize("output_kind", ("absolute", "relative", "symlink", "hardlink"))
def test_drift_command_preserves_manifest_when_output_aliases_it(tmp_path: Path, output_kind: str) -> None:
    repo_root, _ = _drift_repo(tmp_path)
    manifest: Final = repo_root / "tests/rust-python-harness/strategies/drift/drift.json"
    original: Final = manifest.read_bytes()
    alias: Final = repo_root / "report.json"
    if output_kind == "symlink":
        alias.symlink_to(manifest)
    if output_kind == "hardlink":
        alias.hardlink_to(manifest)
    destination: Final = alias if output_kind in {"symlink", "hardlink"} else manifest
    output: Final = os.path.relpath(destination) if output_kind == "relative" else str(destination)
    result: Final = CliRunner().invoke(drift_command(repo_root), ("--output", output))

    assert result.exit_code != 0
    assert "Output path must differ from the drift manifest" in result.output
    assert manifest.read_bytes() == original


def test_run_drift_uses_registered_command_with_repository_context(tmp_path: Path) -> None:
    repo_root, initial_sha = _drift_repo(tmp_path)
    result: Final = CliRunner().invoke(build_cli(load_catalog(), repo_root), ("run", "drift"))

    assert result.exit_code == 0, result.output
    report: Final = DriftReport.model_validate_json(result.output)
    assert report.target_sha == initial_sha
    assert report.checkpoint_state == "unknown"
    assert report.manifest_path == "tests/rust-python-harness/strategies/drift/drift.json"
