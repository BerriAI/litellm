from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Final, Protocol, cast

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
MODULE_PATH: Final = REPO_ROOT / ".circleci" / "scripts" / "python_compatibility_shards.py"


class Shard(Protocol):
    name: str
    test_path: str
    workers: int
    reruns: int
    timeout_minutes: int
    dist: str
    max_failures: int


class CompatibilityModule(Protocol):
    def load_shards(self, repo_root: Path) -> tuple[Shard, ...]: ...

    def select_shard(self, shards: tuple[Shard, ...], node_index: int, node_total: int) -> Shard: ...

    def pytest_command(self, shard: Shard, python_version: str, results_dir: Path) -> tuple[str, ...]: ...


SPEC: Final = importlib.util.spec_from_file_location("python_compatibility_shards", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
raw_module: Final = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(raw_module)
compatibility: Final = cast(CompatibilityModule, raw_module)


def test_loads_and_normalizes_both_github_matrices() -> None:
    shards: Final = compatibility.load_shards(REPO_ROOT)
    standard: Final = next(shard for shard in shards if shard.name == "core-utils")
    proxy: Final = next(shard for shard in shards if shard.name == "key-generation")

    assert len(shards) == len({shard.name for shard in shards}) == 26
    assert (standard.reruns, standard.dist, standard.max_failures) == (1, "loadscope", 10)
    assert (proxy.workers, proxy.reruns, proxy.timeout_minutes) == (0, 2, 20)


def test_commands_preserve_serial_parallel_and_timeout_settings(tmp_path: Path) -> None:
    shards: Final = compatibility.load_shards(REPO_ROOT)
    serial: Final = next(shard for shard in shards if shard.name == "key-generation")
    parallel: Final = next(shard for shard in shards if shard.name == "proxy-utils")
    serial_command: Final = compatibility.pytest_command(serial, "3.10", tmp_path)
    parallel_command: Final = compatibility.pytest_command(parallel, "3.14", tmp_path)

    assert "-n" not in serial_command
    assert parallel_command[parallel_command.index("-n") + 1] == "4"
    assert "--dist=worksteal" in parallel_command
    assert serial_command[:4] == ("timeout", "--signal=TERM", "--kill-after=30s", "20m")
    assert serial_command[4:7] == (".venv/bin/python", "-m", "pytest")
    assert serial_command[-1].endswith("python-3.10-key-generation.xml")
    assert parallel_command[-1].endswith("python-3.14-proxy-utils.xml")


def test_rejects_node_count_drift_and_invalid_indices() -> None:
    shards: Final = compatibility.load_shards(REPO_ROOT)

    for index, total, message in ((0, 25, "expected 26"), (26, 26, "outside 0..25")):
        with pytest.raises(ValueError, match=message):
            compatibility.select_shard(shards, index, total)


@pytest.mark.parametrize("shard_name", ("misc", "proxy-infra"))
def test_wildcard_shards_execute_all_matching_files(
    shard_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard: Final = next(shard for shard in compatibility.load_shards(REPO_ROOT) if shard.name == shard_name)
    paths: Final = shlex.split(shard.test_path)
    for path in paths:
        if "*" in path:
            for suffix in ("first", "second"):
                test_file: Final = tmp_path / path.replace("*", suffix)
                test_file.parent.mkdir(parents=True, exist_ok=True)
                test_file.write_text("def test_matching_file():\n    assert True\n", encoding="utf-8")
        else:
            directory: Final = tmp_path / path
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "test_directory.py").write_text("def test_directory():\n    assert True\n", encoding="utf-8")
    config: Final = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    command: Final = compatibility.pytest_command(shard, "3.13", tmp_path)
    result: Final = subprocess.run(
        (sys.executable, *command[5:], "-c", str(config), "--noconftest", "--import-mode=importlib"),
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(paths) + 1} passed" in result.stdout
