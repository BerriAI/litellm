from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter

WORKFLOWS: Final = (
    (Path(".github/workflows/test-unit.yml"), "unit"),
    (Path(".github/workflows/test-unit-proxy-db.yml"), "proxy-db"),
)


class Shard(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(validation_alias=AliasChoices("shard", "test-group"))
    test_path: str = Field(alias="test-path")
    workers: int = 2
    reruns: int = 2
    timeout_minutes: int = Field(default=20, validation_alias=AliasChoices("timeout-minutes", "timeout"))
    dist: str = "loadscope"
    max_failures: int = Field(default=10, alias="max-failures")


SHARDS: Final = TypeAdapter(tuple[Shard, ...])


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, object], value)


def _entries(repo_root: Path, workflow: Path, job_name: str) -> tuple[Shard, ...]:
    raw: object = yaml.safe_load(  # pyright: ignore[reportAny]  # PyYAML exposes an untyped parser result
        (repo_root / workflow).read_text(encoding="utf-8")
    )
    root: Final = _mapping(raw, str(workflow))
    jobs: Final = _mapping(root.get("jobs"), f"{workflow}: jobs")
    job: Final = _mapping(jobs.get(job_name), f"{workflow}: jobs.{job_name}")
    strategy: Final = _mapping(job.get("strategy"), f"{workflow}: jobs.{job_name}.strategy")
    matrix: Final = _mapping(strategy.get("matrix"), f"{workflow}: jobs.{job_name}.strategy.matrix")
    return SHARDS.validate_python(matrix.get("include"))


def load_shards(repo_root: Path) -> tuple[Shard, ...]:
    shards: Final = tuple(
        shard
        for workflow, job_name in WORKFLOWS
        for shard in _entries(repo_root, workflow, job_name)
    )
    if len({shard.name for shard in shards}) != len(shards):
        raise ValueError("GitHub unit-test shard names must be unique")
    return shards


def select_shard(shards: tuple[Shard, ...], node_index: int, node_total: int) -> Shard:
    if node_total != len(shards):
        raise ValueError(f"CircleCI node total is {node_total}; expected {len(shards)}")
    if not 0 <= node_index < node_total:
        raise ValueError(f"CircleCI node index {node_index} is outside 0..{node_total - 1}")
    return shards[node_index]


def pytest_command(shard: Shard, python_version: str, results_dir: Path) -> tuple[str, ...]:
    slug: Final = re.sub(r"[^a-z0-9]+", "-", shard.name.lower()).strip("-")
    report: Final = results_dir / f"python-{python_version}-{slug}.xml"
    workers: Final = () if shard.workers == 0 else ("-n", str(shard.workers), f"--dist={shard.dist}")
    return (
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{shard.timeout_minutes}m",
        ".venv/bin/python",
        "-m",
        "pytest",
        *shlex.split(shard.test_path),
        "--tb=short",
        "-vv",
        f"--maxfail={shard.max_failures}",
        *workers,
        "--reruns",
        str(shard.reruns),
        "--reruns-delay",
        "1",
        "--durations=20",
        f"--junitxml={report}",
    )


def main() -> int:
    repo_root: Final = Path(__file__).resolve().parents[2]
    shards: Final = load_shards(repo_root)
    shard: Final = select_shard(
        shards,
        int(os.environ["CIRCLE_NODE_INDEX"]),
        int(os.environ["CIRCLE_NODE_TOTAL"]),
    )
    results_dir: Final = repo_root / "test-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    command: Final = pytest_command(shard, os.environ["UV_PYTHON"], results_dir)
    sys.stdout.write(f"Running {shard.name!r}: {shlex.join(command)}\n")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
