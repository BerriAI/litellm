from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from itertools import chain
from pathlib import Path
from typing import Final

import pytest

from client_apps.compat_results import (
    Failed,
    NotApplicable,
    Outcome,
    Passed,
    ResultRow,
    merged_rows,
    results_path,
    shard_dir,
    write_rows,
)
from client_apps.matrix_builder import Manifest, load_manifest

CLIENT_APPS_DIR: Final = Path(__file__).resolve().parent
ROWS: Final = pytest.StashKey[tuple[ResultRow, ...]]()
NO_RESULT_ERROR: Final = "the test passed without reporting a result"


@dataclass(frozen=True, slots=True)
class CellAddress:
    client: str
    feature_id: str
    provider: str


@cache
def client_manifest(client_dir: Path) -> Manifest:
    return load_manifest(client_dir / "manifest.yaml")


def cell_address(test_file: Path, clients_dir: Path = CLIENT_APPS_DIR) -> CellAddress | None:
    resolved: Final = test_file.resolve()
    if not resolved.is_relative_to(clients_dir):
        return None
    match resolved.relative_to(clients_dir).parts:
        case (client, feature_id, filename) if filename.startswith("test_") and filename.endswith(".py"):
            return _manifest_cell(clients_dir / client, feature_id, filename.removeprefix("test_").removesuffix(".py"))
        case _:
            return None


def _manifest_cell(client_dir: Path, feature_id: str, provider: str) -> CellAddress | None:
    if not (client_dir / "manifest.yaml").is_file():
        return None
    manifest: Final = client_manifest(client_dir)
    known_feature: Final = any(feature.id == feature_id for feature in manifest.features)
    if not known_feature or provider not in manifest.providers:
        return None
    return CellAddress(client=client_dir.name, feature_id=feature_id, provider=provider)


def _row(address: CellAddress, model: str | None, outcome: Outcome) -> ResultRow:
    return ResultRow(feature_id=address.feature_id, provider=address.provider, model=model, outcome=outcome)


def _record(item: pytest.Item, row: ResultRow) -> None:
    item.stash[ROWS] = (*item.stash.get(ROWS, ()), row)


class CompatRecorder:
    __slots__ = ("_item", "address")

    def __init__(self, item: pytest.Item, address: CellAddress) -> None:
        self._item: Final = item
        self.address: Final = address

    def passed(self, model: str) -> None:
        _record(self._item, _row(self.address, model, Passed()))

    def failed(self, model: str, error: str) -> None:
        _record(self._item, _row(self.address, model, Failed(error=error)))

    def not_applicable(self, reason: str) -> None:
        _record(self._item, _row(self.address, None, NotApplicable(reason=reason)))


def _fails_cell(report: pytest.TestReport) -> bool:
    return report.failed and report.when in ("setup", "call")


def outcome_rows(
    address: CellAddress, recorded: tuple[ResultRow, ...], report: pytest.TestReport
) -> tuple[ResultRow, ...]:
    if _fails_cell(report) and not any(isinstance(row.outcome, Failed) for row in recorded):
        return (_row(address, None, Failed(error=_crash_message(report))),)
    if report.passed and report.when == "call" and not recorded:
        return (_row(address, None, Failed(error=NO_RESULT_ERROR)),)
    return ()


def _crash_message(report: pytest.TestReport) -> str:
    lines: Final = tuple(line for line in report.longreprtext.splitlines() if line.strip())
    error_lines: Final = tuple(line for line in lines if line.startswith("E "))
    return (error_lines or lines or (f"{report.when} failed",))[-1].removeprefix("E ").strip()


def known_clients() -> tuple[str, ...]:
    return tuple(sorted(manifest.parent.name for manifest in CLIENT_APPS_DIR.glob("*/manifest.yaml")))


def worker_id() -> str | None:
    return os.environ.get("PYTEST_XDIST_WORKER")


def _item_rows(item: pytest.Item, client: str) -> tuple[ResultRow, ...]:
    address: Final = cell_address(item.path)
    return item.stash.get(ROWS, ()) if address is not None and address.client == client else ()


def client_rows(session: pytest.Session, client: str) -> tuple[ResultRow, ...]:
    return tuple(chain.from_iterable(_item_rows(item, client) for item in session.items))


def clear_shards(client: str) -> None:
    shards: Final = shard_dir(results_path(client, os.environ))
    for stale in shards.glob("*.json"):
        stale.unlink()


def publish(client: str, rows: tuple[ResultRow, ...], worker: str | None) -> None:
    results: Final = results_path(client, os.environ)
    if worker is not None:
        if rows:
            write_rows(shard_dir(results) / f"{worker}.json", rows)
        return
    merged: Final = merged_rows(rows, shard_dir(results))
    if merged:
        write_rows(results, merged)
