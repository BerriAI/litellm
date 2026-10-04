from __future__ import annotations

import errno
import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

TESTS_DIR: Final = Path(__file__).resolve().parents[2]
BIND_ERROR_LINE: Final = f"ERROR:    {OSError(errno.EADDRINUSE, os.strerror(errno.EADDRINUSE))}\n"
UNRELATED_CRASH: Final = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'litellm'\n"


@pytest.fixture
def process_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    return importlib.import_module("integration._support.process")


def _written_log(directory: Path, text: str) -> Path:
    log: Final = directory / "owned-proxy.log"
    log.write_text(text)
    return log


def test_lost_port_race_matches_the_bind_error_the_server_logs(process_module: ModuleType, tmp_path: Path) -> None:
    assert process_module._lost_port_race(1, _written_log(tmp_path, BIND_ERROR_LINE))


def test_lost_port_race_ignores_an_exit_for_another_reason(process_module: ModuleType, tmp_path: Path) -> None:
    assert not process_module._lost_port_race(1, _written_log(tmp_path, UNRELATED_CRASH))


def test_lost_port_race_needs_the_process_to_have_exited(process_module: ModuleType, tmp_path: Path) -> None:
    assert not process_module._lost_port_race(None, _written_log(tmp_path, BIND_ERROR_LINE))
