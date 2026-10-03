from __future__ import annotations

import errno
import importlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

TESTS_DIR: Final = Path(__file__).resolve().parents[2]
BIND_ERROR_LINE: Final = f"ERROR:    {OSError(errno.EADDRINUSE, os.strerror(errno.EADDRINUSE))}\n"


@pytest.fixture
def process_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    return importlib.import_module("integration._support.process")


@pytest.fixture
def exited_process() -> subprocess.Popen[bytes]:
    process: Final = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    process.wait()
    return process


@pytest.fixture
def running_process() -> Iterator[subprocess.Popen[bytes]]:
    process: Final = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"])
    try:
        yield process
    finally:
        process.kill()
        process.wait()


def test_lost_port_race_matches_the_bind_error_the_server_logs(
    process_module: ModuleType, exited_process: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    log: Final = tmp_path / "owned-proxy.log"
    log.write_text(BIND_ERROR_LINE)

    assert process_module._lost_port_race(process_module._Launch(exited_process, 0, log))


def test_lost_port_race_ignores_an_exit_for_another_reason(
    process_module: ModuleType, exited_process: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    log: Final = tmp_path / "owned-proxy.log"
    log.write_text("Traceback (most recent call last):\nModuleNotFoundError: No module named 'litellm'\n")

    assert not process_module._lost_port_race(process_module._Launch(exited_process, 0, log))


def test_lost_port_race_needs_the_process_to_have_exited(
    process_module: ModuleType, running_process: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    log: Final = tmp_path / "owned-proxy.log"
    log.write_text(BIND_ERROR_LINE)

    assert not process_module._lost_port_race(process_module._Launch(running_process, 0, log))
