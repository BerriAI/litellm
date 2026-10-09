from __future__ import annotations

import asyncio
import errno
import importlib
import socket
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

TESTS_DIR: Final = Path(__file__).resolve().parents[2]
UNRELATED_CRASH: Final = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'litellm'\n"


@pytest.fixture
def process_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    return importlib.import_module("integration._support.process")


async def _refused_bind(port: int) -> OSError | None:
    try:
        await asyncio.get_running_loop().create_server(asyncio.Protocol, "127.0.0.1", port)
    except OSError as refused:
        return refused
    return None


@pytest.fixture
def bind_error_line() -> Iterator[str]:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        refused: Final = asyncio.run(_refused_bind(held.getsockname()[1]))
        assert refused is not None and refused.errno == errno.EADDRINUSE
        yield f"ERROR:    {refused}\n"


def _written_log(directory: Path, text: str) -> Path:
    log: Final = directory / "owned-proxy.log"
    log.write_text(text)
    return log


def test_lost_port_race_matches_the_bind_error_the_server_logs(
    process_module: ModuleType, bind_error_line: str, tmp_path: Path
) -> None:
    assert process_module._lost_port_race(1, _written_log(tmp_path, bind_error_line))


def test_lost_port_race_ignores_an_exit_for_another_reason(process_module: ModuleType, tmp_path: Path) -> None:
    assert not process_module._lost_port_race(1, _written_log(tmp_path, UNRELATED_CRASH))


def test_lost_port_race_needs_the_process_to_have_exited(
    process_module: ModuleType, bind_error_line: str, tmp_path: Path
) -> None:
    assert not process_module._lost_port_race(None, _written_log(tmp_path, bind_error_line))
