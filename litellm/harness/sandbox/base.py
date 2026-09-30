"""The Sandbox protocol: where a harness runtime runs and which files it can touch."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CompletedRun:
    stdout: str
    stderr: str
    exit_code: int


@runtime_checkable
class Process(Protocol):
    stdin: asyncio.StreamWriter | None
    stdout: asyncio.StreamReader
    stderr: asyncio.StreamReader

    async def wait(self) -> int: ...

    async def kill(self) -> None: ...


@runtime_checkable
class Sandbox(Protocol):
    workdir: str

    async def exec(
        self,
        cmd: list[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> Process: ...

    async def run(
        self,
        cmd: list[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CompletedRun: ...

    async def read(self, path: str) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...

    def host_url(self, port: int) -> str: ...

    async def which(self, binary: str) -> str | None: ...

    async def snapshot(self) -> dict[str, str]: ...

    async def tempdir(self) -> str: ...

    async def close(self) -> None: ...
