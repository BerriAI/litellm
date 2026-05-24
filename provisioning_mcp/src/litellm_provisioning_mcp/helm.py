"""Thin async wrapper around the ``helm`` CLI, scoped to a single namespace."""

from __future__ import annotations

import json
from typing import Awaitable, Callable

from .commands import CommandError, CommandResult, run_command

Runner = Callable[..., Awaitable[CommandResult]]


class HelmRunner:
    def __init__(
        self,
        *,
        namespace: str,
        binary: str = "helm",
        runner: Runner = run_command,
        wait_timeout: int = 600,
    ) -> None:
        self._namespace = namespace
        self._binary = binary
        self._runner = runner
        self._wait_timeout = wait_timeout
        # Allow helm's --wait to run to completion before the subprocess is killed.
        self._command_timeout = wait_timeout + 60

    async def _run(
        self, args: list[str], *, input_text: str | None = None
    ) -> CommandResult:
        full = [self._binary, *args, "--namespace", self._namespace]
        result = await self._runner(
            full, input_text=input_text, timeout=self._command_timeout
        )
        if result.returncode != 0:
            raise CommandError(full, result)
        return result

    async def upgrade_install(
        self, *, release: str, chart_path: str, values_yaml: str
    ) -> CommandResult:
        return await self._run(
            [
                "upgrade",
                release,
                chart_path,
                "--install",
                "--values",
                "-",  # read merged values from stdin
                "--wait",
                "--timeout",
                f"{self._wait_timeout}s",
            ],
            input_text=values_yaml,
        )

    async def uninstall(self, *, release: str) -> CommandResult:
        return await self._run(["uninstall", release, "--ignore-not-found", "--wait"])

    async def status(self, *, release: str) -> dict:
        result = await self._run(["status", release, "--output", "json"])
        return json.loads(result.stdout)

    async def list_releases(self) -> list[dict]:
        result = await self._run(["list", "--output", "json"])
        return json.loads(result.stdout)
