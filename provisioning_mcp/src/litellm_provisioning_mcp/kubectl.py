"""Thin async wrapper around the ``kubectl`` CLI, scoped to a single namespace."""

from __future__ import annotations

import json
from typing import Awaitable, Callable

from .commands import CommandError, CommandResult, run_command

Runner = Callable[..., Awaitable[CommandResult]]


class KubectlRunner:
    def __init__(
        self,
        *,
        namespace: str,
        binary: str = "kubectl",
        runner: Runner = run_command,
        timeout: int = 180,
    ) -> None:
        self._namespace = namespace
        self._binary = binary
        self._runner = runner
        self._timeout = timeout

    async def _run(
        self, args: list[str], *, input_text: str | None = None
    ) -> CommandResult:
        full = [self._binary, "--namespace", self._namespace, *args]
        result = await self._runner(full, input_text=input_text, timeout=self._timeout)
        if result.returncode != 0:
            raise CommandError(full, result)
        return result

    async def apply(self, manifest_yaml: str) -> CommandResult:
        return await self._run(["apply", "--filename", "-"], input_text=manifest_yaml)

    async def wait_available(self, *, deployment: str, timeout: int) -> CommandResult:
        return await self._run(
            [
                "wait",
                f"deployment/{deployment}",
                "--for=condition=Available",
                f"--timeout={timeout}s",
            ]
        )

    async def delete_by_label(
        self, *, selector: str, kinds: list[str]
    ) -> CommandResult:
        return await self._run(
            ["delete", ",".join(kinds), "--selector", selector, "--ignore-not-found"]
        )

    async def get_pods(self, *, selector: str) -> list[dict]:
        result = await self._run(
            ["get", "pods", "--selector", selector, "--output", "json"]
        )
        return json.loads(result.stdout).get("items", [])
