"""Async subprocess execution for the ``helm`` and ``kubectl`` CLIs.

The runner is a plain coroutine so tests can inject a fake in place of real
process execution. Commands are always invoked as an argv list (never through a
shell) to avoid command injection from caller-supplied values.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    def __init__(self, args: list[str], result: CommandResult) -> None:
        self.args_list = args
        self.result = result
        super().__init__(
            f"command failed (exit {result.returncode}): {' '.join(args)}\n{result.stderr.strip()}"
        )


class CommandTimeout(RuntimeError):
    def __init__(self, args: list[str], timeout: float) -> None:
        self.args_list = args
        self.timeout = timeout
        super().__init__(f"command timed out after {timeout}s: {' '.join(args)}")


async def run_command(
    args: list[str],
    *,
    input_text: str | None = None,
    timeout: float,
) -> CommandResult:
    """Execute ``args`` and return its result. Never raises on non-zero exit."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdin_bytes = input_text.encode() if input_text is not None else None
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_bytes), timeout=timeout
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise CommandTimeout(args, timeout) from exc

    return CommandResult(
        returncode=proc.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )
