"""
Generic handler for CLI harnesses (Claude Code, Codex, OpenCode).

The config (`litellm/llms/<harness>/harness/transformation.py`) says what to run and how to
read it; this handler does every sandbox and process operation: binary check, private dir,
config files, persisted dirs, skills, spawning the turn, streaming stdout lines into the
config's parser, collecting stderr, and killing the process on early exit.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from litellm._logging import verbose_logger
from litellm.constants import (
    HARNESS_RESUME_COMMAND_TIMEOUT_SECONDS,
    HARNESS_STDERR_TAIL_LINES,
    HARNESS_STREAM_READ_CHUNK_BYTES,
)
from litellm.harness.context import SessionContext
from litellm.harness.errors import HarnessError, HarnessInstallFailed, SandboxError
from litellm.harness.handlers.base import BaseHarnessHandler
from litellm.harness.sandbox.base import CompletedRun, Process, Sandbox
from litellm.harness.types import Event
from litellm.llms.base_llm.harness.transformation import (
    BaseCLIHarnessConfig,
    HarnessSessionCopy,
    HarnessSessionSetup,
    HarnessTurnError,
)
from litellm.llms.base_llm.harness.utils import decode_json_line, read_skill_files

# Link <private_dir>/<dir> to a LiteLLM-owned cache dir so a later session can resume.
PERSIST_DIR_SCRIPT: Final = (
    'd="${HOME:-/tmp}/.cache/litellm-harness/$2"; mkdir -p "$d" && mkdir -p "$(dirname "$1")" && ln -sfn "$d" "$1"'
)


async def iter_stream_lines(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Newline-delimited lines without StreamReader's 64KiB readline limit."""
    buffer = b""
    while True:
        chunk = await stream.read(HARNESS_STREAM_READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            yield line
    if buffer:
        yield buffer


async def drain_stderr(stream: asyncio.StreamReader, tail: deque[str]) -> None:  # mutable-ok: stderr ring
    async for line in iter_stream_lines(stream):
        tail.append(line.decode("utf-8", errors="replace"))


async def send_stdin(proc: Process, data: str) -> None:
    if proc.stdin is None:
        raise SandboxError("harness process has no stdin")
    proc.stdin.write(data.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()


async def private_dir_for(sandbox: Sandbox) -> str:
    tempdir = getattr(sandbox, "tempdir", None)
    if tempdir is None:
        raise SandboxError(f"{type(sandbox).__name__} has no tempdir(); CLI harnesses need a private config dir")
    path: str = await tempdir()
    return path


def sandbox_path(private_dir: str, path: str) -> str:
    return path if path.startswith("/") else f"{private_dir}/{path}"


async def persist_dir(sandbox: Sandbox, link_path: str, cache_subpath: str) -> None:
    script_args: Final = ("-c", PERSIST_DIR_SCRIPT, "sh", link_path, cache_subpath)
    cmd: Final = ["sh", *script_args]  # mutable-ok: Sandbox.run takes list[str]
    run = await sandbox.run(cmd)
    if run.exit_code != 0:
        verbose_logger.debug(
            "harness: could not persist %s, resume across sessions disabled: %s", cache_subpath, run.stderr.strip()
        )


@dataclass(frozen=True, slots=True)
class ResumeFailed:
    reason: str


async def bounded_resume_run(
    sandbox: Sandbox, cmd: Sequence[str], env: Mapping[str, str]
) -> CompletedRun | ResumeFailed:
    try:
        run: Final = await sandbox.run(cmd, env=env, timeout=HARNESS_RESUME_COMMAND_TIMEOUT_SECONDS)
    except SandboxError as exc:
        return ResumeFailed(str(exc))
    return run if run.exit_code == 0 else ResumeFailed(run.stderr.strip() or f"exit code {run.exit_code}")


async def import_session_copy(sandbox: Sandbox, copy: HarnessSessionCopy, env: Mapping[str, str]) -> str | ResumeFailed:
    try:
        await sandbox.write(copy.path, copy.data)
    except SandboxError as exc:
        return ResumeFailed(str(exc))
    imported: Final = await bounded_resume_run(sandbox, copy.import_command, env)
    return imported if isinstance(imported, ResumeFailed) else copy.session_id


async def resumable_session_id(
    config: BaseCLIHarnessConfig[object, object],
    sandbox: Sandbox,
    setup_env: Mapping[str, str],
    private_dir: str,
    native_session_id: str,
) -> str | ResumeFailed:
    export_cmd: Final = config.transform_resume_export_command(native_session_id)
    if export_cmd is None:
        return native_session_id
    exported: Final = await bounded_resume_run(sandbox, export_cmd, setup_env)
    if isinstance(exported, ResumeFailed):
        return exported
    try:
        copy: Final = config.transform_resume_copy(exported.stdout, private_dir, sandbox.workdir)
    except HarnessError as exc:
        return ResumeFailed(str(exc))
    if copy is None:
        return native_session_id
    return await import_session_copy(sandbox, copy, setup_env)


async def copy_skills(sandbox: Sandbox, skills: Sequence[str], skills_root: str) -> None:
    for skill in skills:
        name = os.path.basename(os.path.realpath(os.fspath(skill)))
        for rel, data in await asyncio.to_thread(read_skill_files, skill):
            await sandbox.write(f"{skills_root}/{name}/{rel.replace(os.sep, '/')}", data)


class CLIHarnessHandler(BaseHarnessHandler):
    config: BaseCLIHarnessConfig[object, object]

    def __init__(self, config: BaseCLIHarnessConfig) -> None:
        super().__init__(config)
        self._private_dir: str | None = None
        self._setup: HarnessSessionSetup | None = None
        self._native_id: str | None = None
        self._pending_resume: str | None = None
        self._proc: Process | None = None

    async def start(self, ctx: SessionContext) -> None:
        self.config.validate_environment(ctx)
        binary = self.config.get_binary()
        if not await ctx.sandbox.which(binary):
            raise HarnessInstallFailed(
                f"`{binary}` was not found on PATH in the sandbox. Install it with: {self.config.get_install_hint()}"
            )
        private_dir = await private_dir_for(ctx.sandbox)
        setup = self.config.transform_session_setup(ctx, private_dir)
        for link, cache_subpath in setup.persisted_dirs:
            await persist_dir(ctx.sandbox, sandbox_path(private_dir, link), cache_subpath)
        for rel_path, data in setup.files.items():
            await ctx.sandbox.write(sandbox_path(private_dir, rel_path), data)
        if ctx.skills and setup.skills_dir:
            await copy_skills(ctx.sandbox, tuple(ctx.skills), sandbox_path(private_dir, setup.skills_dir))
        self._private_dir = private_dir
        self._setup = setup

    async def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        if self._setup is None or self._private_dir is None:
            raise RuntimeError("CLIHarnessHandler.turn() called before start()")
        if self._pending_resume is not None:
            self._native_id = await self._prepared_session_id(ctx, self._setup, self._private_dir, self._pending_resume)
            self._pending_resume = None
        request = self.config.transform_turn_request(ctx, self._setup, self._private_dir, prompt, self._native_id)
        argv: Final = list(request.argv)  # mutable-ok: Sandbox.exec takes list[str]
        proc = await ctx.sandbox.exec(argv, env=request.env, cwd=request.cwd)
        self._proc = proc
        tail: Final[deque[str]] = deque(maxlen=HARNESS_STDERR_TAIL_LINES)  # mutable-ok: bounded stderr ring buffer
        stderr_task = asyncio.ensure_future(drain_stderr(proc.stderr, tail))
        state: Final[object] = self.config.create_stream_state()
        exit_code: int | None = None
        try:
            await send_stdin(proc, request.stdin)
            async for raw in iter_stream_lines(proc.stdout):
                line = decode_json_line(raw)
                if line is None:
                    continue
                for event in self.config.transform_stream_line(line, state):
                    yield event
                self._native_id = self.config.get_native_session_id(state) or self._native_id
            exit_code = await proc.wait()
            await stderr_task
        finally:
            self._proc = None
            if exit_code is None:
                # Consumer stopped early, timed out or errored: don't leave the runtime running.
                await proc.kill()
            if not stderr_task.done():
                stderr_task.cancel()
        response = self.config.transform_turn_response(ctx, state, exit_code, tuple(tail))
        ctx.final_text = response.final_text
        ctx.output_json = response.output_json

    async def stop(self, ctx: SessionContext) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            await proc.kill()

    def native_session_id(self) -> str | None:
        return self._native_id

    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        if native_session_id != self._native_id:
            self._native_id = native_session_id
            self._pending_resume = native_session_id

    async def _prepared_session_id(
        self, ctx: SessionContext, setup: HarnessSessionSetup, private_dir: str, saved_id: str
    ) -> str:
        target: Final = await resumable_session_id(self.config, ctx.sandbox, setup.env, private_dir, saved_id)
        match target:
            case ResumeFailed(reason=reason):
                raise HarnessTurnError(f"could not prepare session {saved_id} for resume: {reason}")
            case session_id:
                return session_id
