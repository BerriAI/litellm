"""LocalSandbox: run the harness runtime as a subprocess on this machine."""

from __future__ import annotations

import asyncio
import itertools
import os
import shutil
import signal
import tempfile
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.constants import HARNESS_PROCESS_KILL_GRACE_SECONDS
from litellm.harness.errors import SandboxError
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.sandbox.snapshot import snapshot_local

_SECRET_PREFIXES: Final = (
    "ANTHROPIC_",
    "OPENAI_",
    "LITELLM_",
    "AZURE_",
    "AWS_",
    "GEMINI_",
    "CODEX_",
    "CURSOR_",
    "VERTEX",
    # A parent Claude Code session's socket/session vars make a child `claude` attach to
    # the parent's login instead of the harness token.
    "CLAUDE_CODE_",
    "CLAUDE_PID",
    "CLAUDECODE",
)
_SECRET_NAMES: Final = frozenset({"GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"})
_SECRET_SUBSTRINGS: Final = ("API_KEY", "TOKEN", "SECRET")
_TEMPDIR_PREFIX: Final = "litellm-harness-"


def is_secret_env_name(name: str) -> bool:
    """True if an env var name looks like a provider credential."""
    upper = name.upper()
    if upper in _SECRET_NAMES or upper.startswith(_SECRET_PREFIXES):
        return True
    return any(part in upper for part in _SECRET_SUBSTRINGS)


def filtered_environ(
    base: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """base (default os.environ) without provider secrets, then extra on top."""
    source = os.environ if base is None else base
    kept = ((k, v) for k, v in source.items() if not is_secret_env_name(k))
    overlay = extra.items() if extra else ()
    return MappingProxyType(dict(itertools.chain(kept, overlay)))


def _signal_process(proc: asyncio.subprocess.Process, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


class SubprocessHandle:
    """Process-protocol wrapper around an asyncio subprocess."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is None or proc.stderr is None:
            raise SandboxError("subprocess was started without stdout/stderr pipes")
        self._proc = proc
        self.stdin: asyncio.StreamWriter | None = proc.stdin
        self.stdout: asyncio.StreamReader = proc.stdout
        self.stderr: asyncio.StreamReader = proc.stderr

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def wait(self) -> int:
        return await self._proc.wait()

    async def kill(self) -> None:
        """SIGTERM, wait HARNESS_PROCESS_KILL_GRACE_SECONDS, then SIGKILL."""
        if self._proc.returncode is not None:
            return
        _signal_process(self._proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=HARNESS_PROCESS_KILL_GRACE_SECONDS)
            return
        except asyncio.TimeoutError:
            pass
        _signal_process(self._proc, signal.SIGKILL)
        await self._proc.wait()


async def _read_all(handle: SubprocessHandle) -> tuple[bytes, bytes, int]:
    if handle.stdin is not None:
        handle.stdin.close()
    stdout, stderr = await asyncio.gather(handle.stdout.read(), handle.stderr.read())
    exit_code = await handle.wait()
    return stdout, stderr, exit_code


async def collect_output(handle: SubprocessHandle, cmd: Sequence[str], timeout: float | None) -> CompletedRun:
    """Close stdin, read stdout/stderr to EOF; kill and raise SandboxError on timeout."""
    try:
        stdout, stderr, code = await asyncio.wait_for(_read_all(handle), timeout)
    except asyncio.TimeoutError:
        await handle.kill()
        raise SandboxError(f"command timed out after {timeout}s: {cmd[0]}")
    return CompletedRun(
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        exit_code=code,
    )


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


class LocalSandbox:
    """Sandbox backed by the local filesystem and asyncio subprocesses."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        resolved = os.path.realpath(os.path.abspath(os.fspath(path)))
        if not os.path.isdir(resolved):
            raise SandboxError(f"sandbox path does not exist or is not a directory: {resolved}")
        self.workdir: str = resolved
        self._processes: set[SubprocessHandle] = set()  # mutable-ok: live-process registry (add/discard)
        self._tempdirs: list[str] = []  # mutable-ok: tempdirs created on demand by tempdir(), removed on close()
        self._closed = False

    def __repr__(self) -> str:
        return f"LocalSandbox({self.workdir!r})"

    def _check_open(self) -> None:
        if self._closed:
            raise SandboxError("sandbox is closed")

    def _allowed_roots(self) -> tuple[str, ...]:
        return (self.workdir, *self._tempdirs)

    def resolve_path(self, path: str) -> str:
        """Absolute real path for path; SandboxError if it escapes the sandbox."""
        joined = path if os.path.isabs(path) else os.path.join(self.workdir, path)
        real = os.path.realpath(joined)
        if not any(_is_within(real, root) for root in self._allowed_roots()):
            raise SandboxError(f"path escapes the sandbox: {path}")
        return real

    def _resolve_cwd(self, cwd: str | None) -> str:
        if cwd is None:
            return self.workdir
        resolved = self.resolve_path(cwd)
        if not os.path.isdir(resolved):
            raise SandboxError(f"cwd is not a directory: {cwd}")
        return resolved

    def child_env(self, env: Mapping[str, str] | None = None) -> Mapping[str, str]:
        return filtered_environ(extra=env)

    async def exec(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> SubprocessHandle:
        self._check_open()
        if not cmd:
            raise SandboxError("exec() needs a non-empty command")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._resolve_cwd(cwd),
                env=self.child_env(env),
                start_new_session=True,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise SandboxError(f"could not start {cmd[0]}: {exc}") from exc
        handle = SubprocessHandle(proc)
        self._processes.add(handle)
        return handle

    async def run(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CompletedRun:
        handle = await self.exec(cmd, env=env, cwd=cwd)
        try:
            return await collect_output(handle, cmd, timeout)
        finally:
            self._processes.discard(handle)

    async def read(self, path: str) -> bytes:
        self._check_open()
        resolved = self.resolve_path(path)
        try:
            return await asyncio.to_thread(_read_bytes, resolved)
        except OSError as exc:
            raise SandboxError(f"could not read {path}: {exc}") from exc

    async def write(self, path: str, data: bytes) -> None:
        self._check_open()
        resolved = self.resolve_path(path)
        try:
            await asyncio.to_thread(_write_bytes, resolved, data)
        except OSError as exc:
            raise SandboxError(f"could not write {path}: {exc}") from exc

    def host_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}"

    async def which(self, binary: str) -> str | None:
        return shutil.which(binary, path=self.child_env().get("PATH"))

    async def tempdir(self) -> str:
        """A private temp dir (e.g. for CODEX_HOME), removed on close()."""
        self._check_open()
        path = os.path.realpath(tempfile.mkdtemp(prefix=_TEMPDIR_PREFIX))
        self._tempdirs.append(path)
        return path

    async def snapshot(self) -> Mapping[str, str]:
        self._check_open()
        return await snapshot_local(self.workdir)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        live = tuple(h for h in self._processes if h.returncode is None)
        await asyncio.gather(*(h.kill() for h in live), return_exceptions=True)
        self._processes.clear()
        for path in self._tempdirs:
            shutil.rmtree(path, ignore_errors=True)
        self._tempdirs.clear()

    async def __aenter__(self) -> LocalSandbox:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def local(path: str | os.PathLike[str]) -> LocalSandbox:
    """Sandbox rooted at an existing local directory."""
    return LocalSandbox(path)
