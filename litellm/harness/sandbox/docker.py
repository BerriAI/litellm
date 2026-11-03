"""DockerSandbox: run the harness runtime inside a container via the docker CLI."""

from __future__ import annotations

import asyncio
import os
import posixpath
import shutil
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.constants import HARNESS_SNAPSHOT_SKIP_DIRS
from litellm.harness.errors import SandboxError
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.sandbox.local import SubprocessHandle, collect_output
from litellm.harness.sandbox.snapshot import HARNESS_SNAPSHOT_MAX_FILE_BYTES

DOCKER_HOST_ALIAS: Final = "host.docker.internal"
_SHA256_HEX_LEN: Final = 64
_WRITE_SCRIPT: Final = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
_WHICH_SCRIPT: Final = 'command -v "$1"'


def _snapshot_script() -> str:
    prune = " -o ".join(f"-name '{name}'" for name in sorted(HARNESS_SNAPSHOT_SKIP_DIRS))
    return (
        'cd "$1" && find . -type d \\( '
        + prune
        + " \\) -prune -o -type f -size -"
        + f"{HARNESS_SNAPSHOT_MAX_FILE_BYTES + 1}c"
        + " -exec sha256sum {} +"
    )


def parse_sha256sum(output: str) -> Mapping[str, str]:
    """Parse `sha256sum` lines ("<hex>  ./rel/path") into {rel/path: hex}."""
    return MappingProxyType(
        {
            line[_SHA256_HEX_LEN + 2 :].removeprefix("./"): line[:_SHA256_HEX_LEN]
            for line in output.splitlines()
            if len(line) > _SHA256_HEX_LEN + 2
        }
    )


class DockerSandbox:
    """Sandbox backed by a long-lived `sleep infinity` container."""

    # Harness configs read this to skip a runtime's own nested OS sandbox.
    is_container = True

    def __init__(
        self,
        image: str,
        mounts: Mapping[str | os.PathLike[str], str] | None = None,
        workdir: str = "/workspace",
        env: Mapping[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        if not image:
            raise SandboxError("docker sandbox needs an image")
        if not posixpath.isabs(workdir):
            raise SandboxError(f"docker workdir must be absolute: {workdir}")
        self.image = image
        self.workdir: str = posixpath.normpath(workdir)
        self.mounts: Mapping[str, str] = MappingProxyType(
            {os.path.abspath(os.fspath(host)): container for host, container in (mounts.items() if mounts else ())}
        )
        self.env: Mapping[str, str] = MappingProxyType(dict(env or ()))
        self.name = name
        self.container_id: str | None = None
        self._start_lock = asyncio.Lock()
        self._processes: set[SubprocessHandle] = set()  # mutable-ok: live-process registry (add/discard)
        self._closed = False

    def __repr__(self) -> str:
        return f"DockerSandbox({self.image!r}, workdir={self.workdir!r})"

    def _docker_binary(self) -> str:
        binary = shutil.which("docker")
        if binary is None:
            raise SandboxError(
                "docker sandbox requires the `docker` CLI on PATH; install Docker or use sandbox.local(path)"
            )
        return binary

    async def _spawn(self, args: Sequence[str]) -> SubprocessHandle:
        """Start `docker <args>` with stdin/stdout/stderr pipes."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._docker_binary(),
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise SandboxError(f"could not run docker: {exc}") from exc
        return SubprocessHandle(proc)

    async def _docker(
        self,
        args: Sequence[str],
        *,
        input: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        """Run `docker <args>` to completion; returns (exit_code, stdout, stderr)."""
        handle = await self._spawn(args)
        try:
            return await asyncio.wait_for(_communicate(handle, input), timeout)
        except asyncio.TimeoutError:
            await handle.kill()
            raise SandboxError(f"docker {args[0]} timed out after {timeout}s")

    def run_args(
        self,
    ) -> list[str]:  # mutable-ok: argv is returned as a list, the shape callers and tests compare against
        name_args = ("--name", self.name) if self.name else ()
        mount_args = tuple(
            arg for host, container in self.mounts.items() for arg in ("-v", f"{host}:{container}")
        )  # comprehension-ok: flattens (flag, value) pairs into argv
        env_args = tuple(
            arg for key, value in self.env.items() for arg in ("-e", f"{key}={value}")
        )  # comprehension-ok: flattens (flag, value) pairs into argv
        return [  # mutable-ok: argv is returned as a list, the shape callers and tests compare against
            "run",
            "-d",
            "--rm",
            f"--add-host={DOCKER_HOST_ALIAS}:host-gateway",
            *name_args,
            *mount_args,
            *env_args,
            "-w",
            self.workdir,
            self.image,
            "sleep",
            "infinity",
        ]

    def exec_args(
        self,
        container_id: str,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[str]:  # mutable-ok: argv is returned as a list, the shape callers and tests compare against
        env_args = tuple(
            arg for key, value in (env.items() if env else ()) for arg in ("-e", f"{key}={value}")
        )  # comprehension-ok: flattens (flag, value) pairs into argv
        return [  # mutable-ok: argv is returned as a list, the shape callers and tests compare against
            "exec",
            "-i",
            "-w",
            self.container_path(cwd or self.workdir),
            *env_args,
            container_id,
            *cmd,
        ]

    def container_path(self, path: str) -> str:
        """Absolute container path; relative paths resolve against workdir."""
        joined = path if posixpath.isabs(path) else posixpath.join(self.workdir, path)
        return posixpath.normpath(joined)

    async def start(self) -> str:
        """Start the container if needed and return its id."""
        if self._closed:
            raise SandboxError("sandbox is closed")
        async with self._start_lock:
            if self.container_id is not None:
                return self.container_id
            code, out, err = await self._docker(self.run_args())
            if code != 0:
                raise SandboxError(f"docker run {self.image} failed ({code}): {err.decode(errors='replace').strip()}")
            container_id = out.decode().strip()
            if not container_id:
                raise SandboxError("docker run returned no container id")
            self.container_id = container_id
            return container_id

    async def _exec_capture(self, cmd: Sequence[str], *, input: bytes | None = None) -> tuple[int, bytes, bytes]:
        container_id = await self.start()
        return await self._docker(self.exec_args(container_id, cmd), input=input)

    async def exec(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> SubprocessHandle:
        if not cmd:
            raise SandboxError("exec() needs a non-empty command")
        container_id = await self.start()
        handle = await self._spawn(self.exec_args(container_id, cmd, env=env, cwd=cwd))
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
        target = self.container_path(path)
        code, out, err = await self._exec_capture(("cat", target))
        if code != 0:
            raise SandboxError(f"could not read {target}: {err.decode(errors='replace').strip()}")
        return out

    async def write(self, path: str, data: bytes) -> None:
        target = self.container_path(path)
        code, _, err = await self._exec_capture(("sh", "-c", _WRITE_SCRIPT, "sh", target), input=data)
        if code != 0:
            raise SandboxError(f"could not write {target}: {err.decode(errors='replace').strip()}")

    def host_url(self, port: int) -> str:
        return f"http://{DOCKER_HOST_ALIAS}:{port}"

    async def which(self, binary: str) -> str | None:
        code, out, _ = await self._exec_capture(("sh", "-lc", _WHICH_SCRIPT, "sh", binary))
        found = out.decode(errors="replace").strip()
        return found if code == 0 and found else None

    async def tempdir(self) -> str:
        """A fresh `mktemp -d` directory inside the container."""
        code, out, err = await self._exec_capture(("mktemp", "-d"))
        path = out.decode(errors="replace").strip()
        if code != 0 or not path:
            raise SandboxError(f"mktemp -d failed: {err.decode(errors='replace').strip()}")
        return path

    async def snapshot(self) -> Mapping[str, str]:
        code, out, err = await self._exec_capture(("sh", "-c", _snapshot_script(), "sh", self.workdir))
        if code != 0:
            raise SandboxError(f"snapshot failed: {err.decode(errors='replace').strip()}")
        return parse_sha256sum(out.decode("utf-8", errors="replace"))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        live = tuple(h for h in self._processes if h.returncode is None)
        await asyncio.gather(*(h.kill() for h in live), return_exceptions=True)
        self._processes.clear()
        if self.container_id is not None:
            container_id, self.container_id = self.container_id, None
            await self._docker(
                ["rm", "-f", container_id]  # mutable-ok: argv list, the shape _spawn records and tests assert on
            )

    async def __aenter__(self) -> DockerSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


async def _communicate(handle: SubprocessHandle, data: bytes | None) -> tuple[int, bytes, bytes]:
    if handle.stdin is not None:
        if data:
            handle.stdin.write(data)
            await handle.stdin.drain()
        handle.stdin.close()
    stdout, stderr = await asyncio.gather(handle.stdout.read(), handle.stderr.read())
    return await handle.wait(), stdout, stderr


def docker(
    image: str,
    mounts: Mapping[str | os.PathLike[str], str] | None = None,
    workdir: str = "/workspace",
    env: Mapping[str, str] | None = None,
    name: str | None = None,
) -> DockerSandbox:
    """Sandbox in a new container of `image`, started lazily on first use."""
    return DockerSandbox(image, mounts=mounts, workdir=workdir, env=env, name=name)
