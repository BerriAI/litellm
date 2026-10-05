import asyncio
import json
import os
import sys
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing
from functools import lru_cache
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Final

from pydantic import Field

from .models import Record

_READY: Final = b"\x1eLENS_PYTHON_READY\x1e\n"


class PythonLimits(Record):
    wall_seconds: float = Field(default=60, gt=0)
    cpu_seconds: int = Field(default=30, ge=1)
    memory_bytes: int = Field(default=512 * 1024 * 1024, ge=16 * 1024 * 1024)
    output_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    file_bytes: int = Field(default=16 * 1024 * 1024, ge=1)
    scratch_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    scratch_entries: int = Field(default=2048, ge=1)


class PythonRuntime(Record):
    executable: str
    directories: tuple[str, ...]
    read: tuple[str, ...]
    execute: tuple[str, ...]


_DEFAULT_LIMITS: Final = PythonLimits()


class ExecutionLimit(Exception):
    pass


class PythonInputError(Exception):
    pass


def _bootstrap(limits: PythonLimits) -> str:
    return f"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, ({limits.cpu_seconds}, {limits.cpu_seconds}))
resource.setrlimit(resource.RLIMIT_AS, ({limits.memory_bytes}, {limits.memory_bytes}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({limits.file_bytes}, {limits.file_bytes}))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
import json, sys
sys.stderr.write({_READY.decode()!r})
request = json.load(sys.stdin)
exec(compile(request["code"], "<lens-python>", "exec"), {{"__name__": "__main__", "data": request["data"]}})
"""


def _command(directory: str, limits: PythonLimits) -> tuple[str, ...]:
    if sys.platform != "linux":
        raise OSError("Python analysis requires the native Linux Lens worker with Landlock and seccomp support.")
    runtime: Final = PythonRuntime.model_validate_json(Path(__file__).with_name("python-runtime.json").read_text())
    policy: Final = Path(__file__).with_name("python.seccomp")
    if not policy.is_file():
        raise OSError("The Lens worker is missing its Python syscall policy. Rebuild the matching worker image.")
    reads: Final = tuple(
        ("--landlock-rule", f"path-beneath:read-file,read-dir:{path}")
        if Path(path).is_dir()
        else ("--landlock-rule", f"path-beneath:read-file:{path}")
        for path in runtime.read
    )
    executable: Final = tuple(("--landlock-rule", f"path-beneath:read-file,execute:{path}") for path in runtime.execute)
    directories: Final = tuple(("--landlock-rule", f"path-beneath:read-dir:{path}") for path in runtime.directories)
    return (
        "/usr/bin/setpriv",
        "--no-new-privs",
        "--landlock-access",
        "fs:execute,write-file,read-file,read-dir,remove-dir,remove-file,make-char,make-dir,make-reg,make-sock,"
        "make-fifo,make-block,make-sym,refer,truncate",
        *chain.from_iterable(reads),
        *chain.from_iterable(executable),
        *chain.from_iterable(directories),
        "--landlock-rule",
        "path-beneath:read-file,read-dir,write-file,remove-file,remove-dir,make-dir,make-reg,make-sym,refer,truncate:"
        + directory,
        "--seccomp-filter",
        str(policy),
        runtime.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        "-u",
        "-c",
        _bootstrap(limits),
    )


async def _input_chunks(data: str | AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    if isinstance(data, str):
        for offset in range(0, len(data), 65536):
            yield data[offset : offset + 65536]
        return
    async with aclosing(data):
        async for chunk in data:
            yield chunk


async def _feed(process: asyncio.subprocess.Process, code: str, data: str | AsyncGenerator[str, None]) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write((json.dumps({"code": code})[:-1] + ', "data":').encode())
        async with aclosing(_input_chunks(data)) as chunks:
            async for chunk in chunks:
                process.stdin.write(chunk.encode())
                await process.stdin.drain()
        process.stdin.write(b"}")
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        process.stdin.close()


async def _read(stream: asyncio.StreamReader | None, limit: int, ready: asyncio.Event | None = None) -> bytes:
    assert stream is not None
    chunks: tuple[bytes, ...] = ()  # rebind-ok: collect bounded pipe output until EOF
    size = 0  # rebind-ok: count streamed bytes before retaining another chunk
    while chunk := await stream.read(65536):
        size += len(chunk)
        if size > limit:
            raise ExecutionLimit(f"Python output exceeded {limit} bytes on one stream; output was not delivered.")
        chunks = (*chunks, chunk)
        if ready is not None and not ready.is_set() and b"".join(chunks).startswith(_READY):
            ready.set()
    return b"".join(chunks)


def _walk_error(error: OSError) -> None:
    raise ExecutionLimit("Python scratch storage could not be inspected; execution stopped.") from error


def _scratch_files(directory: str, pid: int) -> Iterator[os.stat_result]:
    for path, directories, files, descriptor in os.fwalk(directory, follow_symlinks=False, onerror=_walk_error):
        if path.count(os.sep) - directory.count(os.sep) > 128:
            raise ExecutionLimit("Python exceeded its scratch directory-depth limit.")
        for name in (*directories, *files):
            try:
                yield os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
    try:
        descriptors: Final = tuple(Path(f"/proc/{pid}/fd").iterdir())
    except FileNotFoundError:
        return
    for descriptor in descriptors:
        try:
            if os.readlink(descriptor).startswith(directory + os.sep):
                yield descriptor.stat()
        except FileNotFoundError:
            continue


def _scratch_usage(directory: str, pid: int, limits: PythonLimits) -> None:
    size = 0  # rebind-ok: count storage across a descriptor-based directory walk
    entries = 0  # rebind-ok: bound both inode consumption and traversal work
    seen: Final[set[tuple[int, int]]] = set()  # mutable-ok: deduplicate bounded tree and open-file inode accounting
    for details in _scratch_files(directory, pid):
        entries += 1
        if (identity := (details.st_dev, details.st_ino)) not in seen:
            size += max(details.st_size, details.st_blocks * 512)
            seen.add(identity)
        if entries > limits.scratch_entries or size > limits.scratch_bytes:
            raise ExecutionLimit("Python exceeded its scratch storage or file-count limit.")
    page_size: Final = os.sysconf("SC_PAGE_SIZE")
    for mapped in _mapped_scratch(directory, pid):
        if mapped in seen:
            continue
        entries += 1
        size += ((limits.file_bytes + page_size - 1) // page_size) * page_size
        seen.add(mapped)
        if entries > limits.scratch_entries or size > limits.scratch_bytes:
            raise ExecutionLimit("Python exceeded its scratch storage or file-count limit.")


def _mapped_scratch(directory: str, pid: int) -> Iterator[tuple[int, int]]:
    prefix: Final = directory.replace("\n", "\\012") + os.sep
    try:
        mappings: Final = Path(f"/proc/{pid}/maps").read_text().splitlines()
    except FileNotFoundError:
        return
    for mapping in mappings:
        if len(fields := mapping.split(maxsplit=5)) < 6 or fields[4] == "0":
            continue
        if fields[5].startswith(prefix):
            major, minor = fields[3].split(":")
            yield os.makedev(int(major, 16), int(minor, 16)), int(fields[4])


async def _monitor(
    process: asyncio.subprocess.Process, directory: str, limits: PythonLimits, ready: asyncio.Event
) -> None:
    while not ready.is_set():
        if process.returncode is not None:
            return
        await asyncio.sleep(0.005)
    try:
        while process.returncode is None:
            _scratch_usage(directory, process.pid, limits)
            await asyncio.sleep(0.05)
        _scratch_usage(directory, process.pid, limits)
    except (PermissionError, ProcessLookupError):
        try:
            await asyncio.wait_for(process.wait(), timeout=0.05)
        except TimeoutError as error:
            raise ExecutionLimit("Python scratch storage could not be inspected; execution stopped.") from error
        _scratch_usage(directory, process.pid, limits)


async def _discard(stream: asyncio.StreamReader | None) -> None:
    if stream is not None:
        while await stream.read(65536):
            pass


def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass


async def _stop(process: asyncio.subprocess.Process) -> None:
    _kill(process)
    await asyncio.gather(_discard(process.stdout), _discard(process.stderr), process.wait())


async def _finish(task: asyncio.Task[None]) -> bool:
    cancelled = False  # rebind-ok: propagate cancellation only after the child has been reaped
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    return cancelled


async def _cancel_spawn(spawn: asyncio.Task[asyncio.subprocess.Process]) -> None:
    await _stop(await spawn)


async def _cleanup(pending: tuple[asyncio.Task[object], ...], process: asyncio.subprocess.Process) -> None:
    await asyncio.gather(*pending, return_exceptions=True)
    await _stop(process)


async def _start(command: tuple[str, ...], directory: str) -> asyncio.subprocess.Process:
    spawn: Final = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=directory,
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "TMPDIR": directory},
            start_new_session=True,
            close_fds=True,
        )
    )
    try:
        return await asyncio.shield(spawn)
    except asyncio.CancelledError:
        await _finish(asyncio.create_task(_cancel_spawn(spawn)))
        raise


def _result(started: float, stdout: bytes = b"", stderr: bytes = b"", code: int | None = None, error: str = "") -> str:
    return json.dumps(
        {
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "exit_code": code,
            "elapsed_seconds": monotonic() - started,
            "error": error,
            "output_complete": not error,
        },
        ensure_ascii=False,
    )


@lru_cache(maxsize=1)
def _python_slots(loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
    count: Final = int(os.environ.get("LENS_PYTHON_CONCURRENCY", "2"))
    if count < 1:
        raise ValueError("LENS_PYTHON_CONCURRENCY must be a positive integer")
    return asyncio.Semaphore(count)


async def execute_python(
    code: str, data: str | AsyncGenerator[str, None], *, limits: PythonLimits = _DEFAULT_LIMITS
) -> str:
    try:
        slots: Final = _python_slots(asyncio.get_running_loop())
    except ValueError as error:
        return _result(monotonic(), error=f"Python confinement unavailable: {error}")
    async with slots:
        return await _execute(code, data, limits)


async def _execute(code: str, data: str | AsyncGenerator[str, None], limits: PythonLimits) -> str:
    started: Final = monotonic()
    with TemporaryDirectory(prefix="lens-python-") as temporary:
        directory: Final = str(Path(temporary).resolve())
        try:
            command: Final = _command(directory, limits)
            process: Final = await _start(command, directory)
        except (OSError, ValueError) as error:
            return _result(started, error=f"Python confinement unavailable: {error}")
        ready: Final = asyncio.Event()
        pending: Final = (
            asyncio.create_task(_feed(process, code, data)),
            asyncio.create_task(_read(process.stdout, limits.output_bytes)),
            asyncio.create_task(_read(process.stderr, limits.output_bytes + len(_READY), ready)),
            asyncio.create_task(process.wait()),
            asyncio.create_task(_monitor(process, directory, limits, ready)),
        )
        try:
            finished, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                task.result()
            if not pending[0].done():
                pending[0].cancel()
                await asyncio.gather(pending[0], return_exceptions=True)
            stdout, stderr, exit_code, _ = await asyncio.wait_for(
                asyncio.gather(*pending[1:]), timeout=limits.wall_seconds
            )
            return _result(
                started,
                stdout,
                stderr.removeprefix(_READY),
                exit_code,
                "Python confinement failed before execution; inspect stderr and the worker image/kernel support."
                if not stderr.startswith(_READY)
                else f"Python was terminated by signal {-exit_code}; a resource limit may have been reached."
                if exit_code < 0
                else f"Python exited with status {exit_code}; inspect stderr for the computation failure."
                if exit_code
                else "",
            )
        except TimeoutError:
            return _result(started, error=f"Python exceeded its {limits.wall_seconds:g}-second elapsed-time limit.")
        except (ExecutionLimit, PythonInputError, OSError) as error:
            return _result(started, error=str(error))
        finally:
            _kill(process)
            for task in pending:
                task.cancel()
            if await _finish(asyncio.create_task(_cleanup(pending, process))):
                raise asyncio.CancelledError
