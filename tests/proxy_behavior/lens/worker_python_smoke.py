import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Final

import pydantic
from lens import python_tool
from lens.python_tool import PythonInputError, PythonLimits, execute_python
from pydantic import BaseModel


class Reply(BaseModel):
    stdout: str
    stderr: str
    exit_code: int | None
    error: str
    output_complete: bool


async def run(code: str, data: str = "{}", *, limits: PythonLimits = PythonLimits()) -> Reply:
    return Reply.model_validate_json(await execute_python(dedent(code), data, limits=limits))


def succeeded(reply: Reply) -> None:
    assert reply.exit_code == 0 and not reply.error and reply.output_complete, reply


async def useful_python() -> None:
    reply: Final = await run(
        """
        import collections, json, math, sqlite3, tempfile
        counts = collections.Counter(p["parent"] for p in data["parts"])
        with tempfile.TemporaryFile() as temporary:
            temporary.write(b"temporary file")
            temporary.seek(0)
            assert temporary.read() == b"temporary file"
        connection = sqlite3.connect("evidence.db")
        connection.execute("create table parts(parent text)")
        connection.executemany("insert into parts values(?)", [(p["parent"],) for p in data["parts"]])
        assert connection.execute("select count(*) from parts").fetchone()[0] == 3
        assert math.sqrt(81) == 9
        print(json.dumps(dict(counts), sort_keys=True))
        """,
        '{"parts":[{"parent":"root"},{"parent":"child"},{"parent":"root"}]}',
    )
    succeeded(reply)
    assert reply.stdout == '{"child": 1, "root": 2}\n', reply
    large: Final = await run(
        'import sys\nprint(data, end="")\nprint(data, end="", file=sys.stderr)', json.dumps("x" * 100000)
    )
    succeeded(large)
    assert large.stdout == large.stderr == "x" * 100000
    for code, status, error in (
        ("1/0", 1, "ZeroDivisionError"),
        ("if :", 1, "SyntaxError"),
        ("raise SystemExit(7)", 7, ""),
    ):
        failed: Final = await run(code)
        assert failed.exit_code == status and error in failed.stderr and failed.error and not failed.output_complete, (
            failed
        )
    print("PASS ordinary Python, nested evidence, SQLite, temporary files, complete output and script errors")


async def boundaries() -> None:
    os.environ["LENS_TEST_SECRET"] = "worker-secret"
    with TemporaryDirectory(prefix="lens-worker-sentinel-") as sibling:
        sentinel: Final = Path(sibling) / "secret"
        sentinel.write_text("private worker content")
        before: Final = sentinel.stat()
        reply: Final = await run(
            """
            import ctypes, errno, json, os, pathlib, socket, sys
            assert os.getenv("LENS_TEST_SECRET") is None
            assert os.getenv("PYTHONPATH") is None
            assert sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
            def denied(action):
                try:
                    action()
                except OSError as error:
                    assert error.errno in (errno.EACCES, errno.EPERM, errno.EXDEV), error
                    return
                raise AssertionError("operation escaped confinement")
            secret = data["sentinel"]
            for path in (secret, "/proc/self/environ", "/app/lens/worker.py", data["package"]):
                denied(lambda: open(path).read())
            denied(lambda: os.listdir("/proc"))
            denied(lambda: open(secret, "w"))
            denied(lambda: os.chmod(secret, 0o777))
            denied(lambda: os.chown(secret, os.getuid(), os.getgid()))
            denied(lambda: os.utime(secret))
            denied(lambda: os.setxattr(secret, "user.lens", b"changed"))
            os.symlink(secret, "symlink")
            denied(lambda: open("symlink").read())
            denied(lambda: open("symlink", "w"))
            denied(lambda: os.link(secret, "hardlink"))
            denied(lambda: os.rename(secret, "renamed"))
            for family, kind in ((socket.AF_INET, socket.SOCK_STREAM), (socket.AF_INET, socket.SOCK_DGRAM),
                                 (socket.AF_UNIX, socket.SOCK_STREAM)):
                denied(lambda: socket.socket(family, kind))
            denied(socket.socketpair)
            denied(os.fork)
            denied(lambda: os.kill(os.getppid(), 0))
            denied(lambda: os.execv("/bin/sh", ["sh", "-c", "exit 0"]))
            library = ctypes.CDLL(None, use_errno=True)
            for name, arguments in (("ptrace", (16, os.getppid(), 0, 0)),
                                    ("process_vm_readv", (os.getppid(), 0, 0, 0, 0, 0)),
                                    ("process_vm_writev", (os.getppid(), 0, 0, 0, 0, 0)),
                                    ("shmget", (0, 4096, 0o1600)), ("syscall", (425, 0, 0))):
                ctypes.set_errno(0)
                assert getattr(library, name)(*arguments) == -1, name
                assert ctypes.get_errno() == errno.EPERM, name
            print("denied")
            """,
            json.dumps({"sentinel": str(sentinel), "package": pydantic.__file__}),
        )
        succeeded(reply)
        assert reply.stdout == "denied\n", reply
        assert sentinel.read_text() == "private worker content"
        assert sentinel.stat().st_mode == before.st_mode and sentinel.stat().st_mtime_ns == before.st_mtime_ns
    print("PASS worker files, secrets, metadata mutation, path escapes, network, process and raw syscall boundaries")


async def resources() -> None:
    wall: Final = await run("import time\ntime.sleep(10)", limits=PythonLimits(wall_seconds=0.2))
    assert "elapsed-time limit" in wall.error and not wall.output_complete, wall
    cpu: Final = await run("while True: pass", limits=PythonLimits(cpu_seconds=1, wall_seconds=5))
    assert cpu.exit_code is not None and cpu.exit_code < 0 and not cpu.output_complete, cpu
    memory: Final = await run("x = bytearray(1024 * 1024 * 1024)", limits=PythonLimits(memory_bytes=64 * 1024 * 1024))
    assert memory.exit_code != 0 and "MemoryError" in memory.stderr and memory.error and not memory.output_complete, (
        memory
    )
    file: Final = await run('open("large", "wb").write(b"x" * 100000)', limits=PythonLimits(file_bytes=1024))
    assert file.exit_code != 0 and "File too large" in file.stderr, file
    output: Final = await run('print("x" * 100000)', limits=PythonLimits(output_bytes=1024))
    assert "output exceeded" in output.error and not output.stdout and not output.output_complete, output
    entries: Final = await run(
        "import pathlib,time\nfor i in range(128): pathlib.Path(str(i)).touch()\ntime.sleep(1)",
        limits=PythonLimits(scratch_entries=16),
    )
    assert "scratch storage" in entries.error, entries
    fast_entries: Final = await run(
        "import pathlib\nfor i in range(128): pathlib.Path(str(i)).touch()",
        limits=PythonLimits(scratch_entries=16),
    )
    assert "scratch storage" in fast_entries.error, fast_entries
    hidden: Final = await run("import ctypes,time\nassert ctypes.CDLL(None).prctl(4,0,0,0,0) == 0\ntime.sleep(1)")
    assert "could not be inspected" in hidden.error and not hidden.output_complete, hidden
    for retained in ("files.append(f)", "maps.append(mmap.mmap(f.fileno(), 1, trackfd=False))\n    f.close()"):
        scratch: Final = await run(
            "import mmap,os,time\nfiles=[]\nmaps=[]\nfor i in range(4):\n"
            '    f=open(str(i), "w+b")\n    f.write(b"x" * 1048576)\n    f.flush()\n'
            "    os.unlink(str(i))\n    " + retained + "\ntime.sleep(1)",
            limits=PythonLimits(file_bytes=1048576, scratch_bytes=1500000),
        )
        assert "scratch storage" in scratch.error, scratch
    deep: Final = await run(
        'import os,time\nfor i in range(1600):\n    os.mkdir("d")\n    os.chdir("d")\ntime.sleep(1)'
    )
    assert "directory-depth limit" in deep.error, deep
    assert not tuple(Path("/tmp").glob("lens-python-*")), "scratch survived a limit failure"
    print("PASS wall, CPU, memory, file, output, inode, unlinked-file, mapped-file and deep-tree limits")


async def ready_directories(count: int) -> tuple[Path, ...]:
    async with asyncio.timeout(5):
        while True:
            paths: Final = tuple(path for path in Path("/tmp").glob("lens-python-*/ready") if path.is_file())
            if len(paths) == count:
                return paths
            await asyncio.sleep(0.01)


async def cancellation_and_pool() -> None:
    code: Final = 'import os,time\nopen("ready", "w").write(str(os.getpid()))\ntime.sleep(10)'
    running: Final = tuple(asyncio.create_task(run(code)) for _ in range(2))
    try:
        paths: Final = await ready_directories(2)
        pids: Final = tuple(int(path.read_text()) for path in paths)
        queued: Final = asyncio.create_task(run('raise AssertionError("cancelled queue entry executed")'))
        await asyncio.sleep(0.05)
        assert len(tuple(Path("/tmp").glob("lens-python-*"))) == 2
        queued.cancel()
        await asyncio.sleep(0)
        queued.cancel()
        cancelled: Final = await asyncio.gather(queued, return_exceptions=True)
        assert isinstance(cancelled[0], asyncio.CancelledError)
    finally:
        for task in running:
            task.cancel()
        await asyncio.sleep(0)
        for task in running:
            task.cancel()
        stopped: Final = await asyncio.gather(*running, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in stopped), stopped
    assert all(not Path(f"/proc/{pid}").exists() for pid in pids), "cancelled child survived"
    assert all(not path.parent.exists() for path in paths), "cancelled scratch survived"
    for _ in range(4):
        spawning: Final = asyncio.create_task(run(code))
        await asyncio.sleep(0)
        spawning.cancel()
        await asyncio.sleep(0)
        spawning.cancel()
        spawned: Final = await asyncio.gather(spawning, return_exceptions=True)
        assert isinstance(spawned[0], asyncio.CancelledError)
    assert not Path(f"/proc/self/task/{os.getpid()}/children").read_text().strip(), "spawn cancellation leaked a child"
    isolated: Final = await asyncio.gather(
        *(
            run(
                'import time\nopen("same", "w").write(data)\ntime.sleep(.1)\nprint(open("same").read())',
                json.dumps(value),
            )
            for value in ("first", "second")
        )
    )
    assert tuple(reply.stdout for reply in isolated) == ("first\n", "second\n"), isolated
    fresh: Final = await run('print("data" in globals(), "f" in globals())')
    succeeded(fresh)
    assert fresh.stdout == "True False\n"
    startups: Final = await asyncio.gather(*(run("print(1)") for _ in range(32)))
    assert all(reply.stdout == "1\n" and not reply.error for reply in startups), startups
    assert not tuple(Path("/tmp").glob("lens-python-*"))
    print("PASS worker-wide pool, queued/running cancellation, reaping, cleanup and concurrent workspace isolation")


async def streamed_input() -> None:
    async def slow():
        yield '{"value":'
        await asyncio.sleep(0.3)
        yield '"complete"}'

    reply: Final = Reply.model_validate_json(
        await execute_python('print(data["value"])', slow(), limits=PythonLimits(wall_seconds=0.2))
    )
    succeeded(reply)
    assert reply.stdout == "complete\n"

    async def missing():
        yield '{"sessions":['
        raise PythonInputError("Unknown span IDs: missing")

    invalid: Final = Reply.model_validate_json(await execute_python('print("must not execute")', missing()))
    assert "Unknown span IDs" in invalid.error and not invalid.stdout and not invalid.output_complete, invalid
    oversized_closed: Final = asyncio.Event()

    async def oversized():
        try:
            yield '"'
            for _ in range(2048):
                yield "x" * 65536
            yield '"'
        finally:
            oversized_closed.set()

    oversized_reply: Final = Reply.model_validate_json(
        await execute_python(
            'print("must not execute")', oversized(), limits=PythonLimits(memory_bytes=64 * 1024 * 1024)
        )
    )
    assert oversized_reply.error and not oversized_reply.stdout and not oversized_reply.output_complete, oversized_reply
    assert oversized_closed.is_set()
    entered: Final = asyncio.Event()
    closed: Final = asyncio.Event()

    async def stalled():
        try:
            yield '{"value":'
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    pending: Final = asyncio.create_task(execute_python('print("must not execute")', stalled()))
    await asyncio.wait_for(entered.wait(), timeout=5)
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    stopped: Final = await asyncio.gather(pending, return_exceptions=True)
    assert isinstance(stopped[0], asyncio.CancelledError) and closed.is_set()
    assert not tuple(Path("/tmp").glob("lens-python-*"))
    assert not Path(f"/proc/self/task/{os.getpid()}/children").read_text().strip()
    print("PASS streamed input, separate fetch/computation timing, missing selectors and stalled-source cancellation")


def unavailable_policy() -> None:
    source: Final = Path(python_tool.__file__).parent
    with TemporaryDirectory(prefix="lens-policy-smoke-") as directory:
        package: Final = Path(directory) / "lens"
        package.mkdir()
        for name in ("__init__.py", "models.py", "python_tool.py", "python-runtime.json"):
            shutil.copyfile(source / name, package / name)
        for invalid in (False, True):
            if invalid:
                (package / "python.seccomp").write_bytes(b"invalid syscall policy")
            process: Final = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    "import asyncio; from lens.python_tool import execute_python; "
                    "print(asyncio.run(execute_python('print(123456)', '{}')))",
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            reply: Final = Reply.model_validate_json(process.stdout)
            assert reply.error and not reply.output_complete and not reply.stdout, reply
            assert "confinement" in reply.error.lower(), reply
    print("PASS missing and invalid syscall policy fail closed")


async def main() -> None:
    assert sys.platform == "linux" and os.geteuid() != 0, "run inside the native non-root worker image"
    os.environ["LENS_PYTHON_CONCURRENCY"] = "2"
    await useful_python()
    await boundaries()
    await resources()
    await cancellation_and_pool()
    await streamed_input()
    unavailable_policy()
    print("Python confinement smoke passed")


if __name__ == "__main__":
    asyncio.run(main())
