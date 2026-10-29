import asyncio
import hashlib
import shutil
import subprocess
from typing import Optional

import pytest

from litellm import sandbox
from litellm.harness.errors import SandboxError
from litellm.harness.sandbox import DockerSandbox, Sandbox
from litellm.harness.sandbox.docker import parse_sha256sum

DOCKER_IMAGE = "alpine:3.20"
CID = "cid123"


class FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeHandle:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", code: int = 0):
        self.stdin = FakeStdin()
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self.returncode: Optional[int] = code
        self._code = code

    async def wait(self) -> int:
        return self._code

    async def kill(self) -> None:
        return None


class Recorder:
    """Stands in for DockerSandbox._spawn; scripted responses by docker subcommand."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.handles: list[FakeHandle] = []
        self.responses: dict[str, FakeHandle] = {}

    async def __call__(self, args: list[str]) -> FakeHandle:
        self.calls.append(args)
        handle = self.responses.pop(args[0], None)
        if handle is None:
            handle = FakeHandle(stdout=f"{CID}\n".encode() if args[0] == "run" else b"")
        self.handles.append(handle)
        return handle


@pytest.fixture
def fake(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(DockerSandbox, "_spawn", lambda self, args: rec(args))
    return rec


def test_run_args():
    box = sandbox.docker(
        "img:1",
        mounts={"/host/src": "/workspace"},
        env={"A": "1"},
        name="h1",
    )
    assert isinstance(box, Sandbox)
    assert box.run_args() == [
        "run",
        "-d",
        "--rm",
        "--add-host=host.docker.internal:host-gateway",
        "--name",
        "h1",
        "-v",
        "/host/src:/workspace",
        "-e",
        "A=1",
        "-w",
        "/workspace",
        "img:1",
        "sleep",
        "infinity",
    ]
    assert box.host_url(8080) == "http://host.docker.internal:8080"


def test_relative_workdir_rejected():
    with pytest.raises(SandboxError):
        sandbox.docker("img", workdir="rel")


async def test_lazy_start_and_exec_args(fake):
    box = sandbox.docker("img")
    assert fake.calls == []
    await box.exec(["echo", "hi"], env={"K": "V"}, cwd="sub")
    await box.exec(["true"])
    assert fake.calls[0][0] == "run"
    assert [c for c in fake.calls if c[0] == "run"] == [fake.calls[0]]
    assert fake.calls[1] == [
        "exec",
        "-i",
        "-w",
        "/workspace/sub",
        "-e",
        "K=V",
        CID,
        "echo",
        "hi",
    ]
    assert fake.calls[2] == ["exec", "-i", "-w", "/workspace", CID, "true"]


async def test_run_start_failure(fake):
    fake.responses["run"] = FakeHandle(stderr=b"no such image", code=125)
    box = sandbox.docker("img")
    with pytest.raises(SandboxError, match="no such image"):
        await box.run(["echo"])


async def test_read_write_which_tempdir(fake):
    box = sandbox.docker("img")
    await box.start()

    fake.responses["exec"] = FakeHandle(stdout=b"content")
    assert await box.read("a.txt") == b"content"
    assert fake.calls[-1][-2:] == ["cat", "/workspace/a.txt"]

    await box.write("d/b.txt", b"payload")
    assert fake.calls[-1][-5:-1] == ["sh", "-c", fake.calls[-1][-3], "sh"]
    assert fake.calls[-1][-1] == "/workspace/d/b.txt"
    assert fake.handles[-1].stdin.data == b"payload"
    assert fake.handles[-1].stdin.closed

    fake.responses["exec"] = FakeHandle(stdout=b"/usr/bin/codex\n")
    assert await box.which("codex") == "/usr/bin/codex"
    assert fake.calls[-1][-5:] == ["sh", "-lc", 'command -v "$1"', "sh", "codex"]

    fake.responses["exec"] = FakeHandle(code=1)
    assert await box.which("nope") is None

    fake.responses["exec"] = FakeHandle(stdout=b"/tmp/tmp.abc\n")
    assert await box.tempdir() == "/tmp/tmp.abc"

    fake.responses["exec"] = FakeHandle(stderr=b"No such file", code=1)
    with pytest.raises(SandboxError, match="No such file"):
        await box.read("missing")


async def test_snapshot_parses_output(fake):
    box = sandbox.docker("img")
    digest = "a" * 64
    fake.responses["exec"] = FakeHandle(
        stdout=f"{digest}  ./x.txt\n{digest}  ./dir/with space.txt\n".encode()
    )
    snap = await box.snapshot()
    assert snap == {"x.txt": digest, "dir/with space.txt": digest}
    script = fake.calls[-1][-3]
    assert "-name '.git'" in script and "-prune" in script
    assert fake.calls[-1][-1] == "/workspace"


async def test_close_removes_container(fake):
    box = sandbox.docker("img")
    await box.start()
    await box.close()
    assert fake.calls[-1] == ["rm", "-f", CID]
    with pytest.raises(SandboxError):
        await box.start()


async def test_close_without_start_is_noop(fake):
    await sandbox.docker("img").close()
    assert fake.calls == []


async def test_missing_docker_binary(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    with pytest.raises(SandboxError, match="docker"):
        await sandbox.docker("img").start()


def test_parse_sha256sum_ignores_junk():
    assert parse_sha256sum("garbage\n\n") == {}


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=20
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _docker_usable(), reason="docker daemon not available")
async def test_real_docker_roundtrip():
    box = sandbox.docker(DOCKER_IMAGE, workdir="/workspace")
    try:
        result = await box.run(["echo", "hello"], timeout=120)
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0

        await box.write("seed.txt", b"seed")
        await box.write("sub/out.txt", b"from host")
        assert await box.read("sub/out.txt") == b"from host"
        await box.write("node_modules/skip.js", b"x")

        assert await box.which("sh") is not None
        assert await box.which("definitely-not-a-binary-xyz") is None
        tmp = await box.tempdir()
        assert tmp.startswith("/")

        snap = await box.snapshot()
        assert snap == {
            "seed.txt": hashlib.sha256(b"seed").hexdigest(),
            "sub/out.txt": hashlib.sha256(b"from host").hexdigest(),
        }
    finally:
        await box.close()
