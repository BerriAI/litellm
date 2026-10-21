import os
import sys

import pytest

from litellm import sandbox
from litellm.harness.errors import SandboxError
from litellm.harness.sandbox import LocalSandbox, Process, Sandbox
from litellm.harness.sandbox.local import filtered_environ, is_secret_env_name

PY = sys.executable


@pytest.fixture
async def sbx(tmp_path):
    box = sandbox.local(tmp_path)
    yield box
    await box.close()


def test_local_requires_existing_dir(tmp_path):
    with pytest.raises(SandboxError):
        sandbox.local(tmp_path / "missing")


def test_local_resolves_absolute_and_satisfies_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ws").mkdir()
    box = sandbox.local("ws")
    assert isinstance(box, LocalSandbox)
    assert isinstance(box, Sandbox)
    assert box.workdir == os.path.realpath(tmp_path / "ws")
    assert box.host_url(4321) == "http://127.0.0.1:4321"


async def test_run_collects_output(sbx):
    result = await sbx.run(
        [
            PY,
            "-c",
            "import os,sys;print(os.getcwd());print('err',file=sys.stderr);sys.exit(3)",
        ]
    )
    assert result.stdout.strip() == sbx.workdir
    assert result.stderr.strip() == "err"
    assert result.exit_code == 3


async def test_run_cwd_inside_workdir(sbx):
    os.mkdir(os.path.join(sbx.workdir, "sub"))
    result = await sbx.run([PY, "-c", "import os;print(os.getcwd())"], cwd="sub")
    assert result.stdout.strip() == os.path.join(sbx.workdir, "sub")
    with pytest.raises(SandboxError):
        await sbx.run([PY, "-c", "pass"], cwd="/")


async def test_exec_streams_stdin(sbx):
    proc = await sbx.exec([PY, "-c", "import sys;print(sys.stdin.read().upper())"])
    assert isinstance(proc, Process)
    assert proc.stdin is not None
    proc.stdin.write(b"hello")
    await proc.stdin.drain()
    proc.stdin.close()
    assert (await proc.stdout.read()).strip() == b"HELLO"
    assert await proc.wait() == 0


async def test_run_timeout_kills(sbx):
    with pytest.raises(SandboxError, match="timed out"):
        await sbx.run([PY, "-c", "import time;time.sleep(30)"], timeout=0.5)


async def test_missing_binary_raises(sbx):
    with pytest.raises(SandboxError):
        await sbx.run(["definitely-not-a-binary-xyz"])


async def test_close_kills_live_processes(tmp_path):
    box = sandbox.local(tmp_path)
    proc = await box.exec([PY, "-c", "import time;time.sleep(30)"])
    await box.close()
    assert proc.returncode is not None
    with pytest.raises(SandboxError):
        await box.run([PY, "-c", "pass"])


async def test_read_write_roundtrip(sbx):
    await sbx.write("a/b/c.txt", b"data")
    assert await sbx.read("a/b/c.txt") == b"data"
    abs_path = os.path.join(sbx.workdir, "a", "b", "c.txt")
    assert await sbx.read(abs_path) == b"data"


@pytest.mark.parametrize("bad", ["../escape.txt", "a/../../escape.txt", "/etc/passwd"])
async def test_path_escape_rejected(sbx, bad):
    with pytest.raises(SandboxError, match="escapes"):
        await sbx.read(bad)
    with pytest.raises(SandboxError, match="escapes"):
        await sbx.write(bad, b"x")


async def test_symlink_escape_rejected(sbx, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    os.symlink(outside, os.path.join(sbx.workdir, "link"))
    with pytest.raises(SandboxError, match="escapes"):
        await sbx.write("link/x.txt", b"x")


async def test_tempdir_is_allowed_and_cleaned(tmp_path):
    box = sandbox.local(tmp_path)
    tmp = await box.tempdir()
    assert os.path.isdir(tmp)
    target = os.path.join(tmp, "config.toml")
    await box.write(target, b"k = 1")
    assert await box.read(target) == b"k = 1"
    await box.close()
    assert not os.path.exists(tmp)


async def test_env_filters_provider_secrets(sbx, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
    monkeypatch.setenv("HARNESS_TEST_PLAIN", "visible")
    script = (
        "import os;"
        "print(os.environ.get('ANTHROPIC_API_KEY','<none>'));"
        "print(os.environ.get('OPENAI_BASE_URL','<none>'));"
        "print(os.environ.get('GITHUB_TOKEN','<none>'));"
        "print(os.environ.get('HARNESS_TEST_PLAIN','<none>'));"
        "print(os.environ.get('ANTHROPIC_BASE_URL','<none>'))"
    )
    result = await sbx.run(
        [PY, "-c", script], env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}
    )
    assert result.stdout.split() == [
        "<none>",
        "<none>",
        "<none>",
        "visible",
        "http://127.0.0.1:1",
    ]


@pytest.mark.parametrize(
    "name,secret",
    [
        ("ANTHROPIC_API_KEY", True),
        ("AWS_REGION", True),
        ("VERTEXAI_PROJECT", True),
        ("GOOGLE_APPLICATION_CREDENTIALS", True),
        ("MY_API_KEY", True),
        ("SLACK_BOT_TOKEN", True),
        ("CLIENT_SECRET", True),
        ("PATH", False),
        ("HOME", False),
    ],
)
def test_is_secret_env_name(name, secret):
    assert is_secret_env_name(name) is secret


def test_filtered_environ_overlay_wins():
    env = filtered_environ(
        {"PATH": "/bin", "OPENAI_API_KEY": "x"}, {"PATH": "/usr/bin"}
    )
    assert env == {"PATH": "/usr/bin"}


async def test_which_uses_filtered_path(sbx):
    assert await sbx.which("sh") is not None
    assert await sbx.which("definitely-not-a-binary-xyz") is None


async def test_snapshot_skips_dirs(sbx):
    await sbx.write("keep.txt", b"k")
    await sbx.write(".git/HEAD", b"ref")
    await sbx.write("node_modules/x/index.js", b"x")
    snap = await sbx.snapshot()
    assert list(snap) == ["keep.txt"]
