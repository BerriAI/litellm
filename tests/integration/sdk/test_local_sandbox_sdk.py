import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import psutil
import pytest
import pytest_asyncio

from integration._support.client import eventually
from integration._support.process import kill_stray_process
from litellm import sandbox
from litellm.harness.sandbox import LocalSandbox

WRITE_AT_PWD: Final = "import os;open(os.path.join(os.environ['PWD'],'hello.txt'),'w').write('hi')"
SLEEPER_SPAWNED_ONCE_STDIN_CLOSES: Final = (
    "import subprocess,sys;sys.stdin.read();"
    "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
    "open('grandchild.pid','w').write(str(p.pid));p.wait()"
)


@pytest_asyncio.fixture(loop_scope="function")
async def box(tmp_path: Path) -> AsyncIterator[LocalSandbox]:
    local: Final = sandbox.local(tmp_path)
    yield local
    await local.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_child_trusting_pwd_writes_inside_its_cwd(
    box: LocalSandbox,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    elsewhere: Final = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.setenv("PWD", str(elsewhere))
    os.mkdir(os.path.join(box.workdir, "sub"))

    top: Final = await box.run([sys.executable, "-I", "-c", WRITE_AT_PWD])
    nested: Final = await box.run([sys.executable, "-I", "-c", WRITE_AT_PWD], cwd="sub", env={"PWD": str(elsewhere)})

    assert (top.exit_code, nested.exit_code) == (0, 0)
    assert os.listdir(elsewhere) == []
    assert await box.read("hello.txt") == b"hi"
    assert await box.read("sub/hello.txt") == b"hi"


def running(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.asyncio(loop_scope="function")
async def test_cancelled_run_stops_the_whole_command(box: LocalSandbox) -> None:
    run: Final = asyncio.ensure_future(box.run([sys.executable, "-I", "-c", SLEEPER_SPAWNED_ONCE_STDIN_CLOSES]))
    pid_file: Final = Path(box.workdir) / "grandchild.pid"
    grandchild: Final = int(
        await asyncio.to_thread(eventually, lambda: pid_file.read_text() if pid_file.exists() else "", bool)
    )
    try:
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert not eventually(lambda: running(grandchild), lambda alive: not alive, return_last_on_timeout=True)
    finally:
        kill_stray_process(grandchild)
