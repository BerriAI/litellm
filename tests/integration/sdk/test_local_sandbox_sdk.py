import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import pytest
import pytest_asyncio

from litellm import sandbox
from litellm.harness.sandbox import LocalSandbox

WRITE_AT_PWD: Final = "import os;open(os.path.join(os.environ['PWD'],'hello.txt'),'w').write('hi')"


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
