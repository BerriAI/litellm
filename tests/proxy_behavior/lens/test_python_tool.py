import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.lens.python_tool import execute_python


@pytest.mark.skipif(sys.platform == "linux", reason="This check covers unsupported source-development hosts")
@pytest.mark.asyncio
async def test_python_fails_closed_outside_native_worker() -> None:
    result: Final = json.loads(await execute_python('print("must not execute")', "{}"))
    assert result["stdout"] == ""
    assert result["exit_code"] is None
    assert result["output_complete"] is False
    assert "native Linux Lens worker" in result["error"]


def test_python_boundaries_in_native_worker_image() -> None:
    image: Final = os.environ.get("LENS_TEST_WORKER_IMAGE")
    if not image:
        pytest.skip("Set LENS_TEST_WORKER_IMAGE to run confinement checks against the native worker image")
    script: Final = Path(__file__).with_name("worker_python_smoke.py").read_text()
    result: Final = subprocess.run(
        (
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--network",
            "none",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=1g",
            "--entrypoint",
            "python",
            "-i",
            image,
            "-",
        ),
        input=script,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Python confinement smoke passed" in result.stdout
