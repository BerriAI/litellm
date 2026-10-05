import asyncio
import json
import os
import signal
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Final

_BOOTSTRAP: Final = """
import json
import sys

request = json.load(sys.stdin)
namespace = {"__name__": "__main__", "data": json.loads(request["data"])}
exec(compile(request["code"], "<lens-python>", "exec"), namespace)
"""


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass


async def execute_python(code: str, data: str) -> str:
    """Execute local benchmark code with host filesystem permissions, not sandbox isolation."""
    started: Final = monotonic()
    payload: Final = json.dumps({"code": code, "data": data}).encode("utf-8")
    with TemporaryDirectory(prefix="lens-python-") as directory:
        try:
            process: Final = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-X",
                "utf8",
                "-u",
                "-c",
                _BOOTSTRAP,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=directory,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                start_new_session=True,
            )
        except OSError as error:
            return json.dumps(
                {"stdout": "", "stderr": str(error), "exit_code": None, "elapsed_seconds": monotonic() - started}
            )
        try:
            stdout, stderr = await process.communicate(payload)
        except BaseException:
            _kill_process_group(process)
            await process.wait()
            raise
        _kill_process_group(process)
        return json.dumps(
            {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": process.returncode,
                "elapsed_seconds": monotonic() - started,
            },
            ensure_ascii=False,
        )
