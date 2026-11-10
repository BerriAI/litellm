import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import psutil
import pytest
from fastapi import HTTPException

from litellm.proxy.discovery_endpoints.skill_scanner_adapter import server
from tests.integration._support.process import _wait_until_gone, group_members, stop_processes


@pytest.mark.parametrize("mode", ("deadline", "cap", "complete"))
async def test_adapter_bounds_worker_lifetime_and_reaps_its_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str,
) -> None:
    script: Final = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(child.pid, flush=True)\n"
        + ("pass\n" if mode == "complete" else "time.sleep(30)\n")
    )
    child: Final = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
    )
    try:
        assert child.stdout is not None
        descendant: Final = psutil.Process(int(await asyncio.wait_for(child.stdout.readline(), 5)))
        if mode == "complete":
            await asyncio.wait_for(child.wait(), 5)
        output: Final = tmp_path / "report.json"
        output.write_text('{"complete": true}')

        async def launch(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            return child

        monkeypatch.setattr(server.asyncio, "create_subprocess_exec", launch)
        monkeypatch.setattr(server, "SCAN_TIMEOUT", 0.05 if mode == "cap" else 30)
        deadline: Final = (
            None if mode == "cap"
            else datetime.now(timezone.utc) + timedelta(seconds=0.05 if mode == "deadline" else 30)
        )
        result: Final = server.run_worker("unused.zip", output, False, deadline)
        if mode == "complete":
            assert await asyncio.wait_for(result, 5) == {"complete": True}
        else:
            with pytest.raises(HTTPException) as raised:
                await asyncio.wait_for(result, 5)
            assert raised.value.status_code == 504
        assert child.returncode is not None
        _wait_until_gone(descendant, 3)
    finally:
        await asyncio.to_thread(stop_processes, group_members(child.pid), 0.1)
        await child.wait()
