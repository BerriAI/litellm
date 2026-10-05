import asyncio
import json
import os
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.lens.python_tool import execute_python


@pytest.mark.asyncio
async def test_python_aggregates_nested_evidence_without_corrupting_reply_with_printed_json() -> None:
    evidence: Final = json.dumps(
        {"parts": [{"parent_span_id": "root", "kind": "tool"}, {"parent_span_id": "child", "kind": "tool"}]}
    )
    result: Final = json.loads(
        await execute_python(
            "import collections, json, sys\n"
            'print(json.dumps(dict(collections.Counter(p["parent_span_id"] for p in data["parts"]))))\n'
            'print("tracé", file=sys.stderr)',
            evidence,
        )
    )
    assert result["stdout"] == '{"root": 1, "child": 1}\n'
    assert result["stderr"] == "tracé\n"
    assert result["exit_code"] == 0
    assert result["elapsed_seconds"] > 0


@pytest.mark.asyncio
async def test_python_preserves_large_stdout_and_stderr() -> None:
    result: Final = json.loads(
        await execute_python(
            'import sys\nprint(data, end="")\nprint(data, end="", file=sys.stderr)', json.dumps("a" * 100000)
        )
    )
    assert result["stdout"] == "a" * 100000
    assert result["stderr"] == "a" * 100000
    assert result["exit_code"] == 0


@pytest.mark.parametrize(
    ("code", "exit_code", "error"),
    (("1 / 0", 1, "ZeroDivisionError"), ("if :", 1, "SyntaxError"), ("raise SystemExit(7)", 7, "")),
)
@pytest.mark.asyncio
async def test_python_reports_script_failures_as_tool_results(code: str, exit_code: int, error: str) -> None:
    result: Final = json.loads(await execute_python(code, "{}"))
    assert result["stdout"] == ""
    assert result["exit_code"] == exit_code
    assert error in result["stderr"]


@pytest.mark.asyncio
async def test_python_does_not_inherit_worker_secrets_or_python_startup_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LENS_TEST_SECRET", "private-worker-value")
    monkeypatch.setenv("PYTHONPATH", "private-worker-pythonpath")
    result: Final = json.loads(
        await execute_python(
            'import json, os\nprint(json.dumps({"secret": os.getenv("LENS_TEST_SECRET"), '
            '"pythonpath": os.getenv("PYTHONPATH"), "path": os.getenv("PATH")}))',
            "{}",
        )
    )
    environment: Final = json.loads(result["stdout"])
    assert environment == {"secret": None, "pythonpath": None, "path": os.defpath}
    assert os.environ["LENS_TEST_SECRET"] == "private-worker-value"
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_python_uses_fresh_variables_and_removes_temporary_working_files() -> None:
    first: Final = json.loads(
        await execute_python(
            'import pathlib\nsecret = 1\npathlib.Path("scratch.txt").write_text("scratch")\nprint(pathlib.Path.cwd())',
            "{}",
        )
    )
    second: Final = json.loads(await execute_python('print("secret" in globals())', "{}"))
    assert first["exit_code"] == 0
    assert not Path(first["stdout"].strip()).exists()
    assert second["stdout"] == "False\n"
    assert second["exit_code"] == 0


async def _wait_for_file(path: Path) -> None:
    while not path.exists():
        await asyncio.sleep(0.01)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
@pytest.mark.asyncio
async def test_python_cancellation_kills_descendants_and_reaps_its_process(tmp_path: Path) -> None:
    import fcntl

    ready: Final = tmp_path / "ready.json"
    lock: Final = tmp_path / "descendant.lock"
    child_code: Final = (
        "import fcntl, json, os, pathlib, time\n"
        f"lock = open({str(lock)!r}, 'w')\n"
        "fcntl.flock(lock, fcntl.LOCK_EX)\n"
        f"pathlib.Path({str(ready)!r}).write_text(json.dumps({{'parent': os.getppid(), 'child': os.getpid()}}))\n"
        "time.sleep(3600)"
    )
    task: Final = asyncio.create_task(
        execute_python(f"import subprocess, sys\nsubprocess.run([sys.executable, '-c', {child_code!r}])", "{}")
    )
    try:
        await asyncio.wait_for(_wait_for_file(ready), timeout=5)
        parent_pid: Final = json.loads(ready.read_text())["parent"]
        with lock.open() as locked:
            with pytest.raises(BlockingIOError):
                fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        with lock.open() as unlocked:
            fcntl.flock(unlocked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ProcessLookupError):
            os.kill(parent_pid, 0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
