import faulthandler
import os
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Final, TextIO

import pytest
import pytest_asyncio

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

CALLBACK_LISTS: Final = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "input_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "_async_input_callback",
)


@pytest.fixture(scope="session", autouse=True)
def sdk_hang_stacks() -> Iterator[None]:
    destination: Final = Path(os.environ["INTEGRATION_RESULTS_DIR"]) / "sdk-thread-stacks.log"
    phases: Final = destination.with_name("sdk-active-test.log")
    finished: Final = threading.Event()
    with destination.open("w", buffering=1) as output:
        faulthandler.dump_traceback_later(30, repeat=True, file=output)
        watchdog: Final = threading.Thread(target=_stop_stalled_stream_case, args=(phases, output, finished), daemon=True)
        watchdog.start()
        try:
            yield
        finally:
            finished.set()
            watchdog.join(timeout=10)
            faulthandler.cancel_dump_traceback_later()


def _stop_stalled_stream_case(phases: Path, output: TextIO, finished: threading.Event) -> None:
    while not finished.wait(5):
        lines: Final = phases.read_text().splitlines() if phases.exists() else []
        fields: Final = lines[-1].split(" ", 2) if lines else []
        if (
            len(fields) == 3
            and fields[1] == "start"
            and "test_responses_bridge_stream_errors_sdk.py" in fields[2]
            and time.monotonic() - float(fields[0]) > 60
        ):
            output.write("Diagnostic watchdog stopped a stalled Responses streaming case\n")
            faulthandler.dump_traceback(file=output, all_threads=True)
            output.flush()
            os._exit(124)


def _record_sdk_phase(nodeid: str, phase: str) -> None:
    destination: Final = Path(os.environ["INTEGRATION_RESULTS_DIR"]) / "sdk-active-test.log"
    with destination.open("a", buffering=1) as output:
        output.write(f"{time.monotonic():.6f} {phase} {nodeid}\n")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Iterator[object]:
    _record_sdk_phase(item.nodeid, "start")
    try:
        return (yield)
    finally:
        _record_sdk_phase(item.nodeid, "finished")


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def isolate_litellm_callback_lists(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    for name in CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, list(getattr(litellm, name)))
    try:
        yield
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await GLOBAL_LOGGING_WORKER.stop()
