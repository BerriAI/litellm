import faulthandler
import os
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Final

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
    with destination.open("w", buffering=1) as output:
        faulthandler.dump_traceback_later(30, repeat=True, file=output)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()


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
