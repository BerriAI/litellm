import asyncio
import logging
from typing import Final

import pytest

from litellm.litellm_core_utils.socket_logging import SocketLogging, collect_socket_logging, track_socket_logging


@pytest.mark.asyncio
async def test_a_socket_waits_for_the_logging_its_turns_started() -> None:
    turn_may_finish: Final = asyncio.Event()
    with collect_socket_logging() as socket_logging:
        turn_logging: Final = asyncio.create_task(track_socket_logging(turn_may_finish.wait()))
        settled: Final = asyncio.create_task(socket_logging.finished())
        await asyncio.sleep(0.01)
        waited_for_the_turn: Final = not settled.done()
        turn_may_finish.set()
        await asyncio.wait_for(settled, timeout=1)

    assert (waited_for_the_turn, turn_logging.done()) == (True, True)


@pytest.mark.asyncio
async def test_a_socket_stops_waiting_on_logging_that_outlasts_its_wait_and_leaves_it_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    socket_logging: Final = SocketLogging(wait_seconds=0.01)
    turn_logging: Final = asyncio.create_task(socket_logging.tracked(asyncio.Event().wait()))

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        await asyncio.wait_for(socket_logging.finished(), timeout=1)

    assert not turn_logging.done()
    assert "1 WebSocket turn logging tasks were still running" in caplog.text
    turn_logging.cancel()


@pytest.mark.asyncio
async def test_logging_started_outside_the_socket_is_not_waited_for() -> None:
    elsewhere: Final = asyncio.create_task(track_socket_logging(asyncio.Event().wait()))
    with collect_socket_logging() as socket_logging:
        await asyncio.wait_for(socket_logging.finished(), timeout=1)

    assert not elsewhere.done()
    elsewhere.cancel()
