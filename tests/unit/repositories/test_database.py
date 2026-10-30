import logging
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.repositories.database import Database


@pytest.mark.asyncio
async def test_database_disconnects_and_logs_when_stopping_background_tasks_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prisma: Final = MagicMock()
    prisma.stop_background_tasks = AsyncMock(side_effect=RuntimeError("watchdog failed"))
    prisma.disconnect = AsyncMock()

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
        await Database(prisma=prisma).aclose()

    prisma.disconnect.assert_awaited_once()
    assert "watchdog failed" in caplog.text


@pytest.mark.asyncio
async def test_database_aclose_returns_and_logs_when_disconnect_fails(caplog: pytest.LogCaptureFixture) -> None:
    prisma: Final = MagicMock()
    prisma.stop_background_tasks = AsyncMock()
    prisma.disconnect = AsyncMock(side_effect=RuntimeError("engine hung"))

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
        await Database(prisma=prisma).aclose()

    prisma.stop_background_tasks.assert_awaited_once()
    prisma.disconnect.assert_awaited_once()
    assert "engine hung" in caplog.text
