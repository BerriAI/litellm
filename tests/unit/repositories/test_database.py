from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.repositories.database import Database


@pytest.mark.asyncio
async def test_database_disconnects_even_when_stopping_background_tasks_fails() -> None:
    prisma: Final = MagicMock()
    prisma.stop_background_tasks = AsyncMock(side_effect=RuntimeError("watchdog failed"))
    prisma.disconnect = AsyncMock()

    with pytest.raises(RuntimeError, match="watchdog failed"):
        await Database(prisma=prisma).aclose()

    prisma.disconnect.assert_awaited_once()
