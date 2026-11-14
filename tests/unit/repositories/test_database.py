from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.repositories.database import Database


@pytest.mark.asyncio
async def test_closing_the_database_closes_prisma() -> None:
    prisma: Final = MagicMock()
    prisma.aclose = AsyncMock()
    database: Final = await Database.open(prisma)

    prisma.aclose.assert_not_awaited()
    await database.aclose()

    prisma.aclose.assert_awaited_once()
