import asyncio
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm import _db


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_lifespan_closes_database_when_owner_fails(failure):
    prisma: Final = MagicMock()
    prisma.disconnect = AsyncMock()
    database: Final = _db.from_prisma(prisma)

    async def fail():
        async with _db.lifespan(database) as current:
            assert current is database
            prisma.disconnect.assert_not_awaited()
            raise failure("owner failed")

    with pytest.raises(failure):
        await fail()
    prisma.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_without_database_preserves_owner_error():
    async def fail():
        async with _db.lifespan(None) as database:
            assert database is None
            raise RuntimeError("owner failed")

    with pytest.raises(RuntimeError, match="owner failed"):
        await fail()
