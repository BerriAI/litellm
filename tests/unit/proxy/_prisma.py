"""Prisma test doubles with an empty gateway configuration table."""

from typing import Final
from unittest.mock import AsyncMock, MagicMock


def prisma_with_empty_config() -> MagicMock:
    database: Final = MagicMock()
    database.litellm_config.find_unique = AsyncMock(return_value=None)
    return MagicMock(db=database, writer_db=database)
