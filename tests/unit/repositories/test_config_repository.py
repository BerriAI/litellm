from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.repositories.config_repository import ConfigRepository


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "expected"),
    (
        ('{"status":"resolved"}', {"status": "resolved"}),
        ({"status": "dismissed"}, {"status": "dismissed"}),
        (None, None),
    ),
)
async def test_reads_decode_legacy_json_and_current_objects_from_writer(stored: object, expected: object) -> None:
    record: Final = None if stored is None else SimpleNamespace(param_name="status", param_value=stored)
    find_unique: Final = AsyncMock(return_value=record)
    prisma: Final = SimpleNamespace(writer_db=SimpleNamespace(litellm_config=SimpleNamespace(find_unique=find_unique)))
    repository: Final = ConfigRepository(prisma, use_writer=True)
    row: Final = await repository.get_param("status")
    assert (row.param_value if row is not None else None) == expected
    find_unique.assert_awaited_once_with(where={"param_name": "status"})
