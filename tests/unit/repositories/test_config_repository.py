from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.repositories.config_repository import ConfigRepository


@pytest.mark.asyncio
async def test_batch_reads_decode_legacy_json_and_current_objects_from_writer() -> None:
    find_many: Final = AsyncMock(
        return_value=(
            SimpleNamespace(param_name="first", param_value='{"status":"resolved"}'),
            SimpleNamespace(param_name="second", param_value={"status": "dismissed"}),
        )
    )
    prisma: Final = SimpleNamespace(writer_db=SimpleNamespace(litellm_config=SimpleNamespace(find_many=find_many)))
    repository: Final = ConfigRepository(prisma, use_writer=True)
    rows: Final = await repository.get_params(("first", "second", "absent"))
    assert {row.param_name: row.param_value for row in rows} == {
        "first": {"status": "resolved"},
        "second": {"status": "dismissed"},
    }
    find_many.assert_awaited_once_with(where={"param_name": {"in": ("first", "second", "absent")}})
    assert await repository.get_params(()) == ()
    assert find_many.await_count == 1
