from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litellm.repositories.config_repository import ConfigRepository


@pytest.mark.asyncio
async def test_batch_reads_decode_legacy_json_and_current_objects_from_writer():
    find_many = AsyncMock(
        return_value=(
            SimpleNamespace(param_name="first", param_value='{"status":"resolved"}'),
            SimpleNamespace(param_name="second", param_value={"status": "dismissed"}),
        )
    )
    prisma = SimpleNamespace(writer_db=SimpleNamespace(litellm_config=SimpleNamespace(find_many=find_many)))
    repository = ConfigRepository(prisma, use_writer=True)
    rows = await repository.get_params(("first", "second", "absent"))
    assert {row.param_name: row.param_value for row in rows} == {
        "first": {"status": "resolved"},
        "second": {"status": "dismissed"},
    }
    find_many.assert_awaited_once_with(where={"param_name": {"in": ("first", "second", "absent")}})
    assert await repository.get_params(()) == ()
    assert find_many.await_count == 1
