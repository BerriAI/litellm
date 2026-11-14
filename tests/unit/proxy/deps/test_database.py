from typing import Final

import pytest
from starlette.requests import Request

from litellm.proxy.deps.database import DatabaseStateMissingError, get_database


@pytest.mark.asyncio
async def test_missing_lifespan_is_not_treated_as_a_disabled_database():
    request: Final = Request({"type": "http", "state": {}})
    with pytest.raises(DatabaseStateMissingError):
        await get_database(request)
