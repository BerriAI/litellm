import importlib.util
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import LiteLLM_EndUserTable, UserAPIKeyAuth


def _load_blocked_user_list_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "enterprise"
        / "enterprise_hooks"
        / "blocked_user_list.py"
    )
    spec = importlib.util.spec_from_file_location("blocked_user_list", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_blocked_user_check_uses_global_prisma_when_init_had_none(monkeypatch):
    blocked_user_list = _load_blocked_user_list_module()
    _ENTERPRISE_BlockedUserList = blocked_user_list._ENTERPRISE_BlockedUserList

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_unique = AsyncMock(
        return_value=LiteLLM_EndUserTable(user_id="blocked-user", blocked=True)
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    hook = _ENTERPRISE_BlockedUserList(prisma_client=None)
    cache = DualCache()

    with pytest.raises(HTTPException) as exc_info:
        await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=cache,
            data={"user": "blocked-user"},
            call_type="completion",
        )

    assert "blocked" in str(exc_info.value).lower()
    mock_prisma.db.litellm_endusertable.find_unique.assert_awaited_once()
