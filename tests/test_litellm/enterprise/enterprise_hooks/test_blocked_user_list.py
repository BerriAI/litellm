import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth


@pytest.fixture
def hook():
    from enterprise.enterprise_hooks.blocked_user_list import (
        _ENTERPRISE_BlockedUserList,
    )

    return _ENTERPRISE_BlockedUserList(prisma_client=None)


async def _call(hook, user_id: str):
    return await hook.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data={"user": user_id},
        call_type="completion",
    )


@pytest.mark.asyncio
async def test_hook_reads_blocked_user_list_on_every_call(monkeypatch, hook):
    """Regression: the hook used to keep the list object it read at init, so /customer/unblock
    rebinding litellm.blocked_user_list left the hook rejecting an already unblocked customer.
    """
    monkeypatch.setattr(litellm, "blocked_user_list", ["blocked-1"])

    with pytest.raises(HTTPException) as exc_info:
        await _call(hook, "blocked-1")
    assert "blocked-1" in str(exc_info.value.detail)

    monkeypatch.setattr(litellm, "blocked_user_list", [])

    assert await _call(hook, "blocked-1") is None


@pytest.mark.asyncio
async def test_hook_keeps_filepath_backed_list(monkeypatch, tmp_path):
    from enterprise.enterprise_hooks.blocked_user_list import (
        _ENTERPRISE_BlockedUserList,
    )

    blocked_users_file = tmp_path / "blocked_users.txt"
    blocked_users_file.write_text("blocked-1")
    monkeypatch.setattr(litellm, "blocked_user_list", str(blocked_users_file))

    file_backed_hook = _ENTERPRISE_BlockedUserList(prisma_client=None)

    monkeypatch.setattr(litellm, "blocked_user_list", [])

    with pytest.raises(HTTPException):
        await _call(file_backed_hook, "blocked-1")
