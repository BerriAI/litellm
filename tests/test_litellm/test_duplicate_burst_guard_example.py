from collections.abc import Mapping
from typing import cast

import pytest
from fastapi import HTTPException

from cookbook.litellm_proxy_server.duplicate_burst_guard import DuplicateBurstGuard
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

USER_API_KEY = UserAPIKeyAuth(user_id="user-1")
CACHE = DualCache()
CALL_TYPE: CallTypesLiteral = "completion"


def _chat_request(user_prompt: str) -> dict[str, object]:
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Use the shared operating policy."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }


@pytest.mark.asyncio
async def test_duplicate_burst_guard_blocks_repeated_prompt() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request = _chat_request("Summarize invoice A")

    assert (
        await guard.async_pre_call_hook(USER_API_KEY, CACHE, request, CALL_TYPE)
        == request
    )

    with pytest.raises(HTTPException) as exc_info:
        await guard.async_pre_call_hook(USER_API_KEY, CACHE, request, CALL_TYPE)

    assert exc_info.value.status_code == 429
    detail = exc_info.value.detail
    assert isinstance(detail, Mapping)
    detail_data = cast(Mapping[str, object], detail)
    assert detail_data.get("error") == "Duplicate request burst detected"


@pytest.mark.asyncio
async def test_duplicate_burst_guard_uses_last_user_prompt() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_b = _chat_request("Summarize invoice B")

    await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, _chat_request("Summarize invoice A"), CALL_TYPE
    )
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, CALL_TYPE
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_uses_completion_prompt() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a: dict[str, object] = {
        "model": "gpt-4o-mini",
        "prompt": "Summarize invoice A",
    }
    request_b: dict[str, object] = {
        "model": "gpt-4o-mini",
        "prompt": "Summarize invoice B",
    }

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, CALL_TYPE)
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, CALL_TYPE
    )

    assert accepted == request_b
