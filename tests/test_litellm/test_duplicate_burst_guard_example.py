from collections.abc import Mapping
from typing import cast

import pytest
from fastapi import HTTPException

from cookbook.litellm_proxy_server.duplicate_burst_guard import DuplicateBurstGuard
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

USER_API_KEY = UserAPIKeyAuth(user_id="user-1")
OTHER_USER_API_KEY = UserAPIKeyAuth(user_id="user-2")
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


@pytest.mark.asyncio
async def test_duplicate_burst_guard_avoids_delimiter_collisions() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a = _chat_request("c")
    request_a["messages"] = [
        {"role": "system", "content": "a|b"},
        {"role": "user", "content": "c"},
    ]
    request_b = _chat_request("b|c")
    request_b["messages"] = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b|c"},
    ]

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, CALL_TYPE)
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, CALL_TYPE
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_multimodal_content() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a: dict[str, object] = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://a.test/1.png"}},
                ],
            }
        ],
    }
    request_b: dict[str, object] = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://a.test/2.png"}},
                ],
            }
        ],
    }

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, CALL_TYPE)
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, CALL_TYPE
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_prefers_request_user_identity() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a = _chat_request("Summarize invoice A")
    request_a["user"] = "end-user-a"
    request_b = _chat_request("Summarize invoice A")
    request_b["user"] = "end-user-b"

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, CALL_TYPE)
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, CALL_TYPE
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_full_chat_context() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a = _chat_request("What should I do next?")
    request_a["messages"] = [
        {"role": "system", "content": "Use the shared operating policy."},
        {"role": "user", "content": "Invoice A is overdue."},
        {"role": "assistant", "content": "Ask for payment status."},
        {"role": "user", "content": "What should I do next?"},
    ]
    request_b = _chat_request("What should I do next?")
    request_b["messages"] = [
        {"role": "system", "content": "Use the shared operating policy."},
        {"role": "user", "content": "Invoice B has a credit memo."},
        {"role": "assistant", "content": "Check whether it was applied."},
        {"role": "user", "content": "What should I do next?"},
    ]

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, CALL_TYPE)
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, CALL_TYPE
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_output_options() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a = _chat_request("Summarize invoice A")
    request_a["response_format"] = {"type": "json_object"}
    request_b = _chat_request("Summarize invoice A")
    request_b["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "summarize_invoice",
                "parameters": {"type": "object"},
            },
        }
    ]

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, CALL_TYPE)
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, CALL_TYPE
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_non_chat_input() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a: dict[str, object] = {
        "model": "text-embedding-3-small",
        "input": "invoice A",
    }
    request_b: dict[str, object] = {
        "model": "text-embedding-3-small",
        "input": "invoice B",
    }

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, "aembedding")
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, "aembedding"
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_ignores_ephemeral_call_ids() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a = _chat_request("Summarize invoice A")
    request_a["litellm_call_id"] = "call-a"
    request_b = _chat_request("Summarize invoice A")
    request_b["litellm_call_id"] = "call-b"

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, CALL_TYPE)

    with pytest.raises(HTTPException) as exc_info:
        await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_b, CALL_TYPE)

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_authenticated_key_identity() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request = _chat_request("Summarize invoice A")
    request["metadata"] = {"session_id": "shared-session"}

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request, CALL_TYPE)
    accepted = await guard.async_pre_call_hook(
        OTHER_USER_API_KEY, CACHE, request, CALL_TYPE
    )

    assert accepted == request


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_rerank_fields() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a: dict[str, object] = {
        "model": "rerank-english-v3.0",
        "query": "invoice A",
        "documents": ["invoice A is overdue"],
    }
    request_b: dict[str, object] = {
        "model": "rerank-english-v3.0",
        "query": "invoice B",
        "documents": ["invoice B has a credit memo"],
    }

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, "rerank")
    accepted = await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_b, "rerank")

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_responses_context() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a: dict[str, object] = {
        "model": "gpt-4o-mini",
        "input": "continue",
        "previous_response_id": "resp-a",
        "instructions": "Use policy A",
    }
    request_b: dict[str, object] = {
        "model": "gpt-4o-mini",
        "input": "continue",
        "previous_response_id": "resp-b",
        "instructions": "Use policy B",
    }

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, "aresponses")
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, "aresponses"
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_top_level_system_prompt() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a: dict[str, object] = {
        "model": "claude-sonnet-4-5",
        "system": "Use policy A",
        "messages": [{"role": "user", "content": "Summarize invoice A"}],
    }
    request_b: dict[str, object] = {
        "model": "claude-sonnet-4-5",
        "system": "Use policy B",
        "messages": [{"role": "user", "content": "Summarize invoice A"}],
    }

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, CALL_TYPE)
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, CALL_TYPE
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_includes_search_fields() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request_a: dict[str, object] = {
        "model": "search-model",
        "query": "invoice A",
        "vector_store_id": "store-a",
    }
    request_b: dict[str, object] = {
        "model": "search-model",
        "query": "invoice B",
        "vector_store_id": "store-b",
    }

    await guard.async_pre_call_hook(USER_API_KEY, CACHE, request_a, "asearch")
    accepted = await guard.async_pre_call_hook(
        USER_API_KEY, CACHE, request_b, "asearch"
    )

    assert accepted == request_b


@pytest.mark.asyncio
async def test_duplicate_burst_guard_skips_image_edits() -> None:
    guard = DuplicateBurstGuard(max_calls=1, window_seconds=60)
    request: dict[str, object] = {
        "model": "gpt-image-1",
        "prompt": "Add the logo",
        "image": object(),
        "mask": object(),
    }

    assert (
        await guard.async_pre_call_hook(USER_API_KEY, CACHE, request, "aimage_edit")
        == request
    )
    assert (
        await guard.async_pre_call_hook(USER_API_KEY, CACHE, request, "aimage_edit")
        == request
    )
