import json
from collections.abc import AsyncIterator, Sequence
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks import PROXY_HOOKS
from litellm.proxy.hooks.message_thread_ownership import MessageThreadOwnership
from litellm.proxy.utils import InternalUsageCache

OWNER: Final = UserAPIKeyAuth(token="hashed-owner-key")
INTRUDER: Final = UserAPIKeyAuth(token="hashed-intruder-key")
CREATE: Final = {"type": "create"}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def continue_from(message_id: str) -> dict[str, object]:
    return {"thread": {"type": "continue", "previous_message_id": message_id}, "messages": []}


def anthropic_message(message_id: str) -> dict[str, object]:
    return {"id": message_id, "type": "message", "role": "assistant", "content": []}


def sse_frame(event: dict[str, object]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def message_start(message_id: str) -> dict[str, object]:
    return {"type": "message_start", "message": anthropic_message(message_id)}


def hook() -> MessageThreadOwnership:
    return MessageThreadOwnership(internal_usage_cache=InternalUsageCache(DualCache()))


async def chunks(items: Sequence[object]) -> AsyncIterator[object]:
    for item in items:
        yield item


async def served(
    ownership: MessageThreadOwnership, items: Sequence[object], request_data: dict[str, object]
) -> list[object]:
    return [
        chunk
        async for chunk in ownership.async_post_call_streaming_iterator_hook(
            user_api_key_dict=OWNER, response=chunks(items), request_data=request_data
        )
    ]


async def rejection(ownership: MessageThreadOwnership, key: UserAPIKeyAuth, message_id: str) -> HTTPException | None:
    try:
        await ownership.async_pre_call_hook(
            user_api_key_dict=key, cache=DualCache(), data=continue_from(message_id), call_type="anthropic_messages"
        )
    except HTTPException as rejected:
        return rejected
    return None


async def allows(ownership: MessageThreadOwnership, key: UserAPIKeyAuth, message_id: str) -> bool:
    rejected: Final = await rejection(ownership, key, message_id)
    if rejected is None:
        return True
    assert rejected.status_code == 404
    error: Final = _JSON_OBJECT.validate_json(rejected.detail)["error"]
    assert isinstance(error, dict)
    assert error["type"] == "not_found_error"
    assert error["details"] == {"error_code": "thread_not_found"}
    return False


def test_the_hook_loads_with_the_proxy_hooks() -> None:
    assert PROXY_HOOKS["message_thread_ownership"] is MessageThreadOwnership


@pytest.mark.asyncio
async def test_only_the_key_that_created_a_thread_can_continue_it() -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data={"thread": CREATE}, user_api_key_dict=OWNER, response=anthropic_message("msg_created")
    )

    assert await allows(ownership, OWNER, "msg_created")
    assert not await allows(ownership, INTRUDER, "msg_created")


@pytest.mark.asyncio
async def test_without_a_key_only_the_signed_in_identity_that_created_a_thread_can_continue_it() -> None:
    ownership: Final = hook()
    alice: Final = UserAPIKeyAuth(user_id="alice", team_id="team-1")
    bob: Final = UserAPIKeyAuth(user_id="bob", team_id="team-1")
    await ownership.async_post_call_success_hook(
        data={"thread": CREATE}, user_api_key_dict=alice, response=anthropic_message("msg_signed_in")
    )

    assert await allows(ownership, alice, "msg_signed_in")
    assert not await allows(ownership, bob, "msg_signed_in")


@pytest.mark.asyncio
async def test_a_continue_naming_a_message_this_proxy_never_served_in_a_thread_is_thread_not_found() -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data={}, user_api_key_dict=OWNER, response=anthropic_message("msg_unthreaded")
    )

    assert not await allows(ownership, OWNER, "msg_unthreaded")
    assert not await allows(ownership, OWNER, "msg_unknown")


@pytest.mark.asyncio
async def test_each_continued_turn_hands_its_own_message_to_the_same_key() -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data={"thread": CREATE}, user_api_key_dict=OWNER, response=anthropic_message("msg_first")
    )
    await ownership.async_post_call_success_hook(
        data=continue_from("msg_first"), user_api_key_dict=OWNER, response=anthropic_message("msg_second")
    )

    assert await allows(ownership, OWNER, "msg_second")
    assert not await allows(ownership, INTRUDER, "msg_second")


@pytest.mark.asyncio
async def test_a_reply_served_again_to_another_key_stays_with_the_key_that_got_it_first() -> None:
    ownership: Final = hook()
    for key in (OWNER, INTRUDER):
        await ownership.async_post_call_success_hook(
            data={"thread": CREATE}, user_api_key_dict=key, response=anthropic_message("msg_cached")
        )

    assert await allows(ownership, OWNER, "msg_cached")
    assert not await allows(ownership, INTRUDER, "msg_cached")


@pytest.mark.asyncio
async def test_other_call_types_pass_untouched() -> None:
    data: Final = continue_from("msg_unknown")

    assert (
        await hook().async_pre_call_hook(
            user_api_key_dict=INTRUDER, cache=DualCache(), data=data, call_type="acompletion"
        )
        is None
    )


@pytest.mark.parametrize(
    "stream",
    (
        pytest.param(lambda frame: (frame,), id="whole_frame"),
        pytest.param(lambda frame: (frame[:25], frame[25:60], frame[60:]), id="torn_frame"),
        pytest.param(lambda frame: (frame.decode()[:30], frame.decode()[30:]), id="torn_text_frame"),
        pytest.param(
            lambda frame: (sse_frame({"type": "ping"}) + frame[:30], frame[30:]), id="after_a_ping_in_one_chunk"
        ),
    ),
)
@pytest.mark.asyncio
async def test_a_streamed_thread_turn_hands_its_message_to_its_key_and_streams_unchanged(stream) -> None:
    ownership: Final = hook()
    items: Final = (*stream(sse_frame(message_start("msg_streamed"))), sse_frame({"type": "message_stop"}))

    assert await served(ownership, items, {"thread": CREATE}) == list(items)
    assert await allows(ownership, OWNER, "msg_streamed")
    assert not await allows(ownership, INTRUDER, "msg_streamed")


@pytest.mark.asyncio
async def test_a_streamed_thread_turn_of_event_dicts_hands_its_message_to_its_key() -> None:
    ownership: Final = hook()
    items: Final = (message_start("msg_dict_stream"), {"type": "message_stop"})

    assert await served(ownership, items, {"thread": CREATE}) == list(items)
    assert await allows(ownership, OWNER, "msg_dict_stream")


@pytest.mark.asyncio
async def test_a_stream_without_a_thread_hands_out_nothing() -> None:
    ownership: Final = hook()
    items: Final = (sse_frame(message_start("msg_plain_stream")), sse_frame({"type": "message_stop"}))

    assert await served(ownership, items, {}) == list(items)
    assert not await allows(ownership, OWNER, "msg_plain_stream")


@pytest.mark.asyncio
async def test_a_message_start_after_another_event_is_not_read_as_the_threads_message() -> None:
    ownership: Final = hook()
    items: Final = (
        sse_frame({"type": "content_block_start", "index": 0}),
        sse_frame(message_start("msg_late")),
    )

    assert await served(ownership, items, {"thread": CREATE}) == list(items)
    assert not await allows(ownership, OWNER, "msg_late")
