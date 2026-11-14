import asyncio
import datetime
import json
from collections.abc import AsyncIterator, Sequence
from typing import Final

import httpx
import pytest
from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.message_thread_ownership import MessageThreadOwnership
from litellm.types.utils import CallTypesLiteral
from litellm.proxy.utils import InternalUsageCache, ProxyLogging

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


def sent_with(anthropic_beta: str, **data: object) -> dict[str, object]:
    logging_obj: Final = Logging(
        model="claude-opus-4-8",
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="call",
        function_id="function",
    )
    sent: Final = httpx.Request("POST", "https://api.anthropic.com/v1/messages", headers={"anthropic-beta": anthropic_beta})
    logging_obj.model_call_details["httpx_response"] = httpx.Response(200, request=sent)
    logging_obj.post_call(original_response="{}", additional_args={"complete_input_dict": {}})
    return {**data, "litellm_logging_obj": logging_obj}


def hook() -> MessageThreadOwnership:
    return MessageThreadOwnership(message_thread_owner_cache=InternalUsageCache(DualCache()))


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


async def rejection(
    ownership: MessageThreadOwnership, key: UserAPIKeyAuth, message_id: str, call_type: CallTypesLiteral
) -> HTTPException | None:
    try:
        await ownership.async_pre_call_hook(
            user_api_key_dict=key, cache=DualCache(), data=continue_from(message_id), call_type=call_type
        )
    except HTTPException as rejected:
        return rejected
    return None


async def allows(
    ownership: MessageThreadOwnership,
    key: UserAPIKeyAuth,
    message_id: str,
    call_type: CallTypesLiteral = "anthropic_messages",
) -> bool:
    rejected: Final = await rejection(ownership, key, message_id, call_type)
    if rejected is None:
        return True
    assert rejected.status_code == 404
    error: Final = _JSON_OBJECT.validate_json(rejected.detail)["error"]
    assert isinstance(error, dict)
    assert error["type"] == "not_found_error"
    assert error["details"] == {"error_code": "thread_not_found"}
    return False


@pytest.mark.asyncio
async def test_the_proxy_starts_with_a_hook_that_keeps_each_thread_with_its_key() -> None:
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging._add_proxy_hooks()
    await proxy_logging.post_call_success_hook(
        data={"thread": CREATE}, response=anthropic_message("msg_started"), user_api_key_dict=OWNER
    )

    with pytest.raises(HTTPException) as rejected:
        await proxy_logging.pre_call_hook(
            user_api_key_dict=INTRUDER, data=continue_from("msg_started"), call_type="anthropic_messages"
        )
    assert rejected.value.status_code == 404


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
async def test_without_a_key_one_managed_agent_cannot_continue_another_agents_thread() -> None:
    ownership: Final = hook()
    planner: Final = UserAPIKeyAuth(agent_id="agent-planner")
    coder: Final = UserAPIKeyAuth(agent_id="agent-coder")
    await ownership.async_post_call_success_hook(
        data={"thread": CREATE}, user_api_key_dict=planner, response=anthropic_message("msg_agent")
    )

    assert await allows(ownership, planner, "msg_agent")
    assert not await allows(ownership, coder, "msg_agent")


@pytest.mark.parametrize("call_type", ("anthropic_messages", "acompletion", "aresponses"))
@pytest.mark.asyncio
async def test_a_continue_naming_a_message_this_proxy_never_served_in_a_thread_is_thread_not_found(
    call_type: CallTypesLiteral,
) -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data={}, user_api_key_dict=OWNER, response=anthropic_message("msg_unthreaded")
    )

    assert not await allows(ownership, OWNER, "msg_unthreaded", call_type)
    assert not await allows(ownership, OWNER, "msg_unknown", call_type)
    assert not await allows(ownership, INTRUDER, "msg_unknown", call_type)


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
async def test_a_worker_that_writes_after_another_worker_claimed_the_reply_keeps_the_first_owner(
    router_redis: tuple[RedisCache, object],
) -> None:
    redis_cache, _ = router_redis
    late_write_started: Final = asyncio.Event()
    owner_stored: Final = asyncio.Event()

    class WritesAfterTheOwner(RedisCache):
        async def async_set_cache(
            self,
            key: str | None,
            value: object,
            **kwargs: object,  # kwargs-ok: RedisCache write signature
        ) -> None:
            late_write_started.set()
            await owner_stored.wait()
            await super().async_set_cache(key, value, **kwargs)

    first: Final = MessageThreadOwnership(
        message_thread_owner_cache=InternalUsageCache(DualCache(redis_cache=redis_cache))
    )
    late: Final = MessageThreadOwnership(
        message_thread_owner_cache=InternalUsageCache(
            DualCache(redis_cache=WritesAfterTheOwner(url="redis://router-unit.example:6379/0"))
        )
    )

    async def owner_claims() -> None:
        await late_write_started.wait()
        await first.async_post_call_success_hook(
            data={"thread": CREATE}, user_api_key_dict=OWNER, response=anthropic_message("msg_raced")
        )
        owner_stored.set()

    await asyncio.gather(
        late.async_post_call_success_hook(
            data={"thread": CREATE}, user_api_key_dict=INTRUDER, response=anthropic_message("msg_raced")
        ),
        owner_claims(),
    )

    for worker in (first, late):
        assert await allows(worker, OWNER, "msg_raced")
        assert not await allows(worker, INTRUDER, "msg_raced")


@pytest.mark.asyncio
async def test_every_redis_command_for_a_thread_binding_is_traced_under_the_message_thread_owners_family(
    router_redis: tuple[RedisCache, asyncio.Queue[tuple[tuple[object, ...], str | None, str | None]]],
) -> None:
    redis_cache, commands = router_redis
    creating_worker: Final = MessageThreadOwnership(
        message_thread_owner_cache=InternalUsageCache(DualCache(redis_cache=redis_cache))
    )
    continuing_worker: Final = MessageThreadOwnership(
        message_thread_owner_cache=InternalUsageCache(DualCache(redis_cache=redis_cache))
    )
    await creating_worker.async_post_call_success_hook(
        data={"thread": CREATE}, user_api_key_dict=OWNER, response=anthropic_message("msg_traced")
    )

    assert await allows(continuing_worker, OWNER, "msg_traced")
    sent: Final = tuple(commands.get_nowait() for _ in range(commands.qsize()))
    thread_commands: Final = tuple((str(args[0]), target) for args, target, _ in sent if args[0] != "CLIENT")
    assert {command for command, _ in thread_commands} == {"GET", "SET"}
    assert {target for _, target in thread_commands} == {"message_thread_owners"}


@pytest.mark.parametrize("call_type", ("acompletion", "aresponses"))
@pytest.mark.asyncio
async def test_a_thread_continue_on_another_route_stays_with_the_key_that_created_the_thread(
    call_type: CallTypesLiteral,
) -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data={"thread": CREATE}, user_api_key_dict=OWNER, response=anthropic_message("msg_cross_route")
    )

    assert await allows(ownership, OWNER, "msg_cross_route", call_type)
    assert not await allows(ownership, INTRUDER, "msg_cross_route", call_type)


@pytest.mark.parametrize("call_type", ("anthropic_messages", "acompletion", "aresponses"))
@pytest.mark.asyncio
async def test_a_continue_sent_in_extra_body_stays_with_the_key_that_created_the_thread(
    call_type: CallTypesLiteral,
) -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data={"thread": CREATE}, user_api_key_dict=OWNER, response=anthropic_message("msg_extra_body")
    )
    in_extra_body: Final = {"messages": [], "extra_body": {"thread": continue_from("msg_extra_body")["thread"]}}

    assert (
        await ownership.async_pre_call_hook(
            user_api_key_dict=OWNER, cache=DualCache(), data=in_extra_body, call_type=call_type
        )
        is None
    )
    with pytest.raises(HTTPException) as rejected:
        await ownership.async_pre_call_hook(
            user_api_key_dict=INTRUDER, cache=DualCache(), data=in_extra_body, call_type=call_type
        )
    assert rejected.value.status_code == 404


@pytest.mark.asyncio
async def test_an_owned_top_level_continue_does_not_let_another_keys_extra_body_continue_through() -> None:
    ownership: Final = hook()
    for key, message_id in ((OWNER, "msg_owner"), (INTRUDER, "msg_intruder")):
        await ownership.async_post_call_success_hook(
            data={"thread": CREATE}, user_api_key_dict=key, response=anthropic_message(message_id)
        )
    mixed: Final = {**continue_from("msg_intruder"), "extra_body": {"thread": continue_from("msg_owner")["thread"]}}

    with pytest.raises(HTTPException) as rejected:
        await ownership.async_pre_call_hook(
            user_api_key_dict=INTRUDER, cache=DualCache(), data=mixed, call_type="acompletion"
        )
    assert rejected.value.status_code == 404


@pytest.mark.parametrize(
    "create",
    (
        pytest.param(
            lambda ownership, data: ownership.async_post_call_success_hook(
                data=data, user_api_key_dict=OWNER, response=anthropic_message("msg_body_create")
            ),
            id="reply",
        ),
        pytest.param(
            lambda ownership, data: served(
                ownership, (sse_frame(message_start("msg_body_create")), sse_frame({"type": "message_stop"})), data
            ),
            id="stream",
        ),
    ),
)
@pytest.mark.asyncio
async def test_a_thread_created_in_extra_body_stays_with_the_key_that_created_it(create) -> None:
    ownership: Final = hook()
    await create(ownership, {"messages": [], "extra_body": {"thread": CREATE}})

    assert await allows(ownership, OWNER, "msg_body_create", "acompletion")
    assert not await allows(ownership, INTRUDER, "msg_body_create", "acompletion")


@pytest.mark.asyncio
async def test_a_pass_through_continue_of_a_message_the_proxy_never_bound_is_left_to_the_upstream() -> None:
    assert await allows(hook(), OWNER, "msg_unknown", "pass_through_endpoint")
    assert await allows(hook(), INTRUDER, "msg_unknown", "pass_through_endpoint")


@pytest.mark.asyncio
async def test_a_pass_through_continue_of_another_keys_thread_is_thread_not_found() -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data={"thread": CREATE}, user_api_key_dict=OWNER, response=anthropic_message("msg_bound")
    )

    assert await allows(ownership, OWNER, "msg_bound", "pass_through_endpoint")
    assert not await allows(ownership, INTRUDER, "msg_bound", "pass_through_endpoint")


@pytest.mark.asyncio
async def test_a_continue_resends_the_betas_its_create_turn_sent_next_to_its_own_extra_headers() -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data=sent_with("advisor-tool-2026-03-01,message-threads-2026-08-12", thread=CREATE),
        user_api_key_dict=OWNER,
        response=anthropic_message("msg_advisor"),
    )
    continued: Final = await ownership.async_pre_call_hook(
        user_api_key_dict=OWNER,
        cache=DualCache(),
        data={
            **continue_from("msg_advisor"),
            "extra_headers": {"Anthropic-Beta": "context-1m-2025-08-07", "x-trace": "abc"},
        },
        call_type="anthropic_messages",
    )

    assert continued is not None
    assert continued["extra_headers"] == {
        "x-trace": "abc",
        "anthropic-beta": "advisor-tool-2026-03-01,context-1m-2025-08-07,message-threads-2026-08-12",
    }
    assert continued["thread"] == continue_from("msg_advisor")["thread"]


@pytest.mark.asyncio
async def test_the_create_turns_betas_reach_every_later_turn_of_the_thread() -> None:
    ownership: Final = hook()
    items: Final = (sse_frame(message_start("msg_streamed_first")), sse_frame({"type": "message_stop"}))
    await served(ownership, items, sent_with("advisor-tool-2026-03-01,message-threads-2026-08-12", thread=CREATE))
    second: Final = await ownership.async_pre_call_hook(
        user_api_key_dict=OWNER, cache=DualCache(), data=continue_from("msg_streamed_first"), call_type="anthropic_messages"
    )
    assert second is not None
    second_headers: Final = second["extra_headers"]
    assert isinstance(second_headers, dict)
    await ownership.async_post_call_success_hook(
        data=sent_with(str(second_headers["anthropic-beta"]), **continue_from("msg_streamed_first")),
        user_api_key_dict=OWNER,
        response=anthropic_message("msg_second_turn"),
    )
    third: Final = await ownership.async_pre_call_hook(
        user_api_key_dict=OWNER, cache=DualCache(), data=continue_from("msg_second_turn"), call_type="anthropic_messages"
    )

    assert third is not None
    assert third["extra_headers"] == {"anthropic-beta": "advisor-tool-2026-03-01,message-threads-2026-08-12"}


@pytest.mark.parametrize("call_type", ("acompletion", "aresponses"))
@pytest.mark.asyncio
async def test_a_continue_on_chat_or_responses_resends_the_betas_its_create_turn_sent(
    call_type: CallTypesLiteral,
) -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data=sent_with("advisor-tool-2026-03-01,message-threads-2026-08-12", thread=CREATE),
        user_api_key_dict=OWNER,
        response=anthropic_message("msg_other_route"),
    )
    continued: Final = await ownership.async_pre_call_hook(
        user_api_key_dict=OWNER, cache=DualCache(), data=continue_from("msg_other_route"), call_type=call_type
    )

    assert continued is not None
    assert continued["extra_headers"] == {"anthropic-beta": "advisor-tool-2026-03-01,message-threads-2026-08-12"}


CLIENT_BETAS: Final = {"custom_llm_provider": "anthropic,azure_ai", "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"}}
CLIENT_VERSION: Final = {"custom_llm_provider": "anthropic", "extra_headers": {"anthropic-version": "2023-06-01"}}
CLIENT_BETAS_WITH_THE_CREATES: Final = {
    "custom_llm_provider": "anthropic,azure_ai",
    "extra_headers": {"anthropic-beta": "advisor-tool-2026-03-01,context-1m-2025-08-07,message-threads-2026-08-12"},
}


@pytest.mark.parametrize(
    ("client_headers", "expected"),
    (
        (CLIENT_BETAS, CLIENT_BETAS_WITH_THE_CREATES),
        ([CLIENT_BETAS, CLIENT_VERSION], (CLIENT_BETAS_WITH_THE_CREATES, CLIENT_VERSION)),
        (CLIENT_VERSION, CLIENT_VERSION),
    ),
)
@pytest.mark.parametrize("call_type", ("acompletion", "aresponses", "anthropic_messages"))
@pytest.mark.asyncio
async def test_a_continue_that_sends_its_own_betas_keeps_the_create_turns_betas_next_to_them(
    call_type: CallTypesLiteral, client_headers: object, expected: object
) -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data=sent_with("advisor-tool-2026-03-01,message-threads-2026-08-12", thread=CREATE),
        user_api_key_dict=OWNER,
        response=anthropic_message("msg_client_betas"),
    )
    continued: Final = await ownership.async_pre_call_hook(
        user_api_key_dict=OWNER,
        cache=DualCache(),
        data={**continue_from("msg_client_betas"), "provider_specific_header": client_headers},
        call_type=call_type,
    )

    assert continued is not None
    assert continued["provider_specific_header"] == expected
    assert continued["extra_headers"] == {"anthropic-beta": "advisor-tool-2026-03-01,message-threads-2026-08-12"}


@pytest.mark.asyncio
async def test_a_pass_through_continue_is_checked_but_left_unchanged() -> None:
    ownership: Final = hook()
    await ownership.async_post_call_success_hook(
        data=sent_with("advisor-tool-2026-03-01", thread=CREATE),
        user_api_key_dict=OWNER,
        response=anthropic_message("msg_other_route"),
    )

    assert (
        await ownership.async_pre_call_hook(
            user_api_key_dict=OWNER,
            cache=DualCache(),
            data=continue_from("msg_other_route"),
            call_type="pass_through_endpoint",
        )
        is None
    )


@pytest.mark.asyncio
async def test_thread_bindings_never_push_the_proxys_rate_limit_counters_out_of_memory() -> None:
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    ownership: Final = MessageThreadOwnership(message_thread_owner_cache=proxy_logging.message_thread_owner_cache)
    await proxy_logging.internal_usage_cache.async_set_cache(
        key="rpm-counter", value=1, litellm_parent_otel_span=None, ttl=60
    )
    for turn in range(250):
        await ownership.async_post_call_success_hook(
            data={"thread": CREATE}, user_api_key_dict=OWNER, response=anthropic_message(f"msg_{turn}")
        )

    assert await proxy_logging.internal_usage_cache.async_get_cache(key="rpm-counter", litellm_parent_otel_span=None) == 1
    assert await allows(ownership, OWNER, "msg_0")


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
