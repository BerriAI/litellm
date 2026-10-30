import asyncio
from contextvars import Context

import pytest
from fastapi import HTTPException

from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
    RequestRateLimiterStash,
    get_request_stash,
    isolated_mcp_tool_call,
)
from litellm.proxy.utils import InternalUsageCache


async def admit(
    handler: PROXY_MaxParallelRequestsHandler_v3, cache: DualCache, auth: UserAPIKeyAuth, call_id: str
) -> RequestRateLimiterStash:
    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=cache,
        data={"litellm_call_id": call_id},
        call_type="call_mcp_tool",
    )
    stash = get_request_stash()
    assert stash is not None
    return stash


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_children", [False, True])
async def test_children_keep_parent_slot_until_outer_finishes(cancel_children: bool) -> None:
    cache = DualCache()
    handler = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(cache))
    auth = UserAPIKeyAuth(api_key="code-mode-parent", max_parallel_requests=1)
    ready = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def independent_request() -> None:
        with isolated_mcp_tool_call():
            await admit(handler, cache, auth, "independent")
            await handler.async_release_max_parallel_requests_on_disconnect(auth)

    with isolated_mcp_tool_call():
        parent = await admit(handler, cache, auth, "outer")
        acquisition = parent.parallel_slot
        assert acquisition is not None

        async def child(index: int) -> RequestRateLimiterStash:
            with isolated_mcp_tool_call():
                try:
                    owned = await admit(handler, cache, auth, f"child-{index}")
                    assert owned is not parent
                    assert owned.parallel_slot is None
                finally:
                    ready[index].set()
                try:
                    await release.wait()
                    await handler.async_log_success_event(
                        kwargs={"litellm_call_id": f"child-{index}"},
                        response_obj=None,
                        start_time=None,
                        end_time=None,
                    )
                    return owned
                finally:
                    await handler.async_release_max_parallel_requests_on_disconnect(auth)

        children = [asyncio.create_task(child(index)) for index in range(2)]
        await asyncio.gather(*(event.wait() for event in ready))
        if cancel_children:
            for task in children:
                task.cancel()
            outcomes = await asyncio.gather(*children, return_exceptions=True)
            assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
        else:
            release.set()
            first, second = await asyncio.gather(*children)
            assert first is not second
            assert first.owner_litellm_call_id == "child-0"
            assert second.owner_litellm_call_id == "child-1"
        assert get_request_stash() is parent
        assert parent.owner_litellm_call_id == "outer"
        assert parent.parallel_slot is acquisition
        with pytest.raises(HTTPException, match="max_parallel_requests"):
            await Context().run(asyncio.create_task, independent_request())
        await handler.async_release_max_parallel_requests_on_disconnect(auth)
        await Context().run(asyncio.create_task, independent_request())


@pytest.mark.asyncio
async def test_children_still_consume_rpm_and_enforce_unborrowed_concurrency() -> None:
    cache = DualCache()
    handler = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(cache))
    parent_auth = UserAPIKeyAuth(api_key="code-mode-rpm", max_parallel_requests=1, rpm_limit=2)
    other_auth = UserAPIKeyAuth(api_key="code-mode-other", max_parallel_requests=1)
    other_ready = asyncio.Event()
    release_other = asyncio.Event()

    async def hold_other() -> None:
        with isolated_mcp_tool_call():
            await admit(handler, cache, other_auth, "other")
            other_ready.set()
            try:
                await release_other.wait()
            finally:
                await handler.async_release_max_parallel_requests_on_disconnect(other_auth)

    other = Context().run(asyncio.create_task, hold_other())
    await other_ready.wait()
    try:
        with isolated_mcp_tool_call():
            await admit(handler, cache, parent_auth, "outer")
            try:
                with isolated_mcp_tool_call():
                    await admit(handler, cache, parent_auth, "child-1")
                with pytest.raises(HTTPException, match="requests"):
                    with isolated_mcp_tool_call():
                        await admit(handler, cache, parent_auth, "child-2")
                with pytest.raises(HTTPException, match="max_parallel_requests"):
                    with isolated_mcp_tool_call():
                        await admit(handler, cache, other_auth, "different-child-scope")
            finally:
                await handler.async_release_max_parallel_requests_on_disconnect(parent_auth)
    finally:
        release_other.set()
        await other


@pytest.mark.asyncio
async def test_cancelled_child_refunds_only_its_own_token_reservation() -> None:
    cache = DualCache()
    handler = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(cache))
    auth = UserAPIKeyAuth(api_key="code-mode-token-cancellation", tpm_limit=2)
    with isolated_mcp_tool_call():
        parent = await admit(handler, cache, auth, "outer")
        assert parent.reserved_tokens == 1
        with isolated_mcp_tool_call():
            child = await admit(handler, cache, auth, "cancelled-child")
            assert child.reserved_tokens == 1
            for _ in range(2):
                await handler.async_post_call_failure_hook(
                    request_data={"litellm_call_id": "cancelled-child"},
                    original_exception=HTTPException(status_code=499, detail="Cancelled"),
                    user_api_key_dict=auth,
                )
            assert child.reservation_released
        assert not parent.reservation_released
        with isolated_mcp_tool_call():
            await admit(handler, cache, auth, "replacement-child")
        with pytest.raises(HTTPException, match="tokens"):
            with isolated_mcp_tool_call():
                await admit(handler, cache, auth, "over-limit-child")


@pytest.mark.asyncio
async def test_cancelled_outer_does_not_refund_completed_child_usage() -> None:
    from litellm.types.utils import ModelResponse, Usage

    cache = DualCache()
    handler = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(cache))
    auth = UserAPIKeyAuth(api_key="code-mode-completed-child", tpm_limit=4)
    with isolated_mcp_tool_call():
        await admit(handler, cache, auth, "outer")
        with isolated_mcp_tool_call():
            await admit(handler, cache, auth, "completed-child")
            await handler.async_log_success_event(
                kwargs={
                    "litellm_call_id": "completed-child",
                    "standard_logging_object": {"metadata": {"user_api_key_hash": auth.api_key}},
                },
                response_obj=ModelResponse(usage=Usage(prompt_tokens=2, completion_tokens=0, total_tokens=2)),
                start_time=None,
                end_time=None,
            )
        for _ in range(2):
            await handler.async_post_call_failure_hook(
                request_data={"litellm_call_id": "outer"},
                original_exception=HTTPException(status_code=499, detail="Cancelled"),
                user_api_key_dict=auth,
            )
        for index in range(2):
            with isolated_mcp_tool_call():
                await admit(handler, cache, auth, f"remaining-capacity-{index}")
        with pytest.raises(HTTPException, match="tokens"):
            with isolated_mcp_tool_call():
                await admit(handler, cache, auth, "completed-child-charge-still-counts")


def test_request_stash_preserves_existing_positional_constructor() -> None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitResponse

    response = RateLimitResponse(overall_code="OK", statuses=[])
    stash = RequestRateLimiterStash("original-owner", response)
    assert stash.owner_litellm_call_id == "original-owner"
    assert stash.rate_limit_response is response
    assert stash.borrowed_parallel_keys == frozenset()
