"""The SSE keepalive task must share the parallel-request slot with its caller."""

import asyncio

from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _request_stash,
    get_or_create_request_stash,
    get_request_stash,
)


def test_sse_keepalive_fork_shares_the_parallel_request_slot():
    async def process():
        get_or_create_request_stash().owner_litellm_call_id = "call-1"

    async def without_seed():
        _request_stash.set(None)
        task = asyncio.ensure_future(process())
        await asyncio.wait((task,))
        return get_request_stash()

    async def with_seed():
        _request_stash.set(None)
        get_or_create_request_stash()
        task = asyncio.ensure_future(process())
        await asyncio.wait((task,))
        return get_request_stash()

    leaked = asyncio.run(without_seed())
    shared = asyncio.run(with_seed())

    assert leaked is None
    assert shared is not None
    assert shared.owner_litellm_call_id == "call-1"
