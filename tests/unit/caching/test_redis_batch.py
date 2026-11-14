"""The request scope: one set of native batches per request, nested scopes join it, and the shutdown drain sends
every post-call batch still waiting."""

from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import pytest

from litellm.caching.redis_batch import (
    RequestRedisBatches,
    active_request_redis_batches,
    drain_post_call_redis_batches,
    request_redis_batch_scope,
)

from .redis_batch_double import FakeNativeRequest, fake_send


def test_nested_request_scopes_share_the_outer_request_and_close_with_it() -> None:
    assert active_request_redis_batches() is None
    with request_redis_batch_scope() as outer:
        assert active_request_redis_batches() is outer
        with request_redis_batch_scope() as inner:
            assert inner is outer
        assert active_request_redis_batches() is outer
    assert active_request_redis_batches() is None


def test_a_request_scope_that_built_its_native_batches_is_freed_once_it_closes() -> None:
    def serve_request() -> weakref.ref[RequestRedisBatches]:
        with request_redis_batch_scope() as batches:
            batches.native()
            return weakref.ref(batches)

    closed: Final = serve_request()
    gc.collect()
    assert closed() is None


@dataclass
class _DeclaresWhileFlushingPost(FakeNativeRequest):
    declare_once: list[Callable[[], object]] = field(default_factory=list)

    async def flush_post(self) -> None:
        await super().flush_post()
        while self.declare_once:
            self.declare_once.pop()()


@pytest.mark.asyncio
async def test_a_post_call_write_declared_while_the_post_call_batch_goes_out_is_still_sent_by_the_shutdown_drain() -> (
    None
):
    native: Final = _DeclaresWhileFlushingPost(post_call_deadline=60.0)
    with request_redis_batch_scope(native_factory=lambda _deadline: native, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        request.native_post_call()
        native.declare_once.append(request.native_post_call)
        await request.flush_post_call()
        await drain_post_call_redis_batches()
    assert native.events == ["flush_post", "flush_post"]


@dataclass
class _HeldPostFlush(FakeNativeRequest):
    sent: asyncio.Event = field(default_factory=asyncio.Event)

    async def flush_post(self) -> None:
        await self.sent.wait()
        await super().flush_post()


@pytest.mark.asyncio
async def test_a_shutdown_drain_that_overlaps_a_post_call_flush_waits_for_its_writes() -> None:
    native: Final = _HeldPostFlush(post_call_deadline=60.0)
    with request_redis_batch_scope(native_factory=lambda _deadline: native, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        request.native_post_call()
        flush: Final = asyncio.create_task(request.flush_post_call())
        await asyncio.sleep(0)
        drain: Final = asyncio.create_task(drain_post_call_redis_batches())
        await asyncio.sleep(0)
        drained_while_in_flight: Final = drain.done()
        native.sent.set()
        await asyncio.gather(flush, drain)
    assert not drained_while_in_flight
