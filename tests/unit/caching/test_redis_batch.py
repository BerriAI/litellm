"""The request scope: one set of native batches per request, and nested scopes join it."""

from __future__ import annotations

import gc
import weakref
from typing import Final

from litellm.caching.redis_batch import RequestRedisBatches, active_request_redis_batches, request_redis_batch_scope


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
