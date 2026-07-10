"""
Tests that the router's max_parallel_requests semaphore is a process-lifetime
object tied to the deployment, not an expiring cache entry.

Regression tests: the semaphore used to live in the router's InMemoryCache,
which force-applies
default_ttl (600s). On every expiry it was silently recreated with all permits
free while in-flight requests still held permits on the old object, transiently
over-admitting up to 2x max_parallel_requests. Before the recreate-on-miss
existed, expiry disabled concurrency limiting entirely.
"""

import asyncio
import os
import sys
from typing import Optional

import anyio
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm import Router
from litellm.types.router import Deployment

MODEL_ID = "test-deployment-id"


def _model_list(max_parallel_requests: Optional[int] = 1) -> list:
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key",
                **({"max_parallel_requests": max_parallel_requests} if max_parallel_requests is not None else {}),
            },
            "model_info": {"id": MODEL_ID},
        }
    ]


def _get_limiter(router: Router) -> Optional[anyio.CapacityLimiter]:
    return router._get_client(
        deployment=router.model_list[0],
        kwargs={},
        client_type="max_parallel_requests",
    )


@pytest.mark.asyncio
async def test_semaphore_survives_router_cache_expiry():
    """
    The semaphore must keep enforcing even if every router cache entry is
    dropped (InMemoryCache applies default_ttl=600s to entries stored without
    a ttl, and evicts under size pressure).
    """
    router = Router(model_list=_model_list())

    limiter = _get_limiter(router)
    assert isinstance(limiter, anyio.CapacityLimiter)
    await limiter.acquire()

    router.cache.in_memory_cache.cache_dict.clear()
    router.cache.in_memory_cache.ttl_dict.clear()

    limiter_after = _get_limiter(router)
    assert limiter_after is limiter
    assert limiter_after.borrowed_tokens == 1
    limiter.release()


def test_delete_deployment_drops_semaphore():
    router = Router(model_list=_model_list())
    _get_limiter(router)
    assert MODEL_ID in router._max_parallel_requests_semaphores

    router.delete_deployment(id=MODEL_ID)
    assert MODEL_ID not in router._max_parallel_requests_semaphores


@pytest.mark.asyncio
async def test_upsert_deployment_resizes_limiter_without_resetting_active_requests():
    router = Router(model_list=_model_list(max_parallel_requests=2))
    limiter = _get_limiter(router)
    assert limiter is not None
    first_borrower = object()
    second_borrower = object()
    waiting_borrower = object()
    await limiter.acquire_on_behalf_of(first_borrower)
    await limiter.acquire_on_behalf_of(second_borrower)

    router.upsert_deployment(
        Deployment(
            model_name="gpt-3.5-turbo",
            litellm_params={
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key",
                "max_parallel_requests": 1,
            },  # type: ignore
            model_info={"id": MODEL_ID},
        )
    )

    limiter_after = _get_limiter(router)
    assert limiter_after is limiter
    assert limiter_after.total_tokens == 1
    assert limiter_after.borrowed_tokens == 2

    waiting_acquire = asyncio.create_task(limiter_after.acquire_on_behalf_of(waiting_borrower))
    await asyncio.sleep(0)
    assert not waiting_acquire.done()

    limiter_after.release_on_behalf_of(first_borrower)
    await asyncio.sleep(0)
    assert not waiting_acquire.done()

    limiter_after.release_on_behalf_of(second_borrower)
    await waiting_acquire
    limiter_after.release_on_behalf_of(waiting_borrower)


def test_set_model_list_keeps_survivors_and_prunes_removed():
    router = Router(model_list=_model_list())
    limiter = _get_limiter(router)

    router.set_model_list(_model_list())
    assert router._max_parallel_requests_semaphores[MODEL_ID] is limiter

    router.set_model_list([])
    assert router._max_parallel_requests_semaphores == {}


@pytest.mark.asyncio
async def test_set_model_list_resizes_limiter_without_resetting_active_requests():
    router = Router(model_list=_model_list(max_parallel_requests=2))
    limiter = _get_limiter(router)
    assert limiter is not None
    first_borrower = object()
    second_borrower = object()
    waiting_borrower = object()
    await limiter.acquire_on_behalf_of(first_borrower)
    await limiter.acquire_on_behalf_of(second_borrower)

    router.set_model_list(_model_list(max_parallel_requests=1))

    limiter_after = _get_limiter(router)
    assert limiter_after is limiter
    assert limiter_after.total_tokens == 1
    assert limiter_after.borrowed_tokens == 2

    waiting_acquire = asyncio.create_task(limiter_after.acquire_on_behalf_of(waiting_borrower))
    await asyncio.sleep(0)
    assert not waiting_acquire.done()

    limiter_after.release_on_behalf_of(first_borrower)
    await asyncio.sleep(0)
    assert not waiting_acquire.done()

    limiter_after.release_on_behalf_of(second_borrower)
    await waiting_acquire
    limiter_after.release_on_behalf_of(waiting_borrower)


def test_set_model_list_preserves_limiter_when_effective_limit_is_unchanged():
    router = Router(model_list=_model_list(max_parallel_requests=2))
    limiter = _get_limiter(router)
    assert limiter is not None
    reloaded_model_list = _model_list(max_parallel_requests=None)
    reloaded_model_list[0]["litellm_params"]["rpm"] = 2

    router.set_model_list(reloaded_model_list)

    assert _get_limiter(router) is limiter
    assert limiter.total_tokens == 2


def test_set_model_list_removes_limiter_when_limit_is_disabled():
    router = Router(model_list=_model_list(max_parallel_requests=1))
    assert _get_limiter(router) is not None

    router.set_model_list(_model_list(max_parallel_requests=None))

    assert _get_limiter(router) is None
    assert MODEL_ID not in router._max_parallel_requests_semaphores
