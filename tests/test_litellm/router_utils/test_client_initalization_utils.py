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

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm import Router
from litellm.types.router import Deployment

MODEL_ID = "test-deployment-id"


def _model_list(max_parallel_requests: int = 1) -> list:
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key",
                "max_parallel_requests": max_parallel_requests,
            },
            "model_info": {"id": MODEL_ID},
        }
    ]


def _get_semaphore(router: Router) -> Optional[asyncio.Semaphore]:
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

    semaphore = _get_semaphore(router)
    assert isinstance(semaphore, asyncio.Semaphore)
    await semaphore.acquire()  # an in-flight request holds the only permit

    # simulate the 600s TTL elapsing / size-pressure eviction
    router.cache.in_memory_cache.cache_dict.clear()
    router.cache.in_memory_cache.ttl_dict.clear()

    semaphore_after = _get_semaphore(router)
    assert semaphore_after is semaphore
    assert semaphore_after.locked()  # the in-flight permit still counts
    semaphore.release()


def test_delete_deployment_drops_semaphore():
    router = Router(model_list=_model_list())
    _get_semaphore(router)
    assert MODEL_ID in router._max_parallel_requests_semaphores

    router.delete_deployment(id=MODEL_ID)
    assert MODEL_ID not in router._max_parallel_requests_semaphores


def test_upsert_deployment_with_new_params_resets_semaphore():
    router = Router(model_list=_model_list(max_parallel_requests=1))
    semaphore = _get_semaphore(router)

    router.upsert_deployment(
        Deployment(
            model_name="gpt-3.5-turbo",
            litellm_params={
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key",
                "max_parallel_requests": 2,
            },  # type: ignore
            model_info={"id": MODEL_ID},
        )
    )

    semaphore_after = _get_semaphore(router)
    assert semaphore_after is not semaphore
    assert semaphore_after is not None
    assert semaphore_after._value == 2  # reflects the updated cap


def test_set_model_list_keeps_survivors_and_prunes_removed():
    router = Router(model_list=_model_list())
    semaphore = _get_semaphore(router)

    # hot-reload with unchanged models must not reset the semaphore
    router.set_model_list(_model_list())
    assert router._max_parallel_requests_semaphores[MODEL_ID] is semaphore

    # removing the deployment must not leak its semaphore
    router.set_model_list([])
    assert router._max_parallel_requests_semaphores == {}
