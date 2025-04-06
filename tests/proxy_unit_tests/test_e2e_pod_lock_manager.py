import os
import sys
import traceback
import uuid
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute
import httpx
import json

load_dotenv()
import io
import os
import time

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging

import pytest
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    new_user,
    user_info,
    user_update,
)
from litellm.proxy.auth.auth_checks import get_key_object
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_key_fn,
    generate_key_fn,
    generate_key_helper_fn,
    info_key_fn,
    list_keys,
    regenerate_key_fn,
    update_key_fn,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    new_team,
    team_info,
    update_team,
)
from litellm.proxy.proxy_server import (
    LitellmUserRoles,
    audio_transcriptions,
    chat_completion,
    completion,
    embeddings,
    image_generation,
    model_list,
    moderations,
    user_api_key_auth,
)
from litellm.proxy.management_endpoints.customer_endpoints import (
    new_end_user,
)
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    global_spend,
    spend_key_fn,
    spend_user_fn,
    view_spend_logs,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token, update_spend

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from starlette.datastructures import URL

from litellm.caching.caching import DualCache, RedisCache
from litellm.proxy._types import (
    DynamoDBArgs,
    GenerateKeyRequest,
    KeyRequest,
    LiteLLM_UpperboundKeyGenerateParams,
    NewCustomerRequest,
    NewTeamRequest,
    NewUserRequest,
    ProxyErrorTypes,
    ProxyException,
    UpdateKeyRequest,
    UpdateTeamRequest,
    UpdateUserRequest,
    UserAPIKeyAuth,
)

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


request_data = {
    "model": "azure-gpt-3.5",
    "messages": [
        {"role": "user", "content": "this is my new test. respond in 50 lines"}
    ],
}

global_redis_cache = RedisCache(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    password=os.getenv("REDIS_PASSWORD"),
)


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    ### add connection pool + pool timeout args
    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    # Assuming PrismaClient is a class that needs to be instantiated
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    # Reset litellm.proxy.proxy_server.prisma_client to None
    litellm.proxy.proxy_server.litellm_proxy_budget_name = (
        f"litellm-proxy-budget-{time.time()}"
    )
    litellm.proxy.proxy_server.user_custom_key_generate = None

    return prisma_client


async def setup_db_connection(prisma_client):
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()


@pytest.mark.asyncio
async def test_pod_lock_acquisition_when_no_active_lock():
    """Test if a pod can acquire a lock when no lock is active"""
    cronjob_id = str(uuid.uuid4())
    lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)

    # Attempt to acquire lock
    result = await lock_manager.acquire_lock()

    assert result == True, "Pod should be able to acquire lock when no lock exists"

    # Verify in database
    lock_key = PodLockManager.get_redis_lock_key(cronjob_id)
    lock_record = await global_redis_cache.async_get_cache(lock_key)
    print("lock_record=", lock_record)
    assert lock_record == lock_manager.pod_id



@pytest.mark.asyncio
async def test_pod_lock_acquisition_after_completion():
    """Test if a new pod can acquire lock after previous pod completes"""
    cronjob_id = str(uuid.uuid4())
    # First pod acquires and releases lock
    first_lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    await first_lock_manager.acquire_lock()
    await first_lock_manager.release_lock()

    # Second pod attempts to acquire lock
    second_lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    result = await second_lock_manager.acquire_lock()

    assert result == True, "Second pod should acquire lock after first pod releases it"

    # Verify in redis
    lock_key = PodLockManager.get_redis_lock_key(cronjob_id)
    lock_record = await global_redis_cache.async_get_cache(lock_key)
    assert lock_record == second_lock_manager.pod_id


@pytest.mark.asyncio
async def test_pod_lock_acquisition_after_expiry():
    """Test if a new pod can acquire lock after previous pod's lock expires"""
    cronjob_id = str(uuid.uuid4())
    # First pod acquires lock
    first_lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    await first_lock_manager.acquire_lock()

    # release the lock from the first pod
    await first_lock_manager.release_lock()

    # Second pod attempts to acquire lock
    second_lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    result = await second_lock_manager.acquire_lock()

    assert (
        result == True
    ), "Second pod should acquire lock after first pod's lock expires"

    # Verify in redis
    lock_key = PodLockManager.get_redis_lock_key(cronjob_id)
    lock_record = await global_redis_cache.async_get_cache(lock_key)
    assert lock_record == second_lock_manager.pod_id


@pytest.mark.asyncio
async def test_pod_lock_release():
    """Test if a pod can successfully release its lock"""
    cronjob_id = str(uuid.uuid4())
    lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)

    # Acquire and then release lock
    await lock_manager.acquire_lock()
    await lock_manager.release_lock()

    # Verify in redis
    lock_key = PodLockManager.get_redis_lock_key(cronjob_id)
    lock_record = await global_redis_cache.async_get_cache(lock_key)
    assert lock_record is None


@pytest.mark.asyncio
async def test_concurrent_lock_acquisition():
    """Test that only one pod can acquire the lock when multiple pods try simultaneously"""

    cronjob_id = str(uuid.uuid4())
    # Create multiple lock managers simulating different pods
    lock_manager1 = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    lock_manager2 = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    lock_manager3 = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)

    # Try to acquire locks concurrently
    results = await asyncio.gather(
        lock_manager1.acquire_lock(),
        lock_manager2.acquire_lock(),
        lock_manager3.acquire_lock(),
    )

    # Only one should succeed
    print("all results=", results)
    assert sum(results) == 1, "Only one pod should acquire the lock"

    # Verify in redis
    lock_key = PodLockManager.get_redis_lock_key(cronjob_id)
    lock_record = await global_redis_cache.async_get_cache(lock_key)
    assert lock_record in [
        lock_manager1.pod_id,
        lock_manager2.pod_id,
        lock_manager3.pod_id,
    ]



@pytest.mark.asyncio
async def test_lock_acquisition_with_expired_ttl():
    """Test that a pod can acquire a lock when existing lock has expired TTL"""
    cronjob_id = str(uuid.uuid4())
    first_lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)

    # First pod acquires lock with a very short TTL to simulate expiration
    short_ttl = 1  # 1 second
    lock_key = PodLockManager.get_redis_lock_key(cronjob_id)
    await global_redis_cache.async_set_cache(
        lock_key,
        first_lock_manager.pod_id,
        ttl=short_ttl,
    )

    # Wait for the lock to expire
    await asyncio.sleep(short_ttl + 0.5)  # Wait slightly longer than the TTL

    # Second pod tries to acquire without explicit release
    second_lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    result = await second_lock_manager.acquire_lock()

    assert result == True, "Should acquire lock when existing lock has expired TTL"

    # Verify in Redis
    lock_record = await global_redis_cache.async_get_cache(lock_key)
    print("lock_record=", lock_record)
    assert lock_record == second_lock_manager.pod_id


@pytest.mark.asyncio
async def test_release_expired_lock():
    """Test that a pod cannot release a lock that has been taken over by another pod"""
    cronjob_id = str(uuid.uuid4())
    
    # First pod acquires lock with a very short TTL
    first_lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    short_ttl = 1  # 1 second
    lock_key = PodLockManager.get_redis_lock_key(cronjob_id)
    await global_redis_cache.async_set_cache(
        lock_key,
        first_lock_manager.pod_id,
        ttl=short_ttl,
    )

    # Wait for the lock to expire
    await asyncio.sleep(short_ttl + 0.5)  # Wait slightly longer than the TTL

    # Second pod acquires the lock
    second_lock_manager = PodLockManager(cronjob_id=cronjob_id, redis_cache=global_redis_cache)
    await second_lock_manager.acquire_lock()

    # First pod attempts to release its lock
    await first_lock_manager.release_lock()

    # Verify that second pod's lock is still active
    lock_record = await global_redis_cache.async_get_cache(lock_key)
    assert lock_record == second_lock_manager.pod_id