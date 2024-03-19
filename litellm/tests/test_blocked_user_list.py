# What is this?
## This tests the blocked user pre call hook for the proxy server


import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv
from fastapi import Request

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.proxy.enterprise.enterprise_hooks.blocked_user_list import (
    _ENTERPRISE_BlockedUserList,
)
from litellm import Router, mock_completion
from litellm.proxy.utils import ProxyLogging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token

import pytest, logging, asyncio
import litellm, asyncio
from litellm.proxy.proxy_server import (
    new_user,
    generate_key_fn,
    user_api_key_auth,
    user_update,
    delete_key_fn,
    info_key_fn,
    update_key_fn,
    generate_key_fn,
    generate_key_helper_fn,
    spend_user_fn,
    spend_key_fn,
    view_spend_logs,
    user_info,
    block_user,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token
from litellm._logging import verbose_proxy_logger

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from litellm.proxy._types import (
    NewUserRequest,
    GenerateKeyRequest,
    DynamoDBArgs,
    KeyRequest,
    UpdateKeyRequest,
    GenerateKeyRequest,
    BlockUsers,
)
from litellm.proxy.utils import DBClient
from starlette.datastructures import URL
from litellm.caching import DualCache

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    ### add connection pool + pool timeout args
    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    # Assuming DBClient is a class that needs to be instantiated
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    # Reset litellm.proxy.proxy_server.prisma_client to None
    litellm.proxy.proxy_server.custom_db_client = None
    litellm.proxy.proxy_server.litellm_proxy_budget_name = (
        f"litellm-proxy-budget-{time.time()}"
    )
    litellm.proxy.proxy_server.user_custom_key_generate = None

    return prisma_client


@pytest.mark.asyncio
async def test_block_user_check(prisma_client):
    """
    - Set a blocked user as a litellm module value
    - Test to see if a call with that user id is made, an error is raised
    - Test to see if a call without that user is passes
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    litellm.blocked_user_list = ["user_id_1"]

    blocked_user_obj = _ENTERPRISE_BlockedUserList(
        prisma_client=litellm.proxy.proxy_server.prisma_client
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    ## Case 1: blocked user id passed
    try:
        await blocked_user_obj.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            call_type="completion",
            data={"user_id": "user_id_1"},
        )
        pytest.fail(f"Expected call to fail")
    except Exception as e:
        pass

    ## Case 2: normal user id passed
    try:
        await blocked_user_obj.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            call_type="completion",
            data={"user_id": "user_id_2"},
        )
    except Exception as e:
        pytest.fail(f"An error occurred - {str(e)}")


@pytest.mark.asyncio
async def test_block_user_db_check(prisma_client):
    """
    - Block end user via "/user/block"
    - Check returned value
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    _block_users = BlockUsers(user_ids=["user_id_1"])
    result = await block_user(data=_block_users)
    result = result["blocked_users"]
    assert len(result) == 1
    assert result[0].user_id == "user_id_1"
    assert result[0].blocked == True
