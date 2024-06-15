# What is this?
## This tests the batch update spend logic on the proxy server


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
from litellm import Router, mock_completion
from litellm.proxy.utils import ProxyLogging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token

import pytest, logging, asyncio
import litellm, asyncio
from litellm.proxy.proxy_server import (
    user_api_key_auth,
    block_user,
)
from litellm.proxy.spend_reporting_endpoints.spend_management_endpoints import (
    spend_user_fn,
    spend_key_fn,
    view_spend_logs,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    new_user,
    user_update,
    user_info,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_key_fn,
    info_key_fn,
    update_key_fn,
    generate_key_fn,
    generate_key_helper_fn,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token, update_spend
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
async def test_batch_update_spend(prisma_client):
    prisma_client.user_list_transactons["test-litellm-user-5"] = 23
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    await update_spend(
        prisma_client=litellm.proxy.proxy_server.prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging_obj,
    )
