"""
Tests the following endpoints used by the UI 

/global/spend/logs
/global/spend/keys
/global/spend/models
/global/activity
/global/activity/model


For all tests - test the following:
- Response is valid 
- Response for Admin User is different from response from Internal User
"""

import os
import sys
import traceback
from litellm._uuid import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

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

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    new_user,
    user_info,
    user_update,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_key_fn,
    generate_key_fn,
    generate_key_helper_fn,
    info_key_fn,
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
    model_list,
    moderations,
    user_api_key_auth,
)
from litellm.proxy.management_endpoints.customer_endpoints import (
    new_end_user,
)
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    global_spend,
    global_spend_logs,
    global_spend_models,
    global_spend_keys,
    spend_key_fn,
    spend_user_fn,
    view_spend_logs,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token, update_spend

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from starlette.datastructures import URL

from litellm.caching.caching import DualCache
from litellm.types.proxy.management_endpoints.ui_sso import LiteLLM_UpperboundKeyGenerateParams
from litellm.proxy._types import (
    DynamoDBArgs,
    GenerateKeyRequest,
    RegenerateKeyRequest,
    KeyRequest,
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


@pytest.mark.asyncio()
async def test_view_daily_spend_ui(prisma_client):
    print("prisma client=", prisma_client)
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    await litellm.proxy.proxy_server.prisma_client.connect()
    from litellm.proxy.proxy_server import user_api_key_cache

    spend_logs_for_admin = await global_spend_logs(
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-1234",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
        api_key=None,
    )

    print("spend_logs_for_admin=", spend_logs_for_admin)

    spend_logs_for_internal_user = await global_spend_logs(
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-1234", user_role=LitellmUserRoles.INTERNAL_USER, user_id="1234"
        ),
        api_key=None,
    )

    print("spend_logs_for_internal_user=", spend_logs_for_internal_user)

    # Calculate total spend for admin
    admin_total_spend = sum(log.get("spend", 0) for log in spend_logs_for_admin)

    # Calculate total spend for internal user (0 in this case, but we'll keep it generic)
    internal_user_total_spend = sum(
        log.get("spend", 0) for log in spend_logs_for_internal_user
    )

    print("total_spend_for_admin=", admin_total_spend)
    print("total_spend_for_internal_user=", internal_user_total_spend)

    assert (
        admin_total_spend > internal_user_total_spend
    ), "Admin should have more spend than internal user"


@pytest.mark.asyncio
async def test_global_spend_models(prisma_client):
    print("prisma client=", prisma_client)
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    await litellm.proxy.proxy_server.prisma_client.connect()

    # Test for admin user
    models_spend_for_admin = await global_spend_models(
        limit=10,
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-1234",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
    )

    print("models_spend_for_admin=", models_spend_for_admin)

    # Test for internal user
    models_spend_for_internal_user = await global_spend_models(
        limit=10,
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-1234", user_role=LitellmUserRoles.INTERNAL_USER, user_id="1234"
        ),
    )

    print("models_spend_for_internal_user=", models_spend_for_internal_user)

    # Assertions
    assert isinstance(models_spend_for_admin, list), "Admin response should be a list"
    assert isinstance(
        models_spend_for_internal_user, list
    ), "Internal user response should be a list"

    # Check if the response has the expected shape for both admin and internal user
    expected_keys = ["model", "total_spend"]

    if len(models_spend_for_admin) > 0:
        assert all(
            key in models_spend_for_admin[0] for key in expected_keys
        ), f"Admin response should contain keys: {expected_keys}"
        assert isinstance(
            models_spend_for_admin[0]["model"], str
        ), "Model should be a string"
        assert isinstance(
            models_spend_for_admin[0]["total_spend"], (int, float)
        ), "Total spend should be a number"

    if len(models_spend_for_internal_user) > 0:
        assert all(
            key in models_spend_for_internal_user[0] for key in expected_keys
        ), f"Internal user response should contain keys: {expected_keys}"
        assert isinstance(
            models_spend_for_internal_user[0]["model"], str
        ), "Model should be a string"
        assert isinstance(
            models_spend_for_internal_user[0]["total_spend"], (int, float)
        ), "Total spend should be a number"

    # Check if the lists are sorted by total_spend in descending order
    if len(models_spend_for_admin) > 1:
        assert all(
            models_spend_for_admin[i]["total_spend"]
            >= models_spend_for_admin[i + 1]["total_spend"]
            for i in range(len(models_spend_for_admin) - 1)
        ), "Admin response should be sorted by total_spend in descending order"

    if len(models_spend_for_internal_user) > 1:
        assert all(
            models_spend_for_internal_user[i]["total_spend"]
            >= models_spend_for_internal_user[i + 1]["total_spend"]
            for i in range(len(models_spend_for_internal_user) - 1)
        ), "Internal user response should be sorted by total_spend in descending order"

    # Check if admin has access to more or equal models compared to internal user
    assert len(models_spend_for_admin) >= len(
        models_spend_for_internal_user
    ), "Admin should have access to at least as many models as internal user"

    # Check if the response contains expected fields
    if len(models_spend_for_admin) > 0:
        assert all(
            key in models_spend_for_admin[0] for key in ["model", "total_spend"]
        ), "Admin response should contain model, total_spend, and total_tokens"

    if len(models_spend_for_internal_user) > 0:
        assert all(
            key in models_spend_for_internal_user[0] for key in ["model", "total_spend"]
        ), "Internal user response should contain model, total_spend, and total_tokens"


@pytest.mark.asyncio
async def test_global_spend_keys(prisma_client):
    print("prisma client=", prisma_client)
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    await litellm.proxy.proxy_server.prisma_client.connect()

    # Test for admin user
    keys_spend_for_admin = await global_spend_keys(
        limit=10,
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-1234",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
    )

    print("keys_spend_for_admin=", keys_spend_for_admin)

    # Test for internal user
    keys_spend_for_internal_user = await global_spend_keys(
        limit=10,
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-1234", user_role=LitellmUserRoles.INTERNAL_USER, user_id="1234"
        ),
    )

    print("keys_spend_for_internal_user=", keys_spend_for_internal_user)

    # Assertions
    assert isinstance(keys_spend_for_admin, list), "Admin response should be a list"
    assert isinstance(
        keys_spend_for_internal_user, list
    ), "Internal user response should be a list"

    # Check if admin has access to more or equal keys compared to internal user
    assert len(keys_spend_for_admin) >= len(
        keys_spend_for_internal_user
    ), "Admin should have access to at least as many keys as internal user"

    # Check if the response contains expected fields
    if len(keys_spend_for_admin) > 0:
        assert all(
            key in keys_spend_for_admin[0]
            for key in ["api_key", "total_spend", "key_alias", "key_name"]
        ), "Admin response should contain api_key, total_spend, key_alias, and key_name"

    if len(keys_spend_for_internal_user) > 0:
        assert all(
            key in keys_spend_for_internal_user[0]
            for key in ["api_key", "total_spend", "key_alias", "key_name"]
        ), "Internal user response should contain api_key, total_spend, key_alias, and key_name"
