import os
import sys
import traceback
import uuid
import datetime as dt
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
    image_generation,
    model_list,
    moderations,
    new_end_user,
    user_api_key_auth,
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

from litellm.caching import DualCache
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
    RegenerateKeyRequest,
    UpdateTeamRequest,
    UpdateUserRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import DBClient

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


################ Unit Tests for testing regeneration of keys ###########
@pytest.mark.asyncio()
async def test_regenerate_api_key(prisma_client):
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    import uuid

    # generate new key
    key_alias = f"test_alias_regenerate_key-{uuid.uuid4()}"
    spend = 100
    max_budget = 400
    models = ["fake-openai-endpoint"]
    new_key = await generate_key_fn(
        data=GenerateKeyRequest(
            key_alias=key_alias, spend=spend, max_budget=max_budget, models=models
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    generated_key = new_key.key
    print(generated_key)

    # assert the new key works as expected
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body():
        return_string = f'{{"model": "fake-openai-endpoint"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body
    result = await user_api_key_auth(request=request, api_key=f"Bearer {generated_key}")
    print(result)

    # regenerate the key
    new_key = await regenerate_key_fn(
        key=generated_key,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )
    print("response from regenerate_key_fn", new_key)

    # assert the new key works as expected
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body_2():
        return_string = f'{{"model": "fake-openai-endpoint"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body_2
    result = await user_api_key_auth(request=request, api_key=f"Bearer {new_key.key}")
    print(result)

    # assert the old key stops working
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body_3():
        return_string = f'{{"model": "fake-openai-endpoint"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body_3
    try:
        result = await user_api_key_auth(
            request=request, api_key=f"Bearer {generated_key}"
        )
        print(result)
        pytest.fail(f"This should have failed!. the key has been regenerated")
    except Exception as e:
        print("got expected exception", e)
        assert "Invalid proxy server token passed" in e.message

    # Check that the regenerated key has the same spend, max_budget, models and key_alias
    assert new_key.spend == spend, f"Expected spend {spend} but got {new_key.spend}"
    assert (
        new_key.max_budget == max_budget
    ), f"Expected max_budget {max_budget} but got {new_key.max_budget}"
    assert (
        new_key.key_alias == key_alias
    ), f"Expected key_alias {key_alias} but got {new_key.key_alias}"
    assert (
        new_key.models == models
    ), f"Expected models {models} but got {new_key.models}"

    assert new_key.key_name == f"sk-...{new_key.key[-4:]}"

    pass


@pytest.mark.asyncio()
async def test_regenerate_api_key_with_new_alias_and_expiration(prisma_client):
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    import uuid

    # generate new key
    key_alias = f"test_alias_regenerate_key-{uuid.uuid4()}"
    spend = 100
    max_budget = 400
    models = ["fake-openai-endpoint"]
    new_key = await generate_key_fn(
        data=GenerateKeyRequest(
            key_alias=key_alias, spend=spend, max_budget=max_budget, models=models
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    generated_key = new_key.key
    print(generated_key)

    # regenerate the key with new alias and expiration
    new_key = await regenerate_key_fn(
        key=generated_key,
        data=RegenerateKeyRequest(
            key_alias="very_new_alias",
            duration="30d",
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )
    print("response from regenerate_key_fn", new_key)

    # assert the alias and duration are updated
    assert new_key.key_alias == "very_new_alias"

    # assert the new key expires 30 days from now
    now = datetime.now(dt.timezone.utc)
    assert new_key.expires > now + dt.timedelta(days=29)
    assert new_key.expires < now + dt.timedelta(days=31)


@pytest.mark.asyncio()
async def test_regenerate_key_ui(prisma_client):
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    import uuid

    # generate new key
    key_alias = f"test_alias_regenerate_key-{uuid.uuid4()}"
    spend = 100
    max_budget = 400
    models = ["fake-openai-endpoint"]
    new_key = await generate_key_fn(
        data=GenerateKeyRequest(
            key_alias=key_alias, spend=spend, max_budget=max_budget, models=models
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    generated_key = new_key.key
    print(generated_key)

    # assert the new key works as expected
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body():
        return_string = f'{{"model": "fake-openai-endpoint"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body
    result = await user_api_key_auth(request=request, api_key=f"Bearer {generated_key}")
    print(result)

    # regenerate the key
    new_key = await regenerate_key_fn(
        key=generated_key,
        data=RegenerateKeyRequest(duration=""),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )
    print("response from regenerate_key_fn", new_key)
