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
from litellm.proxy.management_endpoints.team_endpoints import list_team
from litellm.proxy._types import *
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    new_user,
    user_info,
    user_update,
    get_users,
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

from litellm.caching.caching import DualCache
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


################ Unit Tests for testing regeneration of keys ###########
@pytest.mark.asyncio()
async def test_regenerate_api_key(prisma_client):
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

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


@pytest.mark.asyncio
async def test_get_users(prisma_client):
    """
    Tests /users/list endpoint

    Admin UI calls this endpoint to list all Internal Users
    """
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Create some test users
    test_users = [
        NewUserRequest(
            user_id=f"test_user_{i}",
            user_role=(
                LitellmUserRoles.INTERNAL_USER.value
                if i % 2 == 0
                else LitellmUserRoles.PROXY_ADMIN.value
            ),
        )
        for i in range(5)
    ]
    for user in test_users:
        await new_user(
            user,
            UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="admin",
            ),
        )

    # Test get_users without filters
    result = await get_users(
        role=None,
        page=1,
        page_size=20,
    )
    print("get users result", result)
    assert "users" in result

    for user in result["users"]:
        assert "user_id" in user
        assert "spend" in user
        assert "user_email" in user
        assert "user_role" in user
        assert "key_count" in user

    # Clean up test users
    for user in test_users:
        await prisma_client.db.litellm_usertable.delete(where={"user_id": user.user_id})


@pytest.mark.asyncio
async def test_get_users_key_count(prisma_client):
    """
    Test that verifies the key_count in get_users increases when a new key is created for a user
    """
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Get initial user list and select the first user
    initial_users = await get_users(role=None, page=1, page_size=20)
    print("initial_users", initial_users)
    assert len(initial_users["users"]) > 0, "No users found to test with"

    test_user = initial_users["users"][0]
    initial_key_count = test_user["key_count"]

    # Create a new key for the selected user
    new_key = await generate_key_fn(
        data=GenerateKeyRequest(
            user_id=test_user["user_id"],
            key_alias=f"test_key_{uuid.uuid4()}",
            models=["fake-model"],
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="admin",
        ),
    )

    # Get updated user list and check key count
    updated_users = await get_users(role=None, page=1, page_size=20)
    print("updated_users", updated_users)
    updated_key_count = None
    for user in updated_users["users"]:
        if user["user_id"] == test_user["user_id"]:
            updated_key_count = user["key_count"]
            break

    assert updated_key_count is not None, "Test user not found in updated users list"
    assert (
        updated_key_count == initial_key_count + 1
    ), f"Expected key count to increase by 1, but got {updated_key_count} (was {initial_key_count})"


async def cleanup_existing_teams(prisma_client):
    all_teams = await prisma_client.db.litellm_teamtable.find_many()
    for team in all_teams:
        await prisma_client.delete_data(team_id_list=[team.team_id], table_name="team")


@pytest.mark.asyncio
async def test_list_teams(prisma_client):
    """
    Tests /team/list endpoint to verify it returns both keys and members_with_roles
    """
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Delete all existing teams first
    await cleanup_existing_teams(prisma_client)

    # Create a test team with members
    team_id = f"test_team_{uuid.uuid4()}"
    team_alias = f"test_team_alias_{uuid.uuid4()}"
    test_team = await new_team(
        data=NewTeamRequest(
            team_id=team_id,
            team_alias=team_alias,
            members_with_roles=[
                Member(role="admin", user_id="test_user_1"),
                Member(role="user", user_id="test_user_2"),
            ],
            models=["gpt-4"],
            tpm_limit=1000,
            rpm_limit=1000,
            budget_duration="30d",
            max_budget=1000,
        ),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234", user_id="admin"
        ),
    )

    # Create a key for the team
    test_key = await generate_key_fn(
        data=GenerateKeyRequest(
            team_id=team_id,
            key_alias=f"test_key_{uuid.uuid4()}",
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234", user_id="admin"
        ),
    )

    # Get team list
    teams = await list_team(
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234", user_id="admin"
        ),
        user_id=None,
    )

    print("teams", teams)

    # Find our test team in the response
    test_team_response = None
    for team in teams:
        if team.team_id == team_id:
            test_team_response = team
            break

    assert (
        test_team_response is not None
    ), f"Could not find test team {team_id} in response"

    # Verify members_with_roles
    assert (
        len(test_team_response.members_with_roles) == 3
    ), "Expected 3 members in team"  # 2 members + 1 team admin
    member_roles = {m.role for m in test_team_response.members_with_roles}
    assert "admin" in member_roles, "Expected admin role in members"
    assert "user" in member_roles, "Expected user role in members"

    # Verify all required fields in TeamListResponseObject
    assert (
        test_team_response.team_id == team_id
    ), f"team_id should be expected value {team_id}"
    assert (
        test_team_response.team_alias == team_alias
    ), f"team_alias should be expected value {team_alias}"
    assert test_team_response.spend is not None, "spend should not be None"
    assert (
        test_team_response.max_budget == 1000
    ), f"max_budget should be expected value 1000"
    assert test_team_response.models == [
        "gpt-4"
    ], f"models should be expected value ['gpt-4']"
    assert (
        test_team_response.tpm_limit == 1000
    ), f"tpm_limit should be expected value 1000"
    assert (
        test_team_response.rpm_limit == 1000
    ), f"rpm_limit should be expected value 1000"
    assert (
        test_team_response.budget_reset_at is not None
    ), "budget_reset_at should not be None since budget_duration is 30d"

    # Verify keys are returned
    assert len(test_team_response.keys) > 0, "Expected at least one key for team"
    assert any(
        k.team_id == team_id for k in test_team_response.keys
    ), "Expected to find team key in response"

    # Clean up
    await prisma_client.delete_data(team_id_list=[team_id], table_name="team")
