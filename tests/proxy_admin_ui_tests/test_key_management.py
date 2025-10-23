import os
import sys
import traceback
from litellm._uuid import uuid
import datetime as dt
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute
from unittest.mock import MagicMock, patch

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
from litellm.proxy._types import (
    DynamoDBArgs,
    GenerateKeyRequest,
    KeyRequest,
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
from litellm.types.proxy.management_endpoints.ui_sso import LiteLLM_UpperboundKeyGenerateParams

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    ### add connection pool + pool timeout args.
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
    print("regenerating key: {}".format(generated_key))
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
    from litellm._uuid import uuid

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
    from litellm._uuid import uuid

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
        assert isinstance(user, LiteLLM_UserTable)

    # Clean up test users
    for user in test_users:
        await prisma_client.db.litellm_usertable.delete(where={"user_id": user.user_id})


@pytest.mark.asyncio
async def test_get_users_filters_dashboard_keys(prisma_client):
    """
    Tests that /users/list endpoint doesn't return keys with team_id='litellm-dashboard'

    The dashboard keys should be filtered out from the response
    """
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Create a test user
    new_user_id = f"test_user_with_keys-{uuid.uuid4()}"
    test_user = NewUserRequest(
        user_id=new_user_id,
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        auto_create_key=False,
    )

    await new_user(
        test_user,
        UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="admin",
        ),
    )

    # Create two keys for the user - one with team_id="litellm-dashboard" and one without
    regular_key = await generate_key_helper_fn(
        user_id=test_user.user_id,
        request_type="key",
        team_id="litellm-dashboard",  # This key should be included in the response
        models=[],
        aliases={},
        config={},
        spend=0,
        duration=None,
    )

    regular_key = await generate_key_helper_fn(
        user_id=test_user.user_id,
        request_type="key",
        team_id="NEW_TEAM",  # This key should be included in the response
        models=[],
        aliases={},
        config={},
        spend=0,
        duration=None,
    )

    regular_key = await generate_key_helper_fn(
        user_id=test_user.user_id,
        request_type="key",
        team_id=None,  # This key should be included in the response
        models=[],
        aliases={},
        config={},
        spend=0,
        duration=None,
    )

    # Test get_users for the specific user
    result = await get_users(
        user_ids=test_user.user_id,
        role=None,
        page=1,
        page_size=20,
    )

    print("get users result", result)
    assert "users" in result
    assert len(result["users"]) == 1

    # Verify the key count is correct (should be 1, not counting dashboard keys)
    user = result["users"][0]
    assert user.user_id == test_user.user_id
    assert user.key_count == 2  # Only count the regular keys, not the UI dashboard key

    # Clean up test user and keys
    await prisma_client.db.litellm_usertable.delete(
        where={"user_id": test_user.user_id}
    )


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
    initial_key_count = test_user.key_count

    # Create a new key for the selected user
    new_key = await generate_key_fn(
        data=GenerateKeyRequest(
            user_id=test_user.user_id,
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
        if user.user_id == test_user.user_id:
            updated_key_count = user.key_count
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


def test_is_team_key():
    from litellm.proxy.management_endpoints.key_management_endpoints import _is_team_key

    assert _is_team_key(GenerateKeyRequest(team_id="test_team_id"))
    assert not _is_team_key(GenerateKeyRequest(user_id="test_user_id"))


def test_team_key_generation_team_member_check():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _team_key_generation_check,
    )
    from fastapi import HTTPException
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    litellm.key_generation_settings = {
        "team_key_generation": {"allowed_team_member_roles": ["admin"]}
    }

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test_team_id",
        team_alias="test_team_alias",
        members_with_roles=[Member(role="admin", user_id="test_user_id")],
    )

    assert _team_key_generation_check(
        team_table=team_table,
        user_api_key_dict=UserAPIKeyAuth(
            user_id="test_user_id",
            user_role=LitellmUserRoles.INTERNAL_USER,
            api_key="sk-1234",
            team_member=Member(role="admin", user_id="test_user_id"),
        ),
        data=GenerateKeyRequest(),
        route=KeyManagementRoutes.KEY_GENERATE,
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test_team_id",
        team_alias="test_team_alias",
        members_with_roles=[Member(role="user", user_id="test_user_id")],
    )

    with pytest.raises(HTTPException):
        _team_key_generation_check(
            team_table=team_table,
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                api_key="sk-1234",
                user_id="test_user_id",
                team_member=Member(role="user", user_id="test_user_id"),
            ),
            data=GenerateKeyRequest(),
            route=KeyManagementRoutes.KEY_GENERATE,
        )


@pytest.mark.parametrize(
    "team_key_generation_settings, input_data, expected_result",
    [
        ({"required_params": ["tags"]}, GenerateKeyRequest(tags=["test_tags"]), True),
        ({}, GenerateKeyRequest(), True),
        (
            {"required_params": ["models"]},
            GenerateKeyRequest(tags=["test_tags"]),
            False,
        ),
    ],
)
@pytest.mark.parametrize("key_type", ["team_key", "personal_key"])
def test_key_generation_required_params_check(
    team_key_generation_settings, input_data, expected_result, key_type
):
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _team_key_generation_check,
        _personal_key_generation_check,
    )
    from litellm.types.utils import (
        TeamUIKeyGenerationConfig,
        StandardKeyGenerationConfig,
        PersonalUIKeyGenerationConfig,
    )
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from fastapi import HTTPException

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-1234",
        user_id="test_user_id",
        team_id="test_team_id",
        team_member=None,
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test_team_id",
        team_alias="test_team_alias",
        members_with_roles=[Member(role="admin", user_id="test_user_id")],
    )

    if key_type == "team_key":
        litellm.key_generation_settings = StandardKeyGenerationConfig(
            team_key_generation=TeamUIKeyGenerationConfig(
                **team_key_generation_settings
            )
        )
    elif key_type == "personal_key":
        litellm.key_generation_settings = StandardKeyGenerationConfig(
            personal_key_generation=PersonalUIKeyGenerationConfig(
                **team_key_generation_settings
            )
        )

    if expected_result:
        if key_type == "team_key":
            assert _team_key_generation_check(
                team_table=team_table,
                user_api_key_dict=user_api_key_dict,
                data=input_data,
                route=KeyManagementRoutes.KEY_GENERATE,
            )
        elif key_type == "personal_key":
            assert _personal_key_generation_check(
                user_api_key_dict=user_api_key_dict,
                data=input_data,
            )
    else:
        if key_type == "team_key":
            with pytest.raises(HTTPException):
                _team_key_generation_check(
                    team_table=team_table,
                    user_api_key_dict=user_api_key_dict,
                    data=input_data,
                    route=KeyManagementRoutes.KEY_GENERATE,
                )
        elif key_type == "personal_key":
            with pytest.raises(HTTPException):
                _personal_key_generation_check(user_api_key_dict, input_data)


def test_personal_key_generation_check():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _personal_key_generation_check,
    )
    from fastapi import HTTPException

    litellm.key_generation_settings = {
        "personal_key_generation": {"allowed_user_roles": ["proxy_admin"]}
    }

    assert _personal_key_generation_check(
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234", user_id="admin"
        ),
        data=GenerateKeyRequest(),
    )

    with pytest.raises(HTTPException):
        _personal_key_generation_check(
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                api_key="sk-1234",
                user_id="admin",
            ),
            data=GenerateKeyRequest(),
        )


@pytest.mark.parametrize(
    "update_request_data, non_default_values, existing_metadata, expected_result",
    [
        (
            {"metadata": {"test": "new"}},
            {"metadata": {"test": "new"}},
            {"test": "test"},
            {"metadata": {"test": "new"}},
        ),
        (
            {"tags": ["new_tag"]},
            {},
            {"tags": ["old_tag"]},
            {"metadata": {"tags": ["new_tag"]}},
        ),
        (
            {"enforced_params": ["metadata.tags"]},
            {},
            {"tags": ["old_tag"]},
            {"metadata": {"tags": ["old_tag"], "enforced_params": ["metadata.tags"]}},
        ),
    ],
)
def test_prepare_metadata_fields(
    update_request_data, non_default_values, existing_metadata, expected_result
):
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        prepare_metadata_fields,
    )

    args = {
        "data": UpdateKeyRequest(
            key="sk-1qGQUJJTcljeaPfzgWRrXQ", **update_request_data
        ),
        "non_default_values": non_default_values,
        "existing_metadata": existing_metadata,
    }

    updated_non_default_values = prepare_metadata_fields(**args)
    assert updated_non_default_values == expected_result


@pytest.mark.asyncio
async def test_key_update_with_model_specific_params(prisma_client):
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    from litellm.proxy.management_endpoints.key_management_endpoints import (
        update_key_fn,
    )
    from litellm.proxy._types import UpdateKeyRequest

    new_key = await generate_key_fn(
        data=GenerateKeyRequest(models=["gpt-4"]),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    generated_key = new_key.key
    token_hash = new_key.token_id
    print(generated_key)

    request = Request(scope={"type": "http"})
    request._url = URL(url="/update/key")

    args = {
        "key_alias": f"test-key_{uuid.uuid4()}",
        "duration": None,
        "models": ["all-team-models"],
        "spend": 0,
        "max_budget": None,
        "user_id": "default_user_id",
        "team_id": None,
        "max_parallel_requests": None,
        "metadata": {
            "model_tpm_limit": {"fake-openai-endpoint": 10},
            "model_rpm_limit": {"fake-openai-endpoint": 0},
        },
        "tpm_limit": None,
        "rpm_limit": None,
        "budget_duration": None,
        "allowed_cache_controls": [],
        "soft_budget": None,
        "config": {},
        "permissions": {},
        "model_max_budget": {},
        "send_invite_email": None,
        "model_rpm_limit": None,
        "model_tpm_limit": None,
        "guardrails": None,
        "blocked": None,
        "aliases": {},
        "key": token_hash,
        "budget_id": None,
        "key_name": "sk-...2GWA",
        "expires": None,
        "token_id": token_hash,
        "litellm_budget_table": None,
        "token": token_hash,
    }
    await update_key_fn(
        request=request,
        data=UpdateKeyRequest(**args),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )


@pytest.mark.asyncio
async def test_list_key_helper(prisma_client):
    """
    Test _list_key_helper function with various scenarios:
    1. Basic pagination
    2. Filtering by user_id
    3. Filtering by team_id
    4. Filtering by key_alias
    5. Return full object vs token only
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _list_key_helper,
    )

    # Setup - create multiple test keys
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Create test data
    test_user_id = f"test_user_{uuid.uuid4()}"
    test_team_id = f"test_team_{uuid.uuid4()}"
    test_key_alias = f"test_alias_{uuid.uuid4()}"

    # Create test data with clear patterns
    test_keys = []

    # 1. Create 2 keys for test user + test team
    for i in range(2):
        key = await generate_key_fn(
            data=GenerateKeyRequest(
                user_id=test_user_id,
                team_id=test_team_id,
                key_alias=f"team_key_{uuid.uuid4()}",  # Make unique with UUID
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="admin",
            ),
        )
        test_keys.append(key)

    # 2. Create 1 key for test user (no team)
    key = await generate_key_fn(
        data=GenerateKeyRequest(
            user_id=test_user_id,
            key_alias=test_key_alias,  # Already unique from earlier UUID generation
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="admin",
        ),
    )
    test_keys.append(key)

    # 3. Create 2 keys for other users
    for i in range(2):
        key = await generate_key_fn(
            data=GenerateKeyRequest(
                user_id=f"other_user_{i}",
                key_alias=f"other_key_{uuid.uuid4()}",  # Make unique with UUID
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="admin",
            ),
        )
        test_keys.append(key)

    # Test 1: Basic pagination
    result = await _list_key_helper(
        prisma_client=prisma_client,
        page=1,
        size=2,
        user_id=None,
        team_id=None,
        key_alias=None,
        key_hash=None,
        organization_id=None,
    )
    assert len(result["keys"]) == 2, "Should return exactly 2 keys"
    assert result["total_count"] >= 5, "Should have at least 5 total keys"
    assert result["current_page"] == 1
    assert isinstance(result["keys"][0], str), "Should return token strings by default"

    # Test 2: Filter by user_id
    result = await _list_key_helper(
        prisma_client=prisma_client,
        page=1,
        size=10,
        user_id=test_user_id,
        team_id=None,
        key_alias=None,
        key_hash=None,
        organization_id=None,
    )
    assert len(result["keys"]) == 3, "Should return exactly 3 keys for test user"

    # Test 3: Filter by team_id
    result = await _list_key_helper(
        prisma_client=prisma_client,
        page=1,
        size=10,
        user_id=None,
        team_id=test_team_id,
        key_alias=None,
        key_hash=None,
        organization_id=None,
    )
    assert len(result["keys"]) == 2, "Should return exactly 2 keys for test team"

    # Test 4: Filter by key_alias
    result = await _list_key_helper(
        prisma_client=prisma_client,
        page=1,
        size=10,
        user_id=None,
        team_id=None,
        key_alias=test_key_alias,
        key_hash=None,
        organization_id=None,
    )
    assert len(result["keys"]) == 1, "Should return exactly 1 key with test alias"

    # Test 5: Return full object
    result = await _list_key_helper(
        prisma_client=prisma_client,
        page=1,
        size=10,
        user_id=test_user_id,
        team_id=None,
        key_alias=None,
        key_hash=None,
        return_full_object=True,
        organization_id=None,
    )
    assert all(
        isinstance(key, UserAPIKeyAuth) for key in result["keys"]
    ), "Should return UserAPIKeyAuth objects"
    assert len(result["keys"]) == 3, "Should return exactly 3 keys for test user"

    # Clean up test keys
    for key in test_keys:
        await delete_key_fn(
            data=KeyRequest(keys=[key.key]),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="admin",
            ),
        )


@pytest.mark.asyncio
async def test_list_key_helper_team_filtering(prisma_client):
    """
    Test _list_key_helper function's team filtering behavior:
    1. Create keys with different team_ids (None, litellm-dashboard, other)
    2. Verify filtering excludes litellm-dashboard keys
    3. Verify keys with team_id=None are included
    4. Test with pagination to ensure behavior is consistent across pages
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _list_key_helper,
    )
    from litellm._uuid import uuid

    # Setup
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Create test data with different team_ids
    test_keys = []

    # Create 3 keys with team_id=None
    for i in range(3):
        key = await generate_key_fn(
            data=GenerateKeyRequest(
                key_alias=f"no_team_key_{i}.{uuid.uuid4()}",
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="admin",
            ),
        )
        test_keys.append(key)

    # Create 2 keys with team_id=litellm-dashboard
    for i in range(2):
        key = await generate_key_fn(
            data=GenerateKeyRequest(
                team_id="litellm-dashboard",
                key_alias=f"dashboard_key_{i}.{uuid.uuid4()}",
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="admin",
            ),
        )
        test_keys.append(key)

    # Create 2 keys with a different team_id
    other_team_id = f"other_team_{uuid.uuid4()}"
    for i in range(2):
        key = await generate_key_fn(
            data=GenerateKeyRequest(
                team_id=other_team_id,
                key_alias=f"other_team_key_{i}.{uuid.uuid4()}",
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="admin",
            ),
        )
        test_keys.append(key)

    try:
        # Test 1: Get all keys with pagination (exclude litellm-dashboard)
        all_keys = []
        page = 1
        max_pages_to_check = 3  # Only check the first 3 pages

        while page <= max_pages_to_check:
            result = await _list_key_helper(
                prisma_client=prisma_client,
                size=100,
                page=page,
                user_id=None,
                team_id=None,
                key_alias=None,
                key_hash=None,
                return_full_object=True,
                organization_id=None,
            )

            all_keys.extend(result["keys"])

            if page >= result["total_pages"] or page >= max_pages_to_check:
                break
            page += 1

        # Verify results
        print(f"Total keys found: {len(all_keys)}")
        for key in all_keys:
            print(f"Key team_id: {key.team_id}, alias: {key.key_alias}")

        # Verify no litellm-dashboard keys are present
        dashboard_keys = [k for k in all_keys if k.team_id == "litellm-dashboard"]
        assert len(dashboard_keys) == 0, "Should not include litellm-dashboard keys"

        # Verify keys with team_id=None are included
        no_team_keys = [k for k in all_keys if k.team_id is None]
        assert (
            len(no_team_keys) > 0
        ), f"Expected more than 0 keys with no team, got {len(no_team_keys)}"

    finally:
        # Clean up test keys
        for key in test_keys:
            await delete_key_fn(
                data=KeyRequest(keys=[key.key]),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="admin",
                ),
            )


@pytest.mark.asyncio
@patch("litellm.proxy.management_endpoints.key_management_endpoints.get_team_object")
async def test_key_generate_always_db_team(mock_get_team_object):
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_fn,
    )

    setattr(litellm.proxy.proxy_server, "prisma_client", MagicMock())
    mock_get_team_object.return_value = None
    try:
        await generate_key_fn(
            data=GenerateKeyRequest(team_id="1234"),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="admin",
            ),
        )
    except Exception as e:
        print(f"Error: {e}")

    mock_get_team_object.assert_called_once()
    assert mock_get_team_object.call_args.kwargs["check_db_only"] == True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_model, should_pass",
    [
        ("gpt-4o", True),  # Should pass - exact match in aliases
        ("gpt-4o-team1", True),  # Should pass - team has access to this deployment
        ("gpt-4o-mini", False),  # Should fail - not in aliases
        ("o-3", False),  # Should fail - not in aliases
    ],
)
async def test_team_model_alias(prisma_client, requested_model, should_pass):
    """
    Test team model alias functionality:
    1. Create team with model alias = `{gpt-4o: gpt-4o-team1}`
    2. Generate key for that team with model = `gpt-4o`
    3. Verify chat completion request works with aliased model = `gpt-4o`
    """
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Create team with model alias
    team_id = f"test_team_{uuid.uuid4()}"
    await new_team(
        data=NewTeamRequest(
            team_id=team_id,
            team_alias=f"test_team_alias_{uuid.uuid4()}",
            models=["gpt-4o-team1"],
            model_aliases={"gpt-4o": "gpt-4o-team1"},
        ),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234", user_id="admin"
        ),
    )

    # Generate key for the team
    new_key = await generate_key_fn(
        data=GenerateKeyRequest(
            team_id=team_id,
            models=["gpt-4o-team1"],
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234", user_id="admin"
        ),
    )

    generated_key = new_key.key

    # Test chat completion request
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body():
        return_string = f'{{"model": "{requested_model}"}}'
        return return_string.encode()

    request.body = return_body

    if should_pass:
        # Verify the key works with the aliased model
        result = await user_api_key_auth(
            request=request, api_key=f"Bearer {generated_key}"
        )

        assert result.models == [
            "gpt-4o-team1"
        ], "Expected model list to contain aliased model"
        assert result.team_model_aliases == {
            "gpt-4o": "gpt-4o-team1"
        }, "Expected model aliases to be present"
    else:
        # Verify the key fails with non-aliased models
        with pytest.raises(Exception) as exc_info:
            await user_api_key_auth(request=request, api_key=f"Bearer {generated_key}")
        assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied
