"""
RBAC tests
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
from unittest.mock import MagicMock
import pytest

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.auth_checks import get_user_object
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_key_fn,
    generate_key_fn,
    generate_key_helper_fn,
    info_key_fn,
    regenerate_key_fn,
    update_key_fn,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
from litellm.proxy.management_endpoints.organization_endpoints import (
    new_organization,
    organization_member_add,
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
from starlette.datastructures import URL

from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token, update_spend

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from starlette.datastructures import URL

from litellm.caching.caching import DualCache
from litellm.proxy._types import *

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


"""
RBAC Tests

1. Add a user to an organization
    - test 1 - if organization_id does exist expect to create a new user and user, organization relation

2. org admin creates team in his org → success 

3. org admin adds new internal user to his org → success 

4. org admin creates team and internal user not in his org → fail both
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_role",
    [
        LitellmUserRoles.ORG_ADMIN,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    ],
)
async def test_create_new_user_in_organization(prisma_client, user_role):
    """

    Add a member to an organization and assert the user object is created with the correct organization memberships / roles
    """
    master_key = "sk-1234"
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    setattr(litellm.proxy.proxy_server, "llm_router", MagicMock())

    await litellm.proxy.proxy_server.prisma_client.connect()

    created_user_id = f"new-user-{uuid.uuid4()}"

    response = await new_organization(
        data=NewOrganizationRequest(
            organization_alias=f"new-org-{uuid.uuid4()}",
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_id=created_user_id,
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
    )

    org_id = response.organization_id

    response = await organization_member_add(
        data=OrganizationMemberAddRequest(
            organization_id=org_id,
            member=OrgMember(role=user_role, user_id=created_user_id),
        ),
        http_request=None,
    )

    print("new user response", response)

    # call get_user_object

    user_object = await get_user_object(
        user_id=created_user_id,
        prisma_client=prisma_client,
        user_api_key_cache=DualCache(),
        user_id_upsert=False,
    )

    print("user object", user_object)

    assert user_object.organization_memberships is not None

    _membership = user_object.organization_memberships[0]

    assert _membership.user_id == created_user_id
    assert _membership.organization_id == org_id

    if user_role != None:
        assert _membership.user_role == user_role
    else:
        assert _membership.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_org_admin_create_team_permissions(prisma_client):
    """
    Create a new org admin

    org admin creates a new team in their org -> success
    """
    import json

    master_key = "sk-1234"
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    setattr(litellm.proxy.proxy_server, "llm_router", MagicMock())

    await litellm.proxy.proxy_server.prisma_client.connect()

    response = await new_organization(
        data=NewOrganizationRequest(
            organization_alias=f"new-org-{uuid.uuid4()}",
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
    )

    org_id = response.organization_id
    created_user_id = f"new-user-{uuid.uuid4()}"
    response = await organization_member_add(
        data=OrganizationMemberAddRequest(
            organization_id=org_id,
            member=OrgMember(role=LitellmUserRoles.ORG_ADMIN, user_id=created_user_id),
        ),
        http_request=None,
    )

    # create key with the response["user_id"]
    # proxy admin will generate key for org admin
    _new_key = await generate_key_fn(
        data=GenerateKeyRequest(user_id=created_user_id),
        user_api_key_dict=UserAPIKeyAuth(user_id=created_user_id),
    )

    new_key = _new_key.key

    print("user api key auth response", response)

    # Create /team/new request -> expect auth to pass
    request = Request(scope={"type": "http"})
    request._url = URL(url="/team/new")

    async def return_body():
        body = {"organization_id": org_id}
        return bytes(json.dumps(body), "utf-8")

    request.body = return_body
    response = await user_api_key_auth(request=request, api_key="Bearer " + new_key)

    # after auth - actually create team now
    response = await new_team(
        data=NewTeamRequest(
            organization_id=org_id,
        ),
        http_request=request,
        user_api_key_dict=UserAPIKeyAuth(
            user_id=response.user_id,
        ),
    )

    print("response from new team")


@pytest.mark.asyncio
async def test_org_admin_create_user_permissions(prisma_client):
    """
    1. Create a new org admin

    2. org admin adds a new member to their org -> success (using using /organization/member_add)

    """
    import json

    master_key = "sk-1234"
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    setattr(litellm.proxy.proxy_server, "llm_router", MagicMock())

    await litellm.proxy.proxy_server.prisma_client.connect()

    # create new org
    response = await new_organization(
        data=NewOrganizationRequest(
            organization_alias=f"new-org-{uuid.uuid4()}",
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
    )
    # Create Org Admin
    org_id = response.organization_id
    created_user_id = f"new-user-{uuid.uuid4()}"
    response = await organization_member_add(
        data=OrganizationMemberAddRequest(
            organization_id=org_id,
            member=OrgMember(role=LitellmUserRoles.ORG_ADMIN, user_id=created_user_id),
        ),
        http_request=None,
    )

    # create key with for Org Admin
    _new_key = await generate_key_fn(
        data=GenerateKeyRequest(user_id=created_user_id),
        user_api_key_dict=UserAPIKeyAuth(user_id=created_user_id),
    )

    new_key = _new_key.key

    print("user api key auth response", response)

    # Create /organization/member_add request -> expect auth to pass
    request = Request(scope={"type": "http"})
    request._url = URL(url="/organization/member_add")

    async def return_body():
        body = {"organization_id": org_id}
        return bytes(json.dumps(body), "utf-8")

    request.body = return_body
    response = await user_api_key_auth(request=request, api_key="Bearer " + new_key)

    # after auth - actually actually add new user to organization
    new_internal_user_for_org = f"new-org-user-{uuid.uuid4()}"
    response = await organization_member_add(
        data=OrganizationMemberAddRequest(
            organization_id=org_id,
            member=OrgMember(
                role=LitellmUserRoles.INTERNAL_USER, user_id=new_internal_user_for_org
            ),
        ),
        http_request=request,
    )

    print("response from new team")


@pytest.mark.asyncio
async def test_org_admin_create_user_team_wrong_org_permissions(prisma_client):
    """
    Create a new org admin

    org admin creates a new user and new team in orgs they are not part of -> expect error
    """
    import json

    master_key = "sk-1234"
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    setattr(litellm.proxy.proxy_server, "llm_router", MagicMock())

    await litellm.proxy.proxy_server.prisma_client.connect()
    created_user_id = f"new-user-{uuid.uuid4()}"
    response = await new_organization(
        data=NewOrganizationRequest(
            organization_alias=f"new-org-{uuid.uuid4()}",
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
    )

    response2 = await new_organization(
        data=NewOrganizationRequest(
            organization_alias=f"new-org-{uuid.uuid4()}",
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
    )

    org1_id = response.organization_id  # has an admin

    org2_id = response2.organization_id  # does not have an org admin

    # Create Org Admin for Org1
    created_user_id = f"new-user-{uuid.uuid4()}"
    response = await organization_member_add(
        data=OrganizationMemberAddRequest(
            organization_id=org1_id,
            member=OrgMember(role=LitellmUserRoles.ORG_ADMIN, user_id=created_user_id),
        ),
        http_request=None,
    )

    _new_key = await generate_key_fn(
        data=GenerateKeyRequest(
            user_id=created_user_id,
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.ORG_ADMIN,
            user_id=created_user_id,
        ),
    )

    new_key = _new_key.key

    print("user api key auth response", response)

    # Add a new request in organization=org_without_admins -> expect fail (organization/member_add)
    request = Request(scope={"type": "http"})
    request._url = URL(url="/organization/member_add")

    async def return_body():
        body = {"organization_id": org2_id}
        return bytes(json.dumps(body), "utf-8")

    request.body = return_body

    try:
        response = await user_api_key_auth(request=request, api_key="Bearer " + new_key)
        pytest.fail(
            f"This should have failed!. creating a user in an org without admins"
        )
    except Exception as e:
        print("got exception", e)
        print("exception.message", e.message)
        assert (
            "You do not have a role within the selected organization. Passed organization_id"
            in e.message
        )

    # Create /team/new request in organization=org_without_admins -> expect fail
    request = Request(scope={"type": "http"})
    request._url = URL(url="/team/new")

    async def return_body():
        body = {"organization_id": org2_id}
        return bytes(json.dumps(body), "utf-8")

    request.body = return_body

    try:
        response = await user_api_key_auth(request=request, api_key="Bearer " + new_key)
        pytest.fail(
            f"This should have failed!. Org Admin creating a team in an org where they are not an admin"
        )
    except Exception as e:
        print("got exception", e)
        print("exception.message", e.message)
        assert (
            "You do not have the required role to call" in e.message
            and org2_id in e.message
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route, user_role, expected_result",
    [
        # Proxy Admin checks
        ("/global/spend/logs", LitellmUserRoles.PROXY_ADMIN, True),
        ("/key/delete", LitellmUserRoles.PROXY_ADMIN, True),
        ("/key/generate", LitellmUserRoles.PROXY_ADMIN, True),
        ("/key/regenerate", LitellmUserRoles.PROXY_ADMIN, True),
        # # Internal User checks - allowed routes
        ("/global/spend/logs", LitellmUserRoles.INTERNAL_USER, True),
        ("/key/delete", LitellmUserRoles.INTERNAL_USER, True),
        ("/key/generate", LitellmUserRoles.INTERNAL_USER, True),
        ("/key/82akk800000000jjsk/regenerate", LitellmUserRoles.INTERNAL_USER, True),
        # Internal User Viewer
        ("/key/generate", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, False),
        (
            "/key/82akk800000000jjsk/regenerate",
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            False,
        ),
        ("/key/delete", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, False),
        ("/team/new", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, False),
        ("/team/delete", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, False),
        ("/team/update", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, False),
        # Proxy Admin Viewer
        ("/global/spend/logs", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, True),
        ("/key/delete", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, False),
        ("/key/generate", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, False),
        (
            "/key/82akk800000000jjsk/regenerate",
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            False,
        ),
        ("/team/new", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, False),
        ("/team/delete", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, False),
        ("/team/update", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, False),
        # Internal User checks - disallowed routes
        ("/organization/member_add", LitellmUserRoles.INTERNAL_USER, False),
    ],
)
async def test_user_role_permissions(prisma_client, route, user_role, expected_result):
    """Test user role based permissions for different routes"""
    try:
        # Setup
        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
        await litellm.proxy.proxy_server.prisma_client.connect()

        # Admin - admin creates a new user
        user_api_key_dict = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        )

        request = NewUserRequest(user_role=user_role)
        new_user_response = await new_user(request, user_api_key_dict=user_api_key_dict)
        user_id = new_user_response.user_id

        # Generate key for new user with team_id="litellm-dashboard"
        key_response = await generate_key_fn(
            data=GenerateKeyRequest(user_id=user_id, team_id="litellm-dashboard"),
            user_api_key_dict=user_api_key_dict,
        )
        generated_key = key_response.key
        bearer_token = "Bearer " + generated_key

        # Create request with route
        request = Request(scope={"type": "http"})
        request._url = URL(url=route)

        # Test authorization
        if expected_result is True:
            # Should pass without error
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print(f"Auth passed as expected for {route} with role {user_role}")
        else:
            # Should raise an error
            with pytest.raises(Exception) as exc_info:
                await user_api_key_auth(request=request, api_key=bearer_token)
            print(f"Auth failed as expected for {route} with role {user_role}")
            print(f"Error message: {str(exc_info.value)}")

    except Exception as e:
        if expected_result:
            pytest.fail(f"Expected success but got exception: {str(e)}")
        else:
            print(f"Got expected exception: {str(e)}")
