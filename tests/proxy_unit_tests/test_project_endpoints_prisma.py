import os
import sys
import traceback
from litellm._uuid import uuid
from datetime import datetime, timezone
from unittest import mock

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute
import httpx

load_dotenv()
import io
import os
import time

sys.path.insert(
    0, os.path.abspath("../..")
)
import asyncio
import logging

import pytest

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_endpoints.team_endpoints import (
    new_team,
    team_info,
    update_team,
)
from litellm.proxy.management_endpoints.project_endpoints import (
    new_project,
    update_project,
    delete_project,
    project_info,
)
from litellm.proxy.proxy_server import (
    LitellmUserRoles,
    user_api_key_auth,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from starlette.datastructures import URL

from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    NewProjectRequest,
    UpdateProjectRequest,
    DeleteProjectRequest,
    NewTeamRequest,
    ProxyException,
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


@pytest.mark.asyncio
async def test_new_project(prisma_client):
    """
    Test creating a new project with budget, models, and metadata.
    """
    try:
        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

        await litellm.proxy.proxy_server.prisma_client.connect()

        # Create a team first
        _team_id = f"project-test-team_{uuid.uuid4()}"
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        # Create a project
        project_data = NewProjectRequest(
            project_alias="test-project",
            description="Test project for unit testing",
            team_id=_team_id,
            metadata={
                "use_case_id": "TEST-001",
                "responsible_ai_id": "RAI-001"
            },
            models=["gpt-4", "gpt-3.5-turbo"],
            max_budget=100.0,
            model_rpm_limit={"gpt-4": 100},
            model_tpm_limit={"gpt-4": 1000},
        )

        response = await new_project(
            data=project_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        print("New project response:", response)
        
        # Assertions
        assert response.project_id is not None
        assert response.project_alias == "test-project"
        assert response.description == "Test project for unit testing"
        assert response.team_id == _team_id
        assert response.models == ["gpt-4", "gpt-3.5-turbo"]
        # model_rpm_limit and model_tpm_limit are stored in metadata
        assert response.metadata["use_case_id"] == "TEST-001"
        assert response.metadata["responsible_ai_id"] == "RAI-001"
        assert response.metadata["model_rpm_limit"] == {"gpt-4": 100}
        assert response.metadata["model_tpm_limit"] == {"gpt-4": 1000}
        assert response.litellm_budget_table is not None
        assert response.litellm_budget_table.max_budget == 100.0

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio
async def test_update_project(prisma_client):
    """
    Test updating an existing project's budget, models, and metadata.
    """
    try:
        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

        await litellm.proxy.proxy_server.prisma_client.connect()

        # Create a team first
        _team_id = f"project-test-team_{uuid.uuid4()}"
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        # Create a project
        project_data = NewProjectRequest(
            project_alias="test-project-update",
            description="Original description",
            team_id=_team_id,
            metadata={
                "use_case_id": "TEST-002",
            },
            models=["gpt-4"],
            max_budget=50.0,
        )

        create_response = await new_project(
            data=project_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        print("Created project:", create_response)
        project_id = create_response.project_id

        # Update the project
        update_data = UpdateProjectRequest(
            project_id=project_id,
            project_alias="test-project-updated",
            description="Updated description",
            metadata={
                "use_case_id": "TEST-002-UPDATED",
                "additional_field": "new_value"
            },
            models=["gpt-4", "gpt-3.5-turbo", "claude-3"],
            max_budget=200.0,
            model_rpm_limit={"gpt-4": 200, "claude-3": 50},
            model_tpm_limit={"gpt-4": 2000, "claude-3": 500},
        )

        update_response = await update_project(
            data=update_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        print("Updated project response:", update_response)

        # Assertions
        assert update_response.project_id == project_id
        assert update_response.project_alias == "test-project-updated"
        assert update_response.description == "Updated description"
        assert update_response.models == ["gpt-4", "gpt-3.5-turbo", "claude-3"]
        # model_rpm_limit and model_tpm_limit are stored in metadata
        assert update_response.metadata["use_case_id"] == "TEST-002-UPDATED"
        assert update_response.metadata["additional_field"] == "new_value"
        assert update_response.metadata["model_rpm_limit"] == {"gpt-4": 200, "claude-3": 50}
        assert update_response.metadata["model_tpm_limit"] == {"gpt-4": 2000, "claude-3": 500}
        assert update_response.litellm_budget_table is not None
        assert update_response.litellm_budget_table.max_budget == 200.0

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio
async def test_delete_project(prisma_client):
    """
    Test deleting a project.
    """
    try:
        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

        await litellm.proxy.proxy_server.prisma_client.connect()

        # Create a team first
        _team_id = f"project-test-team_{uuid.uuid4()}"
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        # Create a project
        project_data = NewProjectRequest(
            project_alias="test-project-delete",
            team_id=_team_id,
            models=["gpt-4"],
            max_budget=50.0,
        )

        create_response = await new_project(
            data=project_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        print("Created project:", create_response)
        project_id = create_response.project_id

        # Delete the project
        delete_data = DeleteProjectRequest(
            project_ids=[project_id]
        )

        delete_response = await delete_project(
            data=delete_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        print("Delete project response:", delete_response)

        # Assertions - delete_project returns a list of deleted project objects
        assert isinstance(delete_response, list)
        assert len(delete_response) == 1
        assert delete_response[0].project_id == project_id

        # Try to get info on the deleted project - should fail or return None
        try:
            await project_info(
                project_id=project_id,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
            pytest.fail("Expected to fail when fetching deleted project")
        except Exception as e:
            print("Expected error when fetching deleted project:", e)
            # This is expected behavior

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio
async def test_project_info(prisma_client):
    """
    Test getting project info.
    """
    try:
        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

        await litellm.proxy.proxy_server.prisma_client.connect()

        # Create a team first
        _team_id = f"project-test-team_{uuid.uuid4()}"
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        # Create a project
        project_data = NewProjectRequest(
            project_alias="test-project-info",
            description="Test project info endpoint",
            team_id=_team_id,
            metadata={
                "use_case_id": "TEST-003",
                "cost_center": "engineering"
            },
            models=["gpt-4", "claude-3"],
            max_budget=150.0,
            model_rpm_limit={"gpt-4": 150},
            model_tpm_limit={"gpt-4": 1500},
        )

        create_response = await new_project(
            data=project_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        print("Created project:", create_response)
        project_id = create_response.project_id

        # Get project info
        info_response = await project_info(
            project_id=project_id,
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        print("Project info response:", info_response)

        # Assertions - project_info returns the project object directly
        assert info_response.project_id == project_id
        assert info_response.project_alias == "test-project-info"
        assert info_response.description == "Test project info endpoint"
        assert info_response.team_id == _team_id
        assert info_response.models == ["gpt-4", "claude-3"]
        # model_rpm_limit and model_tpm_limit are stored in metadata
        assert info_response.metadata["use_case_id"] == "TEST-003"
        assert info_response.metadata["cost_center"] == "engineering"
        assert info_response.metadata["model_rpm_limit"] == {"gpt-4": 150}
        assert info_response.metadata["model_tpm_limit"] == {"gpt-4": 1500}
        assert info_response.litellm_budget_table is not None
        assert info_response.litellm_budget_table.max_budget == 150.0

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")

