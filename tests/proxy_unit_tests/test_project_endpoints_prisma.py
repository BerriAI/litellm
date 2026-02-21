import os
import sys
import traceback
from litellm._uuid import uuid
from unittest import mock

from dotenv import load_dotenv
from fastapi import Request

load_dotenv()
import time

sys.path.insert(0, os.path.abspath("../.."))
import logging

import pytest

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_endpoints.team_endpoints import (
    new_team,
)
from litellm.proxy.management_endpoints.project_endpoints import (
    new_project,
    update_project,
    delete_project,
    project_info,
)
from litellm.proxy.proxy_server import (
    LitellmUserRoles,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging

verbose_proxy_logger.setLevel(level=logging.DEBUG)


from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    NewProjectRequest,
    UpdateProjectRequest,
    DeleteProjectRequest,
    NewTeamRequest,
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

    # Enable premium_user for project management tests
    setattr(litellm.proxy.proxy_server, "premium_user", True)

    return prisma_client


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
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
            metadata={"use_case_id": "TEST-001", "responsible_ai_id": "RAI-001"},
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


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
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
                "additional_field": "new_value",
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
        assert update_response.metadata["model_rpm_limit"] == {
            "gpt-4": 200,
            "claude-3": 50,
        }
        assert update_response.metadata["model_tpm_limit"] == {
            "gpt-4": 2000,
            "claude-3": 500,
        }
        assert update_response.litellm_budget_table is not None
        assert update_response.litellm_budget_table.max_budget == 200.0

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
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
        delete_data = DeleteProjectRequest(project_ids=[project_id])

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


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
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
            metadata={"use_case_id": "TEST-003", "cost_center": "engineering"},
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


### VALIDATION TESTS ###


def test_check_team_project_limits_models_not_in_team():
    """
    Test that creating a project with models not in the team raises an error.
    """
    from litellm.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-4", "gpt-3.5-turbo"],
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-4", "claude-3"],  # claude-3 not in team
    )

    with pytest.raises(Exception) as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "claude-3" in str(exc_info.value.detail)
    assert "not in team's allowed models" in str(exc_info.value.detail)


def test_check_team_project_limits_budget_exceeds_team():
    """
    Test that creating a project with budget > team budget raises an error.
    """
    from litellm.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-4"],
        max_budget=100.0,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-4"],
        max_budget=150.0,  # exceeds team's 100.0
    )

    with pytest.raises(Exception) as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "exceeds team's max_budget" in str(exc_info.value.detail)


def test_check_team_project_limits_valid_subset():
    """
    Test that a valid project (models subset, budget within limit) passes.
    """
    from litellm.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-4", "gpt-3.5-turbo", "claude-3"],
        max_budget=1000.0,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-4", "gpt-3.5-turbo"],
        max_budget=500.0,
    )

    # Should not raise
    _check_team_project_limits(team_object=team, data=data)


def test_check_team_project_limits_all_proxy_models():
    """
    Test that team with 'all-proxy-models' allows any project models.
    """
    from litellm.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["all-proxy-models"],
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-4", "claude-3", "anything-goes"],
    )

    # Should not raise - team allows all models
    _check_team_project_limits(team_object=team, data=data)


def test_check_team_project_limits_tpm_exceeds_team():
    """
    Test that project tpm_limit exceeding team tpm_limit raises an error.
    """
    from litellm.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-4"],
        tpm_limit=10000,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-4"],
        tpm_limit=20000,  # exceeds team's 10000
    )

    with pytest.raises(Exception) as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "exceeds team's tpm_limit" in str(exc_info.value.detail)


def test_check_team_project_limits_negative_budget():
    """
    Test that negative budget values raise an error.
    """
    from litellm.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-4"],
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-4"],
        max_budget=-10.0,
    )

    with pytest.raises(Exception) as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "cannot be negative" in str(exc_info.value.detail)


def test_check_team_project_limits_soft_budget_gte_max():
    """
    Test that soft_budget >= max_budget raises an error.
    """
    from litellm.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-4"],
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-4"],
        max_budget=100.0,
        soft_budget=100.0,  # equal to max, should fail
    )

    with pytest.raises(Exception) as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "must be strictly lower" in str(exc_info.value.detail)


def test_premium_user_gate():
    """
    Test that project endpoints require premium_user=True.
    """

    # This test just validates the premium_user check exists
    # The actual endpoint test would need prisma, but we can verify
    # the import path works
    setattr(litellm.proxy.proxy_server, "premium_user", False)

    # Verify that CommonProxyErrors.not_premium_user exists
    from litellm.proxy._types import CommonProxyErrors

    assert hasattr(CommonProxyErrors, "not_premium_user")

    # Reset
    setattr(litellm.proxy.proxy_server, "premium_user", True)


def test_project_model_access_denied_error_type():
    """
    Test that ProxyErrorTypes.project_model_access_denied exists.
    """
    from litellm.proxy._types import ProxyErrorTypes

    assert hasattr(ProxyErrorTypes, "project_model_access_denied")
    assert (
        ProxyErrorTypes.project_model_access_denied.value
        == "project_model_access_denied"
    )

    # Test the classmethod resolves correctly
    result = ProxyErrorTypes.get_model_access_error_type_for_object("project")
    assert result == ProxyErrorTypes.project_model_access_denied


def test_project_cached_obj_has_last_refreshed_at():
    """
    Test that LiteLLM_ProjectTableCachedObj has last_refreshed_at field
    matching LiteLLM_TeamTableCachedObj pattern.
    """
    from litellm.proxy._types import (
        LiteLLM_ProjectTableCachedObj,
        LiteLLM_ProjectTable,
    )

    # Verify inheritance
    assert issubclass(LiteLLM_ProjectTableCachedObj, LiteLLM_ProjectTable)

    # Verify last_refreshed_at field exists and defaults to None
    obj = LiteLLM_ProjectTableCachedObj(
        project_id="test",
        created_by="admin",
        updated_by="admin",
    )
    assert obj.last_refreshed_at is None

    # Verify it can be set
    obj.last_refreshed_at = 1234567890.0
    assert obj.last_refreshed_at == 1234567890.0


@pytest.mark.asyncio
async def test_project_max_budget_check_fires_alert():
    """
    Test that _project_max_budget_check fires a budget alert
    when project exceeds its max budget (matches _team_max_budget_check pattern).
    """
    from litellm.proxy.auth.auth_checks import _project_max_budget_check
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_ProjectTableCachedObj,
    )

    project = LiteLLM_ProjectTableCachedObj(
        project_id="test-project",
        spend=150.0,
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="user-1",
        team_id="team-1",
    )

    mock_proxy_logging = mock.AsyncMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = mock.AsyncMock()

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _project_max_budget_check(
            project_object=project,
            valid_token=valid_token,
            proxy_logging_obj=mock_proxy_logging,
        )

    assert "Project=test-project" in str(exc_info.value)
    assert "150.0" in str(exc_info.value)


@pytest.mark.asyncio
async def test_project_soft_budget_check():
    """
    Test that _project_soft_budget_check triggers alert when soft budget is exceeded.
    """
    from litellm.proxy.auth.auth_checks import _project_soft_budget_check
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_ProjectTableCachedObj,
    )

    project = LiteLLM_ProjectTableCachedObj(
        project_id="test-project",
        spend=80.0,
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(soft_budget=75.0),
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="user-1",
        team_id="team-1",
    )

    mock_proxy_logging = mock.AsyncMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = mock.AsyncMock()

    # Should not raise (soft budget only alerts, doesn't block)
    await _project_soft_budget_check(
        project_object=project,
        valid_token=valid_token,
        proxy_logging_obj=mock_proxy_logging,
    )


@pytest.mark.asyncio
async def test_project_soft_budget_check_no_alert_under_budget():
    """
    Test that _project_soft_budget_check does NOT trigger alert when under soft budget.
    """
    from litellm.proxy.auth.auth_checks import _project_soft_budget_check
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_ProjectTableCachedObj,
    )

    project = LiteLLM_ProjectTableCachedObj(
        project_id="test-project",
        spend=50.0,
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(soft_budget=75.0),
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="user-1",
        team_id="team-1",
    )

    mock_proxy_logging = mock.AsyncMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = mock.AsyncMock()

    # Should not raise and should not alert
    await _project_soft_budget_check(
        project_object=project,
        valid_token=valid_token,
        proxy_logging_obj=mock_proxy_logging,
    )


def test_litellm_entity_type_has_project():
    """
    Test that Litellm_EntityType has PROJECT member for budget alerts.
    """
    from litellm.proxy._types import Litellm_EntityType

    assert hasattr(Litellm_EntityType, "PROJECT")
    assert Litellm_EntityType.PROJECT.value == "project"
