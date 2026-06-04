"""
Tests for backend domain models.
"""

from datetime import datetime

import pytest

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.credentials import CreateCredentialItem, CredentialItem
from litellm.models.model import LiteLLM_ProxyModelTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.team import (
    LiteLLM_DeletedTeamTable,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
)
from litellm.models.user import LiteLLM_UserTable
from litellm.models.verification_token import (
    LiteLLM_DeletedVerificationToken,
    LiteLLM_VerificationToken,
)


class TestBudget:
    def test_budget_creation(self):
        budget = LiteLLM_BudgetTable(
            budget_id="test-budget-id",
            max_budget=100.0,
            soft_budget=80.0,
            tpm_limit=1000,
            rpm_limit=100,
            model_max_budget={"gpt-4": 50.0},
            budget_duration="monthly",
            allowed_models=["gpt-4"],
        )
        assert budget.budget_id == "test-budget-id"
        assert budget.max_budget == 100.0
        assert budget.soft_budget == 80.0
        assert budget.tpm_limit == 1000
        assert budget.rpm_limit == 100
        assert budget.model_max_budget == {"gpt-4": 50.0}
        assert budget.budget_duration == "monthly"
        assert budget.allowed_models == ["gpt-4"]

    def test_budget_defaults(self):
        budget = LiteLLM_BudgetTable()
        assert budget.budget_id is None
        assert budget.max_budget is None
        assert budget.allowed_models is None


class TestCredentials:
    def test_credentials_creation(self):
        creds = CredentialItem(
            credential_name="test-cred",
            credential_values={"api_key": "secret123"},
            credential_info={"provider": "openai"},
        )
        assert creds.credential_name == "test-cred"
        assert creds.credential_values["api_key"] == "secret123"
        assert creds.credential_info["provider"] == "openai"

    def test_create_credential_item_accepts_model_id(self):
        item = CreateCredentialItem(
            credential_name="from-model",
            credential_info={},
            model_id="model-123",
        )
        assert item.model_id == "model-123"
        assert item.credential_values is None

    def test_create_credential_item_requires_values_or_model_id(self):
        with pytest.raises(
            ValueError, match="Either credential_values or model_id must be set"
        ):
            CreateCredentialItem(credential_name="bad", credential_info={})


class TestModel:
    def test_model_creation(self):
        model = LiteLLM_ProxyModelTable(
            model_id="test-model-id",
            model_name="gpt-4",
            litellm_params={"model": "gpt-4", "api_key": "test"},
            model_info={"team_id": "team-123", "team_public_model_name": "my-gpt4"},
        )
        assert model.model_id == "test-model-id"
        assert model.model_name == "gpt-4"
        assert model.team_id == "team-123"
        assert model.team_public_model_name == "my-gpt4"

    def test_is_blocked(self):
        model_blocked = LiteLLM_ProxyModelTable(
            model_id="m1", model_name="test", litellm_params={}, blocked=True
        )
        model_unblocked = LiteLLM_ProxyModelTable(
            model_id="m2", model_name="test", litellm_params={}, blocked=False
        )
        assert model_blocked.is_blocked
        assert not model_unblocked.is_blocked

    def test_parses_json_string_fields(self):
        model = LiteLLM_ProxyModelTable(
            model_id="m1",
            model_name="gpt-4",
            litellm_params='{"model": "gpt-4"}',
            model_info='{"team_id": "t1"}',
        )
        assert model.litellm_params == {"model": "gpt-4"}
        assert model.model_info == {"team_id": "t1"}

    def test_team_helpers_none_when_no_model_info(self):
        model = LiteLLM_ProxyModelTable(
            model_id="m1", model_name="gpt-4", litellm_params={}, model_info=None
        )
        assert model.team_id is None
        assert model.team_public_model_name is None


class TestObjectPermission:
    def test_object_permission_creation(self):
        perm = LiteLLM_ObjectPermissionTable(
            object_permission_id="test-perm-id",
            mcp_servers=["server1", "server2"],
            vector_stores=["vs1"],
            agents=["agent1"],
            models=["gpt-4"],
            blocked_tools=["dangerous_tool"],
        )
        assert perm.object_permission_id == "test-perm-id"
        assert len(perm.mcp_servers) == 2
        assert perm.vector_stores == ["vs1"]
        assert perm.agents == ["agent1"]
        assert perm.models == ["gpt-4"]
        assert perm.blocked_tools == ["dangerous_tool"]

    def test_object_permission_tool_permissions(self):
        perm = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-tools",
            mcp_tool_permissions={"server1": ["tool1", "tool2"]},
        )
        assert perm.mcp_tool_permissions == {"server1": ["tool1", "tool2"]}


class TestOrganization:
    def test_organization_creation(self):
        org = LiteLLM_OrganizationTable(
            organization_id="org-123",
            organization_alias="My Org",
            budget_id="budget-123",
            models=["gpt-4", "claude-3"],
            spend=50.0,
            created_by="admin",
            updated_by="admin",
        )
        assert org.organization_id == "org-123"
        assert org.organization_alias == "My Org"
        assert len(org.models) == 2


class TestProject:
    def test_project_creation(self):
        project = LiteLLM_ProjectTable(
            project_id="proj-123",
            project_alias="My Project",
            team_id="team-123",
            blocked=False,
        )
        assert project.project_id == "proj-123"
        assert not project.is_blocked


class TestTeam:
    def test_team_creation(self):
        team = LiteLLM_TeamTable(
            team_id="team-123",
            team_alias="Engineering",
            admins=["user1"],
            members=["user2", "user3"],
            models=["gpt-4"],
            max_budget=1000.0,
            spend=100.0,
        )
        assert team.team_id == "team-123"
        assert team.team_alias == "Engineering"
        assert team.admins == ["user1"]
        assert team.members == ["user2", "user3"]
        assert team.models == ["gpt-4"]
        assert team.max_budget == 1000.0

    def test_members_with_roles_parsing(self):
        team = LiteLLM_TeamTable(
            team_id="t2",
            members_with_roles=[
                {"user_id": "user1", "role": "admin"},
                {"user_id": "user2", "role": "user"},
            ],
        )
        assert len(team.members_with_roles) == 2
        assert team.members_with_roles[0].user_id == "user1"
        assert team.members_with_roles[0].role == "admin"

    def test_members_with_roles_empty_dict_coerced(self):
        team = LiteLLM_TeamTable(team_id="t3", members_with_roles={})
        assert team.members_with_roles == []

    def test_json_string_fields_parsed(self):
        team = LiteLLM_TeamTable(
            team_id="t4",
            metadata='{"k": "v"}',
            model_max_budget='{"gpt-4": 5.0}',
        )
        assert team.metadata == {"k": "v"}
        assert team.model_max_budget == {"gpt-4": 5.0}

    def test_cached_team(self):
        cached = LiteLLM_TeamTableCachedObj(
            team_id="t1", last_refreshed_at=1234567890.0
        )
        assert cached.last_refreshed_at == 1234567890.0

    def test_deleted_team(self):
        deleted = LiteLLM_DeletedTeamTable(
            team_id="t1",
            deleted_by="admin",
            deleted_at=datetime.utcnow(),
        )
        assert deleted.deleted_by == "admin"
        assert deleted.deleted_at is not None


class TestUser:
    def test_user_creation(self):
        user = LiteLLM_UserTable(
            user_id="user-123",
            user_email="test@example.com",
            teams=["team1", "team2"],
            max_budget=100.0,
            spend=25.0,
        )
        assert user.user_id == "user-123"
        assert user.user_email == "test@example.com"
        assert len(user.teams) == 2

    def test_is_over_budget(self):
        user = LiteLLM_UserTable(user_id="u1", max_budget=100.0, spend=150.0)
        user_no_budget = LiteLLM_UserTable(user_id="u2", spend=1000.0)

        assert user.is_over_budget()
        assert not user_no_budget.is_over_budget()

    def test_has_model_access(self):
        user_with_models = LiteLLM_UserTable(user_id="u1", models=["gpt-4"])
        user_no_models = LiteLLM_UserTable(user_id="u2", models=[])

        assert user_with_models.has_model_access("gpt-4")
        assert not user_with_models.has_model_access("gpt-3")
        assert user_no_models.has_model_access("any-model")


class TestVerificationToken:
    def test_verification_token_creation(self):
        token = LiteLLM_VerificationToken(
            token="sk-test123",
            key_name="Test Key",
            user_id="user-123",
            team_id="team-123",
            max_budget=100.0,
            spend=25.0,
            models=["gpt-4"],
            blocked=True,
            allowed_routes=["/chat/completions"],
        )
        assert token.token == "sk-test123"
        assert token.key_name == "Test Key"
        assert token.user_id == "user-123"
        assert token.team_id == "team-123"
        assert token.blocked is True
        assert token.models == ["gpt-4"]
        assert token.allowed_routes == ["/chat/completions"]

    def test_expires_accepts_string_and_datetime(self):
        as_str = LiteLLM_VerificationToken(token="t1", expires="2024-12-31T23:59:59Z")
        as_dt = LiteLLM_VerificationToken(token="t2", expires=datetime.utcnow())
        assert as_str.expires == "2024-12-31T23:59:59Z"
        assert isinstance(as_dt.expires, datetime)

    def test_deleted_verification_token(self):
        deleted = LiteLLM_DeletedVerificationToken(
            token="t1",
            deleted_by="admin",
            deleted_at=datetime.utcnow(),
        )
        assert deleted.deleted_by == "admin"
        assert deleted.deleted_at is not None
        assert deleted.token == "t1"
