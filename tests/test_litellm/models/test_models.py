"""
Tests for backend domain models.
"""

from datetime import datetime, timedelta

import pytest

from litellm.models.budget import Budget
from litellm.models.credentials import CreateCredentialItem, CredentialItem
from litellm.models.model import Model
from litellm.models.object_permission import ObjectPermission
from litellm.models.organization import Organization
from litellm.models.project import Project
from litellm.models.team import CachedTeam, DeletedTeam, Team, TeamMember
from litellm.models.user import User
from litellm.models.verification_token import (
    DeletedVerificationToken,
    VerificationToken,
)


class TestBudget:
    def test_budget_creation(self):
        budget = Budget(
            budget_id="test-budget-id",
            max_budget=100.0,
            soft_budget=80.0,
            tpm_limit=1000,
            rpm_limit=100,
        )
        assert budget.budget_id == "test-budget-id"
        assert budget.max_budget == 100.0
        assert budget.soft_budget == 80.0
        assert budget.tpm_limit == 1000
        assert budget.rpm_limit == 100

    def test_is_over_budget(self):
        budget = Budget(max_budget=100.0)
        assert not budget.is_over_budget(50.0)
        assert budget.is_over_budget(100.0)
        assert budget.is_over_budget(150.0)

    def test_is_over_budget_no_limit(self):
        budget = Budget()
        assert not budget.is_over_budget(1000000.0)

    def test_is_approaching_soft_budget(self):
        budget = Budget(soft_budget=80.0)
        assert not budget.is_approaching_soft_budget(50.0)
        assert budget.is_approaching_soft_budget(80.0)
        assert budget.is_approaching_soft_budget(100.0)

    def test_should_reset_budget(self):
        past_time = datetime.utcnow() - timedelta(hours=1)
        future_time = datetime.utcnow() + timedelta(hours=1)

        budget_past = Budget(budget_reset_at=past_time)
        budget_future = Budget(budget_reset_at=future_time)
        budget_none = Budget()

        assert budget_past.should_reset_budget()
        assert not budget_future.should_reset_budget()
        assert not budget_none.should_reset_budget()


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
        model = Model(
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
        model_blocked = Model(
            model_name="test", litellm_params={}, blocked=True
        )
        model_unblocked = Model(
            model_name="test", litellm_params={}, blocked=False
        )
        assert model_blocked.is_blocked
        assert not model_unblocked.is_blocked


class TestObjectPermission:
    def test_object_permission_creation(self):
        perm = ObjectPermission(
            object_permission_id="test-perm-id",
            mcp_servers=["server1", "server2"],
            vector_stores=["vs1"],
            agents=["agent1"],
            blocked_tools=["dangerous_tool"],
        )
        assert perm.object_permission_id == "test-perm-id"
        assert len(perm.mcp_servers) == 2
        assert perm.has_mcp_server_access("server1")
        assert not perm.has_mcp_server_access("server3")
        assert perm.has_vector_store_access("vs1")
        assert perm.has_agent_access("agent1")
        assert perm.is_tool_blocked("dangerous_tool")
        assert not perm.is_tool_blocked("safe_tool")

    def test_get_allowed_tools_for_server(self):
        perm = ObjectPermission(
            mcp_tool_permissions={"server1": ["tool1", "tool2"]}
        )
        assert perm.get_allowed_tools_for_server("server1") == ["tool1", "tool2"]
        assert perm.get_allowed_tools_for_server("server2") is None


class TestOrganization:
    def test_organization_creation(self):
        org = Organization(
            organization_id="org-123",
            organization_alias="My Org",
            budget_id="budget-123",
            models=["gpt-4", "claude-3"],
            spend=50.0,
        )
        assert org.organization_id == "org-123"
        assert org.organization_alias == "My Org"
        assert len(org.models) == 2


class TestProject:
    def test_project_creation(self):
        project = Project(
            project_id="proj-123",
            project_alias="My Project",
            team_id="team-123",
            blocked=False,
        )
        assert project.project_id == "proj-123"
        assert not project.is_blocked


class TestTeam:
    def test_team_creation(self):
        team = Team(
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
        assert team.is_admin("user1")
        assert not team.is_admin("user2")
        assert team.is_member("user2")
        assert team.is_member("user1")
        assert not team.is_member("user4")

    def test_has_model_access(self):
        team_with_models = Team(team_id="t1", models=["gpt-4", "claude-3"])
        team_no_models = Team(team_id="t2", models=[])

        assert team_with_models.has_model_access("gpt-4")
        assert not team_with_models.has_model_access("gpt-3")
        assert team_no_models.has_model_access("any-model")

    def test_is_over_budget(self):
        team = Team(team_id="t1", max_budget=100.0, spend=150.0)
        team_no_budget = Team(team_id="t2", spend=1000.0)

        assert team.is_over_budget()
        assert not team_no_budget.is_over_budget()

    def test_members_with_roles_parsing(self):
        team_dict = Team(
            team_id="t1",
            members_with_roles={"user1": "admin", "user2": "user"},
        )
        assert len(team_dict.members_with_roles) == 2

        team_list = Team(
            team_id="t2",
            members_with_roles=[
                {"user_id": "user1", "role": "admin"},
                {"user_id": "user2", "role": "user"},
            ],
        )
        assert len(team_list.members_with_roles) == 2

    def test_cached_team(self):
        cached = CachedTeam(
            team_id="t1", last_refreshed_at=1234567890.0
        )
        assert cached.last_refreshed_at == 1234567890.0

    def test_deleted_team(self):
        deleted = DeletedTeam(
            team_id="t1",
            deleted_by="admin",
            deleted_at=datetime.utcnow(),
        )
        assert deleted.deleted_by == "admin"
        assert deleted.deleted_at is not None


class TestUser:
    def test_user_creation(self):
        user = User(
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
        user = User(user_id="u1", max_budget=100.0, spend=150.0)
        user_no_budget = User(user_id="u2", spend=1000.0)

        assert user.is_over_budget()
        assert not user_no_budget.is_over_budget()

    def test_has_model_access(self):
        user_with_models = User(user_id="u1", models=["gpt-4"])
        user_no_models = User(user_id="u2", models=[])

        assert user_with_models.has_model_access("gpt-4")
        assert not user_with_models.has_model_access("gpt-3")
        assert user_no_models.has_model_access("any-model")


class TestVerificationToken:
    def test_verification_token_creation(self):
        token = VerificationToken(
            token="sk-test123",
            key_name="Test Key",
            user_id="user-123",
            team_id="team-123",
            max_budget=100.0,
            spend=25.0,
            models=["gpt-4"],
        )
        assert token.token == "sk-test123"
        assert token.key_name == "Test Key"
        assert token.user_id == "user-123"

    def test_is_blocked(self):
        blocked_token = VerificationToken(token="t1", blocked=True)
        unblocked_token = VerificationToken(token="t2", blocked=False)
        none_token = VerificationToken(token="t3", blocked=None)

        assert blocked_token.is_blocked
        assert not unblocked_token.is_blocked
        assert not none_token.is_blocked

    def test_is_expired(self):
        past = datetime.utcnow() - timedelta(hours=1)
        future = datetime.utcnow() + timedelta(hours=1)

        expired_token = VerificationToken(token="t1", expires=past)
        valid_token = VerificationToken(token="t2", expires=future)
        no_expiry = VerificationToken(token="t3")

        assert expired_token.is_expired
        assert not valid_token.is_expired
        assert not no_expiry.is_expired

    def test_is_over_budget(self):
        over_budget = VerificationToken(token="t1", max_budget=100.0, spend=150.0)
        under_budget = VerificationToken(token="t2", max_budget=100.0, spend=50.0)
        no_budget = VerificationToken(token="t3", spend=1000.0)

        assert over_budget.is_over_budget()
        assert not under_budget.is_over_budget()
        assert not no_budget.is_over_budget()

    def test_has_model_access(self):
        token_with_models = VerificationToken(token="t1", models=["gpt-4"])
        token_no_models = VerificationToken(token="t2", models=[])

        assert token_with_models.has_model_access("gpt-4")
        assert not token_with_models.has_model_access("gpt-3")
        assert token_no_models.has_model_access("any-model")

    def test_has_route_access(self):
        token_with_routes = VerificationToken(
            token="t1", allowed_routes=["/chat/completions"]
        )
        token_no_routes = VerificationToken(token="t2", allowed_routes=[])

        assert token_with_routes.has_route_access("/chat/completions")
        assert not token_with_routes.has_route_access("/embeddings")
        assert token_no_routes.has_route_access("/any/route")

    def test_parse_expires_string(self):
        token = VerificationToken(
            token="t1", expires="2024-12-31T23:59:59Z"
        )
        assert isinstance(token.expires, datetime)

    def test_deleted_verification_token(self):
        deleted = DeletedVerificationToken(
            token="t1",
            deleted_by="admin",
            deleted_at=datetime.utcnow(),
        )
        assert deleted.deleted_by == "admin"
        assert deleted.deleted_at is not None
