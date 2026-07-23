"""
Tests for backend domain models.
"""

import json
from datetime import datetime

import pytest

from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.models.budget import (
    LiteLLM_BudgetTable,
    LiteLLM_BudgetTableFull,
    LiteLLM_TeamMemberTable,
)
from litellm.models.config import LiteLLM_Config
from litellm.models.credentials import CreateCredentialItem, CredentialItem
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.managed_files import (
    LiteLLM_ManagedFileTable,
    LiteLLM_ManagedObjectTable,
    LiteLLM_ManagedVectorStoresTable,
)
from litellm.models.mcp_server import LiteLLM_MCPServerTable
from litellm.models.model import LiteLLM_ProxyModelTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.skills import LiteLLM_SkillsTable
from litellm.models.spend_logs import LiteLLM_ErrorLogs, LiteLLM_SpendLogs
from litellm.models.tag import LiteLLM_TagTable
from litellm.models.team import (
    LiteLLM_DeletedTeamTable,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
)
from litellm.models.team_membership import LiteLLM_TeamMembership
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
        with pytest.raises(ValueError, match="Either credential_values or model_id must be set"):
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
        model_blocked = LiteLLM_ProxyModelTable(model_id="m1", model_name="test", litellm_params={}, blocked=True)
        model_unblocked = LiteLLM_ProxyModelTable(model_id="m2", model_name="test", litellm_params={}, blocked=False)
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
        model = LiteLLM_ProxyModelTable(model_id="m1", model_name="gpt-4", litellm_params={}, model_info=None)
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
        cached = LiteLLM_TeamTableCachedObj(team_id="t1", last_refreshed_at=1234567890.0)
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

    def test_password_hash_excluded_from_serialization(self):
        from litellm.proxy._types import LiteLLM_UserTableWithKeyCount

        secret = "$2b$12$abcdefghijklmnopqrstuv"
        user = LiteLLM_UserTable(user_id="u1", user_email="a@b.c", password=secret)

        assert user.password == secret
        assert "password" not in user.model_dump()
        assert "password" not in json.loads(user.model_dump_json())

        with_keys = LiteLLM_UserTableWithKeyCount(user_id="u1", user_email="a@b.c", password=secret, key_count=2)
        assert with_keys.password == secret
        assert "password" not in with_keys.model_dump()
        assert "password" not in json.loads(with_keys.model_dump_json())


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


class TestConfigTable:
    def test_config_creation(self):
        cfg = LiteLLM_Config(param_name="general_settings", param_value={"k": "v"})
        assert cfg.param_name == "general_settings"
        assert cfg.param_value == {"k": "v"}


class TestSkillsTable:
    def test_skills_creation(self):
        skill = LiteLLM_SkillsTable(
            skill_id="s1",
            display_title="My Skill",
            source="custom",
            file_content=b"zipbytes",
            file_name="skill.zip",
        )
        assert skill.skill_id == "s1"
        assert skill.display_title == "My Skill"
        assert skill.file_content == b"zipbytes"

    def test_skills_defaults(self):
        skill = LiteLLM_SkillsTable(skill_id="s2")
        assert skill.source == "custom"
        assert skill.metadata is None


class TestAccessGroupTable:
    def test_access_group_creation(self):
        ag = LiteLLM_AccessGroupTable(
            access_group_id="ag1",
            access_group_name="group-a",
            access_model_names=["gpt-4"],
            assigned_team_ids=["t1"],
        )
        assert ag.access_group_id == "ag1"
        assert ag.access_model_names == ["gpt-4"]
        assert ag.assigned_team_ids == ["t1"]
        assert ag.access_agent_ids == []


class TestTagTable:
    def test_tag_creation(self):
        tag = LiteLLM_TagTable(
            tag_name="prod",
            models=["gpt-4"],
            spend=12.5,
            budget_id="b1",
        )
        assert tag.tag_name == "prod"
        assert tag.models == ["gpt-4"]
        assert tag.spend == 12.5

    def test_tag_set_model_info_coerces_none(self):
        tag = LiteLLM_TagTable(tag_name="t", spend=None, models=None)
        assert tag.spend == 0.0
        assert tag.models == []


class TestEndUserTable:
    def test_end_user_creation(self):
        eu = LiteLLM_EndUserTable(
            user_id="eu1",
            blocked=False,
            spend=5.0,
            allowed_model_region="eu",
            default_model="gpt-4",
        )
        assert eu.user_id == "eu1"
        assert eu.blocked is False
        assert eu.allowed_model_region == "eu"
        assert eu.default_model == "gpt-4"

    def test_end_user_spend_coerced_when_none(self):
        eu = LiteLLM_EndUserTable(user_id="eu2", blocked=True, spend=None)
        assert eu.spend == 0.0


class TestBudgetTableFull:
    def test_full_adds_server_managed_fields(self):
        now = datetime.now()
        budget = LiteLLM_BudgetTableFull(budget_id="b1", max_budget=10.0, created_at=now, budget_reset_at=now)
        assert budget.created_at == now
        assert budget.budget_reset_at == now
        assert budget.max_budget == 10.0

    def test_full_requires_created_at(self):
        with pytest.raises(Exception):
            LiteLLM_BudgetTableFull(budget_id="b1")


class TestTeamMemberTable:
    def test_tracks_user_within_team(self):
        member = LiteLLM_TeamMemberTable(user_id="u1", team_id="t1", spend=3.0, budget_id="b1", max_budget=5.0)
        assert member.user_id == "u1"
        assert member.team_id == "t1"
        assert member.spend == 3.0
        assert member.max_budget == 5.0


class TestTeamMembership:
    def test_safe_get_limits_with_budget_table(self):
        membership = LiteLLM_TeamMembership(
            user_id="u1",
            team_id="t1",
            litellm_budget_table=LiteLLM_BudgetTable(rpm_limit=100, tpm_limit=2000),
        )
        assert membership.safe_get_team_member_rpm_limit() == 100
        assert membership.safe_get_team_member_tpm_limit() == 2000

    def test_safe_get_limits_without_budget_table(self):
        membership = LiteLLM_TeamMembership(user_id="u1", team_id="t1")
        assert membership.safe_get_team_member_rpm_limit() is None
        assert membership.safe_get_team_member_tpm_limit() is None

    def test_full_budget_variant_parsed_for_server_fields(self):
        now = datetime.now()
        membership = LiteLLM_TeamMembership(
            user_id="u1",
            team_id="t1",
            litellm_budget_table={
                "budget_id": "b1",
                "rpm_limit": 7,
                "created_at": now,
                "budget_reset_at": now,
            },
        )
        assert isinstance(membership.litellm_budget_table, LiteLLM_BudgetTableFull)
        assert membership.safe_get_team_member_rpm_limit() == 7


class TestMCPServerTable:
    def test_mcp_server_defaults(self):
        server = LiteLLM_MCPServerTable(server_id="s1", transport="sse")
        assert server.server_id == "s1"
        assert server.transport == "sse"
        assert server.status == "unknown"
        assert server.approval_status == "active"
        assert server.allow_all_keys is False
        assert server.available_on_public_internet is True
        assert server.teams == []
        assert server.env == {}

    def test_mcp_server_requires_transport(self):
        with pytest.raises(Exception):
            LiteLLM_MCPServerTable(server_id="s1")


class TestSpendLogs:
    def test_spend_logs_creation(self):
        log = LiteLLM_SpendLogs(
            request_id="r1",
            api_key="sk-1",
            call_type="completion",
            startTime=None,
            endTime=None,
            messages=None,
            response=None,
        )
        assert log.request_id == "r1"
        assert log.spend == 0.0
        assert log.cache_hit == "False"

    def test_error_logs_creation(self):
        log = LiteLLM_ErrorLogs(request_id="r1", startTime=None, endTime=None, status_code="500")
        assert log.request_id == "r1"
        assert log.status_code == "500"


class TestManagedTables:
    def test_managed_file_table(self):
        table = LiteLLM_ManagedFileTable(
            unified_file_id="f1",
            model_mappings={"gpt-4": "file-abc"},
            flat_model_file_ids=["file-abc"],
        )
        assert table.unified_file_id == "f1"
        assert table.model_mappings == {"gpt-4": "file-abc"}
        assert table.flat_model_file_ids == ["file-abc"]

    def test_managed_object_table_requires_purpose(self):
        with pytest.raises(Exception):
            LiteLLM_ManagedObjectTable(unified_object_id="o1", model_object_id="m1", file_object={})

    def test_managed_vector_stores_table(self):
        table = LiteLLM_ManagedVectorStoresTable(
            vector_store_id="vs1",
            custom_llm_provider="openai",
            vector_store_name=None,
            vector_store_description=None,
            vector_store_metadata=None,
            created_at=None,
            updated_at=None,
            litellm_credential_name=None,
            litellm_params=None,
            team_id=None,
            user_id=None,
        )
        assert table.vector_store_id == "vs1"
        assert table.custom_llm_provider == "openai"
