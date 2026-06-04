"""
Tests for gateway repository layer.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.models.budget import Budget
from litellm.models.credentials import Credentials
from litellm.models.model import Model
from litellm.models.object_permission import ObjectPermission
from litellm.models.organization import Organization
from litellm.models.project import Project
from litellm.models.team import Team
from litellm.models.user import User
from litellm.models.verification_token import VerificationToken
from litellm.repositories.base_repository import BaseRepository
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.repositories.model_repository import ModelRepository
from litellm.repositories.object_permission_repository import (
    ObjectPermissionRepository,
)
from litellm.repositories.organization_repository import OrganizationRepository
from litellm.repositories.project_repository import ProjectRepository
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)


class MockRecord:
    """Mock database record for testing."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data if data is not None else {}

    def dict(self) -> Dict[str, Any]:
        return self._data.copy()

    def model_dump(self) -> Dict[str, Any]:
        return self._data.copy()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)


class MockTable:
    """Mock Prisma table for testing."""

    def __init__(self):
        self._records: Dict[str, Dict[str, Any]] = {}

    async def find_unique(self, where: Dict[str, Any]) -> Optional[MockRecord]:
        key_field = list(where.keys())[0]
        key_value = where[key_field]
        data = self._records.get(key_value)
        return MockRecord(data) if data else None

    async def find_many(
        self,
        where: Optional[Dict[str, Any]] = None,
        skip: Optional[int] = None,
        take: Optional[int] = None,
        order: Optional[Dict[str, str]] = None,
    ) -> List[MockRecord]:
        records = list(self._records.values())
        return [MockRecord(r) for r in records]

    async def create(self, data: Dict[str, Any]) -> MockRecord:
        record_data = data
        self._records[record_data.get("id", str(len(self._records)))] = record_data
        return MockRecord(record_data)

    async def update(
        self, where: Dict[str, Any], data: Dict[str, Any]
    ) -> Optional[MockRecord]:
        key_field = list(where.keys())[0]
        key_value = where[key_field]
        if key_value in self._records:
            for field, value in data.items():
                if isinstance(value, dict) and "push" in value:
                    current = self._records[key_value].get(field, [])
                    push_val = value["push"]
                    if isinstance(push_val, list):
                        current.extend(push_val)
                    else:
                        current.append(push_val)
                    self._records[key_value][field] = current
                else:
                    self._records[key_value][field] = value
            return MockRecord(self._records[key_value])
        return None

    async def delete(self, where: Dict[str, Any]) -> Optional[MockRecord]:
        key_field = list(where.keys())[0]
        key_value = where[key_field]
        data = self._records.pop(key_value, None)
        return MockRecord(data) if data else None

    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        return len(self._records)

    async def upsert(self, where: Dict[str, Any], data: Dict[str, Any]) -> MockRecord:
        key_field = list(where.keys())[0]
        key_value = where[key_field]
        if key_value in self._records:
            self._records[key_value].update(data.get("update", {}))
        else:
            self._records[key_value] = data.get("create", {})
        return MockRecord(self._records[key_value])


class MockPrismaClient:
    """Mock Prisma client for testing."""

    def __init__(self):
        self.db = MagicMock()
        self.db.litellm_budgettable = MockTable()
        self.db.litellm_proxymodeltable = MockTable()
        self.db.litellm_teamtable = MockTable()
        self.db.litellm_deletedteamtable = MockTable()
        self.db.litellm_usertable = MockTable()
        self.db.litellm_verificationtoken = MockTable()
        self.db.litellm_deletedverificationtoken = MockTable()
        self.db.litellm_config = MockTable()
        self.db.litellm_organizationtable = MockTable()
        self.db.litellm_projecttable = MockTable()
        self.db.litellm_objectpermissiontable = MockTable()
        self.db.litellm_credentialstable = MockTable()


class TestBaseRepository:
    @pytest.fixture
    def prisma_client(self):
        return MockPrismaClient()

    def test_prisma_client_none_raises(self):
        class TestRepo(BaseRepository[Budget]):
            @property
            def table(self):
                return None

            @property
            def model_class(self):
                return Budget

        repo = TestRepo(None)
        with pytest.raises(RuntimeError, match="No DB Connected"):
            _ = repo.prisma_client

    @pytest.mark.asyncio
    async def test_find_many(self, prisma_client):
        repo = BudgetRepository(prisma_client)
        prisma_client.db.litellm_budgettable._records = {
            "b1": {"budget_id": "b1", "max_budget": 100.0},
            "b2": {"budget_id": "b2", "max_budget": 200.0},
        }
        budgets = await repo.find_many()
        assert len(budgets) == 2

    @pytest.mark.asyncio
    async def test_count(self, prisma_client):
        repo = BudgetRepository(prisma_client)
        prisma_client.db.litellm_budgettable._records = {
            "b1": {"budget_id": "b1"},
            "b2": {"budget_id": "b2"},
        }
        count = await repo.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_exists(self, prisma_client):
        repo = BudgetRepository(prisma_client)
        prisma_client.db.litellm_budgettable._records = {
            "b1": {"budget_id": "b1"},
        }
        assert await repo.exists("b1", id_field="budget_id")
        assert not await repo.exists("nonexistent", id_field="budget_id")


class TestBudgetRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return BudgetRepository(client)

    @pytest.mark.asyncio
    async def test_create_budget(self, repo):
        budget = await repo.create_budget(
            created_by="test-user",
            max_budget=100.0,
            soft_budget=80.0,
            tpm_limit=1000,
        )
        assert budget.max_budget == 100.0
        assert budget.soft_budget == 80.0
        assert budget.tpm_limit == 1000

    @pytest.mark.asyncio
    async def test_create_budget_all_fields(self, repo):
        budget = await repo.create_budget(
            created_by="test-user",
            max_budget=100.0,
            soft_budget=80.0,
            max_parallel_requests=10,
            tpm_limit=1000,
            rpm_limit=100,
            model_max_budget={"gpt-4": 50.0},
            budget_duration="monthly",
            allowed_models=["gpt-4", "gpt-3.5-turbo"],
        )
        assert budget.max_budget == 100.0
        assert budget.max_parallel_requests == 10

    @pytest.mark.asyncio
    async def test_update_budget(self, repo):
        await repo.create_budget(created_by="test-user", max_budget=100.0)
        repo._prisma_client.db.litellm_budgettable._records["budget-1"] = {
            "budget_id": "budget-1",
            "max_budget": 100.0,
        }

        updated = await repo.update_budget(
            budget_id="budget-1",
            updated_by="test-user",
            max_budget=200.0,
        )
        assert updated.max_budget == 200.0

    @pytest.mark.asyncio
    async def test_delete_budget(self, repo):
        repo._prisma_client.db.litellm_budgettable._records["budget-1"] = {
            "budget_id": "budget-1",
            "max_budget": 100.0,
        }
        deleted = await repo.delete_budget("budget-1")
        assert deleted is not None
        assert "budget-1" not in repo._prisma_client.db.litellm_budgettable._records

    @pytest.mark.asyncio
    async def test_find_by_id(self, repo):
        repo._prisma_client.db.litellm_budgettable._records["budget-1"] = {
            "budget_id": "budget-1",
            "max_budget": 100.0,
        }
        budget = await repo.find_by_id("budget-1")
        assert budget is not None
        assert budget.budget_id == "budget-1"


class TestModelRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return ModelRepository(client)

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: f"encrypted_{v}",
    )
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_create_model_encrypts_params(self, mock_decrypt, mock_encrypt, repo):
        model = await repo.create_model(
            model_name="gpt-4",
            litellm_params={"api_key": "sk-secret"},
            created_by="test-user",
        )
        assert model is not None
        mock_encrypt.assert_called()

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_find_all(self, mock_decrypt, repo):
        repo._prisma_client.db.litellm_proxymodeltable._records = {
            "m1": {
                "model_id": "m1",
                "model_name": "gpt-4",
                "litellm_params": '{"model": "gpt-4"}',
                "blocked": False,
            },
            "m2": {
                "model_id": "m2",
                "model_name": "claude-3",
                "litellm_params": '{"model": "claude-3"}',
                "blocked": False,
            },
        }
        models = await repo.find_all()
        assert len(models) == 2

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_find_unblocked(self, mock_decrypt, repo):
        repo._prisma_client.db.litellm_proxymodeltable._records = {
            "m1": {
                "model_id": "m1",
                "model_name": "gpt-4",
                "litellm_params": '{"model": "gpt-4"}',
                "blocked": False,
            },
        }
        models = await repo.find_unblocked()
        assert len(models) == 1

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_find_by_name(self, mock_decrypt, repo):
        repo._prisma_client.db.litellm_proxymodeltable._records = {
            "m1": {
                "model_id": "m1",
                "model_name": "gpt-4",
                "litellm_params": '{"model": "gpt-4"}',
            },
        }
        models = await repo.find_by_name("gpt-4")
        assert len(models) == 1

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_update_model(self, mock_decrypt, mock_encrypt, repo):
        repo._prisma_client.db.litellm_proxymodeltable._records["m1"] = {
            "model_id": "m1",
            "model_name": "gpt-4",
            "litellm_params": '{"model": "gpt-4"}',
            "blocked": False,
        }
        updated = await repo.update_model(
            model_id="m1",
            updated_by="test-user",
            blocked=True,
        )
        assert updated.blocked is True

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_delete_model(self, mock_decrypt, repo):
        repo._prisma_client.db.litellm_proxymodeltable._records["m1"] = {
            "model_id": "m1",
            "model_name": "gpt-4",
            "litellm_params": '{"model": "gpt-4"}',
        }
        deleted = await repo.delete_model("m1")
        assert deleted is not None

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_block_unblock_model(self, mock_decrypt, mock_encrypt, repo):
        repo._prisma_client.db.litellm_proxymodeltable._records["m1"] = {
            "model_id": "m1",
            "model_name": "gpt-4",
            "litellm_params": '{"model": "gpt-4"}',
            "blocked": False,
        }
        blocked = await repo.block_model("m1", "admin")
        assert blocked.blocked is True

        unblocked = await repo.unblock_model("m1", "admin")
        assert unblocked.blocked is False


class TestTeamRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return TeamRepository(client)

    @pytest.mark.asyncio
    async def test_create_team(self, repo):
        team = await repo.create_team(
            team_id="team-123",
            team_alias="Engineering",
            admins=["user1"],
            members=["user2", "user3"],
        )
        assert team.team_id == "team-123"
        assert team.team_alias == "Engineering"

    @pytest.mark.asyncio
    async def test_create_team_all_fields(self, repo):
        team = await repo.create_team(
            team_id="team-123",
            team_alias="Engineering",
            organization_id="org-1",
            admins=["admin1"],
            members=["user1"],
            members_with_roles={"user1": "developer"},
            metadata={"dept": "engineering"},
            max_budget=1000.0,
            soft_budget=800.0,
            models=["gpt-4"],
            max_parallel_requests=10,
            tpm_limit=50000,
            rpm_limit=500,
            budget_duration="monthly",
            object_permission_id="perm-1",
        )
        assert team.team_id == "team-123"
        assert team.organization_id == "org-1"

    @pytest.mark.asyncio
    async def test_update_team(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Test",
            "admins": [],
            "members": [],
            "models": [],
        }
        updated = await repo.update_team(
            team_id="team-1",
            team_alias="Updated Team",
            blocked=True,
        )
        assert updated.team_alias == "Updated Team"

    @pytest.mark.asyncio
    async def test_add_member(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Test",
            "admins": [],
            "members": ["user1"],
            "models": [],
        }

        team = await repo.add_member("team-1", "user2")
        assert "user2" in team.members

    @pytest.mark.asyncio
    async def test_add_member_nonexistent_team(self, repo):
        result = await repo.add_member("nonexistent", "user1")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_member(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Test",
            "admins": [],
            "members": ["user1", "user2"],
            "models": [],
        }

        team = await repo.remove_member("team-1", "user2")
        assert "user2" not in team.members

    @pytest.mark.asyncio
    async def test_add_admin(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Test",
            "admins": [],
            "members": [],
            "models": [],
        }
        team = await repo.add_admin("team-1", "admin1")
        assert "admin1" in team.admins

    @pytest.mark.asyncio
    async def test_remove_admin(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Test",
            "admins": ["admin1", "admin2"],
            "members": [],
            "models": [],
        }
        team = await repo.remove_admin("team-1", "admin2")
        assert "admin2" not in team.admins

    @pytest.mark.asyncio
    async def test_add_models(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Test",
            "admins": [],
            "members": [],
            "models": ["gpt-3.5-turbo"],
        }
        team = await repo.add_models("team-1", ["gpt-4"])
        assert "gpt-4" in team.models

    @pytest.mark.asyncio
    async def test_remove_models(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Test",
            "admins": [],
            "members": [],
            "models": ["gpt-3.5-turbo", "gpt-4"],
        }
        team = await repo.remove_models("team-1", ["gpt-4"])
        assert "gpt-4" not in team.models

    @pytest.mark.asyncio
    async def test_update_spend(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Test",
            "admins": [],
            "members": [],
            "models": [],
            "spend": 0.0,
        }
        team = await repo.update_spend("team-1", 50.0)
        assert team.spend == 50.0

    @pytest.mark.asyncio
    async def test_find_by_alias(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "team_alias": "Engineering",
            "admins": [],
            "members": [],
            "models": [],
        }
        team = await repo.find_by_alias("Engineering")
        assert team is not None
        assert team.team_id == "team-1"

    @pytest.mark.asyncio
    async def test_find_by_organization_id(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "organization_id": "org-1",
            "admins": [],
            "members": [],
            "models": [],
        }
        teams = await repo.find_by_organization_id("org-1")
        assert len(teams) == 1

    @pytest.mark.asyncio
    async def test_find_by_member(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "admins": [],
            "members": ["user1"],
            "models": [],
        }
        teams = await repo.find_by_member("user1")
        assert len(teams) == 1

    @pytest.mark.asyncio
    async def test_find_by_admin(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-1"] = {
            "team_id": "team-1",
            "admins": ["admin1"],
            "members": [],
            "models": [],
        }
        teams = await repo.find_by_admin("admin1")
        assert len(teams) == 1


class TestUserRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return UserRepository(client)

    @pytest.mark.asyncio
    async def test_create_user(self, repo):
        user = await repo.create_user(
            user_id="user-123",
            user_email="test@example.com",
            teams=["team1"],
        )
        assert user.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_create_user_all_fields(self, repo):
        user = await repo.create_user(
            user_id="user-123",
            user_alias="testuser",
            team_id="team-1",
            sso_user_id="sso-123",
            organization_id="org-1",
            password="hashed_password",
            teams=["team1", "team2"],
            user_role="admin",
            max_budget=500.0,
            user_email="test@example.com",
            models=["gpt-4"],
            metadata={"department": "engineering"},
            max_parallel_requests=5,
            tpm_limit=10000,
            rpm_limit=100,
            budget_duration="monthly",
            allowed_cache_controls=["no-cache"],
            policies=["policy-1"],
            object_permission_id="perm-1",
        )
        assert user.user_id == "user-123"
        assert user.user_alias == "testuser"

    @pytest.mark.asyncio
    async def test_update_user(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "teams": [],
            "models": [],
        }
        updated = await repo.update_user(
            user_id="user-1",
            user_email="updated@example.com",
        )
        assert updated.user_email == "updated@example.com"

    @pytest.mark.asyncio
    async def test_delete_user(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "teams": [],
            "models": [],
        }
        deleted = await repo.delete_user("user-1")
        assert deleted is not None

    @pytest.mark.asyncio
    async def test_add_to_team(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "teams": ["team1"],
            "models": [],
        }

        user = await repo.add_to_team("user-1", "team2")
        assert "team2" in user.teams

    @pytest.mark.asyncio
    async def test_add_to_team_nonexistent_user(self, repo):
        result = await repo.add_to_team("nonexistent", "team1")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_from_team(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "teams": ["team1", "team2"],
            "models": [],
        }
        user = await repo.remove_from_team("user-1", "team2")
        assert "team2" not in user.teams

    @pytest.mark.asyncio
    async def test_update_spend(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "teams": [],
            "models": [],
            "spend": 0.0,
        }
        user = await repo.update_spend("user-1", 25.0)
        assert user.spend == 25.0

    @pytest.mark.asyncio
    async def test_find_by_email(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "user_email": "test@example.com",
            "teams": [],
            "models": [],
        }
        user = await repo.find_by_email("test@example.com")
        assert user is not None

    @pytest.mark.asyncio
    async def test_find_by_sso_id(self, repo):
        repo._prisma_client.db.litellm_usertable._records["sso-123"] = {
            "user_id": "user-1",
            "sso_user_id": "sso-123",
            "teams": [],
            "models": [],
        }
        user = await repo.find_by_sso_id("sso-123")
        assert user is not None

    @pytest.mark.asyncio
    async def test_find_by_organization_id(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "organization_id": "org-1",
            "teams": [],
            "models": [],
        }
        users = await repo.find_by_organization_id("org-1")
        assert len(users) == 1

    @pytest.mark.asyncio
    async def test_find_by_team_id(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "teams": ["team-1"],
            "models": [],
        }
        users = await repo.find_by_team_id("team-1")
        assert len(users) == 1


class TestVerificationTokenRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return VerificationTokenRepository(client)

    @pytest.mark.asyncio
    async def test_create_token(self, repo):
        token = await repo.create_token(
            token="sk-test123",
            key_name="Test Key",
            user_id="user-123",
            max_budget=100.0,
        )
        assert token.token == "sk-test123"
        assert token.key_name == "Test Key"

    @pytest.mark.asyncio
    async def test_create_token_all_fields(self, repo):
        token = await repo.create_token(
            token="sk-test123",
            key_name="Test Key",
            key_alias="test-alias",
            max_budget=100.0,
            expires=datetime(2025, 12, 31),
            models=["gpt-4"],
            aliases={"alias1": "value1"},
            config={"setting": "value"},
            user_id="user-123",
            team_id="team-1",
            agent_id="agent-1",
            project_id="project-1",
            max_parallel_requests=5,
            metadata={"key": "value"},
            tpm_limit=10000,
            rpm_limit=100,
            budget_duration="monthly",
            allowed_cache_controls=["no-cache"],
            allowed_routes=["/v1/completions"],
            permissions={"read": True},
            org_id="org-1",
            created_by="admin",
            object_permission_id="perm-1",
            access_group_ids=["group-1"],
            budget_id="budget-1",
        )
        assert token.token == "sk-test123"

    @pytest.mark.asyncio
    async def test_update_token(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "blocked": False,
        }
        updated = await repo.update_token(
            token="sk-test",
            key_name="Updated Key",
        )
        assert updated.key_name == "Updated Key"

    @pytest.mark.asyncio
    async def test_block_token(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "blocked": False,
        }

        token = await repo.block_token("sk-test", updated_by="admin")
        assert token.blocked is True

    @pytest.mark.asyncio
    async def test_unblock_token(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "blocked": True,
        }
        token = await repo.unblock_token("sk-test", updated_by="admin")
        assert token.blocked is False

    @pytest.mark.asyncio
    async def test_update_spend(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "spend": 0.0,
        }
        token = await repo.update_spend("sk-test", 15.0)
        assert token.spend == 15.0

    @pytest.mark.asyncio
    async def test_update_last_active(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
        }
        token = await repo.update_last_active("sk-test")
        assert token.last_active is not None

    @pytest.mark.asyncio
    async def test_find_by_alias(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "key_alias": "my-key",
        }
        token = await repo.find_by_alias("my-key")
        assert token is not None

    @pytest.mark.asyncio
    async def test_find_by_user_id(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "user_id": "user-1",
        }
        tokens = await repo.find_by_user_id("user-1")
        assert len(tokens) == 1

    @pytest.mark.asyncio
    async def test_find_by_team_id(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "team_id": "team-1",
        }
        tokens = await repo.find_by_team_id("team-1")
        assert len(tokens) == 1

    @pytest.mark.asyncio
    async def test_find_by_project_id(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "project_id": "project-1",
        }
        tokens = await repo.find_by_project_id("project-1")
        assert len(tokens) == 1


class TestOrganizationRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return OrganizationRepository(client)

    @pytest.mark.asyncio
    async def test_create_organization(self, repo):
        org = await repo.create_organization(
            organization_alias="Acme Corp",
            budget_id="budget-1",
            created_by="admin",
        )
        assert org.organization_alias == "Acme Corp"

    @pytest.mark.asyncio
    async def test_create_organization_all_fields(self, repo):
        org = await repo.create_organization(
            organization_alias="Acme Corp",
            budget_id="budget-1",
            created_by="admin",
            organization_id="org-123",
            metadata={"industry": "tech"},
            models=["gpt-4"],
            object_permission_id="perm-1",
        )
        assert org.organization_alias == "Acme Corp"

    @pytest.mark.asyncio
    async def test_update_organization(self, repo):
        repo._prisma_client.db.litellm_organizationtable._records["org-1"] = {
            "organization_id": "org-1",
            "organization_alias": "Old Name",
        }
        updated = await repo.update_organization(
            organization_id="org-1",
            updated_by="admin",
            organization_alias="New Name",
        )
        assert updated.organization_alias == "New Name"

    @pytest.mark.asyncio
    async def test_delete_organization(self, repo):
        repo._prisma_client.db.litellm_organizationtable._records["org-1"] = {
            "organization_id": "org-1",
            "organization_alias": "Acme",
        }
        deleted = await repo.delete_organization("org-1")
        assert deleted is not None

    @pytest.mark.asyncio
    async def test_update_spend(self, repo):
        repo._prisma_client.db.litellm_organizationtable._records["org-1"] = {
            "organization_id": "org-1",
            "organization_alias": "Acme",
            "spend": 0.0,
        }
        org = await repo.update_spend("org-1", 100.0)
        assert org.spend == 100.0

    @pytest.mark.asyncio
    async def test_find_by_alias(self, repo):
        repo._prisma_client.db.litellm_organizationtable._records["org-1"] = {
            "organization_id": "org-1",
            "organization_alias": "Acme",
        }
        org = await repo.find_by_alias("Acme")
        assert org is not None


class TestProjectRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return ProjectRepository(client)

    @pytest.mark.asyncio
    async def test_create_project(self, repo):
        project = await repo.create_project(
            created_by="admin",
            project_alias="My Project",
        )
        assert project.project_alias == "My Project"

    @pytest.mark.asyncio
    async def test_create_project_all_fields(self, repo):
        project = await repo.create_project(
            created_by="admin",
            project_id="proj-123",
            project_alias="My Project",
            description="A test project",
            team_id="team-1",
            budget_id="budget-1",
            metadata={"env": "dev"},
            models=["gpt-4"],
            model_rpm_limit={"gpt-4": 100},
            model_tpm_limit={"gpt-4": 10000},
            object_permission_id="perm-1",
        )
        assert project.project_alias == "My Project"

    @pytest.mark.asyncio
    async def test_update_project(self, repo):
        repo._prisma_client.db.litellm_projecttable._records["proj-1"] = {
            "project_id": "proj-1",
            "project_alias": "Old Name",
        }
        updated = await repo.update_project(
            project_id="proj-1",
            updated_by="admin",
            project_alias="New Name",
            blocked=True,
        )
        assert updated.project_alias == "New Name"

    @pytest.mark.asyncio
    async def test_delete_project(self, repo):
        repo._prisma_client.db.litellm_projecttable._records["proj-1"] = {
            "project_id": "proj-1",
        }
        deleted = await repo.delete_project("proj-1")
        assert deleted is not None

    @pytest.mark.asyncio
    async def test_update_spend(self, repo):
        repo._prisma_client.db.litellm_projecttable._records["proj-1"] = {
            "project_id": "proj-1",
            "spend": 0.0,
        }
        project = await repo.update_spend("proj-1", 50.0)
        assert project.spend == 50.0

    @pytest.mark.asyncio
    async def test_find_by_alias(self, repo):
        repo._prisma_client.db.litellm_projecttable._records["proj-1"] = {
            "project_id": "proj-1",
            "project_alias": "MyProject",
        }
        project = await repo.find_by_alias("MyProject")
        assert project is not None

    @pytest.mark.asyncio
    async def test_find_by_team_id(self, repo):
        repo._prisma_client.db.litellm_projecttable._records["proj-1"] = {
            "project_id": "proj-1",
            "team_id": "team-1",
        }
        projects = await repo.find_by_team_id("team-1")
        assert len(projects) == 1


class TestObjectPermissionRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return ObjectPermissionRepository(client)

    @pytest.mark.asyncio
    async def test_create_permission(self, repo):
        perm = await repo.create_permission(
            mcp_servers=["server1"],
            models=["gpt-4"],
        )
        assert perm.mcp_servers == ["server1"]

    @pytest.mark.asyncio
    async def test_create_permission_all_fields(self, repo):
        perm = await repo.create_permission(
            mcp_servers=["server1"],
            mcp_access_groups=["group1"],
            mcp_tool_permissions={"tool1": ["read", "write"]},
            vector_stores=["store1"],
            agents=["agent1"],
            agent_access_groups=["agent-group1"],
            models=["gpt-4"],
            blocked_tools=["tool2"],
            mcp_toolsets=["toolset1"],
            search_tools=["search1"],
        )
        assert perm.mcp_servers == ["server1"]
        assert perm.agents == ["agent1"]

    @pytest.mark.asyncio
    async def test_update_permission(self, repo):
        repo._prisma_client.db.litellm_objectpermissiontable._records["perm-1"] = {
            "object_permission_id": "perm-1",
            "models": ["gpt-3.5-turbo"],
        }
        updated = await repo.update_permission(
            object_permission_id="perm-1",
            models=["gpt-4"],
        )
        assert updated.models == ["gpt-4"]

    @pytest.mark.asyncio
    async def test_delete_permission(self, repo):
        repo._prisma_client.db.litellm_objectpermissiontable._records["perm-1"] = {
            "object_permission_id": "perm-1",
        }
        deleted = await repo.delete_permission("perm-1")
        assert deleted is not None


class TestCredentialsRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return CredentialsRepository(client)

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.credentials_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: f"encrypted_{v}",
    )
    @patch(
        "litellm.repositories.credentials_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_create_credentials(self, mock_decrypt, mock_encrypt, repo):
        cred = await repo.create_credentials(
            credential_name="my-api-key",
            credential_values={"api_key": "secret123"},
            created_by="admin",
        )
        assert cred.credential_name == "my-api-key"
        mock_encrypt.assert_called()

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.credentials_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: f"encrypted_{v}",
    )
    @patch(
        "litellm.repositories.credentials_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_create_credentials_with_info(self, mock_decrypt, mock_encrypt, repo):
        cred = await repo.create_credentials(
            credential_name="my-api-key",
            credential_values={"api_key": "secret123"},
            created_by="admin",
            credential_info={"provider": "openai"},
        )
        assert cred.credential_name == "my-api-key"

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.credentials_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    @patch(
        "litellm.repositories.credentials_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_update_credentials(self, mock_decrypt, mock_encrypt, repo):
        repo._prisma_client.db.litellm_credentialstable._records["cred-1"] = {
            "credential_id": "cred-1",
            "credential_name": "my-key",
            "credential_values": {"api_key": "old"},
        }
        updated = await repo.update_credentials(
            credential_id="cred-1",
            updated_by="admin",
            credential_values={"api_key": "new"},
        )
        assert updated is not None

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.credentials_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_delete_credentials(self, mock_decrypt, repo):
        repo._prisma_client.db.litellm_credentialstable._records["cred-1"] = {
            "credential_id": "cred-1",
            "credential_name": "my-key",
            "credential_values": {"api_key": "secret"},
        }
        deleted = await repo.delete_credentials("cred-1")
        assert deleted is not None

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.credentials_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_find_by_name(self, mock_decrypt, repo):
        repo._prisma_client.db.litellm_credentialstable._records["my-key"] = {
            "credential_id": "cred-1",
            "credential_name": "my-key",
            "credential_values": {"api_key": "secret"},
        }
        cred = await repo.find_by_name("my-key")
        assert cred is not None


class TestConfigRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return ConfigRepository(client)

    def test_deep_merge_dicts_db_wins(self, repo):
        dst = {"a": 1, "b": {"c": 2}}
        src = {"a": 10, "b": {"d": 3}}
        repo._deep_merge_dicts(dst, src)
        assert dst["a"] == 10
        assert dst["b"]["c"] == 2
        assert dst["b"]["d"] == 3

    def test_deep_merge_dicts_skips_none(self, repo):
        dst = {"a": 1}
        src = {"a": None, "b": 2}
        repo._deep_merge_dicts(dst, src)
        assert dst["a"] == 1
        assert dst["b"] == 2

    def test_deep_merge_dicts_skips_empty_list(self, repo):
        dst = {"models": ["gpt-4"]}
        src = {"models": []}
        repo._deep_merge_dicts(dst, src)
        assert dst["models"] == ["gpt-4"]

    @pytest.mark.asyncio
    async def test_get_param(self, repo):
        repo._prisma_client.db.litellm_config._records["general_settings"] = {
            "param_name": "general_settings",
            "param_value": '{"master_key": "test"}',
        }
        param = await repo.get_param("general_settings")
        assert param is not None
        assert param.param_name == "general_settings"
        assert param.param_value["master_key"] == "test"

    @pytest.mark.asyncio
    async def test_set_param(self, repo):
        param = await repo.set_param("test_param", {"key": "value"})
        assert param.param_name == "test_param"
        assert param.param_value == {"key": "value"}

    @pytest.mark.asyncio
    async def test_delete_param(self, repo):
        repo._prisma_client.db.litellm_config._records["test_param"] = {
            "param_name": "test_param",
            "param_value": "{}",
        }
        result = await repo.delete_param("test_param")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_param_nonexistent(self, repo):
        async def mock_delete(where):
            raise Exception("Not found")

        repo._prisma_client.db.litellm_config.delete = mock_delete
        result = await repo.delete_param("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_all_params(self, repo):
        repo._prisma_client.db.litellm_config._records = {
            "param1": {"param_name": "param1", "param_value": '{"a": 1}'},
            "param2": {"param_name": "param2", "param_value": '{"b": 2}'},
        }
        params = await repo.get_all_params()
        assert len(params) == 2

    @pytest.mark.asyncio
    async def test_reconcile_config_skips_when_store_model_false(self, repo):
        yaml_config = {"general_settings": {"key": "value"}}
        result = await repo.reconcile_config(yaml_config, store_model_in_db=False)
        assert result == yaml_config

    @pytest.mark.asyncio
    async def test_prefetch_params(self, repo):
        repo._prisma_client.db.litellm_config._records["general_settings"] = {
            "param_name": "general_settings",
            "param_value": "{}",
        }
        await repo.prefetch_params(["general_settings"])
