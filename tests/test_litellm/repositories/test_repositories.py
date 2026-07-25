"""
Tests for gateway repository layer.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.models.base import DomainModel
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.credentials import CredentialItem
from litellm.models.team import LiteLLM_TeamTable
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

    def __init__(self, pk_field: Optional[str] = None):
        self._records: Dict[str, Dict[str, Any]] = {}
        self._pk_field = pk_field

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
        record_data = dict(data)
        if self._pk_field and self._pk_field not in record_data:
            record_data[self._pk_field] = f"{self._pk_field}-{len(self._records)}"
        key = (
            record_data.get(self._pk_field)
            if self._pk_field
            else record_data.get("id", str(len(self._records)))
        )
        self._records[key] = record_data
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
        self.db.litellm_proxymodeltable = MockTable(pk_field="model_id")
        self.db.litellm_teamtable = MockTable()
        self.db.litellm_deletedteamtable = MockTable()
        self.db.litellm_usertable = MockTable()
        self.db.litellm_verificationtoken = MockTable()
        self.db.litellm_deletedverificationtoken = MockTable()
        self.db.litellm_config = MockTable()
        self.db.litellm_organizationtable = MockTable()
        self.db.litellm_projecttable = MockTable(pk_field="project_id")
        self.db.litellm_objectpermissiontable = MockTable(
            pk_field="object_permission_id"
        )
        self.db.litellm_credentialstable = MockTable()


class TestBaseRepository:
    @pytest.fixture
    def prisma_client(self):
        return MockPrismaClient()

    def test_prisma_client_none_raises(self):
        class TestRepo(BaseRepository[LiteLLM_BudgetTable]):
            @property
            def table(self):
                return None

            @property
            def model_class(self):
                return LiteLLM_BudgetTable

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

    @pytest.mark.asyncio
    async def test_find_many_with_all_kwargs(self, prisma_client):
        repo = BudgetRepository(prisma_client)
        prisma_client.db.litellm_budgettable._records = {
            "b1": {"budget_id": "b1", "max_budget": 100.0},
        }
        budgets = await repo.find_many(
            where={"budget_id": "b1"}, skip=0, take=10, order={"budget_id": "asc"}
        )
        assert len(budgets) == 1

    def test_record_to_dict_branches(self):
        from litellm.repositories.base_repository import _record_to_dict

        assert _record_to_dict({"a": 1}) == {"a": 1}

        class WithModelDump:
            def model_dump(self):
                return {"src": "model_dump"}

        assert _record_to_dict(WithModelDump()) == {"src": "model_dump"}

        class WithDict:
            def dict(self):
                return {"src": "dict"}

        assert _record_to_dict(WithDict()) == {"src": "dict"}

        assert _record_to_dict([("k", "v")]) == {"k": "v"}


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
        "litellm.repositories.model_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: f"encrypted_{v}",
    )
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_create_model_all_fields(self, mock_decrypt, mock_encrypt, repo):
        model = await repo.create_model(
            model_name="gpt-4-turbo",
            litellm_params={
                "api_key": "sk-secret",
                "api_base": "https://api.openai.com",
            },
            created_by="admin",
            model_id="custom-model-id",
            model_info={"team_id": "team-1", "description": "GPT-4 Turbo model"},
            blocked=True,
        )
        assert model is not None
        assert model.model_name == "gpt-4-turbo"

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: f"encrypted_{v}",
    )
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda v, **kw: v,
    )
    async def test_update_model_all_fields(self, mock_decrypt, mock_encrypt, repo):
        repo._prisma_client.db.litellm_proxymodeltable._records["model-full"] = {
            "model_id": "model-full",
            "model_name": "old-name",
            "litellm_params": '{"api_key": "old"}',
            "blocked": False,
        }
        updated = await repo.update_model(
            model_id="model-full",
            updated_by="admin",
            model_name="new-name",
            litellm_params={"api_key": "new-key"},
            model_info={"updated": True},
            blocked=True,
        )
        assert updated.model_name == "new-name"

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
    @pytest.mark.parametrize(
        "raw_value, expected_ids",
        [
            (
                [
                    {"user_id": "a", "role": "user"},
                    {"user_id": "b", "role": "admin"},
                ],
                ["a", "b"],
            ),
            (json.dumps([{"user_id": "a", "role": "user"}]), ["a"]),
            ({}, []),
            (None, []),
        ],
    )
    async def test_get_members_with_roles_locked(self, repo, raw_value, expected_ids):
        tx = MagicMock()
        tx.query_raw = AsyncMock(return_value=[{"members_with_roles": raw_value}])

        members = await repo.get_members_with_roles_locked(tx, "team-1")

        assert [m.user_id for m in members] == expected_ids
        sql = tx.query_raw.call_args.args[0]
        assert "FOR UPDATE" in sql
        assert tx.query_raw.call_args.args[1] == "team-1"

    @pytest.mark.asyncio
    async def test_get_members_with_roles_locked_missing_row(self, repo):
        tx = MagicMock()
        tx.query_raw = AsyncMock(return_value=[])

        members = await repo.get_members_with_roles_locked(tx, "missing")

        assert members == []

    @pytest.mark.asyncio
    async def test_create_team_all_fields(self, repo):
        team = await repo.create_team(
            team_id="team-123",
            team_alias="Engineering",
            organization_id="org-1",
            admins=["admin1"],
            members=["user1"],
            members_with_roles=[{"user_id": "user1", "role": "user"}],
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
    async def test_update_team_all_fields(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-full"] = {
            "team_id": "team-full",
            "team_alias": "Test",
            "admins": [],
            "members": [],
            "models": [],
        }
        updated = await repo.update_team(
            team_id="team-full",
            team_alias="Fully Updated",
            organization_id="org-new",
            admins=["admin1"],
            members=["member1"],
            members_with_roles=[{"user_id": "user1", "role": "admin"}],
            metadata={"updated": True},
            max_budget=500.0,
            soft_budget=400.0,
            models=["gpt-4", "claude-3"],
            max_parallel_requests=20,
            tpm_limit=100000,
            rpm_limit=1000,
            budget_duration="weekly",
            blocked=False,
            object_permission_id="perm-new",
        )
        assert updated.team_alias == "Fully Updated"

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
            "budget_id": "b1",
            "created_by": "admin",
            "updated_by": "admin",
        }
        updated = await repo.update_organization(
            organization_id="org-1",
            updated_by="admin",
            organization_alias="New Name",
        )
        assert updated.organization_alias == "New Name"

    @pytest.mark.asyncio
    async def test_update_organization_all_fields(self, repo):
        repo._prisma_client.db.litellm_organizationtable._records["org-full"] = {
            "organization_id": "org-full",
            "organization_alias": "Old Name",
            "budget_id": "b1",
            "created_by": "admin",
            "updated_by": "admin",
        }
        updated = await repo.update_organization(
            organization_id="org-full",
            updated_by="admin",
            organization_alias="Fully Updated",
            budget_id="budget-new",
            metadata={"updated": True},
            models=["gpt-4", "claude-3"],
            object_permission_id="perm-new",
        )
        assert updated.organization_alias == "Fully Updated"

    @pytest.mark.asyncio
    async def test_delete_organization(self, repo):
        repo._prisma_client.db.litellm_organizationtable._records["org-1"] = {
            "organization_id": "org-1",
            "organization_alias": "Acme",
            "budget_id": "b1",
            "created_by": "admin",
            "updated_by": "admin",
        }
        deleted = await repo.delete_organization("org-1")
        assert deleted is not None

    @pytest.mark.asyncio
    async def test_update_spend(self, repo):
        repo._prisma_client.db.litellm_organizationtable._records["org-1"] = {
            "organization_id": "org-1",
            "organization_alias": "Acme",
            "spend": 0.0,
            "budget_id": "b1",
            "created_by": "admin",
            "updated_by": "admin",
        }
        org = await repo.update_spend("org-1", 100.0)
        assert org.spend == 100.0

    @pytest.mark.asyncio
    async def test_find_by_alias(self, repo):
        repo._prisma_client.db.litellm_organizationtable._records["org-1"] = {
            "organization_id": "org-1",
            "organization_alias": "Acme",
            "budget_id": "b1",
            "created_by": "admin",
            "updated_by": "admin",
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
    async def test_update_project_all_fields(self, repo):
        repo._prisma_client.db.litellm_projecttable._records["proj-full"] = {
            "project_id": "proj-full",
            "project_alias": "Old Name",
        }
        updated = await repo.update_project(
            project_id="proj-full",
            updated_by="admin",
            project_alias="Fully Updated",
            description="New description",
            team_id="team-new",
            budget_id="budget-new",
            metadata={"updated": True},
            models=["gpt-4", "claude-3"],
            model_rpm_limit={"gpt-4": 200},
            model_tpm_limit={"gpt-4": 20000},
            blocked=False,
            object_permission_id="perm-new",
        )
        assert updated.project_alias == "Fully Updated"

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
    async def test_update_permission_all_fields(self, repo):
        repo._prisma_client.db.litellm_objectpermissiontable._records["perm-full"] = {
            "object_permission_id": "perm-full",
            "models": [],
        }
        updated = await repo.update_permission(
            object_permission_id="perm-full",
            mcp_servers=["server-new"],
            mcp_access_groups=["group-new"],
            mcp_tool_permissions={"tool": ["exec"]},
            vector_stores=["store-new"],
            agents=["agent-new"],
            agent_access_groups=["ag-new"],
            models=["gpt-4", "claude-3"],
            blocked_tools=["blocked-tool"],
            mcp_toolsets=["toolset-new"],
            search_tools=["search-new"],
        )
        assert updated.mcp_servers == ["server-new"]

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
    async def test_create(self, repo):
        record = await repo.create(
            data={
                "credential_name": "my-api-key",
                "credential_values": {"api_key": "encrypted_secret"},
                "credential_info": {"provider": "openai"},
                "created_by": "admin",
                "updated_by": "admin",
            }
        )
        assert record.credential_name == "my-api-key"
        cred = repo._to_model(record)
        assert cred.credential_name == "my-api-key"
        assert cred.credential_info == {"provider": "openai"}
        assert cred.credential_values == {"api_key": "encrypted_secret"}

    @pytest.mark.asyncio
    async def test_find_by_name_returns_stored_values_without_decryption(self, repo):
        repo._prisma_client.db.litellm_credentialstable._records["my-key"] = {
            "credential_id": "cred-1",
            "credential_name": "my-key",
            "credential_values": {"api_key": "encrypted_secret"},
            "credential_info": {"provider": "openai"},
        }
        cred = await repo.find_by_name("my-key")
        assert isinstance(cred, CredentialItem)
        assert cred.credential_values == {"api_key": "encrypted_secret"}
        assert cred.credential_info == {"provider": "openai"}

    @pytest.mark.asyncio
    async def test_find_by_name_missing(self, repo):
        assert await repo.find_by_name("nonexistent") is None

    @pytest.mark.asyncio
    async def test_update_by_name(self, repo):
        repo._prisma_client.db.litellm_credentialstable._records["my-key"] = {
            "credential_id": "cred-1",
            "credential_name": "my-key",
            "credential_values": {"api_key": "old"},
            "credential_info": {},
        }
        await repo.update_by_name(
            "my-key",
            data={"credential_values": {"api_key": "new"}, "updated_by": "admin"},
        )
        cred = await repo.find_by_name("my-key")
        assert cred.credential_values == {"api_key": "new"}

    @pytest.mark.asyncio
    async def test_delete_by_name(self, repo):
        repo._prisma_client.db.litellm_credentialstable._records["my-key"] = {
            "credential_id": "cred-1",
            "credential_name": "my-key",
            "credential_values": {"api_key": "secret"},
            "credential_info": {},
        }
        await repo.delete_by_name("my-key")
        assert await repo.find_by_name("my-key") is None

    @pytest.mark.asyncio
    async def test_find_all(self, repo):
        repo._prisma_client.db.litellm_credentialstable._records["k1"] = {
            "credential_name": "k1",
            "credential_values": {"api_key": "a"},
            "credential_info": {},
        }
        repo._prisma_client.db.litellm_credentialstable._records["k2"] = {
            "credential_name": "k2",
            "credential_values": {"api_key": "b"},
            "credential_info": {},
        }
        records = await repo.find_all()
        assert len(records) == 2

    def test_prisma_client_none_raises(self):
        repo = CredentialsRepository(None)
        with pytest.raises(RuntimeError, match="No DB Connected"):
            _ = repo.table


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

    @pytest.mark.asyncio
    async def test_reconcile_config_with_db_values(self, repo):
        repo._prisma_client.db.litellm_config._records["general_settings"] = {
            "param_name": "general_settings",
            "param_value": '{"master_key": "db-key", "db_only": "from_db"}',
        }
        repo._prisma_client.db.litellm_config._records["router_settings"] = {
            "param_name": "router_settings",
            "param_value": '{"timeout": 60}',
        }
        yaml_config = {
            "general_settings": {"master_key": "yaml-key", "yaml_only": "from_yaml"},
        }
        result = await repo.reconcile_config(yaml_config, store_model_in_db=True)
        assert result["general_settings"]["master_key"] == "db-key"
        assert result["general_settings"]["yaml_only"] == "from_yaml"
        assert result["general_settings"]["db_only"] == "from_db"
        assert result["router_settings"]["timeout"] == 60

    @pytest.mark.asyncio
    @patch("litellm.repositories.config_repository.decrypt_value_helper")
    async def test_reconcile_config_with_environment_variables(
        self, mock_decrypt, repo
    ):
        mock_decrypt.side_effect = lambda value, **kw: f"decrypted_{value}"
        repo._prisma_client.db.litellm_config._records["environment_variables"] = {
            "param_name": "environment_variables",
            "param_value": '{"api_key": "encrypted_key", "secret": "encrypted_secret"}',
        }
        yaml_config = {}
        result = await repo.reconcile_config(yaml_config, store_model_in_db=True)
        assert "environment_variables" in result
        assert "api_key" in result["environment_variables"]
        assert "API_KEY" in result["environment_variables"]

    @pytest.mark.asyncio
    async def test_reconcile_config_none_values_preserved(self, repo):
        repo._prisma_client.db.litellm_config._records["general_settings"] = {
            "param_name": "general_settings",
            "param_value": '{"new_key": "value", "null_key": null}',
        }
        yaml_config = {"general_settings": {"existing": "keep"}}
        result = await repo.reconcile_config(yaml_config, store_model_in_db=True)
        assert result["general_settings"]["existing"] == "keep"
        assert result["general_settings"]["new_key"] == "value"

    def test_update_config_fields_non_dict(self, repo):
        config = {"litellm_settings": "old_value"}
        result = repo._update_config_fields(
            current_config=config,
            param_name="litellm_settings",
            db_param_value="new_value",
        )
        assert result["litellm_settings"] == "new_value"

    def test_update_config_fields_new_param(self, repo):
        config = {}
        result = repo._update_config_fields(
            current_config=config,
            param_name="router_settings",
            db_param_value={"timeout": 30},
        )
        assert result["router_settings"] == {"timeout": 30}

    @patch("litellm.repositories.config_repository.decrypt_value_helper")
    def test_decrypt_env_variables_non_string(self, mock_decrypt, repo):
        mock_decrypt.side_effect = lambda value, **kw: value
        env_vars = {"string_val": "encrypted", "int_val": 123, "bool_val": True}
        result = repo._decrypt_env_variables(env_vars)
        assert result["int_val"] == "123"
        assert result["bool_val"] == "True"

    @patch("litellm.repositories.config_repository.decrypt_value_helper")
    def test_decrypt_env_variables_none_value(self, mock_decrypt, repo):
        mock_decrypt.return_value = None
        env_vars = {"key": "value"}
        result = repo._decrypt_env_variables(env_vars)
        assert "key" not in result


class TestVerificationTokenRepositoryExtended:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return VerificationTokenRepository(client)

    @pytest.mark.asyncio
    async def test_find_active_tokens(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-active"] = {
            "token": "sk-active",
            "blocked": False,
            "expires": None,
        }
        tokens = await repo.find_active_tokens()
        assert len(tokens) >= 1

    @pytest.mark.asyncio
    async def test_delete_token_with_audit(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-delete"] = {
            "token": "sk-delete",
            "key_name": "Delete Me",
            "spend": 0.0,
        }

        class MockTx:
            def __init__(self, client):
                self.litellm_deletedverificationtoken = (
                    client.db.litellm_deletedverificationtoken
                )
                self.litellm_verificationtoken = client.db.litellm_verificationtoken

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        repo._prisma_client.db.tx = lambda: MockTx(repo._prisma_client)
        deleted = await repo.delete_token(
            "sk-delete",
            deleted_by="admin",
            deleted_by_api_key="sk-admin",
            litellm_changed_by="system",
        )
        assert deleted is not None
        assert deleted.token == "sk-delete"

    @pytest.mark.asyncio
    async def test_delete_token_nonexistent(self, repo):
        result = await repo.delete_token("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_token_archive_serialization(self, repo):
        """Archived token must store JSON columns as strings, map org_id onto the
        organization_id column, preserve budget_id, and drop relation-only fields
        that don't exist on LiteLLM_DeletedVerificationToken."""
        repo._prisma_client.db.litellm_verificationtoken._records["sk-arch"] = {
            "token": "sk-arch",
            "key_name": "Archive Me",
            "aliases": json.dumps({"a": "b"}),
            "metadata": json.dumps({"team": "x"}),
            "permissions": json.dumps({"read": True}),
            "spend": 5.0,
            "organization_id": "org-9",
            "budget_id": "budget-9",
            "budget_limits": [{"model": "gpt-4", "budget": 1.0}],
        }

        class MockTx:
            def __init__(self, client):
                self.litellm_deletedverificationtoken = (
                    client.db.litellm_deletedverificationtoken
                )
                self.litellm_verificationtoken = client.db.litellm_verificationtoken

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        repo._prisma_client.db.tx = lambda: MockTx(repo._prisma_client)

        await repo.delete_token("sk-arch", deleted_by="admin")

        archived = list(
            repo._prisma_client.db.litellm_deletedverificationtoken._records.values()
        )[0]

        assert isinstance(archived["aliases"], str)
        assert json.loads(archived["aliases"]) == {"a": "b"}
        assert isinstance(archived["metadata"], str)
        assert isinstance(archived["permissions"], str)

        assert archived["organization_id"] == "org-9"
        assert "org_id" not in archived

        assert archived["budget_id"] == "budget-9"

        for relation_field in (
            "object_permission",
            "litellm_budget_table",
            "budget_limits",
        ):
            assert relation_field not in archived

        assert (
            "sk-arch" not in repo._prisma_client.db.litellm_verificationtoken._records
        )

    @pytest.mark.asyncio
    async def test_find_by_id_maps_org_and_budget_columns(self, repo):
        """Reading a token must surface the organization_id column as org_id and
        populate budget_id rather than silently dropping them."""
        repo._prisma_client.db.litellm_verificationtoken._records["sk-read"] = {
            "token": "sk-read",
            "organization_id": "org-7",
            "budget_id": "budget-7",
        }
        token = await repo.find_by_id("sk-read")
        assert token is not None
        assert token.org_id == "org-7"
        assert token.budget_id == "budget-7"

    @pytest.mark.asyncio
    async def test_update_token_all_fields(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
        }
        updated = await repo.update_token(
            token="sk-test",
            updated_by="admin",
            key_name="Updated",
            key_alias="new-alias",
            max_budget=500.0,
            expires=datetime(2025, 12, 31),
            models=["gpt-4", "gpt-3.5-turbo"],
            aliases={"a": "b"},
            config={"c": "d"},
            max_parallel_requests=10,
            metadata={"m": "data"},
            tpm_limit=5000,
            rpm_limit=50,
            budget_duration="daily",
            allowed_cache_controls=["cache"],
            allowed_routes=["/v1/chat"],
            permissions={"write": True},
            blocked=False,
            object_permission_id="perm-2",
            access_group_ids=["g1", "g2"],
        )
        assert updated.key_name == "Updated"

    @pytest.mark.asyncio
    async def test_to_model_with_json_fields(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-json"] = {
            "token": "sk-json",
            "aliases": '{"alias1": "value1"}',
            "config": '{"setting": "val"}',
            "permissions": '{"read": true}',
            "metadata": '{"key": "value"}',
            "model_spend": '{"gpt-4": 10.0}',
            "model_max_budget": '{"gpt-4": 100.0}',
            "router_settings": '{"timeout": 30}',
            "budget_limits": '[{"limit": 50}]',
            "litellm_budget_table": '{"budget_id": "b1"}',
        }
        token = await repo.find_by_id("sk-json")
        assert token is not None
        assert token.aliases == {"alias1": "value1"}
        assert token.config == {"setting": "val"}


class TestTeamRepositoryExtended:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return TeamRepository(client)

    @pytest.mark.asyncio
    async def test_delete_team_with_audit(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-delete"] = {
            "team_id": "team-delete",
            "team_alias": "Delete Team",
            "members": [],
            "admins": [],
            "models": [],
            "spend": 0.0,
        }

        class MockTx:
            def __init__(self, client):
                self.litellm_deletedteamtable = client.db.litellm_deletedteamtable
                self.litellm_teamtable = client.db.litellm_teamtable

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        repo._prisma_client.db.tx = lambda: MockTx(repo._prisma_client)
        deleted = await repo.delete_team(
            "team-delete",
            deleted_by="admin",
            deleted_by_api_key="sk-admin",
            litellm_changed_by="system",
        )
        assert deleted is not None
        assert deleted.team_id == "team-delete"

    @pytest.mark.asyncio
    async def test_delete_team_nonexistent(self, repo):
        result = await repo.delete_team("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_team_with_full_data(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-full"] = {
            "team_id": "team-full",
            "team_alias": "Full Team",
            "organization_id": "org-1",
            "object_permission_id": "perm-1",
            "members": ["m1", "m2"],
            "admins": ["a1"],
            "members_with_roles": '[{"user_id": "u1", "role": "admin"}]',
            "metadata": '{"key": "value"}',
            "max_budget": 1000.0,
            "soft_budget": 800.0,
            "spend": 150.0,
            "models": ["gpt-4"],
            "max_parallel_requests": 10,
            "tpm_limit": 5000,
            "rpm_limit": 50,
            "budget_duration": "monthly",
            "budget_reset_at": "2025-01-01T00:00:00",
            "blocked": True,
            "model_spend": '{"gpt-4": 100.0}',
            "model_max_budget": '{"gpt-4": 500.0}',
            "router_settings": '{"timeout": 30}',
            "team_member_permissions": ["read"],
            "access_group_ids": ["group-1"],
            "policies": ["policy-1"],
            "model_id": 42,
            "allow_team_guardrail_config": True,
        }

        class MockTx:
            def __init__(self, client):
                self.litellm_deletedteamtable = client.db.litellm_deletedteamtable
                self.litellm_teamtable = client.db.litellm_teamtable

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        repo._prisma_client.db.tx = lambda: MockTx(repo._prisma_client)
        deleted = await repo.delete_team(
            "team-full",
            deleted_by="admin",
            deleted_by_api_key="sk-admin",
            litellm_changed_by="system",
        )
        assert deleted is not None
        assert deleted.team_id == "team-full"
        assert deleted.organization_id == "org-1"
        assert deleted.max_budget == 1000.0

    @pytest.mark.asyncio
    async def test_to_model_with_json_fields(self, repo):
        repo._prisma_client.db.litellm_teamtable._records["team-json"] = {
            "team_id": "team-json",
            "metadata": '{"key": "value"}',
            "model_spend": '{"gpt-4": 10.0}',
            "model_max_budget": '{"gpt-4": 100.0}',
            "router_settings": '{"timeout": 30}',
            "budget_limits": '[{"budget_duration": "1d", "max_budget": 50.0}]',
            "members_with_roles": '[{"user_id": "u1", "role": "admin"}]',
            "members": [],
            "admins": [],
            "models": [],
        }
        team = await repo.find_by_id("team-json")
        assert team is not None
        assert team.metadata == {"key": "value"}
        assert len(team.members_with_roles) == 1
        assert team.members_with_roles[0].user_id == "u1"


class TestUserRepositoryExtended:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return UserRepository(client)

    @pytest.mark.asyncio
    async def test_delete_user_simple(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-delete"] = {
            "user_id": "user-delete",
            "user_email": "delete@example.com",
            "teams": [],
            "models": [],
            "spend": 0.0,
        }
        deleted = await repo.delete_user("user-delete")
        assert deleted is not None
        assert deleted.user_id == "user-delete"

    @pytest.mark.asyncio
    async def test_delete_user_nonexistent(self, repo):
        result = await repo.delete_user("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_user_all_fields(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-update"] = {
            "user_id": "user-update",
            "teams": [],
            "models": [],
        }
        updated = await repo.update_user(
            user_id="user-update",
            user_alias="newalias",
            team_id="team-new",
            sso_user_id="sso-new",
            organization_id="org-1",
            password="new-hashed-pw",
            teams=["team-1", "team-2"],
            user_role="admin",
            max_budget=1000.0,
            user_email="new@example.com",
            models=["gpt-4"],
            metadata={"pref": "dark"},
            max_parallel_requests=20,
            tpm_limit=10000,
            rpm_limit=100,
            budget_duration="monthly",
            allowed_cache_controls=["no-cache"],
            policies=["policy-1"],
            object_permission_id="perm-new",
        )
        assert updated.user_email == "new@example.com"


class TestProjectRepositoryExtended:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return ProjectRepository(client)

    @pytest.mark.asyncio
    async def test_delete_project_simple(self, repo):
        repo._prisma_client.db.litellm_projecttable._records["proj-delete"] = {
            "project_id": "proj-delete",
            "project_alias": "Delete Project",
            "spend": 0.0,
        }
        deleted = await repo.delete_project("proj-delete")
        assert deleted is not None


class TestBudgetRepositoryExtended:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return BudgetRepository(client)

    @pytest.mark.asyncio
    async def test_update_budget_all_fields(self, repo):
        repo._prisma_client.db.litellm_budgettable._records["budget-update"] = {
            "budget_id": "budget-update",
            "max_budget": 100.0,
        }
        updated = await repo.update_budget(
            budget_id="budget-update",
            updated_by="admin",
            max_budget=500.0,
            soft_budget=400.0,
            max_parallel_requests=15,
            tpm_limit=20000,
            rpm_limit=200,
            model_max_budget={"gpt-4": 200.0},
            budget_duration="weekly",
            allowed_models=["gpt-4", "claude-3"],
        )
        assert updated.max_budget == 500.0


class TestModelRepositoryExtended:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return ModelRepository(client)

    @pytest.mark.asyncio
    @patch(
        "litellm.repositories.model_repository.decrypt_value_helper",
        side_effect=lambda value, **kw: value,
    )
    async def test_find_by_team_id(self, mock_decrypt, repo):
        repo._prisma_client.db.litellm_proxymodeltable._records["model-1"] = {
            "model_id": "model-1",
            "model_name": "gpt-4",
            "litellm_params": '{"api_key": "sk-test"}',
            "model_info": '{"team_id": "team-1"}',
            "blocked": False,
        }
        repo._prisma_client.db.litellm_proxymodeltable._records["model-2"] = {
            "model_id": "model-2",
            "model_name": "claude-3",
            "litellm_params": '{"api_key": "sk-other"}',
            "model_info": '{"team_id": "team-2"}',
            "blocked": False,
        }
        models = await repo.find_by_team_id("team-1")
        assert len(models) == 1
        assert models[0].model_name == "gpt-4"


class TestBaseRepositoryExtended:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return BudgetRepository(client)

    @pytest.mark.asyncio
    async def test_find_many_with_pagination(self, repo):
        repo._prisma_client.db.litellm_budgettable._records = {
            "b1": {"budget_id": "b1", "max_budget": 100.0},
            "b2": {"budget_id": "b2", "max_budget": 200.0},
            "b3": {"budget_id": "b3", "max_budget": 300.0},
        }
        budgets = await repo.find_many(skip=0, take=2, order={"budget_id": "asc"})
        assert len(budgets) >= 2

    @pytest.mark.asyncio
    async def test_find_many_with_where(self, repo):
        repo._prisma_client.db.litellm_budgettable._records = {
            "b1": {"budget_id": "b1", "max_budget": 100.0},
        }
        budgets = await repo.find_many(where={"budget_id": "b1"})
        assert len(budgets) >= 1

    @pytest.mark.asyncio
    async def test_to_model_list_with_none(self, repo):
        result = repo._to_model_list([None, None])
        assert result == []


class _SampleDomainModel(DomainModel):
    budget_id: Optional[str] = None
    max_budget: Optional[float] = None


class TestDomainModelExtended:
    def test_from_db_record_none_raises(self):
        with pytest.raises(ValueError, match="Cannot create domain model from None"):
            DomainModel.from_db_record(None)

    def test_from_db_record_dict(self):
        model = _SampleDomainModel.from_db_record(
            {"budget_id": "b1", "max_budget": 100.0}
        )
        assert model.budget_id == "b1"

    def test_from_db_record_model_dump(self):
        class MockRecordWithModelDump:
            def model_dump(self):
                return {"budget_id": "b2", "max_budget": 200.0}

        model = _SampleDomainModel.from_db_record(MockRecordWithModelDump())
        assert model.budget_id == "b2"

    def test_to_db_dict(self):
        model = _SampleDomainModel(budget_id="b3", max_budget=300.0)
        data = model.to_db_dict()
        assert data["budget_id"] == "b3"
        assert data["max_budget"] == 300.0


class TestTeamRepositoryArchiveData:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return TeamRepository(client)

    def test_build_archive_data_minimal_fields(self, repo):

        team = LiteLLM_TeamTable(team_id="team-minimal")
        archive_data = repo._build_archive_data(team)
        assert archive_data["team_id"] == "team-minimal"
        assert archive_data["admins"] == []
        assert archive_data["members"] == []
        assert archive_data["models"] == []
        assert archive_data["spend"] == 0.0
        assert archive_data["blocked"] is False
        assert "team_alias" not in archive_data
        assert "organization_id" not in archive_data
        assert "object_permission_id" not in archive_data
        assert "members_with_roles" not in archive_data
        assert "metadata" not in archive_data
        assert "max_budget" not in archive_data
        assert "soft_budget" not in archive_data
        assert "max_parallel_requests" not in archive_data
        assert "tpm_limit" not in archive_data
        assert "rpm_limit" not in archive_data
        assert "budget_duration" not in archive_data
        assert "budget_reset_at" not in archive_data
        assert "model_spend" not in archive_data
        assert "model_max_budget" not in archive_data
        assert "router_settings" not in archive_data
        assert "model_id" not in archive_data

    def test_build_archive_data_excludes_invalid_columns(self, repo):

        team = LiteLLM_TeamTable(
            team_id="team-1",
            team_alias="My Team",
            admins=["admin1"],
            members=["member1"],
            models=["gpt-4"],
            default_team_member_models=["gpt-3.5-turbo"],
        )
        archive_data = repo._build_archive_data(team)
        assert "default_team_member_models" not in archive_data
        assert "budget_limits" not in archive_data
        assert archive_data["team_id"] == "team-1"
        assert archive_data["team_alias"] == "My Team"
        assert archive_data["admins"] == ["admin1"]
        assert archive_data["members"] == ["member1"]
        assert archive_data["models"] == ["gpt-4"]

    def test_build_archive_data_with_all_valid_fields(self, repo):
        from datetime import datetime

        from litellm.models.team import Member

        team = LiteLLM_TeamTable(
            team_id="team-full",
            team_alias="Full Team",
            organization_id="org-1",
            object_permission_id="perm-1",
            admins=["admin1", "admin2"],
            members=["m1", "m2"],
            members_with_roles=[Member(user_id="u1", role="admin")],
            metadata={"key": "value"},
            max_budget=1000.0,
            soft_budget=800.0,
            spend=150.0,
            models=["gpt-4", "claude-3"],
            max_parallel_requests=10,
            tpm_limit=5000,
            rpm_limit=50,
            budget_duration="monthly",
            budget_reset_at=datetime(2025, 1, 1),
            blocked=True,
            model_spend={"gpt-4": 100.0},
            model_max_budget={"gpt-4": 500.0},
            router_settings={"timeout": 30},
            team_member_permissions=["read"],
            access_group_ids=["group-1"],
            policies=["policy-1"],
            model_id=42,
            allow_team_guardrail_config=True,
        )
        archive_data = repo._build_archive_data(team)
        assert archive_data["team_id"] == "team-full"
        assert archive_data["organization_id"] == "org-1"
        assert archive_data["object_permission_id"] == "perm-1"
        assert archive_data["max_budget"] == 1000.0
        assert archive_data["soft_budget"] == 800.0
        assert archive_data["spend"] == 150.0
        assert archive_data["blocked"] is True
        assert archive_data["model_id"] == 42
        assert archive_data["allow_team_guardrail_config"] is True
        assert "members_with_roles" in archive_data
        assert "metadata" in archive_data
        assert "model_spend" in archive_data
        assert "model_max_budget" in archive_data
        assert "router_settings" in archive_data


class TestConfigRepositoryDeepCopy:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return ConfigRepository(client)

    @pytest.mark.asyncio
    async def test_reconcile_config_does_not_mutate_original(self, repo):
        import copy

        repo._prisma_client.db.litellm_config._records["general_settings"] = {
            "param_name": "general_settings",
            "param_value": '{"db_key": "db_value", "nested": {"db_nested": "from_db"}}',
        }
        original_config = {
            "general_settings": {
                "yaml_key": "yaml_value",
                "nested": {"yaml_nested": "from_yaml"},
            }
        }
        original_copy = copy.deepcopy(original_config)
        result = await repo.reconcile_config(original_config, store_model_in_db=True)
        assert original_config == original_copy
        assert result["general_settings"]["db_key"] == "db_value"
        assert result["general_settings"]["yaml_key"] == "yaml_value"
        assert result["general_settings"]["nested"]["db_nested"] == "from_db"
        assert result["general_settings"]["nested"]["yaml_nested"] == "from_yaml"

    @pytest.mark.asyncio
    async def test_reconcile_config_repeated_calls_independent(self, repo):
        repo._prisma_client.db.litellm_config._records["general_settings"] = {
            "param_name": "general_settings",
            "param_value": '{"db_key": "db_value"}',
        }
        yaml_config = {"general_settings": {"yaml_key": "yaml_value"}}
        result1 = await repo.reconcile_config(yaml_config, store_model_in_db=True)
        result1["general_settings"]["modified"] = "in_result1"
        result2 = await repo.reconcile_config(yaml_config, store_model_in_db=True)
        assert "modified" not in yaml_config.get("general_settings", {})
        assert "modified" not in result2.get("general_settings", {})


class TestPrismaTableRepository:
    def test_table_property_returns_named_delegate(self):
        from litellm.repositories.table_repositories import (
            AgentsRepository,
            PolicyRepository,
        )

        prisma_client = MagicMock()
        agents = AgentsRepository(prisma_client)
        policy = PolicyRepository(prisma_client)

        assert agents.table is prisma_client.db.litellm_agentstable
        assert policy.table is prisma_client.db.litellm_policytable
        assert agents.table is not policy.table

    def test_table_access_raises_without_db(self):
        from litellm.repositories.table_repositories import SpendLogsRepository

        repo = SpendLogsRepository(None)
        with pytest.raises(RuntimeError, match="No DB Connected"):
            _ = repo.table

    def test_each_repository_binds_its_own_table_name(self):
        import litellm.repositories.table_repositories as tr

        prisma_client = MagicMock()
        repos = [
            obj
            for name, obj in vars(tr).items()
            if isinstance(obj, type)
            and issubclass(obj, tr.PrismaTableRepository)
            and obj is not tr.PrismaTableRepository
        ]
        assert len(repos) >= 40
        seen = set()
        for repo_cls in repos:
            name = repo_cls.table_name
            assert name.startswith("litellm_")
            assert name not in seen, f"duplicate table_name {name}"
            seen.add(name)
            assert repo_cls(prisma_client).table is getattr(prisma_client.db, name)


def _json_path_equals(
    metadata: Optional[Dict[str, Any]], path: List[str], expected: Any
) -> bool:
    """Reproduce Postgres jsonb path-equals semantics: a missing path yields
    SQL NULL, which never matches `equals`."""
    value: Any = metadata
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return value == expected


class _ScimAwareUserTable:
    """Fake LiteLLM_UserTable whose count() applies the JSON `where` filter the
    way Postgres would, so count_billable_users is checked against an
    independent model of the filter rather than echoing its own where dict."""

    def __init__(self, metadatas: List[Optional[Dict[str, Any]]]):
        self._metadatas = metadatas

    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        if where is None:
            return len(self._metadatas)
        json_filter = where["metadata"]
        path = json_filter["path"]
        expected = getattr(json_filter["equals"], "data", json_filter["equals"])
        return sum(
            1
            for metadata in self._metadatas
            if _json_path_equals(metadata, path, expected)
        )


class TestCountBillableUsers:
    def _repo(self, metadatas: List[Optional[Dict[str, Any]]]) -> UserRepository:
        client = MockPrismaClient()
        client.db.litellm_usertable = _ScimAwareUserTable(metadatas)
        return UserRepository(client)

    @pytest.mark.asyncio
    async def test_excludes_only_scim_deactivated_users(self):
        repo = self._repo(
            [
                {},
                {"scim_active": True},
                {"scim_active": True},
                {"scim_active": None},
                {"other": "x"},
                {"scim_active": False},
            ]
        )
        assert await repo.count_billable_users() == 5

    @pytest.mark.asyncio
    async def test_absent_null_and_true_all_count_as_billable(self):
        repo = self._repo([{}, {"scim_active": None}, {"scim_active": True}])
        assert await repo.count_billable_users() == 3

    @pytest.mark.asyncio
    async def test_all_deactivated_returns_zero(self):
        repo = self._repo([{"scim_active": False}, {"scim_active": False}])
        assert await repo.count_billable_users() == 0

    @pytest.mark.asyncio
    async def test_floors_at_zero_when_deactivated_exceeds_total(self):
        """The total and deactivated counts are separate queries; a burst of
        deactivations between them must never yield a negative seat count."""

        class _RacyTable:
            async def count(self, where=None):
                return 5 if where is not None else 2

        client = MockPrismaClient()
        client.db.litellm_usertable = _RacyTable()
        repo = UserRepository(client)
        assert await repo.count_billable_users() == 0
