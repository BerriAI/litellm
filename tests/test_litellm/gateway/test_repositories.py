"""
Tests for gateway repository layer.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.backend.models.budget import Budget
from litellm.backend.models.model import Model
from litellm.backend.models.team import Team
from litellm.backend.models.user import User
from litellm.backend.models.verification_token import VerificationToken
from litellm.gateway.repositories.base_repository import BaseRepository
from litellm.gateway.repositories.budget_repository import BudgetRepository
from litellm.gateway.repositories.config_repository import ConfigRepository
from litellm.gateway.repositories.model_repository import ModelRepository
from litellm.gateway.repositories.team_repository import TeamRepository
from litellm.gateway.repositories.user_repository import UserRepository
from litellm.gateway.repositories.verification_token_repository import (
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


class TestModelRepository:
    @pytest.fixture
    def repo(self):
        client = MockPrismaClient()
        return ModelRepository(client)

    @pytest.mark.asyncio
    @patch(
        "litellm.gateway.repositories.model_repository.encrypt_value_helper",
        side_effect=lambda v, **kw: f"encrypted_{v}",
    )
    @patch(
        "litellm.gateway.repositories.model_repository.decrypt_value_helper",
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
        "litellm.gateway.repositories.model_repository.decrypt_value_helper",
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
    async def test_add_to_team(self, repo):
        repo._prisma_client.db.litellm_usertable._records["user-1"] = {
            "user_id": "user-1",
            "teams": ["team1"],
            "models": [],
        }

        user = await repo.add_to_team("user-1", "team2")
        assert "team2" in user.teams


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
    async def test_block_token(self, repo):
        repo._prisma_client.db.litellm_verificationtoken._records["sk-test"] = {
            "token": "sk-test",
            "blocked": False,
        }

        token = await repo.block_token("sk-test", updated_by="admin")
        assert token.blocked is True


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
    async def test_reconcile_config_skips_when_store_model_false(self, repo):
        yaml_config = {"general_settings": {"key": "value"}}
        result = await repo.reconcile_config(yaml_config, store_model_in_db=False)
        assert result == yaml_config
