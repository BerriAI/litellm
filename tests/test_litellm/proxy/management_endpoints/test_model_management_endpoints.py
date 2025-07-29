import json
import os
import sys
import uuid
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
from litellm.proxy._types import (
    LiteLLM_ModelTable,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    ModelManagementAuthChecks,
    clear_cache,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.router import Deployment, LiteLLM_Params, updateDeployment


class MockPrismaClient:
    def __init__(self, team_exists: bool = True, user_admin: bool = True):
        self.team_exists = team_exists
        self.user_admin = user_admin
        self.db = self

    async def find_unique(self, where):
        if self.team_exists:
            return LiteLLM_TeamTable(
                team_id=where["team_id"],
                team_alias="test_team",
                members_with_roles=[
                    Member(
                        user_id="test_user", role="admin" if self.user_admin else "user"
                    )
                ],
            )
        return None

    @property
    def litellm_teamtable(self):
        return self


class MockLLMRouter:
    def __init__(self):
        self.model_list = ["model1", "model2"]
        self.model_names = {"model1": True, "model2": True}
        self.cleared = False

    def get_deployment(self, model_id):
        return {"model_id": model_id} if model_id in self.model_list else None

    def delete_deployment(self, id):
        if id in self.model_list:
            self.model_list.remove(id)
            self.model_names.pop(id, None)


class MockProxyConfig:
    def __init__(self, success=True):
        self.success = success
        self.deployment_called = False

    async def add_deployment(self, prisma_client, proxy_logging_obj):
        self.deployment_called = True
        if not self.success:
            raise Exception("Failed to add deployment")
        return True


class TestModelManagementAuthChecks:
    def setup_method(self):
        """Setup test cases"""
        self.admin_user = UserAPIKeyAuth(
            user_id="test_admin", user_role=LitellmUserRoles.PROXY_ADMIN
        )

        self.normal_user = UserAPIKeyAuth(
            user_id="test_user", user_role=LitellmUserRoles.INTERNAL_USER
        )

        self.team_admin_user = UserAPIKeyAuth(
            user_id="test_user",
            team_id="test_team",
            user_role=LitellmUserRoles.INTERNAL_USER,
        )

    @pytest.mark.asyncio
    async def test_can_user_make_team_model_call_admin_success(self):
        """Test that admin users can make team model calls"""
        result = ModelManagementAuthChecks.can_user_make_team_model_call(
            team_id="test_team", user_api_key_dict=self.admin_user, premium_user=True
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_can_user_make_team_model_call_non_premium_fails(self):
        """Test that non-premium users cannot make team model calls"""
        with pytest.raises(Exception) as exc_info:
            ModelManagementAuthChecks.can_user_make_team_model_call(
                team_id="test_team",
                user_api_key_dict=self.admin_user,
                premium_user=False,
            )
        assert "403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_can_user_make_team_model_call_team_admin_success(self):
        """Test that team admins can make calls for their team"""
        team_obj = LiteLLM_TeamTable(
            team_id="test_team",
            team_alias="test_team",
            members_with_roles=[
                Member(user_id=self.team_admin_user.user_id, role="admin")
            ],
        )

        result = ModelManagementAuthChecks.can_user_make_team_model_call(
            team_id="test_team",
            user_api_key_dict=self.team_admin_user,
            team_obj=team_obj,
            premium_user=True,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_allow_team_model_action_success(self):
        """Test successful team model action"""
        model_params = Deployment(
            model_name="test_model",
            litellm_params=LiteLLM_Params(model="test_model", team_id="test_team"),
            model_info={"team_id": "test_team"},
        )
        prisma_client = MockPrismaClient(team_exists=True)

        result = await ModelManagementAuthChecks.allow_team_model_action(
            model_params=model_params,
            user_api_key_dict=self.admin_user,
            prisma_client=prisma_client,
            premium_user=True,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_allow_team_model_action_non_premium_fails(self):
        """Test team model action fails for non-premium users"""
        model_params = Deployment(
            model_name="test_model",
            litellm_params=LiteLLM_Params(model="test_model", team_id="test_team"),
            model_info={"team_id": "test_team"},
        )
        prisma_client = MockPrismaClient(team_exists=True)

        with pytest.raises(Exception) as exc_info:
            await ModelManagementAuthChecks.allow_team_model_action(
                model_params=model_params,
                user_api_key_dict=self.admin_user,
                prisma_client=prisma_client,
                premium_user=False,
            )
        assert "403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_allow_team_model_action_nonexistent_team_fails(self):
        """Test team model action fails for non-existent team"""
        model_params = Deployment(
            model_name="test_model",
            litellm_params=LiteLLM_Params(
                model="test_model",
            ),
            model_info={"team_id": "nonexistent_team"},
        )
        prisma_client = MockPrismaClient(team_exists=False)

        with pytest.raises(Exception) as exc_info:
            await ModelManagementAuthChecks.allow_team_model_action(
                model_params=model_params,
                user_api_key_dict=self.admin_user,
                prisma_client=prisma_client,
                premium_user=True,
            )
        assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_can_user_make_model_call_admin_success(self):
        """Test that admin users can make any model call"""
        model_params = Deployment(
            model_name="test_model",
            litellm_params=LiteLLM_Params(
                model="test_model",
            ),
            model_info={"team_id": "test_team"},
        )
        prisma_client = MockPrismaClient(team_exists=True)

        result = await ModelManagementAuthChecks.can_user_make_model_call(
            model_params=model_params,
            user_api_key_dict=self.admin_user,
            prisma_client=prisma_client,
            premium_user=True,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_can_user_make_model_call_normal_user_fails(self):
        """Test that normal users cannot make model calls"""
        model_params = Deployment(
            model_name="test_model",
            litellm_params=LiteLLM_Params(
                model="test_model",
            ),
            model_info={"team_id": "test_team"},
        )
        prisma_client = MockPrismaClient(team_exists=True, user_admin=False)

        with pytest.raises(Exception) as exc_info:
            await ModelManagementAuthChecks.can_user_make_model_call(
                model_params=model_params,
                user_api_key_dict=self.normal_user,
                prisma_client=prisma_client,
                premium_user=True,
            )
        assert "403" in str(exc_info.value)


class MockModelTable:
    def __init__(self, model_aliases: Dict[str, str], include: Optional[dict] = None):
        for alias, model in model_aliases.items():
            setattr(self, alias, model)
        self.id = str(uuid.uuid4())
        self.model_aliases = model_aliases


class MockPrismaDB:
    def __init__(self, model_aliases_list):
        self.litellm_modeltable = self
        self.model_aliases_list = model_aliases_list
        self.update_calls = []

    async def find_many(self, include=None):
        print(f"self.model_aliases_list: {self.model_aliases_list}")
        return [LiteLLM_ModelTable(**aliases) for aliases in self.model_aliases_list]

    async def update(self, where, data):
        self.update_calls.append({"where": where, "data": data})
        return None


class MockPrismaWrapper:
    def __init__(self, model_aliases_list):
        self.litellm_modeltable = MockPrismaDB(model_aliases_list)


class TestDeleteTeamModelAlias:
    @pytest.mark.asyncio
    async def test_delete_team_model_alias_success(self):
        """Test successful deletion of a team model alias"""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            delete_team_model_alias,
        )

        # Setup test data
        model_aliases_list = [
            {
                "id": 1,
                "model_aliases": {
                    "alias1": "public_model_1",
                    "alias2": "public_model_2",
                },
                "updated_by": "test_user",
                "created_by": "test_user",
            },
            {
                "id": 2,
                "model_aliases": {
                    "alias3": "public_model_3",
                    "alias4": "public_model_1",
                },
                "updated_by": "test_user",
                "created_by": "test_user",
            },  # public_model_1 appears twice
        ]

        # Create mock prisma client
        mock_prisma = MockPrismaClient(team_exists=True)
        mock_prisma.db = MockPrismaWrapper(model_aliases_list)

        # Call the function
        await delete_team_model_alias(
            public_model_name="public_model_1", prisma_client=mock_prisma
        )

        # Verify results
        mock_db = mock_prisma.db.litellm_modeltable
        assert (
            len(mock_db.update_calls) == 2
        )  # Should have 2 update calls since public_model_1 appears twice

        # Verify first update
        first_update = mock_db.update_calls[0]
        assert first_update["where"] == {"id": 1}
        assert json.loads(first_update["data"]["model_aliases"]) == {
            "alias2": "public_model_2"
        }

        # Verify second update
        second_update = mock_db.update_calls[1]
        assert second_update["where"] == {"id": 2}
        assert json.loads(second_update["data"]["model_aliases"]) == {
            "alias3": "public_model_3"
        }

    @pytest.mark.asyncio
    async def test_delete_team_model_alias_no_matches(self):
        """Test deletion when no matching model alias exists"""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            delete_team_model_alias,
        )

        # Setup test data with no matching model
        model_aliases_list = [
            {
                "id": 1,
                "model_aliases": {
                    "alias1": "public_model_1",
                    "alias2": "public_model_2",
                },
                "updated_by": "test_user",
                "created_by": "test_user",
            },
            {
                "id": 2,
                "model_aliases": {
                    "alias3": "public_model_3",
                    "alias4": "public_model_4",
                },
                "updated_by": "test_user",
                "created_by": "test_user",
            },
        ]

        # Create mock prisma client
        mock_prisma = MockPrismaClient(team_exists=True)
        mock_prisma.db = MockPrismaWrapper(model_aliases_list)

        # Call the function with non-existent model
        await delete_team_model_alias(
            public_model_name="non_existent_model", prisma_client=mock_prisma
        )

        # Verify no updates were made
        mock_db = mock_prisma.db.litellm_modeltable
        assert len(mock_db.update_calls) == 0


class TestClearCache:
    """
    Tests for the clear_cache function in model_management_endpoints.py
    """

    @pytest.mark.asyncio
    async def test_clear_cache_success(self):
        """
        Test that clear_cache successfully clears router model caches and reloads models.
        """
        mock_router = MagicMock()
        mock_router.model_list = ["openai/gpt-4o", "openai/gpt-4o-mini"]

        mock_config = MagicMock()
        mock_config.add_deployment = AsyncMock(return_value=True)

        mock_prisma = MagicMock()
        mock_logging = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", mock_router), patch(
            "litellm.proxy.proxy_server.proxy_config", mock_config
        ), patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
            "litellm.proxy.proxy_server.proxy_logging_obj", mock_logging
        ), patch(
            "litellm.proxy.proxy_server.verbose_proxy_logger"
        ):
            await clear_cache()

            assert len(mock_router.model_list) == 0

            mock_config.add_deployment.assert_called_once_with(
                prisma_client=mock_prisma, proxy_logging_obj=mock_logging
            )
