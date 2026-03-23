import json
import os
import sys
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm._uuid import uuid

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

    async def find_many(self, where):
        return []

    @property
    def litellm_teamtable(self):
        return self

    @property
    def litellm_proxymodeltable(self):
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

            assert len(mock_router.model_list) == 2

            assert len(mock_router.auto_routers) == 0

    @pytest.mark.asyncio
    async def test_clear_cache_preserve_config_models(self):
        """
        Test that clear_cache clears DB models and preserves config models.
        """
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            clear_cache,
        )

        # Create mock router with mixed DB and config models
        mock_router = MagicMock()
        mock_router.model_list = [
            {
                "model_name": "gpt-4",
                "model_info": {"id": "db-model-1", "db_model": True},
                "litellm_params": {"model": "gpt-4"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "model_info": {"id": "config-model-1", "db_model": False},
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
            {
                "model_name": "claude-3",
                "model_info": {"id": "db-model-2", "db_model": True},
                "litellm_params": {"model": "claude-3"},
            },
        ]
        mock_router.delete_deployment = MagicMock(return_value=True)
        mock_router.auto_routers = MagicMock()
        mock_router.auto_routers.clear = MagicMock()

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

            # Should have called delete_deployment for both DB models
            assert mock_router.delete_deployment.call_count == 2
            mock_router.delete_deployment.assert_any_call(id="db-model-1")
            mock_router.delete_deployment.assert_any_call(id="db-model-2")

            # Should have cleared auto routers
            mock_router.auto_routers.clear.assert_called_once()

            # Should have called add_deployment to reload DB models
            mock_config.add_deployment.assert_called_once_with(
                prisma_client=mock_prisma, proxy_logging_obj=mock_logging
            )


class TestUpdatePublicModelGroups:
    """Test that update_public_model_groups correctly sets litellm.public_model_groups
    even when get_config() overwrites it with stale DB values."""

    @pytest.mark.asyncio
    async def test_public_model_groups_set_after_get_config(self):
        """
        Regression test: get_config() internally calls _update_config_from_db which
        sets litellm.public_model_groups to the old DB value. The endpoint must set
        the in-memory value AFTER get_config() so the new value is not overwritten.
        """
        import litellm
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            UpdatePublicModelGroupsRequest,
            update_public_model_groups,
        )

        old_db_models = ["db-model-1", "db-model-2"]
        new_models = ["db-model-1", "db-model-2", "config-model-1", "config-model-2"]

        # Simulate get_config() overwriting litellm.public_model_groups with old DB value
        async def mock_get_config(*args, **kwargs):
            # This simulates _update_config_from_db calling setattr(litellm, "public_model_groups", old_value)
            litellm.public_model_groups = old_db_models
            return {"litellm_settings": {"public_model_groups": old_db_models}}

        mock_proxy_config = MagicMock()
        mock_proxy_config.get_config = mock_get_config
        mock_proxy_config.save_config = AsyncMock()

        admin_user = UserAPIKeyAuth(
            user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
        )

        request = UpdatePublicModelGroupsRequest(model_groups=new_models)

        original_value = getattr(litellm, "public_model_groups", None)
        try:
            with patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ), patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ):
                result = await update_public_model_groups(
                    request=request,
                    user_api_key_dict=admin_user,
                )

            # After the endpoint completes, the in-memory value must reflect
            # the NEW models, not the stale DB value
            assert litellm.public_model_groups == new_models
            assert result["public_model_groups"] == new_models
        finally:
            litellm.public_model_groups = original_value

    @pytest.mark.asyncio
    async def test_useful_links_set_after_get_config(self):
        """
        Regression test: same stale-overwrite bug as public_model_groups applies
        to update_useful_links / public_model_groups_links.
        """
        import litellm
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_useful_links,
        )
        from litellm.types.proxy.management_endpoints.model_management_endpoints import (
            UpdateUsefulLinksRequest,
        )

        old_links = {"Old Doc": "https://old.example.com"}
        new_links = {
            "New Doc": "https://new.example.com",
            "API Ref": "https://api.example.com",
        }

        async def mock_get_config(*args, **kwargs):
            litellm.public_model_groups_links = old_links
            return {"litellm_settings": {"public_model_groups_links": old_links}}

        mock_proxy_config = MagicMock()
        mock_proxy_config.get_config = mock_get_config
        mock_proxy_config.save_config = AsyncMock()

        admin_user = UserAPIKeyAuth(
            user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
        )

        request = UpdateUsefulLinksRequest(useful_links=new_links)

        original_value = getattr(litellm, "public_model_groups_links", None)
        try:
            with patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ):
                result = await update_useful_links(
                    request=request,
                    user_api_key_dict=admin_user,
                )

            assert litellm.public_model_groups_links == new_links
            assert result["useful_links"] == new_links
        finally:
            litellm.public_model_groups_links = original_value


class TestTeamModelSiblingRouting:
    """
    Verify that sibling team deployments (same public model name, different
    api_base) are all reachable through routing — no alias overwrite, no
    collapse to a single deployment.
    """

    @pytest.mark.asyncio
    async def test_no_model_aliases_written_for_team_models(self):
        """
        _add_team_model_to_db must NOT write model_aliases (which caused
        the second sibling to overwrite the first). It should only call
        team_model_add to register the public name on the team's models list.
        """
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _add_team_model_to_db,
        )
        from litellm.types.router import ModelInfo

        team_id = "team_no_alias"
        public_name = "gpt-4.1-mini"

        mock_update_team = AsyncMock()

        async def mock_add_model_to_db(model_params, user_api_key_dict, prisma_client):
            return MagicMock(model_id=str(uuid.uuid4()))

        mock_team_model_add = AsyncMock()

        user = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        prisma_client = MockPrismaClient(team_exists=True)

        for api_base in ["https://eastus.example.com", "https://westus.example.com"]:
            dep = Deployment(
                model_name=public_name,
                litellm_params=LiteLLM_Params(
                    model="azure/gpt-4o-mini",
                    api_key="key",
                    api_base=api_base,
                ),
                model_info=ModelInfo(team_id=team_id),
            )
            with patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.update_team",
                mock_update_team,
            ), patch(
                "litellm.proxy.management_endpoints.model_management_endpoints._add_model_to_db",
                side_effect=mock_add_model_to_db,
            ), patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.team_model_add",
                mock_team_model_add,
            ):
                await _add_team_model_to_db(
                    model_params=dep,
                    user_api_key_dict=user,
                    prisma_client=prisma_client,
                )

        mock_update_team.assert_not_called()
        assert mock_team_model_add.call_count == 2

    @pytest.mark.asyncio
    async def test_router_finds_all_sibling_team_deployments(self):
        """
        When two team deployments share team_public_model_name="gpt-4.1-mini",
        the router's _common_checks_available_deployment must return BOTH as
        healthy_deployments (not collapse to one).
        """
        import litellm

        team_id = "teamA"
        public_name = "gpt-4.1-mini"

        router = litellm.Router(
            model_list=[
                {
                    "model_name": f"model_name_{team_id}_uuid1",
                    "litellm_params": {
                        "model": "azure/gpt-4o-mini",
                        "api_key": "key-1",
                        "api_base": "https://eastus.openai.azure.com",
                    },
                    "model_info": {
                        "team_id": team_id,
                        "team_public_model_name": public_name,
                    },
                },
                {
                    "model_name": f"model_name_{team_id}_uuid2",
                    "litellm_params": {
                        "model": "azure/gpt-4o-mini",
                        "api_key": "key-2",
                        "api_base": "https://westus.openai.azure.com",
                    },
                    "model_info": {
                        "team_id": team_id,
                        "team_public_model_name": public_name,
                    },
                },
            ],
        )

        # map_team_model should return the public name (not an internal UUID)
        result = router.map_team_model(public_name, team_id)
        assert result == public_name

        # _common_checks_available_deployment should return both deployments
        model, healthy = router._common_checks_available_deployment(
            model=public_name,
            request_kwargs={"metadata": {"user_api_key_team_id": team_id}},
        )
        assert isinstance(healthy, list)
        assert len(healthy) == 2
        api_bases = {d["litellm_params"]["api_base"] for d in healthy}
        assert api_bases == {
            "https://eastus.openai.azure.com",
            "https://westus.openai.azure.com",
        }


class TestTeamModelUpdate:
    """Test team model update handles team_id consistently with model creation"""

    @pytest.mark.asyncio
    async def test_patch_model_with_team_id_creates_proper_setup(self):
        """Test PATCH with team_id creates unique model name, alias, and team membership like POST does"""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _update_team_model_in_db,
        )
        from litellm.types.router import ModelInfo

        patch_data = updateDeployment(
            model_name="tenant-azure-gpt4",
            model_info=ModelInfo(
                team_id="test_team_123",
                base_model="azure/gpt-4",
            ),
        )
        db_model = Deployment(
            model_name="original-model",
            litellm_params=LiteLLM_Params(model="test_model"),
            model_info=ModelInfo(),
        )
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        prisma_client = MockPrismaClient(team_exists=True)

        with patch(
            "litellm.proxy.proxy_server.premium_user",
            True,
        ), patch(
            "litellm.proxy.management_endpoints.model_management_endpoints.update_team"
        ) as mock_update_team, patch(
            "litellm.proxy.management_endpoints.model_management_endpoints.team_model_add"
        ) as mock_team_model_add:
            result = await _update_team_model_in_db(
                db_model=db_model,
                patch_data=patch_data,
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,  # type: ignore
            )

            assert result.get("model_name", "").startswith("model_name_test_team_123_")
            assert "team_public_model_name" in str(result.get("model_info", ""))
            # update_team must not be called (no model_aliases writes for team models)
            mock_update_team.assert_not_called()
            # team_model_add must be called to add public name to team's models list
            mock_team_model_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_model_with_team_id_validates_permissions(self):
        """Test PATCH with team_id runs same validation as POST for team permissions"""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _update_team_model_in_db,
        )
        from litellm.types.router import ModelInfo

        patch_data = updateDeployment(
            model_name="tenant-azure-gpt4",
            model_info=ModelInfo(team_id="test_team_123"),
        )
        db_model = Deployment(
            model_name="original-model",
            litellm_params=LiteLLM_Params(model="test_model"),
            model_info=ModelInfo(),
        )
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        prisma_client = MockPrismaClient(team_exists=True, user_admin=False)

        with patch(
            "litellm.proxy.proxy_server.premium_user",
            True,
        ):
            with pytest.raises(Exception) as exc_info:
                await _update_team_model_in_db(
                    db_model=db_model,
                    patch_data=patch_data,
                    user_api_key_dict=user_api_key_dict,
                    prisma_client=prisma_client,  # type: ignore
                )
            assert "403" in str(exc_info.value)


class TestModelInfoEndpoint:
    """Test the model_info endpoint for retrieving individual model information"""

    @pytest.mark.asyncio
    async def test_model_info_accessible_model_success(self):
        """Test model_info returns model data for accessible models"""
        from litellm.proxy.proxy_server import model_info

        # Mock user with access to specific models
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            api_key="test_key",
            models=["gpt-4", "claude-3"],
            team_models=["gpt-3.5-turbo"],
        )

        with patch("litellm.proxy.proxy_server.llm_router") as mock_router, patch(
            "litellm.proxy.proxy_server.get_key_models"
        ) as mock_get_key_models, patch(
            "litellm.proxy.proxy_server.get_team_models"
        ) as mock_get_team_models, patch(
            "litellm.proxy.proxy_server.get_complete_model_list"
        ) as mock_get_complete_models, patch(
            "litellm.get_llm_provider"
        ) as mock_get_provider:
            # Setup mocks
            mock_router.get_model_names.return_value = [
                "gpt-4",
                "claude-3",
                "gpt-3.5-turbo",
            ]
            mock_router.get_model_access_groups.return_value = {}
            mock_get_key_models.return_value = ["gpt-4", "claude-3"]
            mock_get_team_models.return_value = ["gpt-3.5-turbo"]
            mock_get_complete_models.return_value = [
                "gpt-4",
                "claude-3",
                "gpt-3.5-turbo",
            ]
            mock_get_provider.return_value = (None, "openai", None, None)

            # Test accessible model
            result = await model_info(
                model_id="gpt-4", user_api_key_dict=user_api_key_dict
            )

            assert result["id"] == "gpt-4"
            assert result["object"] == "model"
            assert result["owned_by"] == "openai"
            assert "created" in result

    @pytest.mark.asyncio
    async def test_model_info_inaccessible_model_returns_404(self):
        """Test model_info returns 404 for inaccessible models"""
        from fastapi import HTTPException

        from litellm.proxy.proxy_server import model_info

        # Mock user with limited access
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            api_key="test_key",
            models=["gpt-4"],  # Only has access to gpt-4
            team_models=[],
        )

        with patch("litellm.proxy.proxy_server.llm_router") as mock_router, patch(
            "litellm.proxy.proxy_server.get_key_models"
        ) as mock_get_key_models, patch(
            "litellm.proxy.proxy_server.get_team_models"
        ) as mock_get_team_models, patch(
            "litellm.proxy.proxy_server.get_complete_model_list"
        ) as mock_get_complete_models:
            # Setup mocks - user only has access to gpt-4
            mock_router.get_model_names.return_value = ["gpt-4", "claude-3"]
            mock_router.get_model_access_groups.return_value = {}
            mock_get_key_models.return_value = ["gpt-4"]
            mock_get_team_models.return_value = []
            mock_get_complete_models.return_value = ["gpt-4"]  # Only gpt-4 accessible

            # Test inaccessible model should raise 404
            with pytest.raises(HTTPException) as exc_info:
                await model_info(
                    model_id="claude-3",  # Not in user's accessible models
                    user_api_key_dict=user_api_key_dict,
                )

            assert exc_info.value.status_code == 404
            assert "does not exist or is not accessible" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_model_info_team_model_access(self):
        """Test model_info works with team model access"""
        from litellm.proxy.proxy_server import model_info

        # Mock user with team access
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            api_key="test_key",
            team_id="test_team",
            models=[],  # No direct key models
            team_models=["team-model-1"],
        )

        with patch("litellm.proxy.proxy_server.llm_router") as mock_router, patch(
            "litellm.proxy.proxy_server.get_key_models"
        ) as mock_get_key_models, patch(
            "litellm.proxy.proxy_server.get_team_models"
        ) as mock_get_team_models, patch(
            "litellm.proxy.proxy_server.get_complete_model_list"
        ) as mock_get_complete_models, patch(
            "litellm.get_llm_provider"
        ) as mock_get_provider:
            # Setup mocks
            mock_router.get_model_names.return_value = ["team-model-1"]
            mock_router.get_model_access_groups.return_value = {}
            mock_get_key_models.return_value = []
            mock_get_team_models.return_value = ["team-model-1"]
            mock_get_complete_models.return_value = ["team-model-1"]
            mock_get_provider.return_value = (None, "custom", None, None)

            # Test team model access
            result = await model_info(
                model_id="team-model-1", user_api_key_dict=user_api_key_dict
            )

            assert result["id"] == "team-model-1"
            assert result["object"] == "model"
            assert result["owned_by"] == "custom"
