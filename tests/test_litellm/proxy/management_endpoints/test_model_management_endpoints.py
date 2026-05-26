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
    LiteLLM_ProxyModelTable,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    ModelManagementAuthChecks,
    _get_team_deployments,
    clear_cache,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.router import Deployment, LiteLLM_Params, updateDeployment


class MockPrismaClient:
    def __init__(
        self,
        team_exists: bool = True,
        user_admin: bool = True,
        sibling_deployments: list = None,
    ):
        self.team_exists = team_exists
        self.user_admin = user_admin
        self.sibling_deployments = sibling_deployments or []
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

    async def find_many(self, where=None):
        # Filter sibling deployments based on where clause
        if not self.sibling_deployments:
            return []

        results = self.sibling_deployments

        # Support model_name startswith filter (used by _get_team_deployments)
        if where and "model_name" in where:
            model_name_filter = where["model_name"]
            if (
                isinstance(model_name_filter, dict)
                and "startswith" in model_name_filter
            ):
                prefix = model_name_filter["startswith"]
                results = [d for d in results if d.model_name.startswith(prefix)]

        return results

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

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.proxy_config", mock_config),
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
            patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_logging),
            patch("litellm.proxy.proxy_server.verbose_proxy_logger"),
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

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.proxy_config", mock_config),
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
            patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_logging),
            patch("litellm.proxy.proxy_server.verbose_proxy_logger"),
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


class TestUpdateModel:
    """
    Tests for the update_model (POST /model/update) handler.
    """

    @pytest.mark.asyncio
    async def test_update_model_clears_cache_after_db_write(self):
        """
        Regression test for the stale-router bug: POST /model/update must refresh
        the in-memory router after persisting to LiteLLM_ProxyModelTable, otherwise
        model-level guardrails (and any other litellm_params change) silently no-op
        until the APScheduler reload tick fires ~30 s later.
        """
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_model,
        )
        from litellm.types.router import (
            ModelInfo,
            updateDeployment,
            updateLiteLLMParams,
        )

        model_id = "db-model-under-test"

        existing_row = MagicMock()
        existing_row.litellm_params = {
            "model": "openai/gpt-4o-mini",
            "api_key": "sk-existing",
        }
        existing_row.model_dump.return_value = {
            "model_name": "gpt-4o-mini",
            "litellm_params": existing_row.litellm_params,
            "model_info": {"id": model_id},
        }
        existing_row.model_dump_json.return_value = "{}"

        updated_row = MagicMock()
        updated_row.model_dump_json.return_value = "{}"

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(
            return_value=existing_row
        )
        mock_prisma.db.litellm_proxymodeltable.update = AsyncMock(
            return_value=updated_row
        )

        mock_router = MagicMock()
        admin_user = UserAPIKeyAuth(
            user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.ModelManagementAuthChecks.can_user_make_model_call",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.encrypt_value_helper",
                side_effect=lambda value: value,
            ),
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.clear_cache",
                new=AsyncMock(return_value=None),
            ) as mock_clear_cache,
        ):
            await update_model(
                model_params=updateDeployment(
                    litellm_params=updateLiteLLMParams(guardrails=["g1"]),
                    model_info=ModelInfo(id=model_id),
                ),
                user_api_key_dict=admin_user,
            )

            mock_prisma.db.litellm_proxymodeltable.update.assert_awaited_once()
            mock_clear_cache.assert_awaited_once_with()


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
            with (
                patch(
                    "litellm.proxy.proxy_server.proxy_config",
                    mock_proxy_config,
                ),
                patch(
                    "litellm.proxy.proxy_server.store_model_in_db",
                    True,
                ),
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
            with (
                patch(
                    "litellm.proxy.management_endpoints.model_management_endpoints._add_model_to_db",
                    side_effect=mock_add_model_to_db,
                ),
                patch(
                    "litellm.proxy.management_endpoints.model_management_endpoints.team_model_add",
                    mock_team_model_add,
                ),
            ):
                await _add_team_model_to_db(
                    model_params=dep,
                    user_api_key_dict=user,
                    prisma_client=prisma_client,
                )

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
                {
                    "model_name": "global-gpt-4o",
                    "litellm_params": {
                        "model": "azure/gpt-4o",
                        "api_key": "global-key",
                        "api_base": "https://global.openai.azure.com",
                    },
                    "model_info": {},  # No team_id - global deployment
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

    def test_global_deployments_accessible_to_teams(self):
        """Test that global deployments (no team_id) are accessible to all teams"""
        import litellm

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "global-gpt-4o",
                    "litellm_params": {
                        "model": "azure/gpt-4o",
                        "api_key": "global-key",
                        "api_base": "https://global.openai.azure.com",
                    },
                    "model_info": {},  # No team_id - global deployment
                },
            ],
        )

        # Global deployment should be accessible when team_id is provided
        deployments = router._get_all_deployments(
            model_name="global-gpt-4o", team_id="teamA"
        )
        assert len(deployments) == 1
        assert deployments[0]["model_name"] == "global-gpt-4o"

        # should_include_deployment should return True for global deployments
        assert router.should_include_deployment(
            model_name="global-gpt-4o",
            model={"model_name": "global-gpt-4o", "model_info": {}},
            team_id="teamA",
        )


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

        with (
            patch(
                "litellm.proxy.proxy_server.premium_user",
                True,
            ),
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.team_model_add"
            ) as mock_team_model_add,
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.update_team"
            ) as mock_update_team,
        ):
            result = await _update_team_model_in_db(
                db_model=db_model,
                patch_data=patch_data,
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,  # type: ignore
            )

            assert result.get("model_name", "").startswith("model_name_test_team_123_")
            assert "team_public_model_name" in str(result.get("model_info", ""))
            # team_model_add must be called to add public name to team's models list
            mock_team_model_add.assert_called_once()
            # update_team (model_aliases write) must NOT be called in the new implementation
            mock_update_team.assert_not_called()

    @pytest.mark.asyncio
    async def test_rename_preserves_old_name_when_siblings_exist(self):
        """Test that renaming a deployment preserves old public name when sibling deployments still use it"""
        from unittest.mock import MagicMock

        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _update_existing_team_model_assignment,
        )
        from litellm.types.router import ModelInfo

        # Create a deployment being renamed
        db_model = Deployment(
            model_name="model_name_team_123_uuid1",
            litellm_params=LiteLLM_Params(model="azure/gpt-4o-mini"),
            model_info=ModelInfo(
                team_id="team_123", team_public_model_name="old-public-name"
            ),
        )

        # Create a sibling deployment that still uses the old public name
        sibling_deployment = MagicMock()
        sibling_deployment.model_name = "model_name_team_123_uuid2"
        sibling_deployment.model_info = {
            "team_id": "team_123",
            "team_public_model_name": "old-public-name",
        }

        prisma_client = MockPrismaClient(
            team_exists=True, sibling_deployments=[sibling_deployment]
        )

        patch_data = updateDeployment(
            model_name="new-public-name",
            model_info=ModelInfo(team_id="team_123"),
        )

        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.team_model_delete"
            ) as mock_delete,
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.team_model_add"
            ) as mock_add,
        ):
            await _update_existing_team_model_assignment(
                team_id="team_123",
                public_model_name="new-public-name",
                db_model=db_model,
                patch_data=patch_data,
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,  # type: ignore
            )

            # team_model_delete should NOT be called because sibling exists
            mock_delete.assert_not_called()
            # team_model_add should be called to add new public name
            mock_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_first_time_public_name_assignment_adds_team_model(self):
        """If existing team deployment had no public name, first assignment must call team_model_add."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _update_existing_team_model_assignment,
        )
        from litellm.types.router import ModelInfo

        db_model = Deployment(
            model_name="model_name_team_123_uuid1",
            litellm_params=LiteLLM_Params(model="azure/gpt-4o-mini"),
            model_info=ModelInfo(team_id="team_123"),
        )

        patch_data = updateDeployment(
            model_name="new-public-name",
            model_info=ModelInfo(team_id="team_123"),
        )

        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.team_model_delete"
            ) as mock_delete,
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.team_model_add"
            ) as mock_add,
        ):
            await _update_existing_team_model_assignment(
                team_id="team_123",
                public_model_name="new-public-name",
                db_model=db_model,
                patch_data=patch_data,
                user_api_key_dict=user_api_key_dict,
                prisma_client=None,
            )

            mock_add.assert_called_once()
            mock_delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_rename_with_prisma_none_clears_patch_model_name(self):
        """Rename path must clear patch_data.model_name even when prisma is unavailable (P1)."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _update_existing_team_model_assignment,
        )
        from litellm.types.router import ModelInfo

        db_model = Deployment(
            model_name="model_name_team_123_uuid1",
            litellm_params=LiteLLM_Params(model="azure/gpt-4o-mini"),
            model_info=ModelInfo(
                team_id="team_123", team_public_model_name="old-public-name"
            ),
        )
        patch_data = updateDeployment(
            model_name="new-public-name",
            model_info=ModelInfo(team_id="team_123"),
        )
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        await _update_existing_team_model_assignment(
            team_id="team_123",
            public_model_name="new-public-name",
            db_model=db_model,
            patch_data=patch_data,
            user_api_key_dict=user_api_key_dict,
            prisma_client=None,
        )

        assert patch_data.model_name is None

    @pytest.mark.asyncio
    async def test_rename_handles_legacy_string_model_info(self):
        """Test rename path handles legacy string-encoded model_info rows without crashing."""
        from unittest.mock import MagicMock

        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _update_existing_team_model_assignment,
        )
        from litellm.types.router import ModelInfo

        db_model = Deployment(
            model_name="model_name_team_123_uuid1",
            litellm_params=LiteLLM_Params(model="azure/gpt-4o-mini"),
            model_info=ModelInfo(
                team_id="team_123", team_public_model_name="old-public-name"
            ),
        )

        sibling_deployment = MagicMock()
        sibling_deployment.model_name = "model_name_team_123_uuid2"
        sibling_deployment.model_info = (
            '{"team_id":"team_123","team_public_model_name":"old-public-name"}'
        )

        prisma_client = MockPrismaClient(
            team_exists=True, sibling_deployments=[sibling_deployment]
        )

        patch_data = updateDeployment(
            model_name="new-public-name",
            model_info=ModelInfo(team_id="team_123"),
        )

        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.team_model_delete"
            ) as mock_delete,
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.team_model_add"
            ) as mock_add,
        ):
            await _update_existing_team_model_assignment(
                team_id="team_123",
                public_model_name="new-public-name",
                db_model=db_model,
                patch_data=patch_data,
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,  # type: ignore
            )

            mock_delete.assert_not_called()
            mock_add.assert_called_once()

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

        with (
            patch("litellm.proxy.proxy_server.llm_router") as mock_router,
            patch("litellm.proxy.proxy_server.get_key_models") as mock_get_key_models,
            patch("litellm.proxy.proxy_server.get_team_models") as mock_get_team_models,
            patch(
                "litellm.proxy.proxy_server.get_complete_model_list"
            ) as mock_get_complete_models,
            patch("litellm.get_llm_provider") as mock_get_provider,
        ):
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

        with (
            patch("litellm.proxy.proxy_server.llm_router") as mock_router,
            patch("litellm.proxy.proxy_server.get_key_models") as mock_get_key_models,
            patch("litellm.proxy.proxy_server.get_team_models") as mock_get_team_models,
            patch(
                "litellm.proxy.proxy_server.get_complete_model_list"
            ) as mock_get_complete_models,
        ):
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

        with (
            patch("litellm.proxy.proxy_server.llm_router") as mock_router,
            patch("litellm.proxy.proxy_server.get_key_models") as mock_get_key_models,
            patch("litellm.proxy.proxy_server.get_team_models") as mock_get_team_models,
            patch(
                "litellm.proxy.proxy_server.get_complete_model_list"
            ) as mock_get_complete_models,
            patch("litellm.get_llm_provider") as mock_get_provider,
        ):
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


class TestAddAndDeleteModelLifecycle:
    """
    Mock replacement for test_add_and_delete_models in tests/test_models.py.

    The original integration test required a live proxy + OPENAI_API_KEY.
    This test verifies the same lifecycle (add → delete → double-delete fails)
    by calling the endpoint handlers directly with mocked DB.
    """

    @pytest.mark.asyncio
    async def test_add_then_delete_model(self):
        """
        - Add model via add_new_model → returns model_id
        - Delete model via delete_model → returns success
        - Delete same model again → raises (model not found)
        """
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            add_new_model,
            delete_model as delete_model_endpoint,
        )
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            ModelInfoDelete,
        )

        model_id = "lifecycle-test-model-123"
        admin_user = UserAPIKeyAuth(
            user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN
        )

        # Build a real LiteLLM_ProxyModelTable for the DB mock to return
        db_row = LiteLLM_ProxyModelTable(
            model_id=model_id,
            model_name="lifecycle-model",
            litellm_params={"model": "openai/gpt-4.1-nano"},
            model_info={"id": model_id},
            created_by="test-admin",
            updated_by="test-admin",
        )

        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_proxymodeltable = AsyncMock()
        mock_prisma.db.litellm_proxymodeltable.create = AsyncMock(return_value=db_row)
        mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(
            return_value=db_row
        )
        mock_prisma.db.litellm_proxymodeltable.delete = AsyncMock(return_value=db_row)

        mock_proxy_config = MagicMock()
        mock_proxy_config.add_deployment = AsyncMock()

        mock_router = MagicMock()
        mock_router.delete_deployment = MagicMock()

        _PS = "litellm.proxy.proxy_server"
        _ENCRYPT = "litellm.proxy.management_endpoints.model_management_endpoints.encrypt_value_helper"
        with (
            patch(f"{_PS}.prisma_client", mock_prisma),
            patch(f"{_PS}.store_model_in_db", True),
            patch(f"{_PS}.proxy_config", mock_proxy_config),
            patch(f"{_PS}.proxy_logging_obj", MagicMock()),
            patch(f"{_PS}.general_settings", {}),
            patch(f"{_PS}.premium_user", True),
            patch(f"{_PS}.llm_router", mock_router),
            patch(_ENCRYPT, side_effect=lambda value, **kwargs: value),
        ):

            # --- ADD ---
            add_result = await add_new_model(
                model_params=Deployment(
                    model_name="lifecycle-model",
                    litellm_params=LiteLLM_Params(
                        model="openai/gpt-4.1-nano", api_key="fake-key"
                    ),
                    model_info={"id": model_id},
                ),
                user_api_key_dict=admin_user,
            )
            assert add_result.model_id == model_id

            # --- DELETE ---
            delete_result = await delete_model_endpoint(
                model_info=ModelInfoDelete(id=model_id),
                user_api_key_dict=admin_user,
            )
            assert "deleted successfully" in delete_result["message"]

            # --- DELETE again should fail (model not found) ---
            mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(
                return_value=None
            )
            from litellm.proxy.proxy_server import ProxyException

            with pytest.raises(ProxyException) as exc_info:
                await delete_model_endpoint(
                    model_info=ModelInfoDelete(id=model_id),
                    user_api_key_dict=admin_user,
                )
            assert str(exc_info.value.code) == "400"


class TestGetTeamDeployments:
    """Tests for _get_team_deployments which filters by model_name prefix + Python-side team_id check."""

    @pytest.mark.asyncio
    async def test_returns_matching_team_deployments(self):
        """Deployments with matching model_name prefix and team_id are returned."""
        team_id = "team_abc"
        dep = MagicMock()
        dep.model_name = f"model_name_{team_id}_uuid1"
        dep.model_info = {"team_id": team_id, "team_public_model_name": "gpt-4"}

        prisma_client = MockPrismaClient(sibling_deployments=[dep])
        result = await _get_team_deployments(team_id, prisma_client)
        assert len(result) == 1
        assert result[0] is dep

    @pytest.mark.asyncio
    async def test_filters_out_wrong_team_id_in_model_info(self):
        """A deployment whose model_name matches but model_info.team_id differs is excluded."""
        team_id = "team_abc"
        dep = MagicMock()
        dep.model_name = f"model_name_{team_id}_uuid1"
        dep.model_info = {"team_id": "other_team"}

        prisma_client = MockPrismaClient(sibling_deployments=[dep])
        result = await _get_team_deployments(team_id, prisma_client)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_handles_string_encoded_model_info(self):
        """Legacy rows with JSON-string model_info are parsed and filtered correctly."""
        team_id = "team_abc"
        dep = MagicMock()
        dep.model_name = f"model_name_{team_id}_uuid1"
        dep.model_info = json.dumps({"team_id": team_id})

        prisma_client = MockPrismaClient(sibling_deployments=[dep])
        result = await _get_team_deployments(team_id, prisma_client)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_deployments(self):
        """Returns empty list when no deployments exist."""
        prisma_client = MockPrismaClient(sibling_deployments=[])
        result = await _get_team_deployments("team_abc", prisma_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_rows_with_invalid_model_info(self):
        """Rows with non-dict, non-parseable model_info are skipped."""
        team_id = "team_abc"
        dep = MagicMock()
        dep.model_name = f"model_name_{team_id}_uuid1"
        dep.model_info = "not-valid-json"

        prisma_client = MockPrismaClient(sibling_deployments=[dep])
        result = await _get_team_deployments(team_id, prisma_client)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_multiple_deployments_mixed_filtering(self):
        """Only deployments with correct prefix AND team_id are returned."""
        team_id = "team_abc"

        # Matches both prefix and team_id
        dep1 = MagicMock()
        dep1.model_name = f"model_name_{team_id}_uuid1"
        dep1.model_info = {"team_id": team_id}

        # Matches prefix but wrong team_id
        dep2 = MagicMock()
        dep2.model_name = f"model_name_{team_id}_uuid2"
        dep2.model_info = {"team_id": "wrong_team"}

        # Different prefix entirely (won't be returned by mock's startswith filter)
        dep3 = MagicMock()
        dep3.model_name = "model_name_other_team_uuid3"
        dep3.model_info = {"team_id": "other_team"}

        prisma_client = MockPrismaClient(sibling_deployments=[dep1, dep2, dep3])
        result = await _get_team_deployments(team_id, prisma_client)
        assert len(result) == 1
        assert result[0] is dep1


def _build_db_model_for_blocked_test():
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    return Deployment(
        model_name="gpt-4o",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
        model_info=ModelInfo(id="dep-0"),
    )


class TestUpdateDBModelBlocked:
    """`update_db_model` must thread `blocked` through to the Prisma payload only
    when the caller explicitly set it — PATCH semantics: an absent field means
    "leave the stored value untouched"."""

    def test_update_db_model_passes_blocked_true_to_db(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )

        result = update_db_model(
            db_model=_build_db_model_for_blocked_test(),
            updated_patch=updateDeployment(blocked=True),
        )
        assert result["blocked"] is True

    def test_update_db_model_passes_blocked_false_to_db(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )

        result = update_db_model(
            db_model=_build_db_model_for_blocked_test(),
            updated_patch=updateDeployment(blocked=False),
        )
        assert result["blocked"] is False

    def test_update_db_model_omits_blocked_when_patch_is_none(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )

        result = update_db_model(
            db_model=_build_db_model_for_blocked_test(),
            updated_patch=updateDeployment(),
        )
        assert "blocked" not in result


def _build_db_model_with_pricing():
    """Wildcard deployment with custom pricing in litellm_params; Deployment.__init__
    mirrors SPECIAL_MODEL_INFO_PARAMS into model_info, so both blobs hold the rate."""
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    return Deployment(
        model_name="openai/*",
        litellm_params=LiteLLM_Params(
            model="openai/*",
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        ),
        model_info=ModelInfo(id="dep-pricing-0"),
    )


class TestUpdateDBModelClearPricing:
    """Sending an explicit `null` for a pricing field must remove it from both
    `litellm_params` and `model_info` (SPECIAL_MODEL_INFO_PARAMS are mirrored
    between the two by Deployment.__init__).

    Restricted to SPECIAL_MODEL_INFO_PARAMS so non-pricing fields (e.g. team_id)
    cannot be cleared via this path.
    """

    def test_clear_input_cost_removes_from_both_blobs(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import updateLiteLLMParams

        result = update_db_model(
            db_model=_build_db_model_with_pricing(),
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(input_cost_per_token=None)
            ),
        )

        params = json.loads(result["litellm_params"])
        info = json.loads(result["model_info"])
        assert "input_cost_per_token" not in params
        assert "input_cost_per_token" not in info
        # Other pricing untouched
        assert params.get("output_cost_per_token") == 0.000002
        assert info.get("output_cost_per_token") == 0.000002

    def test_clear_output_cost_removes_from_both_blobs(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import updateLiteLLMParams

        result = update_db_model(
            db_model=_build_db_model_with_pricing(),
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(output_cost_per_token=None)
            ),
        )

        params = json.loads(result["litellm_params"])
        info = json.loads(result["model_info"])
        assert "output_cost_per_token" not in params
        assert "output_cost_per_token" not in info

    def test_non_null_pricing_update_still_works(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import updateLiteLLMParams

        result = update_db_model(
            db_model=_build_db_model_with_pricing(),
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(input_cost_per_token=0.000005)
            ),
        )

        params = json.loads(result["litellm_params"])
        assert params["input_cost_per_token"] == 0.000005

    def test_omitted_pricing_field_is_preserved(self):
        """PATCH semantics: fields not in the patch keep their existing value."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import updateLiteLLMParams

        result = update_db_model(
            db_model=_build_db_model_with_pricing(),
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(output_cost_per_token=0.000007)
            ),
        )

        params = json.loads(result["litellm_params"])
        assert params["input_cost_per_token"] == 0.000001
        assert params["output_cost_per_token"] == 0.000007

    def test_null_on_non_pricing_field_does_not_clear(self):
        """Security guard: only SPECIAL_MODEL_INFO_PARAMS can be cleared via null.
        Privileged or unrelated model_info fields (e.g. team_id) must be unaffected
        by the null-clearing path so a team admin can't ungate a team-scoped model.
        """
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import (
            Deployment,
            LiteLLM_Params,
            ModelInfo,
            updateLiteLLMParams,
        )

        db_model = Deployment(
            model_name="openai/*",
            litellm_params=LiteLLM_Params(
                model="openai/*",
                input_cost_per_token=0.000001,
            ),
            model_info=ModelInfo(id="dep-pricing-1", team_id="team-keep-me"),
        )

        # Patch sends a null for api_base (non-SPECIAL field). Must NOT clear team_id
        # or any other non-pricing field from the merged dict.
        result = update_db_model(
            db_model=db_model,
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(api_base=None)
            ),
        )

        info = json.loads(result["model_info"])
        # Pricing still present (not part of this patch)
        assert "input_cost_per_token" in info
        # team_id must survive
        assert info.get("team_id") == "team-keep-me"

    def test_clear_survives_model_info_passthrough_with_old_pricing(self):
        """Realistic UI submit shape: the patch carries BOTH blobs. The
        model_info portion still has the old pricing because the form
        re-serializes the source blob. The litellm_params null must beat the
        model_info merge — i.e. the clear runs after both merges, not between.
        """
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import ModelInfo, updateLiteLLMParams

        result = update_db_model(
            db_model=_build_db_model_with_pricing(),
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(input_cost_per_token=None),
                # The UI passes the OLD model_info blob through unchanged.
                model_info=ModelInfo(
                    id="dep-pricing-0",
                    input_cost_per_token=0.000001,  # stale value from the page state
                ),
            ),
        )

        params = json.loads(result["litellm_params"])
        info = json.loads(result["model_info"])
        assert "input_cost_per_token" not in params
        assert (
            "input_cost_per_token" not in info
        ), "model_info passthrough must not resurrect the cleared override"

    def test_clear_via_model_info_clears_both_blobs(self):
        """The mirror works in the reverse direction too: nulling a pricing field
        via the model_info patch should clear it from litellm_params as well."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import ModelInfo

        result = update_db_model(
            db_model=_build_db_model_with_pricing(),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-pricing-0", input_cost_per_token=None)
            ),
        )

        params = json.loads(result["litellm_params"])
        info = json.loads(result["model_info"])
        assert "input_cost_per_token" not in params
        assert "input_cost_per_token" not in info

    def test_clear_cache_read_cost_removes_from_both_blobs(self):
        """cache_read_input_token_cost was added to SPECIAL_MODEL_INFO_PARAMS so
        the same null-clear path works for cache-read overrides."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import (
            Deployment,
            LiteLLM_Params,
            ModelInfo,
            updateLiteLLMParams,
        )

        db_model = Deployment(
            model_name="openai/*",
            litellm_params=LiteLLM_Params(
                model="openai/*",
                cache_read_input_token_cost=0.0000005,
            ),
            model_info=ModelInfo(id="dep-cache-read-0"),
        )

        result = update_db_model(
            db_model=db_model,
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(cache_read_input_token_cost=None)
            ),
        )

        params = json.loads(result["litellm_params"])
        info = json.loads(result["model_info"])
        assert "cache_read_input_token_cost" not in params
        assert "cache_read_input_token_cost" not in info

    def test_clear_cache_write_cost_removes_from_both_blobs(self):
        """cache_creation_input_token_cost was added to SPECIAL_MODEL_INFO_PARAMS so
        the same null-clear path works for cache-write overrides."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import (
            Deployment,
            LiteLLM_Params,
            ModelInfo,
            updateLiteLLMParams,
        )

        db_model = Deployment(
            model_name="openai/*",
            litellm_params=LiteLLM_Params(
                model="openai/*",
                cache_creation_input_token_cost=0.000003,
            ),
            model_info=ModelInfo(id="dep-cache-write-0"),
        )

        result = update_db_model(
            db_model=db_model,
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(cache_creation_input_token_cost=None)
            ),
        )

        params = json.loads(result["litellm_params"])
        info = json.loads(result["model_info"])
        assert "cache_creation_input_token_cost" not in params
        assert "cache_creation_input_token_cost" not in info

    def test_clear_cache_read_preserves_other_pricing(self):
        """Clearing cache_read must not touch input/output cost overrides."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            update_db_model,
        )
        from litellm.types.router import (
            Deployment,
            LiteLLM_Params,
            ModelInfo,
            updateLiteLLMParams,
        )

        db_model = Deployment(
            model_name="openai/*",
            litellm_params=LiteLLM_Params(
                model="openai/*",
                input_cost_per_token=0.000001,
                output_cost_per_token=0.000002,
                cache_read_input_token_cost=0.0000005,
                cache_creation_input_token_cost=0.000003,
            ),
            model_info=ModelInfo(id="dep-cache-mixed-0"),
        )

        result = update_db_model(
            db_model=db_model,
            updated_patch=updateDeployment(
                litellm_params=updateLiteLLMParams(cache_read_input_token_cost=None)
            ),
        )

        params = json.loads(result["litellm_params"])
        info = json.loads(result["model_info"])
        assert "cache_read_input_token_cost" not in params
        assert "cache_read_input_token_cost" not in info
        # Other pricing untouched in both blobs
        assert params["input_cost_per_token"] == 0.000001
        assert params["output_cost_per_token"] == 0.000002
        assert params["cache_creation_input_token_cost"] == 0.000003
        assert info["input_cost_per_token"] == 0.000001
        assert info["output_cost_per_token"] == 0.000002
        assert info["cache_creation_input_token_cost"] == 0.000003


class TestGetModelInfoWithIdBlocked:
    """`ProxyConfig.get_model_info_with_id` must propagate the DB-level `blocked`
    column into the in-memory `model_info` dict so the router filter can read it."""

    def test_get_model_info_with_id_propagates_blocked_true(self):
        from litellm.proxy.proxy_server import ProxyConfig

        model = MagicMock()
        model.model_id = "dep-1"
        model.model_info = {}
        model.blocked = True
        info = ProxyConfig().get_model_info_with_id(model=model, db_model=True)
        assert info.id == "dep-1"
        assert getattr(info, "blocked") is True

    def test_get_model_info_with_id_defaults_blocked_to_false_when_missing(self):
        from litellm.proxy.proxy_server import ProxyConfig

        model = MagicMock(spec=["model_id", "model_info"])
        model.model_id = "dep-2"
        model.model_info = {}
        info = ProxyConfig().get_model_info_with_id(model=model, db_model=True)
        assert getattr(info, "blocked") is False


class TestPatchModelBlockedAuthGate:
    """Only proxy admins may flip `blocked` — team admins authorized for
    team-scoped models via `can_user_make_model_call` must still be rejected
    when they attempt to toggle the pause flag."""

    @pytest.mark.asyncio
    async def test_team_admin_cannot_toggle_blocked(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            patch_model,
        )

        non_admin = UserAPIKeyAuth(
            user_id="team_admin",
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        existing_row = MagicMock()
        existing_row.litellm_params = {"model": "openai/gpt-4o-mini"}
        existing_row.model_dump.return_value = {
            "model_name": "gpt-4o-mini",
            "litellm_params": existing_row.litellm_params,
            "model_info": {"id": "m1"},
        }
        existing_row.model_dump_json.return_value = "{}"

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(
            return_value=existing_row
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
            patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.ModelManagementAuthChecks.can_user_make_model_call",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(Exception) as exc_info:
                await patch_model(
                    model_id="m1",
                    patch_data=updateDeployment(blocked=True),
                    user_api_key_dict=non_admin,
                )
            err = exc_info.value
            assert getattr(err, "param", "") == "blocked"
            assert "proxy admin" in getattr(err, "message", "").lower()

    @pytest.mark.asyncio
    async def test_proxy_admin_can_toggle_blocked(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            patch_model,
        )

        admin = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        existing_row = MagicMock()
        existing_row.litellm_params = {"model": "openai/gpt-4o-mini"}
        existing_row.model_dump.return_value = {
            "model_name": "gpt-4o-mini",
            "litellm_params": existing_row.litellm_params,
            "model_info": {"id": "m1"},
        }
        existing_row.model_dump_json.return_value = "{}"
        updated_row = MagicMock()
        updated_row.model_dump_json.return_value = "{}"

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(
            return_value=existing_row
        )
        mock_prisma.db.litellm_proxymodeltable.update = AsyncMock(
            return_value=updated_row
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
            patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.ModelManagementAuthChecks.can_user_make_model_call",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.model_management_endpoints.clear_cache",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await patch_model(
                model_id="m1",
                patch_data=updateDeployment(blocked=True),
                user_api_key_dict=admin,
            )
            assert result is updated_row
            mock_prisma.db.litellm_proxymodeltable.update.assert_awaited_once()
