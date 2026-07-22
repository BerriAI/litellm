"""
Test that ProxyConfig.add_deployment honours store_model_in_db.

Regression test for: when store_model_in_db is False the periodic DB sync
must not pull models from LiteLLM_ProxyModelTable. Previously the gate was
only on `_should_load_db_object("models")` (default-on), so a stray UI-added
row could replace config.yaml models on every sync — silently wiping out
deployments declared in config.

Documented behaviour of `store_model_in_db`:
"Allow saving / managing models in DB" — when disabled, the DB is not the
source of truth for models.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.proxy_server import ProxyConfig


class TestAddDeploymentStoreModelInDB:
    """Test add_deployment respects store_model_in_db before reading DB models."""

    @pytest.mark.asyncio
    async def test_skips_db_model_load_when_store_model_in_db_false(self):
        """When store_model_in_db=False (config.yaml), no DB sync of models."""
        proxy_config = ProxyConfig()

        with (
            patch.object(
                proxy_config,
                "_get_models_from_db",
                new_callable=AsyncMock,
            ) as mock_get_models,
            patch.object(
                proxy_config,
                "_update_llm_router",
                new_callable=AsyncMock,
            ) as mock_update_router,
            patch.object(
                proxy_config,
                "_init_non_llm_objects_in_db",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.proxy_server.prefetch_config_params",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.proxy_server.get_config_param",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy.proxy_server.general_settings",
                {"store_model_in_db": False},
            ),
            patch("litellm.proxy.proxy_server.store_model_in_db", False),
        ):
            await proxy_config.add_deployment(
                prisma_client=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )

            mock_get_models.assert_not_called()
            mock_update_router.assert_not_called()

    @pytest.mark.asyncio
    async def test_loads_db_models_when_store_model_in_db_true(self):
        """When store_model_in_db=True (config.yaml), DB sync runs as before."""
        proxy_config = ProxyConfig()

        db_models = [MagicMock(model_id="db-id-1", model_name="some-model")]

        with (
            patch.object(
                proxy_config,
                "_get_models_from_db",
                new_callable=AsyncMock,
                return_value=db_models,
            ) as mock_get_models,
            patch.object(
                proxy_config,
                "_update_llm_router",
                new_callable=AsyncMock,
            ) as mock_update_router,
            patch.object(
                proxy_config,
                "_init_non_llm_objects_in_db",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.proxy_server.prefetch_config_params",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.proxy_server.get_config_param",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy.proxy_server.general_settings",
                {"store_model_in_db": True},
            ),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            await proxy_config.add_deployment(
                prisma_client=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )

            mock_get_models.assert_called_once()
            mock_update_router.assert_called_once()

    @pytest.mark.asyncio
    async def test_loads_db_models_when_module_global_true(self):
        """
        Env var path (STORE_MODEL_IN_DB=True) sets the module-level
        `store_model_in_db` global. Must still trigger DB sync even when
        general_settings does not explicitly set the key.
        """
        proxy_config = ProxyConfig()

        with (
            patch.object(
                proxy_config,
                "_get_models_from_db",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_get_models,
            patch.object(
                proxy_config,
                "_update_llm_router",
                new_callable=AsyncMock,
            ) as mock_update_router,
            patch.object(
                proxy_config,
                "_init_non_llm_objects_in_db",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.proxy_server.prefetch_config_params",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.proxy_server.get_config_param",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("litellm.proxy.proxy_server.general_settings", {}),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            await proxy_config.add_deployment(
                prisma_client=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )

            mock_get_models.assert_called_once()
            mock_update_router.assert_called_once()

    @pytest.mark.asyncio
    async def test_supported_db_objects_gate_still_applies(self):
        """
        Even with store_model_in_db=True, an explicit supported_db_objects list
        that omits "models" must still skip the DB sync. This preserves the
        existing fine-grained gate.
        """
        proxy_config = ProxyConfig()

        with (
            patch.object(
                proxy_config,
                "_get_models_from_db",
                new_callable=AsyncMock,
            ) as mock_get_models,
            patch.object(
                proxy_config,
                "_update_llm_router",
                new_callable=AsyncMock,
            ) as mock_update_router,
            patch.object(
                proxy_config,
                "_init_non_llm_objects_in_db",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.proxy_server.prefetch_config_params",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.proxy_server.get_config_param",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy.proxy_server.general_settings",
                {
                    "store_model_in_db": True,
                    "supported_db_objects": ["mcp", "guardrails"],
                },
            ),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            await proxy_config.add_deployment(
                prisma_client=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )

            mock_get_models.assert_not_called()
            mock_update_router.assert_not_called()
