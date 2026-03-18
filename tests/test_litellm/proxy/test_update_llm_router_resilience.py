"""
Test that _update_llm_router and _delete_deployment are resilient to
config loading failures (e.g. database timeouts).

This addresses a bug where httpcore.ReadTimeout from the Prisma client
during get_config() would prevent ALL DB models from loading into the
router, because the exception propagated up and was caught by the
catch-all handler in _update_llm_router.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.proxy_server import ProxyConfig


def _make_db_model(model_name: str, model_id: str):
    """Helper to create a mock DB model record."""
    record = MagicMock()
    record.model_id = model_id
    record.model_name = model_name
    record.litellm_params = {"model": model_name}
    record.model_info = {"id": model_id}
    record.created_by = "default_user_id"
    record.created_at = None
    record.updated_at = None
    record.updated_by = None
    return record


class TestUpdateLlmRouterResilience:
    """Test _update_llm_router handles get_config failures gracefully."""

    @pytest.mark.asyncio
    async def test_models_loaded_when_get_config_times_out(self):
        """DB models should still be added to the router when get_config() raises a timeout."""
        proxy_config = ProxyConfig()

        db_models = [_make_db_model("gpt-5.1", "db-id-1")]

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = []
        mock_router.get_model_ids.return_value = []

        mock_proxy_logging = MagicMock()

        with (
            patch.object(
                proxy_config,
                "get_config",
                new_callable=AsyncMock,
                side_effect=Exception("httpcore.ReadTimeout"),
            ),
            patch.object(proxy_config, "_add_deployment", return_value=1) as mock_add,
            patch.object(
                proxy_config,
                "_delete_deployment",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.master_key", "sk-test"),
            patch("litellm.proxy.proxy_server.llm_model_list", []),
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            await proxy_config._update_llm_router(
                new_models=db_models,
                proxy_logging_obj=mock_proxy_logging,
            )

            # _add_deployment should still have been called despite get_config failure
            mock_add.assert_called_once_with(db_models=db_models)

    @pytest.mark.asyncio
    async def test_get_config_success_still_works(self):
        """Normal flow should still work when get_config succeeds."""
        proxy_config = ProxyConfig()

        db_models = [_make_db_model("gpt-5.1", "db-id-1")]

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = []
        mock_router.get_model_ids.return_value = []

        mock_proxy_logging = MagicMock()

        with (
            patch.object(
                proxy_config,
                "get_config",
                new_callable=AsyncMock,
                return_value={"model_list": []},
            ),
            patch.object(proxy_config, "_add_deployment", return_value=1) as mock_add,
            patch.object(
                proxy_config,
                "_delete_deployment",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.master_key", "sk-test"),
            patch("litellm.proxy.proxy_server.llm_model_list", []),
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            await proxy_config._update_llm_router(
                new_models=db_models,
                proxy_logging_obj=mock_proxy_logging,
            )

            mock_add.assert_called_once_with(db_models=db_models)


class TestDeleteDeploymentResilience:
    """Test _delete_deployment handles get_config failures gracefully."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_get_config_times_out(self):
        """Should return 0 (no deletions) when get_config fails, not raise."""
        proxy_config = ProxyConfig()

        db_models = [_make_db_model("gpt-5.1", "db-id-1")]

        mock_router = MagicMock()
        mock_router.get_model_ids.return_value = ["db-id-1", "config-id-1"]

        with (
            patch.object(
                proxy_config,
                "get_config",
                new_callable=AsyncMock,
                side_effect=Exception("httpcore.ReadTimeout"),
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.premium_user", False),
        ):
            result = await proxy_config._delete_deployment(db_models=db_models)

            # Should safely return 0 instead of raising
            assert result == 0
            # Should NOT have deleted any deployments
            mock_router.delete_deployment.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_delete_still_works(self):
        """Normal deletion should work when get_config succeeds."""
        proxy_config = ProxyConfig()

        db_models = [_make_db_model("gpt-5.1", "db-id-1")]

        mock_router = MagicMock()
        # Router has a model ID that's not in DB or config -> should be deleted
        mock_router.get_model_ids.return_value = ["db-id-1", "stale-id"]
        mock_router.delete_deployment.return_value = True
        mock_router._generate_model_id = MagicMock(return_value="config-id-1")

        with (
            patch.object(
                proxy_config,
                "get_config",
                new_callable=AsyncMock,
                return_value={"model_list": [
                    {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}, "model_info": {"id": "config-id-1"}}
                ]},
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.premium_user", False),
        ):
            result = await proxy_config._delete_deployment(db_models=db_models)

            # "stale-id" should have been deleted (not in db_models or config)
            assert result == 1
            mock_router.delete_deployment.assert_called_once_with(id="stale-id")
