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
    async def test_returns_none_when_get_config_times_out(self):
        """Should return None (no reconcile ran, desired set unknown) when get_config
        fails, not raise. A caller judging its own reload must not read that as "the db
        wants nothing" and blame the reload for every model it serves."""
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

            # Should safely return None instead of raising
            assert result is None
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
                return_value={
                    "model_list": [
                        {
                            "model_name": "gpt-4",
                            "litellm_params": {"model": "gpt-4"},
                            "model_info": {"id": "config-id-1"},
                        }
                    ]
                },
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.premium_user", False),
        ):
            result = await proxy_config._delete_deployment(db_models=db_models)

            # "stale-id" should have been deleted (not in db_models or config)
            mock_router.delete_deployment.assert_called_once_with(id="stale-id")
            assert result == frozenset({"db-id-1", "config-id-1"}), (
                "the returned set must be what the db + config still want, so a caller can "
                f"tell that eviction apart from a deployment that went missing; got {result}"
            )


class TestDeleteDeploymentResolvesPluginConfigs:
    """Regression: _delete_deployment re-reads the raw config and hashes litellm_params to
    compute the ids the config wants served. The router's ids were hashed from the RESOLVED
    params (plugin dotted paths swapped for live instances), so hashing the raw strings
    computed different ids and evicted every plugin-bearing auto-router one reconcile after
    startup. The fix resolves plugins in _delete_deployment before hashing, like load_config."""

    @staticmethod
    def _write_plugin_module(tmp_path):
        (tmp_path / "rig_classifier.py").write_text(
            "class _Classifier:\n"
            "    async def classify(self, context):\n"
            "        return 'SIMPLE'\n"
            "\n"
            "class _Narrower:\n"
            "    async def run(self, context):\n"
            "        return context\n"
            "\n"
            "classifier_instance = _Classifier()\n"
            "narrower_instance = _Narrower()\n"
        )

    @staticmethod
    def _raw_model_entry():
        return {
            "model_name": "smart-router",
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_default_model": "gpt-4o-mini",
                "complexity_router_config": {
                    "classifier_type": "plugin",
                    "classifier_plugin": "rig_classifier.classifier_instance",
                    "plugins": ["rig_classifier.narrower_instance"],
                    "tiers": {"SIMPLE": "gpt-4o-mini"},
                },
            },
        }

    @pytest.mark.asyncio
    async def test_plugin_bearing_config_model_survives_reconcile(self, tmp_path):
        import copy

        from litellm import Router
        from litellm.proxy.proxy_server import resolve_complexity_router_plugins

        self._write_plugin_module(tmp_path)
        config_file_path = str(tmp_path / "config.yaml")

        resolved_entry = copy.deepcopy(self._raw_model_entry())
        resolve_complexity_router_plugins(
            model_name="smart-router",
            complexity_router_config=resolved_entry["litellm_params"]["complexity_router_config"],
            config_file_path=config_file_path,
        )
        router = Router(
            model_list=[{"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}, resolved_entry]
        )
        assert "smart-router" in router.model_names

        raw_config = {
            "model_list": [
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}},
                self._raw_model_entry(),
            ]
        }
        proxy_config = ProxyConfig()
        with (
            patch.object(proxy_config, "get_config", new_callable=AsyncMock, return_value=raw_config),
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.user_config_file_path", config_file_path),
            patch("litellm.proxy.proxy_server.premium_user", False),
        ):
            await proxy_config._delete_deployment(db_models=[])

        assert "smart-router" in router.model_names

    @pytest.mark.asyncio
    async def test_broken_plugin_module_skips_cleanup_instead_of_evicting(self, tmp_path):
        """A plugin module broken on disk at reconcile time must not evict valid deployments."""
        from litellm import Router
        from litellm.proxy.proxy_server import resolve_complexity_router_plugins

        self._write_plugin_module(tmp_path)
        config_file_path = str(tmp_path / "config.yaml")

        resolved_entry = self._raw_model_entry()
        resolve_complexity_router_plugins(
            model_name="smart-router",
            complexity_router_config=resolved_entry["litellm_params"]["complexity_router_config"],
            config_file_path=config_file_path,
        )
        router = Router(model_list=[resolved_entry])

        raw_entry = self._raw_model_entry()
        raw_entry["litellm_params"]["complexity_router_config"]["classifier_plugin"] = "rig_classifier.missing"
        proxy_config = ProxyConfig()
        with (
            patch.object(proxy_config, "get_config", new_callable=AsyncMock, return_value={"model_list": [raw_entry]}),
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.user_config_file_path", config_file_path),
            patch("litellm.proxy.proxy_server.premium_user", False),
        ):
            result = await proxy_config._delete_deployment(db_models=[])

        assert result is None
        assert "smart-router" in router.model_names
