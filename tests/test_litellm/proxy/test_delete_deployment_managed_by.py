"""
Regression tests for runtime-injected deployments being silently deleted by the
config reconcile (_delete_deployment) when store_model_in_db is enabled.

A deployment injected into the Router at runtime (via set_model_list /
upsert_deployment by a callback or plugin) is present in the router but absent
from both the DB and config.yaml, so the clean-up loop treats it as an orphan
and evicts it on the next 30s tick - silently.

The fix: a deployment whose model_info declares an external owner (`managed_by`)
is not an orphan. The reconcile skips it and logs the skip.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.proxy_server import ProxyConfig


def _make_config_deployment(model_name: str, model_id: str) -> dict:
    """A config.yaml-style deployment entry."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
        "model_info": {"id": model_id},
    }


@pytest.fixture
def runtime_router():
    """A Router holding one config-sourced deployment and one runtime-injected
    deployment that declares an external owner via model_info.managed_by."""
    from litellm.router import Router

    router = Router(
        model_list=[
            {
                "model_name": "plugin/model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                "model_info": {"id": "plugin-1", "managed_by": "my-plugin"},
            },
            _make_config_deployment("gpt-4o", "config-1"),
        ]
    )
    return router


class TestDeleteDeploymentKeepsExternallyManagedModels:
    @pytest.mark.asyncio
    async def test_runtime_injected_deployment_survives_reconcile(self, runtime_router):
        """Deployments declaring model_info.managed_by must not be evicted when
        config.yaml exists and lists only other models."""
        cfg = ProxyConfig()

        async def fake_get_config(config_file_path=None):
            return {"model_list": [_make_config_deployment("gpt-4o", "config-1")]}

        cfg.get_config = fake_get_config

        router = runtime_router
        with patch("litellm.proxy.proxy_server.llm_router", router):
            await cfg._delete_deployment(db_models=[])

        model_ids = sorted(router.get_model_ids())
        assert "plugin-1" in model_ids, "runtime-injected deployment was evicted"
        assert "config-1" in model_ids
        assert len(model_ids) == 2

    @pytest.mark.asyncio
    async def test_orphan_without_owner_is_still_evicted(self):
        """A router deployment that is in neither config nor DB and declares no
        external owner must still be deleted - the keep rule must not become a
        blanket stay-deleted rule."""
        from litellm.router import Router

        router = Router(
            model_list=[
                {
                    "model_name": "legacy/model",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                    "model_info": {"id": "orphan-1"},
                },
                _make_config_deployment("gpt-4o", "config-1"),
            ]
        )
        cfg = ProxyConfig()

        async def fake_get_config(config_file_path=None):
            return {"model_list": [_make_config_deployment("gpt-4o", "config-1")]}

        cfg.get_config = fake_get_config

        with patch("litellm.proxy.proxy_server.llm_router", router):
            await cfg._delete_deployment(db_models=[])

        model_ids = sorted(router.get_model_ids())
        assert "orphan-1" not in model_ids, "orphan without owner should be evicted"
        assert model_ids == ["config-1"]