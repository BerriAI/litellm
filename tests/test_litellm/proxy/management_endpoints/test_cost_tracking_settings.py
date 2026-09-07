"""
Tests for cost tracking settings management endpoints.

Tests the GET and PATCH endpoints for managing cost discount configuration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


import litellm
from litellm.proxy.management_endpoints.cost_tracking_settings import router
from litellm.proxy.proxy_server import app

client = TestClient(app)


class TestCostTrackingSettings:
    """Test suite for cost tracking settings endpoints"""

    @pytest.mark.asyncio
    async def test_get_cost_discount_config_success(self):
        """
        Test GET /config/cost_discount_config endpoint successfully retrieves configuration.
        """
        # Mock the proxy_config to return a config with cost_discount_config
        mock_proxy_config = AsyncMock()
        mock_proxy_config.get_config = AsyncMock(
            return_value={
                "litellm_settings": {
                    "cost_discount_config": {
                        "vertex_ai": 0.05,
                        "gemini": 0.05,
                        "openai": 0.01,
                    }
                }
            }
        )

        mock_prisma_client = MagicMock()

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
        ):
            # Make request
            response = client.get(
                "/config/cost_discount_config",
                headers={"Authorization": "Bearer sk-1234"},
            )

            # Verify response
            assert response.status_code == 200
            response_data = response.json()

            assert "values" in response_data
            assert response_data["values"]["vertex_ai"] == 0.05
            assert response_data["values"]["gemini"] == 0.05
            assert response_data["values"]["openai"] == 0.01

            # Verify get_config was called
            mock_proxy_config.get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cost_discount_config_empty(self):
        """
        Test GET /config/cost_discount_config endpoint returns empty config when not set.
        """
        # Mock the proxy_config to return a config without cost_discount_config
        mock_proxy_config = AsyncMock()
        mock_proxy_config.get_config = AsyncMock(return_value={"litellm_settings": {}})

        mock_prisma_client = MagicMock()

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
        ):
            # Make request
            response = client.get(
                "/config/cost_discount_config",
                headers={"Authorization": "Bearer sk-1234"},
            )

            # Verify response
            assert response.status_code == 200
            response_data = response.json()

            assert "values" in response_data
            assert response_data["values"] == {}

    @pytest.mark.asyncio
    async def test_update_cost_discount_config_success(self):
        """
        Test PATCH /config/cost_discount_config endpoint successfully updates configuration.
        """
        # Mock the proxy_config
        mock_proxy_config = AsyncMock()
        mock_proxy_config.get_config = AsyncMock(return_value={"litellm_settings": {}})
        mock_proxy_config.save_config = AsyncMock()

        mock_prisma_client = MagicMock()
        mock_store_model_in_db = True

        # Test data
        test_discount_config = {
            "vertex_ai": 0.05,
            "gemini": 0.05,
            "openai": 0.01,
        }

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                mock_store_model_in_db,
            ),
            patch.object(litellm, "cost_discount_config", {}),
        ):
            # Make request
            response = client.patch(
                "/config/cost_discount_config",
                json=test_discount_config,
                headers={"Authorization": "Bearer sk-1234"},
            )

            # Verify response
            assert response.status_code == 200
            response_data = response.json()

            assert response_data["status"] == "success"
            assert "message" in response_data
            assert "values" in response_data
            assert response_data["values"]["vertex_ai"] == 0.05
            assert response_data["values"]["gemini"] == 0.05
            assert response_data["values"]["openai"] == 0.01

            # Verify config was saved
            mock_proxy_config.save_config.assert_called_once()

            # Verify litellm.cost_discount_config was updated
            assert litellm.cost_discount_config == test_discount_config

    @pytest.mark.asyncio
    async def test_update_cost_discount_config_invalid_provider(self):
        """
        Test PATCH /config/cost_discount_config endpoint rejects invalid provider names.
        """
        mock_proxy_config = AsyncMock()
        mock_prisma_client = MagicMock()
        mock_store_model_in_db = True

        # Test data with invalid provider
        test_discount_config = {
            "invalid_provider": 0.05,
            "openai": 0.01,
        }

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                mock_store_model_in_db,
            ),
        ):
            # Make request
            response = client.patch(
                "/config/cost_discount_config",
                json=test_discount_config,
                headers={"Authorization": "Bearer sk-1234"},
            )

            # Verify response - should fail with 400
            assert response.status_code == 400
            response_data = response.json()
            assert "error" in response_data["detail"]
            assert "invalid_provider" in response_data["detail"]["error"]

    @pytest.mark.asyncio
    async def test_update_cost_discount_config_invalid_discount_value(self):
        """
        Test PATCH /config/cost_discount_config endpoint rejects discount values outside 0-1 range.
        """
        mock_proxy_config = AsyncMock()
        mock_prisma_client = MagicMock()
        mock_store_model_in_db = True

        # Test data with invalid discount value (> 1)
        test_discount_config = {
            "openai": 1.5,  # Invalid: greater than 1
        }

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                mock_store_model_in_db,
            ),
        ):
            # Make request
            response = client.patch(
                "/config/cost_discount_config",
                json=test_discount_config,
                headers={"Authorization": "Bearer sk-1234"},
            )

            # Verify response - should fail with 400
            assert response.status_code == 400
            response_data = response.json()
            assert "detail" in response_data
            assert "between 0 and 1" in response_data["detail"]

    @pytest.mark.asyncio
    async def test_update_cost_discount_config_no_store_model_in_db(self):
        """
        Test PATCH /config/cost_discount_config endpoint fails when STORE_MODEL_IN_DB is not enabled.
        """
        mock_proxy_config = AsyncMock()
        mock_prisma_client = MagicMock()
        mock_store_model_in_db = False  # Not enabled

        test_discount_config = {
            "openai": 0.05,
        }

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                mock_store_model_in_db,
            ),
        ):
            # Make request
            response = client.patch(
                "/config/cost_discount_config",
                json=test_discount_config,
                headers={"Authorization": "Bearer sk-1234"},
            )

            # Verify response - should fail with 500
            assert response.status_code == 500
            response_data = response.json()
            assert "error" in response_data["detail"]
            assert "STORE_MODEL_IN_DB" in response_data["detail"]["error"]


class TestResolveModelForCostLookup:
    """Tests for _resolve_model_for_cost_lookup base_model resolution."""

    def test_resolves_base_model_for_azure_deployment(self):
        """
        When a model group has base_model set in model_info,
        _resolve_model_for_cost_lookup should return the base_model
        instead of the raw litellm_params.model (Azure deployment name).
        """
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            _resolve_model_for_cost_lookup,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-5.3-codex",
                "litellm_params": {
                    "model": "azure/openai/gpt-5.3-codex",
                    "api_base": "https://fake.openai.azure.com/",
                    "api_key": "fake-key",
                },
                "model_info": {
                    "id": "test-id",
                    "base_model": "azure/gpt-4o",
                },
            }
        ]

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router,
        ):
            resolved = _resolve_model_for_cost_lookup("gpt-5.3-codex")

        assert resolved.model == "azure/gpt-4o"
        mock_router.get_model_list.assert_called_once_with(model_name="gpt-5.3-codex")

    def test_falls_back_to_litellm_params_model_when_no_base_model(self):
        """
        When no base_model is set, should fall back to litellm_params.model.
        """
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            _resolve_model_for_cost_lookup,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                },
                "model_info": {
                    "id": "test-id",
                },
            }
        ]

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router,
        ):
            resolved = _resolve_model_for_cost_lookup("gpt-4")

        assert resolved.model == "openai/gpt-4"

    def test_resolves_base_model_from_litellm_params(self):
        """
        When base_model is in litellm_params (not model_info),
        it should still be resolved.
        """
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            _resolve_model_for_cost_lookup,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "my-azure-model",
                "litellm_params": {
                    "model": "azure/my-custom-deployment",
                    "base_model": "azure/gpt-4o-mini",
                },
                "model_info": {
                    "id": "test-id",
                },
            }
        ]

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router,
        ):
            resolved = _resolve_model_for_cost_lookup("my-azure-model")

        assert resolved.model == "azure/gpt-4o-mini"

    def test_returns_original_model_when_no_router(self):
        """
        When no router is available, should return the original model name.
        """
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            _resolve_model_for_cost_lookup,
        )

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            None,
        ):
            resolved = _resolve_model_for_cost_lookup("azure/openai/gpt-5.3-codex")

        assert resolved.model == "azure/openai/gpt-5.3-codex"
        assert resolved.provider is None

    def test_returns_custom_llm_provider_on_base_model_path(self):
        """base_model path: the custom_llm_provider from litellm_params is
        returned as the second tuple element, unchanged."""
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            _resolve_model_for_cost_lookup,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "my-azure-model",
                "litellm_params": {
                    "model": "azure/my-deployment",
                    "base_model": "azure/gpt-4o",
                    "custom_llm_provider": "azure",
                },
                "model_info": {"id": "test-id"},
            }
        ]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            resolved = _resolve_model_for_cost_lookup("my-azure-model")

        assert resolved.model == "azure/gpt-4o"
        assert resolved.provider == "azure"

    def test_returns_custom_llm_provider_on_resolved_model_path(self):
        """resolved-model path (no base_model): the custom_llm_provider from
        litellm_params is returned alongside litellm_params.model."""
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            _resolve_model_for_cost_lookup,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "custom_llm_provider": "openai",
                },
                "model_info": {"id": "test-id"},
            }
        ]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            resolved = _resolve_model_for_cost_lookup("gpt-4")

        assert resolved.model == "openai/gpt-4"
        assert resolved.provider == "openai"

    def test_resolves_base_model_when_deployment_has_no_litellm_params(self):
        """A deployment can omit litellm_params entirely; base_model from
        model_info must still resolve (the .get default must be {} not None,
        else the later litellm_params.get(...) raises and resolution is lost)."""
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            _resolve_model_for_cost_lookup,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "my-azure-model",
                "model_info": {"base_model": "azure/gpt-4o"},
            }
        ]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            resolved = _resolve_model_for_cost_lookup("my-azure-model")

        assert resolved.model == "azure/gpt-4o"
        assert resolved.provider is None

    def test_resolves_model_when_deployment_has_no_model_info(self):
        """A deployment can omit model_info entirely; litellm_params.model must
        still resolve (the .get default must be {} not None, else the earlier
        model_info.get(...) raises and resolution is lost)."""
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            _resolve_model_for_cost_lookup,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4"},
            }
        ]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            resolved = _resolve_model_for_cost_lookup("gpt-4")

        assert resolved.model == "openai/gpt-4"
        assert resolved.provider is None


class TestEstimateCostOnPremProvider:
    """Regression tests for LIT-5210: /cost/estimate on on-prem deployment aliases."""

    @pytest.mark.asyncio
    async def test_estimate_cost_onprem_model_without_pricing(self):
        """
        On-prem deployments (custom_llm_provider set, model absent from the cost map)
        must not 500 with "LLM Provider NOT provided". The resolved provider has to be
        forwarded to completion_cost so provider inference doesn't run on the bare model.

        completion_cost is intentionally NOT mocked.
        """
        from litellm.proxy._types import CostEstimateRequest
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            estimate_cost,
        )

        request = CostEstimateRequest(
            model="nvidia/zai-org/glm-5.2",
            input_tokens=1000,
            output_tokens=500,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "nvidia/zai-org/glm-5.2",
                "litellm_params": {
                    "model": "zai-org/GLM-5.2",
                    "custom_llm_provider": "openai",
                },
                "model_info": {},
            }
        ]

        saved_model_cost = dict(litellm.model_cost)
        litellm.register_model(
            {
                "openai/zai-org/GLM-5.2": {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "litellm_provider": "openai",
                    "mode": "chat",
                }
            }
        )
        try:
            with patch("litellm.proxy.proxy_server.llm_router", mock_router):
                response = await estimate_cost(request=request, user_api_key_dict=MagicMock())
        finally:
            litellm.model_cost = saved_model_cost

        assert response.model == "nvidia/zai-org/glm-5.2"
        assert response.provider == "openai"
        assert response.cost_per_request == 0.0

    @pytest.mark.asyncio
    async def test_estimate_cost_onprem_model_with_configured_pricing(self):
        """
        On-prem deployments with input/output_cost_per_token configured must estimate a
        real cost using that pricing, not fall back to 0.0.

        completion_cost is intentionally NOT mocked.
        """
        from litellm.proxy._types import CostEstimateRequest
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            estimate_cost,
        )

        request = CostEstimateRequest(
            model="nvidia/zai-org/glm-5.2",
            input_tokens=1000,
            output_tokens=500,
            num_requests_per_day=100,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "nvidia/zai-org/glm-5.2",
                "litellm_params": {
                    "model": "zai-org/GLM-5.2",
                    "custom_llm_provider": "openai",
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000002,
                },
                "model_info": {},
            }
        ]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            response = await estimate_cost(request=request, user_api_key_dict=MagicMock())

        assert response.provider == "openai"
        assert response.cost_per_request == pytest.approx(0.002)
        assert response.input_cost_per_request == pytest.approx(0.001)
        assert response.output_cost_per_request == pytest.approx(0.001)
        assert response.daily_cost == pytest.approx(0.2)
        assert response.input_cost_per_token == pytest.approx(0.000001)
        assert response.output_cost_per_token == pytest.approx(0.000002)

    @pytest.mark.asyncio
    async def test_estimate_cost_onprem_model_with_model_info_pricing(self):
        """
        Custom pricing configured under model_info (how DB / Admin UI added
        deployments store it) must be honored, not just litellm_params pricing.

        completion_cost is intentionally NOT mocked.
        """
        from litellm.proxy._types import CostEstimateRequest
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            estimate_cost,
        )

        request = CostEstimateRequest(
            model="nvidia/zai-org/glm-5.2",
            input_tokens=1000,
            output_tokens=500,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "nvidia/zai-org/glm-5.2",
                "litellm_params": {
                    "model": "zai-org/GLM-5.2",
                    "custom_llm_provider": "openai",
                },
                "model_info": {
                    "input_cost_per_token": 0.000003,
                    "output_cost_per_token": 0.000004,
                },
            }
        ]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            response = await estimate_cost(request=request, user_api_key_dict=MagicMock())

        assert response.provider == "openai"
        assert response.cost_per_request == pytest.approx(0.005)
        assert response.input_cost_per_token == pytest.approx(0.000003)
        assert response.output_cost_per_token == pytest.approx(0.000004)

    @pytest.mark.asyncio
    async def test_estimate_cost_litellm_params_pricing_overrides_model_info(self):
        """
        When pricing is set in both places, litellm_params wins, matching the
        router's cost-map registration precedence.
        """
        from litellm.proxy._types import CostEstimateRequest
        from litellm.proxy.management_endpoints.cost_tracking_settings import (
            estimate_cost,
        )

        request = CostEstimateRequest(
            model="nvidia/zai-org/glm-5.2",
            input_tokens=1000,
            output_tokens=500,
        )

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "nvidia/zai-org/glm-5.2",
                "litellm_params": {
                    "model": "zai-org/GLM-5.2",
                    "custom_llm_provider": "openai",
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000002,
                },
                "model_info": {
                    "input_cost_per_token": 0.000003,
                    "output_cost_per_token": 0.000004,
                },
            }
        ]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            response = await estimate_cost(request=request, user_api_key_dict=MagicMock())

        assert response.cost_per_request == pytest.approx(0.002)
        assert response.input_cost_per_token == pytest.approx(0.000001)
        assert response.output_cost_per_token == pytest.approx(0.000002)




class TestBlockRequestsForModelsWithoutPricing:
    """Test suite for the block_requests_for_models_without_pricing toggle endpoints"""

    @pytest.mark.asyncio
    async def test_get_reflects_in_memory_flag(self):
        with patch.object(litellm, "block_requests_for_models_without_pricing", True):
            response = client.get(
                "/config/block_requests_for_models_without_pricing",
                headers={"Authorization": "Bearer sk-1234"},
            )

        assert response.status_code == 200
        assert response.json() == {"enabled": True}

    @pytest.mark.asyncio
    async def test_patch_persists_and_updates_flag(self):
        mock_proxy_config = AsyncMock()
        mock_proxy_config.get_config = AsyncMock(return_value={"litellm_settings": {}})
        mock_proxy_config.save_config = AsyncMock()

        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
            patch.object(litellm, "block_requests_for_models_without_pricing", False),
        ):
            response = client.patch(
                "/config/block_requests_for_models_without_pricing",
                headers={"Authorization": "Bearer sk-1234"},
                json={"enabled": True},
            )

            assert response.status_code == 200
            assert response.json() == {"enabled": True}
            assert litellm.block_requests_for_models_without_pricing is True

        saved_config = mock_proxy_config.save_config.call_args.kwargs["new_config"]
        assert saved_config["litellm_settings"]["block_requests_for_models_without_pricing"] is True

    def test_peer_workers_pick_up_persisted_flag_on_config_reload(self):
        """A PATCH only mutates the flag on the worker that served it; peer workers must pick the
        persisted value up when they reload litellm_settings from the DB."""
        from litellm.proxy.proxy_server import ProxyConfig

        with patch.object(litellm, "block_requests_for_models_without_pricing", False):
            ProxyConfig()._update_config_fields(
                current_config={},
                param_name="litellm_settings",
                db_param_value={"block_requests_for_models_without_pricing": True},
            )

            assert litellm.block_requests_for_models_without_pricing is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("loads_config_overrides", [True, False])
    async def test_periodic_db_sync_applies_flag_to_peer_worker(self, loads_config_overrides):
        """The ~10s reconcile loop runs _init_non_llm_objects_in_db on every worker; it must apply
        the persisted flag so peers converge without a restart, including when supported_db_objects
        leaves config_overrides out."""
        from types import SimpleNamespace

        from litellm.proxy.proxy_server import ProxyConfig

        config_record = SimpleNamespace(
            param_value={"block_requests_for_models_without_pricing": True, "unsafe_key": "x"}
        )
        with (
            patch.object(litellm, "block_requests_for_models_without_pricing", False),
            patch.object(
                ProxyConfig,
                "_should_load_db_object",
                side_effect=lambda object_type: loads_config_overrides and object_type == "config_overrides",
            ),
            patch.object(ProxyConfig, "_init_hashicorp_vault_config_override", AsyncMock()),
            patch("litellm.proxy.proxy_server.get_config_param", AsyncMock(return_value=config_record)),
        ):
            await ProxyConfig()._init_non_llm_objects_in_db(prisma_client=MagicMock())

            assert litellm.block_requests_for_models_without_pricing is True
            assert not hasattr(litellm, "unsafe_key")

    @pytest.mark.asyncio
    async def test_patch_requires_store_model_in_db(self):
        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.proxy_config", AsyncMock()),
            patch("litellm.proxy.proxy_server.store_model_in_db", False),
        ):
            response = client.patch(
                "/config/block_requests_for_models_without_pricing",
                headers={"Authorization": "Bearer sk-1234"},
                json={"enabled": True},
            )

        assert response.status_code == 500
        assert "error" in response.json()["detail"]


AN_ALIAS = "onprem/alias"
AN_UNDERLYING_MODEL = "vendor/model"
A_MAPPED_MODEL = "openai/mapped-only-model"
INPUT_TOKENS = 1000
OUTPUT_TOKENS = 500


def _router_pricing(**pricing: float) -> MagicMock:
    mock_router = MagicMock()
    mock_router.get_model_list.return_value = [
        {
            "model_name": AN_ALIAS,
            "litellm_params": {
                "model": AN_UNDERLYING_MODEL,
                "custom_llm_provider": "openai",
                **pricing,
            },
            "model_info": {},
        }
    ]
    return mock_router


async def _estimate(mock_router: MagicMock | None, model: str = AN_ALIAS, **overrides: int):
    from litellm.proxy._types import CostEstimateRequest
    from litellm.proxy.management_endpoints.cost_tracking_settings import estimate_cost

    request = CostEstimateRequest(
        model=model,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        **overrides,
    )
    with patch(  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        "litellm.proxy.proxy_server.llm_router", mock_router
    ):
        return await estimate_cost(request=request, user_api_key_dict=MagicMock())


class TestEstimateCostPartiallyPricedDeployments:
    @pytest.mark.asyncio
    async def test_a_deployment_that_prices_only_input_bills_output_at_zero(self):
        response = await _estimate(_router_pricing(input_cost_per_token=0.000001))

        assert response.input_cost_per_token == pytest.approx(0.000001)
        assert response.output_cost_per_token == 0.0
        assert response.cost_per_request == pytest.approx(0.001)

    @pytest.mark.asyncio
    async def test_a_deployment_that_prices_only_output_bills_input_at_zero(self):
        response = await _estimate(_router_pricing(output_cost_per_token=0.000002))

        assert response.input_cost_per_token == 0.0
        assert response.output_cost_per_token == pytest.approx(0.000002)
        assert response.cost_per_request == pytest.approx(0.001)

    @pytest.mark.asyncio
    async def test_a_model_priced_only_by_the_cost_map_reports_that_price_and_provider(self, monkeypatch):
        monkeypatch.setitem(
            litellm.model_cost,
            A_MAPPED_MODEL,
            {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.000006,
                "litellm_provider": "openai",
                "mode": "chat",
            },
        )

        response = await _estimate(None, model=A_MAPPED_MODEL)

        assert response.input_cost_per_token == pytest.approx(0.000005)
        assert response.output_cost_per_token == pytest.approx(0.000006)
        assert response.provider == "openai"


class TestEstimateCostPeriodTotals:
    @pytest.mark.asyncio
    async def test_zero_requests_a_day_reports_no_daily_cost_rather_than_zero(self):
        response = await _estimate(
            _router_pricing(input_cost_per_token=0.000001, output_cost_per_token=0.000002),
            num_requests_per_day=0,
        )

        assert response.daily_cost is None
        assert response.daily_input_cost is None
        assert response.daily_output_cost is None

    @pytest.mark.asyncio
    async def test_daily_totals_scale_every_component_by_the_request_count(self):
        response = await _estimate(
            _router_pricing(input_cost_per_token=0.000001, output_cost_per_token=0.000002),
            num_requests_per_day=100,
        )

        assert response.input_cost_per_request == pytest.approx(0.001)
        assert response.output_cost_per_request == pytest.approx(0.001)
        assert response.daily_input_cost == pytest.approx(0.1)
        assert response.daily_output_cost == pytest.approx(0.1)
        assert response.daily_cost == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_a_month_and_a_day_are_totalled_from_their_own_request_counts(self):
        response = await _estimate(
            _router_pricing(input_cost_per_token=0.000001, output_cost_per_token=0.000002),
            num_requests_per_day=100,
            num_requests_per_month=3000,
        )

        assert response.daily_cost == pytest.approx(0.2)
        assert response.monthly_cost == pytest.approx(6.0)
        assert response.monthly_input_cost == pytest.approx(3.0)
        assert response.monthly_output_cost == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_a_configured_margin_is_totalled_per_period_like_the_other_components(self, monkeypatch):
        monkeypatch.setattr(litellm, "cost_margin_config", {"openai": 0.10})

        response = await _estimate(
            _router_pricing(input_cost_per_token=0.000001, output_cost_per_token=0.000002),
            num_requests_per_day=100,
        )

        assert response.margin_cost_per_request == pytest.approx(0.0002)
        assert response.cost_per_request == pytest.approx(0.0022)
        assert response.daily_margin_cost == pytest.approx(0.02)
        assert response.daily_cost == pytest.approx(0.22)
