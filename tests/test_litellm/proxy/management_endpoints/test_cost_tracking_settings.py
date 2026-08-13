"""
Tests for cost tracking settings management endpoints.

Tests the GET and PATCH endpoints for managing cost discount configuration.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

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
