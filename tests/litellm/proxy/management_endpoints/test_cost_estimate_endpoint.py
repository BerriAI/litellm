"""
Tests for the /cost/estimate endpoint in cost_tracking_settings.py
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import CostEstimateRequest, CostEstimateResponse
from litellm.proxy.management_endpoints.cost_tracking_settings import estimate_cost


class TestCostEstimateEndpoint:
    """Tests for the cost estimation endpoint."""

    @pytest.mark.asyncio
    async def test_estimate_cost_with_margin(self):
        """
        Test that cost estimation returns correct breakdown including margin.
        """
        mock_model_group_info = MagicMock()
        mock_model_group_info.providers = ["openai"]
        mock_model_group_info.input_cost_per_token = 0.00003
        mock_model_group_info.output_cost_per_token = 0.00006

        mock_router = MagicMock()
        mock_router.get_model_group_info.return_value = mock_model_group_info

        request = CostEstimateRequest(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            num_requests=10,
        )

        # Base cost = 0.00003 * 1000 + 0.00006 * 500 = 0.03 + 0.03 = 0.06
        # With 10% margin = 0.06 + 0.006 = 0.066
        with patch("litellm.cost_calculator._apply_cost_margin") as mock_apply_margin:
            mock_apply_margin.return_value = (0.066, 0.10, 0.0, 0.006)

            with patch("litellm.proxy.proxy_server.llm_router", mock_router):
                response = await estimate_cost(
                    request=request,
                    user_api_key_dict=MagicMock(),
                )

        assert response.model == "gpt-4"
        assert response.input_cost_per_request == pytest.approx(0.03)
        assert response.output_cost_per_request == pytest.approx(0.03)
        assert response.margin_cost_per_request == pytest.approx(0.006)
        assert response.cost_per_request == pytest.approx(0.066)
        assert response.total_cost == pytest.approx(0.66)
        assert response.total_margin_cost == pytest.approx(0.06)

    @pytest.mark.asyncio
    async def test_estimate_cost_model_not_found(self):
        """
        Test that 404 is raised when model is not found.
        """
        mock_router = MagicMock()
        mock_router.get_model_group_info.return_value = None

        request = CostEstimateRequest(
            model="nonexistent-model",
            input_tokens=1000,
            output_tokens=500,
        )

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await estimate_cost(
                    request=request,
                    user_api_key_dict=MagicMock(),
                )

            assert exc_info.value.status_code == 404
