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
    async def test_estimate_cost_uses_completion_cost(self):
        """
        Test that cost estimation uses completion_cost and returns breakdown.
        """
        request = CostEstimateRequest(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            num_requests=10,
        )

        with patch(
            "litellm.proxy.management_endpoints.cost_tracking_settings.completion_cost"
        ) as mock_completion_cost:
            mock_completion_cost.return_value = 0.066

            with patch("litellm.get_model_info") as mock_get_model_info:
                mock_get_model_info.return_value = {
                    "input_cost_per_token": 0.00003,
                    "output_cost_per_token": 0.00006,
                    "litellm_provider": "openai",
                }

                response = await estimate_cost(
                    request=request,
                    user_api_key_dict=MagicMock(),
                )

        assert response.model == "gpt-4"
        assert response.cost_per_request == 0.066
        assert response.total_cost == pytest.approx(0.66)
        # Verify completion_cost was called
        mock_completion_cost.assert_called_once()

    @pytest.mark.asyncio
    async def test_estimate_cost_model_not_found(self):
        """
        Test that 404 is raised when model cost calculation fails.
        """
        request = CostEstimateRequest(
            model="nonexistent-model",
            input_tokens=1000,
            output_tokens=500,
        )

        with patch(
            "litellm.proxy.management_endpoints.cost_tracking_settings.completion_cost"
        ) as mock_completion_cost:
            mock_completion_cost.side_effect = Exception("Model not found in cost map")

            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await estimate_cost(
                    request=request,
                    user_api_key_dict=MagicMock(),
                )

            assert exc_info.value.status_code == 404
