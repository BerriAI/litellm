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
    async def test_estimate_cost_daily_and_monthly(self):
        """
        Test that cost estimation calculates daily and monthly costs correctly.
        """
        request = CostEstimateRequest(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            num_requests_per_day=100,
            num_requests_per_month=3000,
        )

        with patch(
            "litellm.proxy.management_endpoints.cost_tracking_settings.completion_cost"
        ) as mock_completion_cost:
            mock_completion_cost.return_value = 0.06

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
        assert response.cost_per_request == 0.06
        assert response.daily_cost == pytest.approx(6.0)  # 0.06 * 100
        assert response.monthly_cost == pytest.approx(180.0)  # 0.06 * 3000

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

    @pytest.mark.asyncio
    async def test_estimate_cost_resolves_router_model_alias(self):
        """
        Test that estimate_cost resolves router model aliases to underlying models.

        When a user selects a model like 'my-gpt4-alias' from the UI (which is a
        router model_name), the endpoint should resolve it to the actual model
        (e.g., 'azure/gpt-4') for cost calculation.

        This prevents the bug where custom model names fail cost lookup because
        they aren't in model_prices_and_context_window.json.
        """
        request = CostEstimateRequest(
            model="my-gpt4-alias",  # Router alias, not actual model name
            input_tokens=1000,
            output_tokens=500,
        )

        # Mock the router to return deployment info
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "my-gpt4-alias",
                "litellm_params": {
                    "model": "azure/gpt-4",  # Actual model for pricing
                    "custom_llm_provider": "azure",
                },
            }
        ]

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router,
        ):
            with patch(
                "litellm.proxy.management_endpoints.cost_tracking_settings.completion_cost"
            ) as mock_completion_cost:
                mock_completion_cost.return_value = 0.05

                with patch("litellm.get_model_info") as mock_get_model_info:
                    mock_get_model_info.return_value = {
                        "input_cost_per_token": 0.00003,
                        "output_cost_per_token": 0.00006,
                        "litellm_provider": "azure",
                    }

                    response = await estimate_cost(
                        request=request,
                        user_api_key_dict=MagicMock(),
                    )

        # Verify router was queried for the alias
        mock_router.get_model_list.assert_called_with(model_name="my-gpt4-alias")

        # Verify completion_cost was called with RESOLVED model, not the alias
        call_args = mock_completion_cost.call_args
        assert call_args.kwargs["model"] == "azure/gpt-4"

        # Verify response contains original model name (for UI display)
        assert response.model == "my-gpt4-alias"
        assert response.cost_per_request == 0.05
        assert response.provider == "azure"

