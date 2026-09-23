"""
Test suite for XAI cost calculation functionality.
"""

import math
import os

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.llms.xai.cost_calculator import (
    _DEFAULT_WEB_SEARCH_COST_PER_CALL,
    _web_search_cost_per_call_from_model_info,
    apply_server_side_tool_usage_details_to_usage,
    cost_per_token,
    cost_per_web_search_request,
)
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


class TestXAICostCalculator:
    """Test suite for XAI cost calculation functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Load the main model cost map directly to ensure we have the latest pricing
        import json

        try:
            with open("model_prices_and_context_window.json", "r") as f:
                model_cost_map = json.load(f)
            litellm.model_cost = model_cost_map
        except FileNotFoundError:
            # Fallback to default behavior
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_tiered_pricing_model_without_tiered_pricing(self):
        litellm.model_cost["xai/flat-rate-fixture"] = {
            "input_cost_per_token": 3e-7,
            "output_cost_per_token": 5e-7,
            "litellm_provider": "xai",
            "mode": "chat",
        }
        usage = Usage(prompt_tokens=250000, completion_tokens=50000, total_tokens=300000)
        prompt_cost, completion_cost = cost_per_token(model="xai/flat-rate-fixture", usage=usage)
        expected_prompt_cost = 250000 * 3e-7
        expected_completion_cost = 50000 * 5e-7
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_web_search_cost_via_server_side_tool_usage_details(self):
        """usage.server_side_tool_usage_details.web_search_calls at default $5/1k."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        setattr(
            usage,
            "server_side_tool_usage_details",
            {
                "web_search_calls": 3,
                "x_search_calls": 0,
                "code_interpreter_calls": 0,
                "file_search_calls": 0,
                "mcp_calls": 0,
                "document_search_calls": 0,
            },
        )

        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})
        assert math.isclose(web_search_cost, 3 * (5.0 / 1000.0), rel_tol=1e-10)

    def test_web_search_cost_uses_model_info_search_context_pricing(self):
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        setattr(usage, "server_side_tool_usage_details", {"web_search_calls": 2})
        model_info = {
            "search_context_cost_per_query": {
                "search_context_size_medium": 0.01,
            }
        }
        web_search_cost = cost_per_web_search_request(usage=usage, model_info=model_info)
        assert math.isclose(web_search_cost, 0.02, rel_tol=1e-10)

    def test_web_search_cost_zero_without_details(self):
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert cost_per_web_search_request(usage=usage, model_info={}) == 0.0

    def test_apply_details_sets_web_search_requests_for_cost_gate(self):
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        apply_server_side_tool_usage_details_to_usage(usage, {"web_search_calls": 2, "x_search_calls": 0})
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.web_search_requests == 2
        assert StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
            response_object=object(), usage=usage
        )

    def test_reported_cost_is_preferred_over_token_math(self):
        """The amount xAI reported, carried on usage.cost by the transformation, is billed.

        It lands entirely on completion cost because xAI does not split its total by
        direction, the same shape the perplexity calculator returns.
        """
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            cost=0.0037756,
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-4-latest", usage=usage)

        assert prompt_cost == 0.0
        assert math.isclose(completion_cost, 0.0037756, rel_tol=1e-10)

    def test_reported_cost_suppresses_web_search_surcharge(self):
        """The reported total already covers server-side tool calls.

        Without the suppression these 3 searches would be billed a second time on
        top of the total xAI already charged.
        """
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=100,
                web_search_requests=3,
            ),
            cost=0.0037756,
        )

        assert cost_per_web_search_request(usage=usage, model_info={}) == 0.0

    def test_web_search_surcharge_suppressed_through_the_dispatcher(self):
        """The suppression has to hold on the path cost tracking actually uses.

        Legacy behaviour stays intact when xAI reports no cost.
        """
        from litellm.llms import get_cost_for_web_search_request

        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        setattr(usage, "server_side_tool_usage_details", {"web_search_calls": 3})

        assert get_cost_for_web_search_request("xai", usage, {}) > 0.0

        reported = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.0037756)
        setattr(reported, "server_side_tool_usage_details", {"web_search_calls": 3})
        assert get_cost_for_web_search_request("xai", reported, {}) == 0.0






    def test_zero_reported_cost_is_honoured(self):
        """A reported zero is a real answer, not a missing value."""
        usage = Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300, cost=0.0)

        assert cost_per_token(model="grok-4-latest", usage=usage) == (0.0, 0.0)

    def test_custom_pricing_beats_the_reported_cost(self):
        response = ModelResponse(
            id="chatcmpl-xai",
            model="grok-4-latest",
            choices=[Choices(index=0, message=Message(role="assistant", content="x"), finish_reason="stop")],
            usage=Usage(prompt_tokens=198, completion_tokens=353, total_tokens=551, cost=0.0009956),
        )

        billed = litellm.completion_cost(
            completion_response=response,
            model="xai/grok-4-latest",
            custom_llm_provider="xai",
            custom_cost_per_token={"input_cost_per_token": 0.001, "output_cost_per_token": 0.001},
            custom_pricing=True,
        )

        assert math.isclose(billed, 0.551, rel_tol=1e-10)

    def test_deployment_custom_pricing_beats_the_reported_cost(self, monkeypatch):
        deployment_id = "xai-deployment-priced-by-the-operator"
        monkeypatch.setitem(
            litellm.model_cost,
            deployment_id,
            {"input_cost_per_token": 0.001, "output_cost_per_token": 0.001, "litellm_provider": "xai", "mode": "chat"},
        )
        response = ModelResponse(
            id="chatcmpl-xai",
            model="grok-4-latest",
            choices=[Choices(index=0, message=Message(role="assistant", content="x"), finish_reason="stop")],
            usage=Usage(prompt_tokens=198, completion_tokens=353, total_tokens=551, cost=0.0009956),
        )

        billed = litellm.completion_cost(
            completion_response=response,
            model="xai/grok-4-latest",
            custom_llm_provider="xai",
            custom_pricing=True,
            router_model_id=deployment_id,
        )

        assert math.isclose(billed, 0.551, rel_tol=1e-10)


class TestXAIWebSearchCostHelpers:
    """Focused coverage for web_search / tool-usage helpers in cost_calculator.py."""

    def test_apply_details_noop_when_details_none(self):
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        apply_server_side_tool_usage_details_to_usage(usage, None)
        assert getattr(usage, "server_side_tool_usage_details", None) is None

    def test_apply_details_sets_attr_but_skips_mirror_when_web_search_zero(self):
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        details = {"web_search_calls": 0, "x_search_calls": 3}
        apply_server_side_tool_usage_details_to_usage(usage, details)
        assert getattr(usage, "server_side_tool_usage_details") == details
        assert usage.prompt_tokens_details is None or usage.prompt_tokens_details.web_search_requests is None

    def test_apply_details_skips_mirror_when_web_search_calls_invalid(self):
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        details = {"web_search_calls": "not-a-number"}
        apply_server_side_tool_usage_details_to_usage(usage, details)
        assert getattr(usage, "server_side_tool_usage_details") == details
        assert usage.prompt_tokens_details is None

    def test_apply_details_updates_existing_prompt_tokens_details(self):
        usage = Usage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=7),
        )
        apply_server_side_tool_usage_details_to_usage(usage, {"web_search_calls": 4})
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 7
        assert usage.prompt_tokens_details.web_search_requests == 4

    def test_web_search_cost_per_call_default_when_model_info_empty(self):
        assert _web_search_cost_per_call_from_model_info({}) == _DEFAULT_WEB_SEARCH_COST_PER_CALL

    def test_web_search_cost_per_call_prefers_medium_over_low(self):
        model_info = {
            "search_context_cost_per_query": {
                "search_context_size_low": 0.001,
                "search_context_size_medium": 0.009,
            }
        }
        assert _web_search_cost_per_call_from_model_info(model_info) == 0.009

    def test_web_search_cost_per_call_falls_back_to_low_then_high(self):
        assert (
            _web_search_cost_per_call_from_model_info(
                {"search_context_cost_per_query": {"search_context_size_low": 0.003}}
            )
            == 0.003
        )
        assert (
            _web_search_cost_per_call_from_model_info(
                {"search_context_cost_per_query": {"search_context_size_high": 0.007}}
            )
            == 0.007
        )

    def test_web_search_cost_per_call_ignores_zero_and_invalid_values(self):
        assert (
            _web_search_cost_per_call_from_model_info(
                {
                    "search_context_cost_per_query": {
                        "search_context_size_medium": 0,
                        "search_context_size_low": "bad",
                    }
                }
            )
            == _DEFAULT_WEB_SEARCH_COST_PER_CALL
        )

    def test_cost_per_web_search_request_zero_when_details_not_mapping(self):
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        setattr(usage, "server_side_tool_usage_details", "invalid")
        assert cost_per_web_search_request(usage=usage, model_info={}) == 0.0

    def test_cost_per_web_search_request_zero_when_web_search_calls_invalid(self):
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        setattr(
            usage,
            "server_side_tool_usage_details",
            {"web_search_calls": object()},
        )
        assert cost_per_web_search_request(usage=usage, model_info={}) == 0.0

    def test_cost_per_web_search_request_zero_when_web_search_calls_zero(self):
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        setattr(
            usage,
            "server_side_tool_usage_details",
            {"web_search_calls": 0, "x_search_calls": 5},
        )
        assert cost_per_web_search_request(usage=usage, model_info={}) == 0.0

    def test_cost_per_web_search_request_uses_default_rate_without_model_pricing(self):
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        setattr(usage, "server_side_tool_usage_details", {"web_search_calls": 4})
        cost = cost_per_web_search_request(usage=usage, model_info={})
        assert math.isclose(cost, 4 * _DEFAULT_WEB_SEARCH_COST_PER_CALL, rel_tol=1e-10)
