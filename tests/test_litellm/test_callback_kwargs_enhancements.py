"""
Tests for callback kwargs enhancements:
1. project_alias in StandardLoggingUserAPIKeyMetadata
2. reasoning_tokens_cost in CostBreakdown
3. input_cost_breakdown / output_cost_breakdown in CostBreakdown
"""

import os
import sys
from datetime import datetime


import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    CostBreakdown,
    InputCostBreakdown,
    ModelResponse,
    OutputCostBreakdown,
    PromptTokensDetailsWrapper,
    StandardLoggingUserAPIKeyMetadata,
    Usage,
)

sys.path.insert(0, os.path.abspath("../../.."))


# ---------------------------------------------------------------------------
# Issue 1: project_alias in callback kwargs
# ---------------------------------------------------------------------------


class TestProjectAliasInCallbackKwargs:
    """Tests that project_alias is exposed in callback kwargs metadata."""

    def test_project_alias_field_exists_on_standard_logging_user_api_key_metadata(
        self,
    ):
        """StandardLoggingUserAPIKeyMetadata TypedDict should have user_api_key_project_alias."""
        metadata = StandardLoggingUserAPIKeyMetadata(
            user_api_key_hash="test-hash",
            user_api_key_alias="test-alias",
            user_api_key_spend=0.0,
            user_api_key_max_budget=None,
            user_api_key_budget_reset_at=None,
            user_api_key_org_id=None,
            user_api_key_team_id="team-1",
            user_api_key_project_id="proj-1",
            user_api_key_project_alias="my-project",
            user_api_key_user_id="user-1",
            user_api_key_user_email=None,
            user_api_key_team_alias="my-team",
            user_api_key_end_user_id=None,
            user_api_key_request_route=None,
            user_api_key_auth_metadata=None,
        )
        assert metadata["user_api_key_project_alias"] == "my-project"

    def test_project_alias_none_when_not_set(self):
        """project_alias should be None when key has no project."""
        metadata = StandardLoggingUserAPIKeyMetadata(
            user_api_key_hash="test-hash",
            user_api_key_alias=None,
            user_api_key_spend=None,
            user_api_key_max_budget=None,
            user_api_key_budget_reset_at=None,
            user_api_key_org_id=None,
            user_api_key_team_id=None,
            user_api_key_project_id=None,
            user_api_key_project_alias=None,
            user_api_key_user_id=None,
            user_api_key_user_email=None,
            user_api_key_team_alias=None,
            user_api_key_end_user_id=None,
            user_api_key_request_route=None,
            user_api_key_auth_metadata=None,
        )
        assert metadata["user_api_key_project_alias"] is None

    def test_get_sanitized_user_information_includes_project_alias(self):
        """get_sanitized_user_information_from_key should populate project_alias from UserAPIKeyAuth."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key-hash",
            key_alias="test-alias",
            user_id="test-user",
            project_id="proj-123",
            project_alias="My Cool Project",
            team_id="team-1",
            team_alias="my-team",
        )

        result = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
            user_api_key_dict=user_api_key_dict
        )

        assert result["user_api_key_project_id"] == "proj-123"
        assert result["user_api_key_project_alias"] == "My Cool Project"

    def test_get_sanitized_user_information_project_alias_none_when_no_project(self):
        """project_alias should be None when UserAPIKeyAuth has no project."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key-hash",
            user_id="test-user",
        )

        result = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
            user_api_key_dict=user_api_key_dict
        )

        assert result["user_api_key_project_id"] is None
        assert result["user_api_key_project_alias"] is None

    def test_project_alias_in_standard_logging_metadata(self):
        """project_alias should flow through to StandardLoggingMetadata in the logging payload."""
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        metadata = {
            "user_api_key_project_id": "proj-123",
            "user_api_key_project_alias": "My Cool Project",
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "my-team",
        }

        result = StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)
        assert result["user_api_key_project_alias"] == "My Cool Project"

    def test_project_alias_defaults_to_none_in_logging_metadata(self):
        """When no project_alias in metadata, it should default to None."""
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        result = StandardLoggingPayloadSetup.get_standard_logging_metadata({})
        assert result["user_api_key_project_alias"] is None

    def test_verification_token_view_has_project_alias(self):
        """LiteLLM_VerificationTokenView should have project_alias field."""
        from litellm.proxy._types import LiteLLM_VerificationTokenView

        token_view = LiteLLM_VerificationTokenView(
            token="test-token",
            project_id="proj-123",
            project_alias="My Project",
        )
        assert token_view.project_alias == "My Project"

    def test_verification_token_view_project_alias_defaults_none(self):
        """project_alias should default to None on LiteLLM_VerificationTokenView."""
        from litellm.proxy._types import LiteLLM_VerificationTokenView

        token_view = LiteLLM_VerificationTokenView(token="test-token")
        assert token_view.project_alias is None


# ---------------------------------------------------------------------------
# Issue 2: reasoning_tokens_cost in CostBreakdown
# ---------------------------------------------------------------------------


class TestReasoningTokensCostInCostBreakdown:
    """Tests that reasoning_tokens_cost is broken out in CostBreakdown."""

    def test_cost_breakdown_includes_reasoning_tokens_cost_field(self):
        """CostBreakdown TypedDict should accept reasoning_tokens_cost."""
        breakdown = CostBreakdown(
            input_cost=0.001,
            output_cost=0.005,
            total_cost=0.006,
            tool_usage_cost=0.0,
            reasoning_tokens_cost=0.003,
        )
        assert breakdown["reasoning_tokens_cost"] == 0.003

    def test_reasoning_tokens_cost_is_subset_of_output_cost(self):
        """reasoning_tokens_cost should be <= output_cost since it's included in it."""
        breakdown = CostBreakdown(
            input_cost=0.001,
            output_cost=0.005,
            total_cost=0.006,
            tool_usage_cost=0.0,
            reasoning_tokens_cost=0.003,
        )
        assert breakdown["reasoning_tokens_cost"] <= breakdown["output_cost"]

    def test_cost_breakdown_reasoning_tokens_cost_zero_when_no_reasoning(self):
        """When no reasoning tokens, reasoning_tokens_cost should be 0."""
        breakdown = CostBreakdown(
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            tool_usage_cost=0.0,
            reasoning_tokens_cost=0.0,
        )
        assert breakdown["reasoning_tokens_cost"] == 0.0

    def test_set_cost_breakdown_stores_reasoning_tokens_cost(self):
        """Logging.set_cost_breakdown should store reasoning_tokens_cost."""
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj = Logging(
            model="o1",
            messages=[{"role": "user", "content": "Think about this"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-123",
            function_id="test-function",
        )

        logging_obj.set_cost_breakdown(
            input_cost=0.001,
            output_cost=0.010,
            total_cost=0.011,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            reasoning_tokens_cost=0.006,
        )

        assert logging_obj.cost_breakdown is not None
        assert logging_obj.cost_breakdown["reasoning_tokens_cost"] == 0.006
        assert logging_obj.cost_breakdown["input_cost"] == 0.001
        assert logging_obj.cost_breakdown["output_cost"] == 0.010

    def test_reasoning_tokens_cost_in_standard_logging_payload(self):
        """reasoning_tokens_cost should appear in StandardLoggingPayload's cost_breakdown."""
        from litellm.litellm_core_utils.litellm_logging import (
            Logging,
            get_standard_logging_object_payload,
        )

        logging_obj = Logging(
            model="o1",
            messages=[{"role": "user", "content": "Think step by step"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-reasoning-123",
            function_id="test-function",
        )

        logging_obj.set_cost_breakdown(
            input_cost=0.001,
            output_cost=0.010,
            total_cost=0.011,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            reasoning_tokens_cost=0.006,
        )

        mock_response = {
            "id": "chatcmpl-reasoning",
            "object": "chat.completion",
            "model": "o1",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 500,
                "total_tokens": 510,
                "completion_tokens_details": {
                    "reasoning_tokens": 400,
                    "text_tokens": 100,
                },
            },
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Here's my answer"},
                    "finish_reason": "stop",
                }
            ],
        }

        kwargs = {
            "model": "o1",
            "messages": [{"role": "user", "content": "Think step by step"}],
            "response_cost": 0.011,
            "custom_llm_provider": "openai",
        }

        payload = get_standard_logging_object_payload(
            kwargs=kwargs,
            init_response_obj=mock_response,
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
            status="success",
        )

        assert payload is not None
        assert payload["cost_breakdown"] is not None
        assert payload["cost_breakdown"]["reasoning_tokens_cost"] == 0.006
        assert payload["cost_breakdown"]["input_cost"] == 0.001
        assert payload["cost_breakdown"]["output_cost"] == 0.010

    def test_cost_breakdown_without_reasoning_tokens_cost_still_works(self):
        """Existing CostBreakdown usage without reasoning_tokens_cost should still work (total=False)."""
        breakdown = CostBreakdown(
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            tool_usage_cost=0.0,
        )
        # reasoning_tokens_cost is optional (total=False on CostBreakdown)
        assert "reasoning_tokens_cost" not in breakdown

    def test_completion_cost_populates_reasoning_tokens_cost(self):
        """completion_cost() should populate reasoning_tokens_cost in the logging object's cost_breakdown."""
        from litellm.cost_calculator import completion_cost
        from litellm.litellm_core_utils.litellm_logging import Logging

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        logging_obj = Logging(
            model="o1",
            messages=[{"role": "user", "content": "Think"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-cost-calc",
            function_id="test-function",
        )

        response = ModelResponse(
            id="chatcmpl-test",
            model="o1",
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=500,
                total_tokens=600,
                completion_tokens_details=CompletionTokensDetailsWrapper(
                    reasoning_tokens=400,
                    text_tokens=100,
                ),
                prompt_tokens_details=PromptTokensDetailsWrapper(
                    text_tokens=100,
                ),
            ),
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Answer"},
                    "finish_reason": "stop",
                }
            ],
        )

        cost = completion_cost(
            completion_response=response,
            model="o1",
            custom_llm_provider="openai",
            litellm_logging_obj=logging_obj,
        )

        assert cost > 0
        assert logging_obj.cost_breakdown is not None
        assert "reasoning_tokens_cost" in logging_obj.cost_breakdown
        # reasoning_tokens_cost should be > 0 since there are 400 reasoning tokens
        assert logging_obj.cost_breakdown["reasoning_tokens_cost"] > 0
        # reasoning_tokens_cost should be <= output_cost
        assert (
            logging_obj.cost_breakdown["reasoning_tokens_cost"]
            <= logging_obj.cost_breakdown["output_cost"]
        )

    def test_completion_cost_no_reasoning_tokens_zero_cost(self):
        """When there are no reasoning tokens, reasoning_tokens_cost should be 0."""
        from litellm.cost_calculator import completion_cost
        from litellm.litellm_core_utils.litellm_logging import Logging

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        logging_obj = Logging(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-no-reasoning",
            function_id="test-function",
        )

        response = ModelResponse(
            id="chatcmpl-test",
            model="gpt-4o",
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
        )

        cost = completion_cost(
            completion_response=response,
            model="gpt-4o",
            custom_llm_provider="openai",
            litellm_logging_obj=logging_obj,
        )

        assert cost > 0
        assert logging_obj.cost_breakdown is not None
        assert logging_obj.cost_breakdown.get("reasoning_tokens_cost", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Issue 3: input_cost_breakdown / output_cost_breakdown in CostBreakdown
# ---------------------------------------------------------------------------


class TestDetailedCostBreakdowns:
    """Tests for per-token-type cost breakdowns (input_cost_breakdown / output_cost_breakdown)."""

    def test_input_cost_breakdown_typed_dict(self):
        """InputCostBreakdown TypedDict should accept all token-type cost fields."""
        breakdown = InputCostBreakdown(
            prompt_token_cost=0.003,
            cached_token_cost=0.0005,
            audio_token_cost=0.001,
            image_token_cost=0.0,
        )
        assert breakdown["prompt_token_cost"] == 0.003
        assert breakdown["cached_token_cost"] == 0.0005

    def test_output_cost_breakdown_typed_dict(self):
        """OutputCostBreakdown TypedDict should accept all token-type cost fields."""
        breakdown = OutputCostBreakdown(
            completion_token_cost=0.006,
            reasoning_token_cost=0.0018,
            audio_token_cost=0.0,
            image_token_cost=0.0,
        )
        assert breakdown["completion_token_cost"] == 0.006
        assert breakdown["reasoning_token_cost"] == 0.0018

    def test_cost_breakdown_includes_input_output_breakdowns(self):
        """CostBreakdown should accept input_cost_breakdown and output_cost_breakdown."""
        breakdown = CostBreakdown(
            input_cost=0.0045,
            input_cost_breakdown=InputCostBreakdown(
                prompt_token_cost=0.003,
                cached_token_cost=0.0005,
                audio_token_cost=0.001,
            ),
            output_cost=0.0078,
            output_cost_breakdown=OutputCostBreakdown(
                completion_token_cost=0.006,
                reasoning_token_cost=0.0018,
            ),
            total_cost=0.0123,
            tool_usage_cost=0.0,
        )
        assert breakdown["input_cost_breakdown"]["prompt_token_cost"] == 0.003
        assert breakdown["output_cost_breakdown"]["reasoning_token_cost"] == 0.0018

    def test_set_cost_breakdown_stores_input_output_breakdowns(self):
        """Logging.set_cost_breakdown should store input/output cost breakdowns."""
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj = Logging(
            model="o1",
            messages=[{"role": "user", "content": "Think"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-breakdown",
            function_id="test-function",
        )

        input_bd = {"prompt_token_cost": 0.003, "cached_token_cost": 0.0005}
        output_bd = {"completion_token_cost": 0.006, "reasoning_token_cost": 0.0018}

        logging_obj.set_cost_breakdown(
            input_cost=0.0035,
            output_cost=0.0078,
            total_cost=0.0113,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            reasoning_tokens_cost=0.0018,
            input_cost_breakdown=input_bd,
            output_cost_breakdown=output_bd,
        )

        assert logging_obj.cost_breakdown is not None
        assert "input_cost_breakdown" in logging_obj.cost_breakdown
        assert "output_cost_breakdown" in logging_obj.cost_breakdown
        assert (
            logging_obj.cost_breakdown["input_cost_breakdown"]["prompt_token_cost"]
            == 0.003
        )
        assert (
            logging_obj.cost_breakdown["output_cost_breakdown"]["reasoning_token_cost"]
            == 0.0018
        )

    def test_completion_cost_populates_input_output_breakdowns_with_reasoning(self):
        """completion_cost() should populate input/output cost breakdowns for reasoning models."""
        from litellm.cost_calculator import completion_cost
        from litellm.litellm_core_utils.litellm_logging import Logging

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        logging_obj = Logging(
            model="o1",
            messages=[{"role": "user", "content": "Think"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-detailed-breakdown",
            function_id="test-function",
        )

        response = ModelResponse(
            id="chatcmpl-test",
            model="o1",
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=500,
                total_tokens=600,
                completion_tokens_details=CompletionTokensDetailsWrapper(
                    reasoning_tokens=400,
                    text_tokens=100,
                ),
                prompt_tokens_details=PromptTokensDetailsWrapper(
                    text_tokens=100,
                ),
            ),
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Answer"},
                    "finish_reason": "stop",
                }
            ],
        )

        cost = completion_cost(
            completion_response=response,
            model="o1",
            custom_llm_provider="openai",
            litellm_logging_obj=logging_obj,
        )

        assert cost > 0
        assert logging_obj.cost_breakdown is not None

        # Should have input_cost_breakdown with prompt_token_cost
        assert "input_cost_breakdown" in logging_obj.cost_breakdown
        input_bd = logging_obj.cost_breakdown["input_cost_breakdown"]
        assert "prompt_token_cost" in input_bd
        assert input_bd["prompt_token_cost"] > 0

        # Should have output_cost_breakdown with reasoning and completion
        assert "output_cost_breakdown" in logging_obj.cost_breakdown
        output_bd = logging_obj.cost_breakdown["output_cost_breakdown"]
        assert "reasoning_token_cost" in output_bd
        assert output_bd["reasoning_token_cost"] > 0
        assert "completion_token_cost" in output_bd
        assert output_bd["completion_token_cost"] > 0

    def test_completion_cost_populates_breakdowns_simple_model(self):
        """completion_cost() should populate breakdowns for a simple model (no reasoning)."""
        from litellm.cost_calculator import completion_cost
        from litellm.litellm_core_utils.litellm_logging import Logging

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        logging_obj = Logging(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-simple-breakdown",
            function_id="test-function",
        )

        response = ModelResponse(
            id="chatcmpl-test",
            model="gpt-4o",
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
        )

        cost = completion_cost(
            completion_response=response,
            model="gpt-4o",
            custom_llm_provider="openai",
            litellm_logging_obj=logging_obj,
        )

        assert cost > 0
        assert logging_obj.cost_breakdown is not None

        # For simple models, should have input_cost_breakdown with prompt_token_cost
        assert "input_cost_breakdown" in logging_obj.cost_breakdown
        assert (
            logging_obj.cost_breakdown["input_cost_breakdown"]["prompt_token_cost"] > 0
        )

        # For simple models, output_cost_breakdown should have completion_token_cost
        assert "output_cost_breakdown" in logging_obj.cost_breakdown
        assert (
            logging_obj.cost_breakdown["output_cost_breakdown"]["completion_token_cost"]
            > 0
        )

        # No reasoning_token_cost in output breakdown since no reasoning tokens
        assert (
            "reasoning_token_cost"
            not in logging_obj.cost_breakdown["output_cost_breakdown"]
        )

    def test_calculate_detailed_cost_breakdowns_directly(self):
        """Test the calculate_detailed_cost_breakdowns function directly."""
        from litellm.litellm_core_utils.llm_cost_calc.utils import (
            calculate_detailed_cost_breakdowns,
        )

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=500,
            total_tokens=600,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=400,
                text_tokens=100,
            ),
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=100,
            ),
        )

        input_bd, output_bd = calculate_detailed_cost_breakdowns(
            model="o1",
            usage=usage,
            custom_llm_provider="openai",
        )

        # Input should have prompt_token_cost
        assert "prompt_token_cost" in input_bd
        assert input_bd["prompt_token_cost"] > 0

        # Output should have both reasoning and completion costs
        assert "reasoning_token_cost" in output_bd
        assert output_bd["reasoning_token_cost"] > 0
        assert "completion_token_cost" in output_bd
        assert output_bd["completion_token_cost"] > 0

    def test_empty_breakdowns_when_no_usage(self):
        """Breakdowns should be empty dicts when there are no token details."""
        from litellm.litellm_core_utils.llm_cost_calc.utils import (
            calculate_detailed_cost_breakdowns,
        )

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        input_bd, output_bd = calculate_detailed_cost_breakdowns(
            model="gpt-4o",
            usage=usage,
            custom_llm_provider="openai",
        )

        # Both breakdowns should be empty (no costs > 0)
        assert len(input_bd) == 0
        assert len(output_bd) == 0
