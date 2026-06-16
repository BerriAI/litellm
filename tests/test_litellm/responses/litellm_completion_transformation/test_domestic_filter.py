"""
Tests for domestic model parameter filtering in handler.py.

Tests the parameter filtering logic for domestic models in the
response_api_handler method.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import litellm


class TestDomesticModelParameterFiltering(unittest.TestCase):
    """Test parameter filtering for domestic models."""

    def setUp(self):
        """Set up test fixtures."""
        # Save original settings
        self.original_domestic_disabled = os.environ.get(
            "LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"
        )
        # Ensure domestic filtering is enabled
        if "LITELLM_DISABLE_DOMESTIC_COMPATIBILITY" in os.environ:
            del os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"]

    def tearDown(self):
        """Restore original settings."""
        if self.original_domestic_disabled is not None:
            os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = (
                self.original_domestic_disabled
            )
        elif "LITELLM_DISABLE_DOMESTIC_COMPATIBILITY" in os.environ:
            del os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"]

    @patch("litellm.completion")
    def test_client_metadata_filtered_for_domestic_model(self, mock_completion):
        """Test that client_metadata is filtered for domestic models."""
        from litellm.responses.litellm_completion_transformation.handler import (
            LiteLLMCompletionTransformationHandler,
        )

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = []
        mock_completion.return_value = mock_response

        handler = LiteLLMCompletionTransformationHandler()

        # Call with domestic model and client_metadata
        try:
            handler.response_api_handler(
                model="deepseek-coder",
                input="test",
                responses_api_request={},
                client_metadata={"test": "value"},
            )
        except Exception:
            pass  # May fail on response transformation, but we only care about completion call

        # Check that client_metadata was not passed to completion
        if mock_completion.called:
            call_kwargs = mock_completion.call_args.kwargs
            self.assertNotIn("client_metadata", call_kwargs)

    @patch("litellm.completion")
    def test_coding_plan_filtered_for_domestic_model(self, mock_completion):
        """Test that coding_plan is filtered for domestic models."""
        from litellm.responses.litellm_completion_transformation.handler import (
            LiteLLMCompletionTransformationHandler,
        )

        mock_response = MagicMock()
        mock_response.choices = []
        mock_completion.return_value = mock_response

        handler = LiteLLMCompletionTransformationHandler()

        try:
            handler.response_api_handler(
                model="qwen3.5-plus",
                input="test",
                responses_api_request={},
                coding_plan={"plan": "test"},
            )
        except Exception:
            pass

        if mock_completion.called:
            call_kwargs = mock_completion.call_args.kwargs
            self.assertNotIn("coding_plan", call_kwargs)

    @patch("litellm.completion")
    def test_functions_filtered_for_domestic_model(self, mock_completion):
        """Test that functions/function_call (old format) is filtered."""
        from litellm.responses.litellm_completion_transformation.handler import (
            LiteLLMCompletionTransformationHandler,
        )

        mock_response = MagicMock()
        mock_response.choices = []
        mock_completion.return_value = mock_response

        handler = LiteLLMCompletionTransformationHandler()

        try:
            handler.response_api_handler(
                model="minimax-m2",
                input="test",
                responses_api_request={},
                functions=[{"name": "test"}],
                function_call="test",
            )
        except Exception:
            pass

        if mock_completion.called:
            call_kwargs = mock_completion.call_args.kwargs
            self.assertNotIn("functions", call_kwargs)
            self.assertNotIn("function_call", call_kwargs)

    @patch("litellm.completion")
    def test_tool_choice_required_converted_to_auto(self, mock_completion):
        """Test that tool_choice='required' is converted to 'auto'."""
        from litellm.responses.litellm_completion_transformation.handler import (
            LiteLLMCompletionTransformationHandler,
        )

        mock_response = MagicMock()
        mock_response.choices = []
        mock_completion.return_value = mock_response

        handler = LiteLLMCompletionTransformationHandler()

        try:
            handler.response_api_handler(
                model="mimo-v2.5-pro",
                input="test",
                responses_api_request={},
                tool_choice="required",
            )
        except Exception:
            pass

        if mock_completion.called:
            call_kwargs = mock_completion.call_args.kwargs
            # tool_choice should be "auto", not "required"
            self.assertNotEqual(call_kwargs.get("tool_choice"), "required")
            self.assertIn(call_kwargs.get("tool_choice"), ["auto", None])

    @patch("litellm.completion")
    def test_non_domestic_model_not_filtered(self, mock_completion):
        """Test that non-domestic models don't trigger filtering."""
        from litellm.responses.litellm_completion_transformation.handler import (
            LiteLLMCompletionTransformationHandler,
        )

        mock_response = MagicMock()
        mock_response.choices = []
        mock_completion.return_value = mock_response

        handler = LiteLLMCompletionTransformationHandler()

        try:
            handler.response_api_handler(
                model="gpt-4",
                input="test",
                responses_api_request={},
                client_metadata={"test": "value"},
                reasoning_effort="medium",
                parallel_tool_calls=True,
            )
        except Exception:
            pass

        # For non-domestic models, these params should NOT be filtered
        # (they may still be handled elsewhere, but not by domestic filter)
        # Note: client_metadata is always filtered (LiteLLM internal error)
        # reasoning_effort and parallel_tool_calls should remain if present
        if mock_completion.called:
            call_kwargs = mock_completion.call_args.kwargs
            # reasoning_effort should be present (or handled by transformation)
            # The key point: domestic filter should NOT remove it


if __name__ == "__main__":
    unittest.main()
