import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


class TestOpenTelemetry(unittest.TestCase):
    @patch("litellm.integrations.opentelemetry.datetime")
    def test_create_guardrail_span_with_valid_info(self, mock_datetime):
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()
        mock_span = MagicMock()
        otel.tracer.start_span.return_value = mock_span

        # Create guardrail information
        guardrail_info = {
            "guardrail_name": "test_guardrail",
            "guardrail_mode": "input",
            "masked_entity_count": {"CREDIT_CARD": 2},
            "guardrail_response": "filtered_content",
            "start_time": 1609459200.0,
            "end_time": 1609459201.0,
        }

        # Create a kwargs dict with standard_logging_object containing guardrail information
        kwargs = {"standard_logging_object": {"guardrail_information": guardrail_info}}

        # Call the method
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        # Assertions
        otel.tracer.start_span.assert_called_once()

        # print all calls to mock_span.set_attribute
        print("Calls to mock_span.set_attribute:")
        for call in mock_span.set_attribute.call_args_list:
            print(call)

        # Check that the span has the correct attributes set
        mock_span.set_attribute.assert_any_call("guardrail_name", "test_guardrail")
        mock_span.set_attribute.assert_any_call("guardrail_mode", "input")
        mock_span.set_attribute.assert_any_call(
            "guardrail_response", "filtered_content"
        )
        mock_span.set_attribute.assert_any_call(
            "masked_entity_count", safe_dumps({"CREDIT_CARD": 2})
        )

        # Verify that the span was ended
        mock_span.end.assert_called_once()

    def test_create_guardrail_span_with_no_info(self):
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()

        # Test with no guardrail information
        kwargs = {"standard_logging_object": {}}
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        # Verify that start_span was never called
        otel.tracer.start_span.assert_not_called()
