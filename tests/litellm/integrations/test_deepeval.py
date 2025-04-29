import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
import os
import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


class TestDeepEvalLogger(unittest.TestCase):
    def setUp(self):
        # First, mock the imported modules that DeepEvalLogger needs
        self.update_current_span_attributes_mock = MagicMock()
        self.observe_mock = MagicMock()
        self.tool_attributes_mock = MagicMock()
        self.llm_attributes_mock = MagicMock()

        # Create module patches before instantiating DeepEvalLogger
        self.module_patches = [
            patch(
                "deepeval.tracing.update_current_span_attributes",
                self.update_current_span_attributes_mock,
            ),
            patch("deepeval.tracing.LlmAttributes", self.llm_attributes_mock),
            patch("deepeval.tracing.observe", self.observe_mock),
            patch("deepeval.tracing.tracing.ToolAttributes", self.tool_attributes_mock),
        ]

        # Start all the patches
        for p in self.module_patches:
            p.start()

        # Now import and instantiate DeepEvalLogger
        from litellm.integrations.deepeval.deepeval import DeepEvalLogger

        self.logger = DeepEvalLogger()

        # Mock the observe decorator to execute the function
        self.observe_mock.return_value = lambda func: func

        # Setup common test variables
        self.start_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
        self.end_time = datetime.datetime(2023, 1, 1, 12, 0, 1)

    def tearDown(self):
        # Stop all patches
        for p in self.module_patches:
            p.stop()

    def test_log_failure_event(self):
        """Test the log_failure_event method."""
        # Setup test data
        kwargs = {
            "model": "gpt-4",
            "input": [{"role": "user", "content": "Hello"}],
            "standard_logging_object": {"error_str": "Test error"},
        }

        response_obj = MagicMock()

        # Test the method and verify it raises an exception
        with self.assertRaises(Exception) as context:
            self.logger.log_failure_event(
                kwargs, response_obj, self.start_time, self.end_time
            )

        # Verify the exception message
        self.assertEqual(str(context.exception), "Test error")

        # Verify observe was called with the correct parameters
        self.observe_mock.assert_called_with(
            type="llm",
            model="gpt-4",
            name="litellm_message_failure",
            error_str="Test error",
        )

        # Verify update_current_span_attributes was called
        self.update_current_span_attributes_mock.assert_called_once()
        # Check LlmAttributes was called with correct parameters
        self.llm_attributes_mock.assert_called_once()
        llm_call_args = self.llm_attributes_mock.call_args[1]
        self.assertEqual(llm_call_args["input"], "user: Hello\n")
        self.assertEqual(llm_call_args["output"], "Test error")

    def test_log_success_event(self):
        """Test the log_success_event method."""
        # Setup mocks for helper methods
        with patch.object(
            self.logger, "_get_cost_per_token", return_value=(0.01, 0.02)
        ), patch.object(self.logger, "_get_token_count", return_value=(10, 20)):
            # Setup test data
            kwargs = {
                "model": "gpt-4",
                "input": [{"role": "user", "content": "Hello"}],
                "litellm_params": {
                    "metadata": {"deepeval_metrics": ["accuracy", "relevance"]}
                },
            }

            # Create a response with content
            choice = MagicMock()
            choice.message.content = "Hi there"
            choice.message.tool_calls = None

            response_obj = MagicMock()
            response_obj.choices = [choice]
            response_obj.usage.prompt_tokens = 10
            response_obj.usage.completion_tokens = 20

            # Call the method
            self.logger.log_success_event(
                kwargs, response_obj, self.start_time, self.end_time
            )

            # Verify observe was called for message success
            self.observe_mock.assert_any_call(
                type="llm",
                model="gpt-4",
                name="litellm_message_success",
                cost_per_output_token=0.02,
                cost_per_input_token=0.01,
                metrics=["accuracy", "relevance"],
            )

            # Verify update_current_span_attributes was called
            self.update_current_span_attributes_mock.assert_called()

            # Verify LlmAttributes was called with correct parameters
            self.llm_attributes_mock.assert_called_with(
                input="user: Hello\n",
                output="Hi there",
                input_token_count=10,
                output_token_count=20,
            )

    def test_log_success_event_with_tool_calls(self):
        """Test the log_success_event method with tool calls."""
        # Setup mocks for helper methods
        with patch.object(
            self.logger, "_get_cost_per_token", return_value=(0.01, 0.02)
        ), patch.object(self.logger, "_get_token_count", return_value=(10, 20)):
            # Setup test data
            kwargs = {
                "model": "gpt-4",
                "input": [{"role": "user", "content": "Get the weather"}],
                "litellm_params": {"metadata": {"deepeval_metrics": ["accuracy"]}},
            }

            # Create a response with tool calls
            function_dict = {
                "name": "get_weather",
                "arguments": '{"location": "New York"}',
            }
            tool_call = MagicMock()
            tool_call.function.to_dict.return_value = function_dict

            choice = MagicMock()
            choice.message.content = None
            choice.message.tool_calls = [tool_call]

            response_obj = MagicMock()
            response_obj.choices = [choice]

            # Call the method
            self.logger.log_success_event(
                kwargs, response_obj, self.start_time, self.end_time
            )

            # Verify observe was called for tool success
            self.observe_mock.assert_any_call(
                type="tool",
                cost_per_output_token=0.02,
                cost_per_input_token=0.01,
                metrics=["accuracy"],
            )

            # Verify ToolAttributes was called with correct parameters
            self.tool_attributes_mock.assert_called_with(
                input_parameters=function_dict, output=None
            )

            # Verify update_current_span_attributes was called
            self.update_current_span_attributes_mock.assert_called()


if __name__ == "__main__":
    unittest.main()
