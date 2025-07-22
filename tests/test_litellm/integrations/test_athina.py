import datetime
import json
import os
import sys
import unittest
from unittest.mock import ANY, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from litellm.integrations.athina import AthinaLogger


class TestAthinaLogger(unittest.TestCase):
    def setUp(self):
        # Set up environment variables for testing
        self.env_patcher = patch.dict(
            "os.environ",
            {
                "ATHINA_API_KEY": "test-api-key",
                "ATHINA_BASE_URL": "https://test.athina.ai",
            },
        )
        self.env_patcher.start()
        self.logger = AthinaLogger()

        # Setup common test variables
        self.start_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
        self.end_time = datetime.datetime(2023, 1, 1, 12, 0, 1)
        self.print_verbose = MagicMock()

    def tearDown(self):
        self.env_patcher.stop()

    def test_init(self):
        """Test the initialization of AthinaLogger"""
        self.assertEqual(self.logger.athina_api_key, "test-api-key")
        self.assertEqual(
            self.logger.athina_logging_url,
            "https://test.athina.ai/api/v1/log/inference",
        )
        self.assertEqual(
            self.logger.headers,
            {"athina-api-key": "test-api-key", "Content-Type": "application/json"},
        )

    @patch("litellm.module_level_client.post")
    def test_log_event_success(self, mock_post):
        """Test successful logging of an event"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Success"
        mock_post.return_value = mock_response

        # Create test data
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "litellm_params": {
                "metadata": {
                    "environment": "test-environment",
                    "prompt_slug": "test-prompt",
                    "customer_id": "test-customer",
                    "customer_user_id": "test-user",
                    "session_id": "test-session",
                    "external_reference_id": "test-ext-ref",
                    "context": "test-context",
                    "expected_response": "test-expected",
                    "user_query": "test-query",
                    "tags": ["test-tag"],
                    "user_feedback": "test-feedback",
                    "model_options": {"test-opt": "test-val"},
                    "custom_attributes": {"test-attr": "test-val"},
                }
            },
        }

        response_obj = MagicMock()
        response_obj.model_dump.return_value = {
            "id": "resp-123",
            "choices": [{"message": {"content": "Hi there"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Call the method
        self.logger.log_event(
            kwargs, response_obj, self.start_time, self.end_time, self.print_verbose
        )

        # Verify the results
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "https://test.athina.ai/api/v1/log/inference")
        self.assertEqual(call_args[1]["headers"], self.logger.headers)

        # Parse and verify the sent data
        sent_data = json.loads(call_args[1]["data"])
        self.assertEqual(sent_data["language_model_id"], "gpt-4")
        self.assertEqual(sent_data["prompt"], kwargs["messages"])
        self.assertEqual(sent_data["prompt_tokens"], 10)
        self.assertEqual(sent_data["completion_tokens"], 5)
        self.assertEqual(sent_data["total_tokens"], 15)
        self.assertEqual(sent_data["response_time"], 1000)  # 1 second = 1000ms
        self.assertEqual(sent_data["customer_id"], "test-customer")
        self.assertEqual(sent_data["session_id"], "test-session")
        self.assertEqual(sent_data["environment"], "test-environment")
        self.assertEqual(sent_data["prompt_slug"], "test-prompt")
        self.assertEqual(sent_data["external_reference_id"], "test-ext-ref")
        self.assertEqual(sent_data["context"], "test-context")
        self.assertEqual(sent_data["expected_response"], "test-expected")
        self.assertEqual(sent_data["user_query"], "test-query")
        self.assertEqual(sent_data["tags"], ["test-tag"])
        self.assertEqual(sent_data["user_feedback"], "test-feedback")
        self.assertEqual(sent_data["model_options"], {"test-opt": "test-val"})
        self.assertEqual(sent_data["custom_attributes"], {"test-attr": "test-val"})
        # Verify the print_verbose was called
        self.print_verbose.assert_called_once_with("Athina Logger Succeeded - Success")

    @patch("litellm.module_level_client.post")
    def test_log_event_error_response(self, mock_post):
        """Test handling of error response from the API"""
        # Setup mock error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        # Create test data
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        response_obj = MagicMock()
        response_obj.model_dump.return_value = {
            "id": "resp-123",
            "choices": [{"message": {"content": "Hi there"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Call the method
        self.logger.log_event(
            kwargs, response_obj, self.start_time, self.end_time, self.print_verbose
        )

        # Verify print_verbose was called with error message
        self.print_verbose.assert_called_once_with(
            "Athina Logger Error - Bad Request, 400"
        )

    @patch("litellm.module_level_client.post")
    def test_log_event_exception(self, mock_post):
        """Test handling of exceptions during logging"""
        # Setup mock to raise exception
        mock_post.side_effect = Exception("Test exception")

        # Create test data
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        response_obj = MagicMock()
        response_obj.model_dump.return_value = {}

        # Call the method
        self.logger.log_event(
            kwargs, response_obj, self.start_time, self.end_time, self.print_verbose
        )

        # Verify print_verbose was called with exception info
        self.print_verbose.assert_called_once()
        self.assertIn(
            "Athina Logger Error - Test exception", self.print_verbose.call_args[0][0]
        )

    @patch("litellm.module_level_client.post")
    def test_log_event_with_tools(self, mock_post):
        """Test logging with tools/functions data"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create test data with tools
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "stream": False,
            "optional_params": {
                "tools": [{"type": "function", "function": {"name": "get_weather"}}]
            },
        }

        response_obj = MagicMock()
        response_obj.model_dump.return_value = {
            "id": "resp-123",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Call the method
        self.logger.log_event(
            kwargs, response_obj, self.start_time, self.end_time, self.print_verbose
        )

        # Verify the results
        sent_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(
            sent_data["tools"],
            [{"type": "function", "function": {"name": "get_weather"}}],
        )


if __name__ == "__main__":
    unittest.main()
