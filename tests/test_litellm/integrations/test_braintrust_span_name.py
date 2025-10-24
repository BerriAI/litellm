import json
import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import litellm
from litellm.integrations.braintrust_logging import BraintrustLogger


class TestBraintrustSpanName(unittest.TestCase):
    """Test custom span_name functionality in Braintrust logging."""

    @patch("litellm.integrations.braintrust_logging.HTTPHandler")
    def test_default_span_name(self, MockHTTPHandler):
        """Test that default span name is 'Chat Completion' when not provided."""
        # Mock HTTP response
        mock_http_handler = Mock()
        mock_http_handler.post.return_value = Mock()
        MockHTTPHandler.return_value = mock_http_handler

        # Setup
        logger = BraintrustLogger(api_key="test-key")
        logger.default_project_id = "test-project-id"

        # Create a properly structured mock response
        response_obj = litellm.ModelResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test response"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        kwargs = {
            "litellm_call_id": "test-call-id",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_params": {"metadata": {}},
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
        }

        # Execute
        logger.log_success_event(kwargs, response_obj, datetime.now(), datetime.now())

        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs["json"]
        self.assertEqual(
            json_data["events"][0]["span_attributes"]["name"], "Chat Completion"
        )

    @patch("litellm.integrations.braintrust_logging.HTTPHandler")
    def test_custom_span_name(self, MockHTTPHandler):
        """Test that custom span name is used when provided in metadata."""
        # Mock HTTP response
        mock_http_handler = Mock()
        mock_http_handler.post.return_value = Mock()
        MockHTTPHandler.return_value = mock_http_handler

        # Setup
        logger = BraintrustLogger(api_key="test-key")
        logger.default_project_id = "test-project-id"

        # Create a properly structured mock response
        response_obj = litellm.ModelResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test response"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        kwargs = {
            "litellm_call_id": "test-call-id",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_params": {"metadata": {"span_name": "Custom Operation"}},
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
        }

        # Execute
        logger.log_success_event(kwargs, response_obj, datetime.now(), datetime.now())

        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs["json"]
        self.assertEqual(
            json_data["events"][0]["span_attributes"]["name"], "Custom Operation"
        )

    @patch("litellm.integrations.braintrust_logging.HTTPHandler")
    def test_span_name_with_other_metadata(self, MockHTTPHandler):
        """Test that span_name works alongside other metadata fields."""
        # Mock HTTP response
        mock_http_handler = Mock()
        mock_http_handler.post.return_value = Mock()
        MockHTTPHandler.return_value = mock_http_handler

        # Setup
        logger = BraintrustLogger(api_key="test-key")
        logger.default_project_id = "test-project-id"

        # Create a properly structured mock response
        response_obj = litellm.ModelResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test response"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        kwargs = {
            "litellm_call_id": "test-call-id",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_params": {
                "metadata": {
                    "span_name": "Multi Metadata Test",
                    "project_id": "custom-project",
                    "user_id": "user123",
                    "session_id": "session456",
                    "environment": "production",
                }
            },
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "standard_logging_object": {
                "user_id": "user123",
            },
        }

        # Execute
        logger.log_success_event(kwargs, response_obj, datetime.now(), datetime.now())

        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs["json"]

        # Check span name
        self.assertEqual(
            json_data["events"][0]["span_attributes"]["name"], "Multi Metadata Test"
        )

        # Check that other metadata is preserved (except for filtered keys)
        event_metadata = json_data["events"][0]["metadata"]
        print(event_metadata)
        self.assertEqual(event_metadata["user_id"], "user123")
        self.assertEqual(event_metadata["session_id"], "session456")
        self.assertEqual(event_metadata["environment"], "production")

        # Span name should be in span_attributes, not in metadata
        self.assertIn("span_name", event_metadata)  # span_name is also kept in metadata

    @patch("litellm.integrations.braintrust_logging.get_async_httpx_client")
    async def test_async_custom_span_name(self, mock_get_http_handler):
        """Test async logging with custom span name."""
        # Mock async HTTP response
        mock_http_handler = MagicMock()
        mock_http_handler.post = MagicMock(return_value=Mock())
        mock_get_http_handler.return_value = mock_http_handler

        # Setup
        logger = BraintrustLogger(api_key="test-key")
        logger.default_project_id = "test-project-id"

        # Create a properly structured mock response
        response_obj = litellm.ModelResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test response"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        kwargs = {
            "litellm_call_id": "test-call-id",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_params": {"metadata": {"span_name": "Async Custom Operation"}},
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
        }

        # Execute
        await logger.async_log_success_event(
            kwargs, response_obj, datetime.now(), datetime.now()
        )

        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs["json"]
        self.assertEqual(
            json_data["events"][0]["span_attributes"]["name"], "Async Custom Operation"
        )

    @patch('litellm.integrations.braintrust_logging.HTTPHandler')
    def test_span_attributes_with_multiple_metadata_fields(self, MockHTTPHandler):
        """Test that span_name works correctly alongside other metadata fields."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test-project-id"}
        mock_http_handler = Mock()
        mock_http_handler.post.return_value = mock_response
        MockHTTPHandler.return_value = mock_http_handler

        # Setup
        logger = BraintrustLogger(api_key="test-key")
        logger.default_project_id = "test-project-id"
        
        # Create a mock response object
        message_mock = Mock()
        message_mock.json = Mock(return_value={"content": "test"})
        
        choice_mock = Mock()
        choice_mock.message = message_mock
        choice_mock.dict = Mock(return_value={"message": {"content": "test"}})
        choice_mock.__getitem__ = Mock(return_value=message_mock)
        
        response_obj = Mock(spec=litellm.ModelResponse)
        response_obj.choices = [choice_mock]
        response_obj.__getitem__ = Mock(return_value=[choice_mock])
        response_obj.usage = litellm.Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        kwargs = {
            "litellm_call_id": "test-call-id",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_params": {
                "metadata": {
                    "span_name": "Multi Metadata Test",
                    "span_id": "span_id",
                    "root_span_id": "root_span_id",
                    "span_parents": "span_parent1,span_parent2",
                    "project_id": "custom-project",
                    "user_id": "user123",
                    "session_id": "session456"
                }
            },
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001
        }
        
        # Execute
        logger.log_success_event(kwargs, response_obj, datetime.now(), datetime.now())
        
        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs['json']
        
        # Check span name
        self.assertEqual(json_data['events'][0]['span_attributes']['name'], 'Multi Metadata Test')
        self.assertEqual(json_data['events'][0]['span_id'], 'span_id')
        self.assertEqual(json_data['events'][0]['root_span_id'], 'root_span_id')
        self.assertEqual(json_data['events'][0]['span_parents'][0], 'span_parent1')
        self.assertEqual(json_data['events'][0]['span_parents'][1], 'span_parent2')
        
        # Check that other metadata is preserved
        event_metadata = json_data['events'][0]['metadata']
        self.assertEqual(event_metadata['user_id'], 'user123')
        self.assertEqual(event_metadata['session_id'], 'session456')


if __name__ == "__main__":
    unittest.main()
