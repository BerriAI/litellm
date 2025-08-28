import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import litellm
from litellm.integrations.braintrust_logging import BraintrustLogger

class TestBraintrustLogger(unittest.TestCase):
    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-env-api-key"})
    @patch.dict(os.environ, {"BRAINTRUST_API_BASE": "https://test-env-api.com/v1"})
    def test_init_with_env_var(self):
        """Test BraintrustLogger initialization with environment variable."""
        logger = BraintrustLogger()
        self.assertEqual(logger.api_key, "test-env-api-key")
        self.assertEqual(logger.api_base, "https://test-env-api.com/v1")
        self.assertEqual(logger.headers["Authorization"], "Bearer test-env-api-key")
        self.assertEqual(logger.headers["Content-Type"], "application/json")

    def test_init_with_explicit_params(self):
        """Test BraintrustLogger initialization with explicit parameters."""
        logger = BraintrustLogger(api_key="explicit-key", api_base="https://custom-api.com/v1")
        self.assertEqual(logger.api_key, "explicit-key")
        self.assertEqual(logger.api_base, "https://custom-api.com/v1")
        self.assertEqual(logger.headers["Authorization"], "Bearer explicit-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_api_key(self):
        """Test BraintrustLogger initialization fails without API key."""
        with self.assertRaises(Exception) as context:
            BraintrustLogger()
        self.assertIn("Missing keys=['BRAINTRUST_API_KEY']", str(context.exception))

    def test_validate_environment_with_api_key(self):
        """Test validate_environment method with valid API key."""
        logger = BraintrustLogger(api_key="test-key")
        # Should not raise an exception
        logger.validate_environment(api_key="test-key")

    def test_validate_environment_missing_api_key(self):
        """Test validate_environment method with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Exception) as context:
                BraintrustLogger(api_key=None)
            self.assertIn("Missing keys=['BRAINTRUST_API_KEY']", str(context.exception))

    @patch('litellm.integrations.braintrust_logging.HTTPHandler')
    def test_log_success_event_with_default_span_name(self, MockHTTPHandler):
        """Test log_success_event uses default span name when not provided."""
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
        # Mock the __getitem__ to support response_obj["choices"][0]["message"]
        choice_mock.__getitem__ = Mock(return_value=message_mock)
        
        response_obj = Mock(spec=litellm.ModelResponse)
        response_obj.choices = [choice_mock]
        # Mock the __getitem__ to support response_obj["choices"]
        response_obj.__getitem__ = Mock(return_value=[choice_mock])
        response_obj.usage = litellm.Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        kwargs = {
            "litellm_call_id": "test-call-id",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_params": {"metadata": {}},
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001
        }
        
        # Execute
        logger.log_success_event(kwargs, response_obj, datetime.now(), datetime.now())
        
        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs['json']
        self.assertEqual(json_data['events'][0]['span_attributes']['name'], 'Chat Completion')

    @patch('litellm.integrations.braintrust_logging.HTTPHandler')
    def test_log_success_event_with_custom_span_name(self, MockHTTPHandler):
        """Test log_success_event uses custom span name when provided."""
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
            "litellm_params": {"metadata": {"span_name": "Custom Operation"}},
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001
        }
        
        # Execute
        logger.log_success_event(kwargs, response_obj, datetime.now(), datetime.now())
        
        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs['json']
        self.assertEqual(json_data['events'][0]['span_attributes']['name'], 'Custom Operation')

    @patch('litellm.integrations.braintrust_logging.get_async_httpx_client')
    async def test_async_log_success_event_with_default_span_name(self, mock_get_http_handler):
        """Test async_log_success_event uses default span name when not provided."""
        # Mock async HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test-project-id"}
        mock_http_handler = MagicMock()
        mock_http_handler.post = MagicMock(return_value=mock_response)
        mock_get_http_handler.return_value = mock_http_handler

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
            "litellm_params": {"metadata": {}},
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001
        }
        
        # Execute
        await logger.async_log_success_event(kwargs, response_obj, datetime.now(), datetime.now())
        
        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs['json']
        self.assertEqual(json_data['events'][0]['span_attributes']['name'], 'Chat Completion')

    @patch('litellm.integrations.braintrust_logging.get_async_httpx_client')
    async def test_async_log_success_event_with_custom_span_name(self, mock_get_http_handler):
        """Test async_log_success_event uses custom span name when provided."""
        # Mock async HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test-project-id"}
        mock_http_handler = MagicMock()
        mock_http_handler.post = MagicMock(return_value=mock_response)
        mock_get_http_handler.return_value = mock_http_handler

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
            "litellm_params": {"metadata": {"span_name": "Async Custom Operation"}},
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001
        }
        
        # Execute
        await logger.async_log_success_event(kwargs, response_obj, datetime.now(), datetime.now())
        
        # Verify
        call_args = mock_http_handler.post.call_args
        self.assertIsNotNone(call_args)
        json_data = call_args.kwargs['json']
        self.assertEqual(json_data['events'][0]['span_attributes']['name'], 'Async Custom Operation')

    @patch('litellm.integrations.braintrust_logging.HTTPHandler')
    def test_span_name_with_multiple_metadata_fields(self, MockHTTPHandler):
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
        
        # Check that other metadata is preserved
        event_metadata = json_data['events'][0]['metadata']
        self.assertEqual(event_metadata['user_id'], 'user123')
        self.assertEqual(event_metadata['session_id'], 'session456')