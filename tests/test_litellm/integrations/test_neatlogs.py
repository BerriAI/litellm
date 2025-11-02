from litellm.integrations.neatlogs.neatlogs import NeatlogsLogger
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os


sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


@pytest.fixture
def neatlogs_config():
    """NeatLogs configuration"""
    return {
        "api_key": "test_api_key",
        "endpoint": "https://app.neatlogs.com/api/data/v2"
    }


@pytest.fixture
def sample_payload():
    """Sample payload for testing"""
    return {
        "session_id": "test_session",
        "agent_id": "test_agent",
        "thread_id": "test_thread",
        "span_id": "test_span_id",
        "trace_id": "test_trace_id",
        "parent_span_id": None,
        "node_type": "llm_call",
        "node_name": "gpt-4",
        "model": "gpt-4",
        "provider": "openai",
        "framework": None,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "cost": 0.002,
        "messages": [{"role": "user", "content": "Hello!"}],
        "completion": "Hello! How can I help you?",
        "timestamp": "2025-01-01T12:00:00.000000",
        "start_time": 1735737600.0,
        "end_time": 1735737602.0,
        "duration": 2.0,
        "tags": ["test"],
        "error_report": None,
        "status": "SUCCESS",
        "api_key": "test_api_key",
    }


def test_neatlogs_logger_initialization():
    """Test NeatlogsLogger initialization with API key"""
    with patch.dict(os.environ, {"NEATLOGS_API_KEY": "test_key"}):
        logger = NeatlogsLogger()
        assert logger.api_key == "test_key"
        assert logger.endpoint == "https://app.neatlogs.com/api/data/v2"


def test_neatlogs_logger_initialization_with_param():
    """Test NeatlogsLogger initialization with API key parameter"""
    logger = NeatlogsLogger(neatlogs_api_key="param_key")
    assert logger.api_key == "param_key"


def test_neatlogs_logger_initialization_no_key():
    """Test NeatlogsLogger initialization without API key"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(Exception, match="NEATLOGS_API_KEY is not set"):
            NeatlogsLogger()


def test_create_neatlogs_payload(sample_payload):
    """Test payload creation"""
    with patch.dict(os.environ, {"NEATLOGS_API_KEY": "test_key"}):
        logger = NeatlogsLogger()

        # Mock datetime objects
        import datetime
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=2)

        # Mock kwargs and response
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
            "litellm_call_id": "test_call_id"
        }

        response_obj = MagicMock()
        response_obj.choices = [MagicMock()]
        response_obj.choices[0].message.content = "Hello! How can I help you?"
        response_obj.usage.prompt_tokens = 10
        response_obj.usage.completion_tokens = 20
        response_obj.usage.total_tokens = 30

        payload = logger.create_neatlogs_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            status="SUCCESS"
        )

        assert payload["model"] == "gpt-4"
        assert payload["status"] == "SUCCESS"
        assert payload["prompt_tokens"] == 10
        assert payload["completion_tokens"] == 20
        assert payload["total_tokens"] == 30
        assert payload["messages"] == [{"role": "user", "content": "Hello!"}]
        assert payload["completion"] == "Hello! How can I help you?"


@patch("litellm.integrations.neatlogs.neatlogs._get_httpx_client")
def test_log_success_event(mock_get_client):
    """Test sync success event logging"""
    with patch.dict(os.environ, {"NEATLOGS_API_KEY": "test_key"}):
        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.raise_for_status = MagicMock()
        mock_response.text = '{"message": "Data saved successfully"}'

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        logger = NeatlogsLogger()

        # Mock datetime objects
        import datetime
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=2)

        # Mock kwargs and response
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
            "litellm_call_id": "test_call_id"
        }

        response_obj = MagicMock()
        response_obj.choices = [MagicMock()]
        response_obj.choices[0].message.content = "Hello! How can I help you?"
        response_obj.usage.prompt_tokens = 10
        response_obj.usage.completion_tokens = 20
        response_obj.usage.total_tokens = 30

        # Call the method
        logger.log_success_event(kwargs, response_obj, start_time, end_time)

        # Verify the API was called with correct format
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check the payload format
        payload = call_args[1]["json"]
        assert "dataDump" in payload
        assert "projectAPIKey" in payload
        assert "externalTraceId" in payload
        assert "timestamp" in payload

        # Verify the dataDump contains our payload as JSON string
        data_dump = json.loads(payload["dataDump"])
        assert data_dump["model"] == "gpt-4"
        assert data_dump["status"] == "SUCCESS"


@patch("litellm.integrations.neatlogs.neatlogs._get_httpx_client")
def test_log_failure_event(mock_get_client):
    """Test sync failure event logging"""
    with patch.dict(os.environ, {"NEATLOGS_API_KEY": "test_key"}):
        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.raise_for_status = MagicMock()
        mock_response.text = '{"message": "Data saved successfully"}'

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        logger = NeatlogsLogger()

        # Mock datetime objects
        import datetime
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=2)

        # Mock kwargs and response
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
            "litellm_call_id": "test_call_id"
        }

        response_obj = MagicMock()
        response_obj.__str__ = MagicMock(return_value="Mock error response")

        # Call the method
        logger.log_failure_event(kwargs, response_obj, start_time, end_time)

        # Verify the API was called
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check the payload format
        payload = call_args[1]["json"]
        assert "dataDump" in payload
        assert "projectAPIKey" in payload
        assert "externalTraceId" in payload
        assert "timestamp" in payload

        # Verify the dataDump contains failure status
        data_dump = json.loads(payload["dataDump"])
        assert data_dump["status"] == "FAILURE"


@patch("litellm.integrations.neatlogs.neatlogs.get_async_httpx_client")
@patch("asyncio.create_task")
def test_async_send_batch(mock_create_task, mock_get_client):
    """Test async batch sending"""
    with patch.dict(os.environ, {"NEATLOGS_API_KEY": "test_key"}):
        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.raise_for_status = MagicMock()
        mock_response.text = '{"message": "Data saved successfully"}'

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        logger = NeatlogsLogger()

        # Add a test payload to the queue
        test_payload = {
            "model": "gpt-4",
            "status": "SUCCESS",
            "timestamp": "2025-01-01T12:00:00.000000"
        }
        logger.log_queue.append(test_payload)

        # Call async_send_batch
        import asyncio
        asyncio.run(logger.async_send_batch())

        # Verify the API was called with correct format
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check the payload format (should be a list for batch)
        payload = call_args[1]["json"]
        assert isinstance(payload, list)
        assert len(payload) == 1

        batch_item = payload[0]
        assert "dataDump" in batch_item
        assert "projectAPIKey" in batch_item
        assert "externalTraceId" in batch_item
        assert "timestamp" in batch_item

        # Verify the queue was cleared
        assert len(logger.log_queue) == 0


def test_payload_structure(sample_payload):
    """Test that payload has all required fields"""
    required_fields = [
        "session_id", "agent_id", "thread_id", "span_id", "trace_id",
        "node_type", "node_name", "model", "provider", "framework",
        "prompt_tokens", "completion_tokens", "total_tokens", "cost",
        "messages", "completion", "timestamp", "start_time", "end_time",
        "duration", "tags", "status", "api_key"
    ]

    for field in required_fields:
        assert field in sample_payload, f"Missing required field: {field}"


def test_payload_data_types(sample_payload):
    """Test that payload fields have correct data types"""
    # String fields
    string_fields = ["session_id", "agent_id", "thread_id", "span_id", "trace_id",
                     "node_type", "node_name", "model", "provider", "framework",
                     "completion", "timestamp", "status", "api_key"]
    for field in string_fields:
        if sample_payload[field] is not None:
            assert isinstance(
                sample_payload[field], str), f"Field {field} should be string"

    # Integer fields
    int_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
    for field in int_fields:
        assert isinstance(sample_payload[field],
                          int), f"Field {field} should be integer"

    # Float fields
    float_fields = ["cost", "start_time", "end_time", "duration"]
    for field in float_fields:
        assert isinstance(sample_payload[field],
                          float), f"Field {field} should be float"

    # List fields
    list_fields = ["messages", "tags"]
    for field in list_fields:
        assert isinstance(sample_payload[field],
                          list), f"Field {field} should be list"


def test_logger_inheritance():
    """Test that NeatlogsLogger properly inherits from required classes"""
    with patch.dict(os.environ, {"NEATLOGS_API_KEY": "test_key"}):
        from litellm.integrations.custom_batch_logger import CustomBatchLogger
        from litellm.integrations.additional_logging_utils import AdditionalLoggingUtils

        logger = NeatlogsLogger()

        # Check inheritance
        assert isinstance(logger, CustomBatchLogger)
        assert isinstance(logger, AdditionalLoggingUtils)

        # Check required attributes exist
        assert hasattr(logger, 'log_queue')
        assert hasattr(logger, 'batch_size')
        assert hasattr(logger, 'flush_interval')
        assert hasattr(logger, 'flush_lock')


def test_health_check_method():
    """Test the async_health_check method exists and is callable"""
    with patch.dict(os.environ, {"NEATLOGS_API_KEY": "test_key"}):
        logger = NeatlogsLogger()

        # Check method exists
        assert hasattr(logger, 'async_health_check')
        assert callable(getattr(logger, 'async_health_check'))


async def test_get_request_response_payload_method():
    """Test the get_request_response_payload method exists"""
    with patch.dict(os.environ, {"NEATLOGS_API_KEY": "test_key"}):
        logger = NeatlogsLogger()

        # Check method exists
        assert hasattr(logger, 'get_request_response_payload')
        assert callable(getattr(logger, 'get_request_response_payload'))

        # Test calling the method
        result = await logger.get_request_response_payload("test_id", None, None)
        assert result is None  # Method returns None as per implementation
