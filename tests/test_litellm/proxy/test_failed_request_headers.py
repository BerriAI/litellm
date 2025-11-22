import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from litellm.proxy.proxy_server import app
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy._types import UserAPIKeyAuth

@pytest.fixture
def client():
    return TestClient(app)

def test_x_litellm_model_id_header_in_exception():
    """
    Directly test the logic in ProxyBaseLLMRequestProcessing._handle_llm_api_exception
    to ensure it extracts model_id from the logging object and passes it to get_custom_headers.
    """
    # Mock dependencies
    mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    mock_user_api_key_dict.allowed_model_region = "us-east-1"
    mock_user_api_key_dict.tpm_limit = 100
    mock_user_api_key_dict.rpm_limit = 10
    mock_user_api_key_dict.max_budget = 100.0
    mock_user_api_key_dict.spend = 5.0
    
    # Use AsyncMock for awaited methods
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock()
    
    # Create a mock exception
    exception = Exception("Test exception")
    
    # Create a mock logging object with model_id in litellm_params
    mock_litellm_logging_obj = MagicMock()
    mock_litellm_logging_obj.litellm_call_id = "test-call-id"
    mock_litellm_logging_obj.litellm_params = {
        "metadata": {
            "model_info": {
                "id": "test-model-id-123"
            }
        }
    }
    
    # Setup the processor with data containing the logging object
    data = {
        "litellm_logging_obj": mock_litellm_logging_obj,
        "model": "gpt-4"
    }
    processor = ProxyBaseLLMRequestProcessing(data=data)
    
    import asyncio
    from litellm.proxy._types import ProxyException
    
    try:
        asyncio.run(processor._handle_llm_api_exception(
            e=exception,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging_obj
        ))
    except ProxyException as pe:
        # Verify the headers in the raised exception
        assert "x-litellm-model-id" in pe.headers
        assert pe.headers["x-litellm-model-id"] == "test-model-id-123"
    except Exception as e:
        pytest.fail(f"Raised unexpected exception type: {type(e)}")

def test_x_litellm_model_id_header_in_exception_fallback_kwargs():
    """
    Test fallback to kwargs if litellm_params is missing/empty
    """
    # Mock dependencies
    mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    mock_user_api_key_dict.allowed_model_region = "us-east-1"
    # Need to mock tpm_limit/rpm_limit etc as they are accessed by get_custom_headers
    mock_user_api_key_dict.tpm_limit = 100
    mock_user_api_key_dict.rpm_limit = 10
    mock_user_api_key_dict.max_budget = 100.0
    mock_user_api_key_dict.spend = 5.0
    
    # Use AsyncMock for awaited methods
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock()
    
    exception = Exception("Test exception")
    
    # Create a mock logging object with model_id in kwargs
    mock_litellm_logging_obj = MagicMock()
    mock_litellm_logging_obj.litellm_call_id = "test-call-id"
    mock_litellm_logging_obj.litellm_params = {} # Empty
    mock_litellm_logging_obj.kwargs = {
        "litellm_params": {
            "metadata": {
                "model_info": {
                    "id": "fallback-model-id-456"
                }
            }
        }
    }
    
    data = {
        "litellm_logging_obj": mock_litellm_logging_obj,
        "model": "gpt-4"
    }
    processor = ProxyBaseLLMRequestProcessing(data=data)
    
    import asyncio
    from litellm.proxy._types import ProxyException
    
    try:
        asyncio.run(processor._handle_llm_api_exception(
            e=exception,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging_obj
        ))
    except ProxyException as pe:
        assert "x-litellm-model-id" in pe.headers
        assert pe.headers["x-litellm-model-id"] == "fallback-model-id-456"
    except Exception as e:
        pytest.fail(f"Raised unexpected exception type: {type(e)}")
