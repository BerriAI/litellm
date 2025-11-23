"""
Test that x-litellm-model-id header is propagated correctly on error responses.

This test suite verifies the `maybe_get_model_id` method
which is responsible for extracting model_id from different locations
depending on the request lifecycle stage.
"""

import pytest
from unittest.mock import MagicMock

from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy._types import UserAPIKeyAuth


def test_maybe_get_model_id_from_litellm_params():
    """
    Test extraction of model_id from logging_obj.litellm_params (used by /v1/chat/completions).
    """
    # Create a ProxyBaseLLMRequestProcessing instance
    processor = ProxyBaseLLMRequestProcessing(data={})

    # Create a mock logging object with model_info in litellm_params
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_params = {
        "model_info": {
            "id": "test-model-id-from-litellm-params"
        }
    }

    # Test extraction
    model_id = processor.maybe_get_model_id(mock_logging_obj)

    assert model_id == "test-model-id-from-litellm-params"


def test_maybe_get_model_id_from_litellm_params_nested():
    """
    Test extraction of model_id from nested metadata in logging_obj.litellm_params.
    """
    processor = ProxyBaseLLMRequestProcessing(data={})

    # Create a mock logging object with model_info nested in metadata
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_params = {
        "metadata": {
            "model_info": {
                "id": "test-model-id-nested"
            }
        }
    }

    # Test extraction
    model_id = processor.maybe_get_model_id(mock_logging_obj)

    assert model_id == "test-model-id-nested"


def test_maybe_get_model_id_from_kwargs():
    """
    Test extraction of model_id from logging_obj.kwargs (fallback path).
    """
    processor = ProxyBaseLLMRequestProcessing(data={})

    # Create a mock logging object with model_info in kwargs
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_params = None
    mock_logging_obj.kwargs = {
        "litellm_params": {
            "model_info": {
                "id": "test-model-id-from-kwargs"
            }
        }
    }

    # Test extraction
    model_id = processor.maybe_get_model_id(mock_logging_obj)

    assert model_id == "test-model-id-from-kwargs"


def test_maybe_get_model_id_from_data():
    """
    Test extraction of model_id from self.data (used by /v1/messages and /v1/responses).
    """
    # Create a processor with model_info in data
    processor = ProxyBaseLLMRequestProcessing(data={
        "litellm_metadata": {
            "model_info": {
                "id": "test-model-id-from-data"
            }
        }
    })

    # Create a mock logging object without model_info
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_params = {}
    mock_logging_obj.kwargs = {}

    # Test extraction - should fall back to self.data
    model_id = processor.maybe_get_model_id(mock_logging_obj)

    assert model_id == "test-model-id-from-data"


def test_maybe_get_model_id_no_logging_obj():
    """
    Test extraction of model_id when logging_obj is None (should use self.data).
    """
    # Create a processor with model_info in data
    processor = ProxyBaseLLMRequestProcessing(data={
        "litellm_metadata": {
            "model_info": {
                "id": "test-model-id-no-logging-obj"
            }
        }
    })

    # Test extraction with None logging_obj
    model_id = processor.maybe_get_model_id(None)

    assert model_id == "test-model-id-no-logging-obj"


def test_maybe_get_model_id_not_found():
    """
    Test extraction of model_id when it's not available anywhere (should return None).
    """
    processor = ProxyBaseLLMRequestProcessing(data={})

    # Create a mock logging object without model_info anywhere
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_params = {}
    mock_logging_obj.kwargs = {}

    # Test extraction - should return None
    model_id = processor.maybe_get_model_id(mock_logging_obj)

    assert model_id is None


def test_maybe_get_model_id_priority_litellm_params_over_data():
    """
    Test that model_id from logging_obj.litellm_params takes priority over self.data.
    """
    # Create a processor with model_info in both places
    processor = ProxyBaseLLMRequestProcessing(data={
        "litellm_metadata": {
            "model_info": {
                "id": "model-id-from-data"
            }
        }
    })

    # Create a mock logging object with model_info
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_params = {
        "model_info": {
            "id": "model-id-from-litellm-params"
        }
    }

    # Test extraction - should prefer litellm_params
    model_id = processor.maybe_get_model_id(mock_logging_obj)

    assert model_id == "model-id-from-litellm-params"


def test_get_custom_headers_includes_model_id():
    """
    Test that get_custom_headers includes x-litellm-model-id when model_id is provided.
    """
    # Create mock user_api_key_dict with all required attributes
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    mock_user_api_key_dict.team_id = "test-team"
    mock_user_api_key_dict.tpm_limit = 1000
    mock_user_api_key_dict.rpm_limit = 100

    # Call get_custom_headers with a model_id
    headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
        user_api_key_dict=mock_user_api_key_dict,
        model_id="test-model-123",
        cache_key="test-cache-key",
        api_base="https://api.example.com",
        version="1.0.0",
        response_cost=0.001,
        request_data={},
        hidden_params={}
    )

    # Verify model_id is in headers
    assert "x-litellm-model-id" in headers
    assert headers["x-litellm-model-id"] == "test-model-123"


def test_get_custom_headers_without_model_id():
    """
    Test that get_custom_headers works correctly when model_id is None or empty.
    """
    # Create mock user_api_key_dict with all required attributes
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    mock_user_api_key_dict.team_id = "test-team"
    mock_user_api_key_dict.tpm_limit = 1000
    mock_user_api_key_dict.rpm_limit = 100

    # Call get_custom_headers without a model_id
    headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
        user_api_key_dict=mock_user_api_key_dict,
        model_id=None,
        cache_key="test-cache-key",
        api_base="https://api.example.com",
        version="1.0.0",
        response_cost=0.001,
        request_data={},
        hidden_params={}
    )

    # x-litellm-model-id should not be in headers (or should be empty/None)
    if "x-litellm-model-id" in headers:
        assert headers["x-litellm-model-id"] in [None, ""]


def test_get_custom_headers_with_empty_string_model_id():
    """
    Test that get_custom_headers handles empty string model_id correctly.
    """
    # Create mock user_api_key_dict with all required attributes
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    mock_user_api_key_dict.team_id = "test-team"
    mock_user_api_key_dict.tpm_limit = 1000
    mock_user_api_key_dict.rpm_limit = 100

    # Call get_custom_headers with empty string model_id
    headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
        user_api_key_dict=mock_user_api_key_dict,
        model_id="",
        cache_key="test-cache-key",
        api_base="https://api.example.com",
        version="1.0.0",
        response_cost=0.001,
        request_data={},
        hidden_params={}
    )

    # x-litellm-model-id should not be in headers (or should be empty)
    if "x-litellm-model-id" in headers:
        assert headers["x-litellm-model-id"] == ""
