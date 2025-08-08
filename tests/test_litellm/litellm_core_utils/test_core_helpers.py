import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.core_helpers import (
    add_metadata_to_request_body,
    get_litellm_metadata_from_kwargs,
)


def test_get_litellm_metadata_from_kwargs():
    kwargs = {
        "litellm_params": {
            "litellm_metadata": {},
            "metadata": {"user_api_key": "1234567890"},
        },
    }
    assert get_litellm_metadata_from_kwargs(kwargs) == {"user_api_key": "1234567890"}


def test_add_missing_spend_metadata_to_litellm_metadata():
    litellm_metadata = {"test_key": "test_value"}
    metadata = {"user_api_key_hash_value": "1234567890"}
    kwargs = {
        "litellm_params": {
            "litellm_metadata": litellm_metadata,
            "metadata": metadata,
        },
    }
    assert get_litellm_metadata_from_kwargs(kwargs) == {
        "test_key": "test_value",
        "user_api_key_hash_value": "1234567890",
    }


def test_preserve_upstream_non_openai_attributes():
    from litellm.litellm_core_utils.core_helpers import (
        preserve_upstream_non_openai_attributes,
    )
    from litellm.types.utils import ModelResponseStream

    model_response = ModelResponseStream(
        id="123",
        object="text_completion",
        created=1715811200,
        model="gpt-3.5-turbo",
    )

    setattr(model_response, "test_key", "test_value")
    preserve_upstream_non_openai_attributes(
        model_response=ModelResponseStream(),
        original_chunk=model_response,
    )

    assert model_response.test_key == "test_value"


def test_add_metadata_to_request_body_with_metadata():
    """Test adding metadata to request body when metadata is present"""
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.7,
    }
    
    litellm_params = {
        "metadata": {
            "user_id": "123",
            "tags": ["test", "unit-test"]
        }
    }
    
    result = add_metadata_to_request_body(request_body, litellm_params)
    
    expected = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.7,
        "metadata": {
            "user_id": "123",
            "tags": ["test", "unit-test"]
        }
    }
    
    assert result == expected


def test_add_metadata_to_request_body_with_litellm_metadata():
    """Test adding metadata to request body when litellm_metadata is present"""
    request_body = {
        "model": "claude-3",
        "messages": [{"role": "user", "content": "test"}]
    }
    
    litellm_params = {
        "litellm_metadata": {
            "user_api_key": "sk-123",
            "proxy_server_request": {"url": "http://localhost"}
        }
    }
    
    result = add_metadata_to_request_body(request_body, litellm_params)
    
    expected = {
        "model": "claude-3",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {
            "user_api_key": "sk-123",
            "proxy_server_request": {"url": "http://localhost"}
        }
    }
    
    assert result == expected


def test_add_metadata_to_request_body_with_both_metadata_sources():
    """Test adding metadata when both metadata and litellm_metadata are present"""
    request_body = {"model": "test-model"}
    
    # litellm_metadata takes priority, but spend-tracking metadata gets merged from metadata
    litellm_params = {
        "metadata": {"user_api_key_hash": "abc123", "user_id": "123"},
        "litellm_metadata": {"proxy_key": "sk-123"}
    }
    
    result = add_metadata_to_request_body(request_body, litellm_params)
    
    expected = {
        "model": "test-model",
        "metadata": {
            "proxy_key": "sk-123",
            "user_api_key_hash": "abc123"  # This gets merged because it contains "user_api_key"
        }
    }
    
    assert result == expected


def test_add_metadata_to_request_body_no_metadata():
    """Test that request body is unchanged when no metadata is present"""
    request_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "test"}],
        "stream": True
    }
    
    litellm_params = {"some_other_param": "value"}
    
    result = add_metadata_to_request_body(request_body, litellm_params)
    
    # Should be unchanged
    assert result == request_body


def test_add_metadata_to_request_body_empty_metadata():
    """Test handling of empty metadata"""
    request_body = {"model": "test"}
    litellm_params = {"metadata": {}, "litellm_metadata": {}}
    
    result = add_metadata_to_request_body(request_body, litellm_params)
    
    # Should be unchanged since metadata is empty
    assert result == request_body


def test_add_metadata_to_request_body_none_metadata():
    """Test handling of None metadata"""
    request_body = {"model": "test"}
    litellm_params = {"metadata": None, "litellm_metadata": None}
    
    result = add_metadata_to_request_body(request_body, litellm_params)
    
    # Should be unchanged since metadata is None
    assert result == request_body
