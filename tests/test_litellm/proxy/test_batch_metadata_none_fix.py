"""
Test for issue #13995: /batches request throws Internal Server Error when metadata=None

This test verifies that the fix for handling None metadata in batch requests works correctly.
"""
import asyncio
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from openai import OpenAI

import litellm
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy._types import UserAPIKeyAuth

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def test_add_key_level_controls_with_none_metadata():
    """
    Test that add_key_level_controls handles None metadata gracefully.
    This is the core fix for issue #13995.
    """
    # Test data
    data = {"metadata": {}}
    metadata_variable_name = "metadata"
    
    # Test with None key_metadata (this was causing the original error)
    result = LiteLLMProxyRequestSetup.add_key_level_controls(
        key_metadata=None,
        data=data,
        _metadata_variable_name=metadata_variable_name
    )
    
    # Should return the data unchanged without throwing an error
    assert result == data
    
    # Test with empty dict key_metadata (should also work)
    result = LiteLLMProxyRequestSetup.add_key_level_controls(
        key_metadata={},
        data=data,
        _metadata_variable_name=metadata_variable_name
    )
    
    # Should return the data unchanged
    assert result == data
    
    # Test with valid key_metadata containing cache settings
    key_metadata_with_cache = {
        "cache": {
            "ttl": 300,
            "s-maxage": 600
        }
    }
    
    result = LiteLLMProxyRequestSetup.add_key_level_controls(
        key_metadata=key_metadata_with_cache,
        data=data.copy(),
        _metadata_variable_name=metadata_variable_name
    )
    
    # Should add cache settings to data
    assert "cache" in result
    assert result["cache"]["ttl"] == 300
    assert result["cache"]["s-maxage"] == 600


def test_add_key_level_controls_simulates_original_issue():
    """
    Test that simulates the original issue scenario more directly.
    This tests the exact code path that was failing in issue #13995.
    """
    # This simulates the scenario where user_api_key_dict.metadata is None
    # which was causing the original "'NoneType' object has no attribute 'get'" error
    
    data = {"metadata": {}}
    metadata_variable_name = "metadata"
    
    # This is the exact call that was failing before the fix
    # user_api_key_dict.metadata was None, causing the error in add_key_level_controls
    try:
        result = LiteLLMProxyRequestSetup.add_key_level_controls(
            key_metadata=None,  # This was the root cause of the issue
            data=data,
            _metadata_variable_name=metadata_variable_name
        )
        
        # If we get here, the fix is working
        assert result == data
        print("✓ Original issue scenario handled correctly - no NoneType error")
        
    except AttributeError as e:
        if "'NoneType' object has no attribute 'get'" in str(e):
            pytest.fail("The fix for issue #13995 is not working - still getting NoneType error")
        else:
            # Some other AttributeError, re-raise it
            raise


def test_batch_create_with_litellm_sdk():
    """
    Test creating a batch using litellm SDK with metadata=None.
    This is a more direct test of the original issue.
    """
    # Mock the OpenAI batches instance to avoid actual API calls
    with patch('litellm.batches.main.openai_batches_instance') as mock_openai_batches:
        # Mock the response
        mock_response = MagicMock()
        mock_response.id = "batch_test123"
        mock_openai_batches.create_batch.return_value = mock_response
        
        # This should not raise an exception
        try:
            response = litellm.create_batch(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id="file-test123",
                metadata=None,  # This was causing the original issue
                custom_llm_provider="openai"
            )
            
            assert response.id == "batch_test123"
            
        except Exception as e:
            if "'NoneType' object has no attribute 'get'" in str(e):
                pytest.fail("The fix for issue #13995 is not working - still getting NoneType error")
            else:
                # Some other exception, re-raise it
                raise


if __name__ == "__main__":
    # Run the tests
    test_add_key_level_controls_with_none_metadata()
    print("✓ test_add_key_level_controls_with_none_metadata passed")
    
    test_add_key_level_controls_simulates_original_issue()
    print("✓ test_add_key_level_controls_simulates_original_issue passed")
    
    test_batch_create_with_litellm_sdk()
    print("✓ test_batch_create_with_litellm_sdk passed")
    
    print("All tests passed! Issue #13995 fix is working correctly.")