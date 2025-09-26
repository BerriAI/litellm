"""
Test for extra_body custom_llm_provider extraction fix
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set up environment
os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"
os.environ["OPENAI_API_KEY"] = "test-openai-key"

import litellm
from litellm.batches.main import retrieve_batch, cancel_batch, list_batches, create_batch


def test_retrieve_batch_extracts_azure_provider_from_extra_body():
    """Test that retrieve_batch correctly extracts azure provider from extra_body"""
    
    with patch('litellm.batches.main.azure_batches_instance') as mock_azure_instance:
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch_test123"
        mock_azure_instance.retrieve_batch.return_value = mock_batch_response
        
        # Call retrieve_batch with azure provider in extra_body (this was broken before the fix)
        result = retrieve_batch(
            batch_id="batch_test123",
            extra_body={"custom_llm_provider": "azure"}
        )
        
        # Verify that the azure instance was called (not openai instance)
        assert mock_azure_instance.retrieve_batch.called
        assert result.id == "batch_test123"


def test_cancel_batch_extracts_azure_provider_from_extra_body():
    """Test that cancel_batch correctly extracts azure provider from extra_body"""
    
    with patch('litellm.batches.main.azure_batches_instance') as mock_azure_instance:
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch_test123"
        mock_azure_instance.cancel_batch.return_value = mock_batch_response
        
        result = cancel_batch(
            batch_id="batch_test123",
            extra_body={"custom_llm_provider": "azure"}
        )
        
        assert mock_azure_instance.cancel_batch.called


def test_list_batches_extracts_azure_provider_from_extra_body():
    """Test that list_batches correctly extracts azure provider from extra_body"""
    
    with patch('litellm.batches.main.azure_batches_instance') as mock_azure_instance:
        mock_batches_response = MagicMock()
        mock_azure_instance.list_batches.return_value = mock_batches_response
        
        result = list_batches(
            extra_body={"custom_llm_provider": "azure"}
        )
        
        assert mock_azure_instance.list_batches.called


def test_create_batch_extracts_azure_provider_from_extra_body():
    """Test that create_batch correctly extracts azure provider from extra_body"""
    
    with patch('litellm.batches.main.azure_batches_instance') as mock_azure_instance:
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch_test123"
        mock_azure_instance.create_batch.return_value = mock_batch_response
        
        result = create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="file_test123",
            extra_body={"custom_llm_provider": "azure"}
        )
        
        assert mock_azure_instance.create_batch.called


def test_retrieve_batch_defaults_to_openai_without_extra_body():
    """Test that retrieve_batch defaults to openai when no provider specified"""
    
    with patch('litellm.batches.main.openai_batches_instance') as mock_openai_instance:
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch_test123"
        mock_openai_instance.retrieve_batch.return_value = mock_batch_response
        
        # Call retrieve_batch without specifying provider (should default to openai)
        result = retrieve_batch(
            batch_id="batch_test123"
        )
        
        # Verify that the openai instance was called
        assert mock_openai_instance.retrieve_batch.called
        assert result.id == "batch_test123"


def test_retrieve_batch_respects_explicit_provider_parameter():
    """Test that retrieve_batch respects explicit custom_llm_provider parameter"""
    
    with patch('litellm.batches.main.azure_batches_instance') as mock_azure_instance:
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch_test123"
        mock_azure_instance.retrieve_batch.return_value = mock_batch_response
        
        # Call retrieve_batch with explicit azure provider parameter
        result = retrieve_batch(
            batch_id="batch_test123",
            custom_llm_provider="azure"
        )
        
        # Verify that the azure instance was called
        assert mock_azure_instance.retrieve_batch.called
        assert result.id == "batch_test123"