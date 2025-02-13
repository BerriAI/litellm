import pytest
from unittest.mock import Mock, patch
import mimetypes
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.prompt_templates.factory import BedrockImageProcessor

# Mock the configurations and classes we need
class MockAmazonConfig:
    def get_supported_image_types(self):
        return ['png', 'jpeg', 'gif', 'webp']
    
    def get_supported_document_types(self):
        return ['pdf', 'docx']


def test_error_mapping_ValueError_to_BadRequest():
    """Test that ValueError gets mapped to BadRequestError with correct message"""
    try:
        # Simulate the ValueError being raised
        raise ValueError(
            "Client error '400 Bad Request': PDF files are not supported. Supported formats are: ['docx']"
        )
    except ValueError as e:
        with pytest.raises(BadRequestError) as exc_info:
            exception_type(
                model="model-name",
                original_exception=e,
                custom_llm_provider="bedrock"
            )
        
        error = exc_info.value
        assert error.status_code == 400
        assert "PDF files are not supported" in str(error)

def test_error_mapping_413_error():
    """Test that 413 status code gets mapped correctly"""
    mock_exception = Mock()
    mock_exception.status_code = 413
    mock_exception.message = "File too large"
    
    with pytest.raises(BadRequestError) as exc_info:
        exception_type(
            model="model-name",
            original_exception=mock_exception,
            custom_llm_provider="replicate"
        )
    
    error = exc_info.value
    assert error.status_code == 400
    assert "ReplicateException" in str(error)
    assert "File too large" in str(error)

def test_validation_edge_cases():
    """Test edge cases in format validation"""
    with patch('litellm.AmazonConverseConfig', return_value=MockAmazonConfig()):
        # Test empty mime type
        with pytest.raises(ValueError) as exc_info:
            BedrockImageProcessor._validate_format("", "")
        assert "Client error '400 Bad Request'" in str(exc_info.value)
        
        # Test invalid mime type
        with pytest.raises(ValueError) as exc_info:
            BedrockImageProcessor._validate_format("invalid/type", "format")
        assert "Client error '400 Bad Request'" in str(exc_info.value)
