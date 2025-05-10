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
    print("\n--- Running test_error_mapping_ValueError_to_BadRequest ---")
    try:
        # Simulate the ValueError being raised
        error_message = "Client error '400 Bad Request': PDF files are not supported. Supported formats are: ['docx']"
        print(f"Raising ValueError with message: {error_message}")
        
        raise ValueError(error_message)
    except ValueError as e:
        print(f"Caught ValueError: {e}")
        
        with pytest.raises(BadRequestError) as exc_info:
            print("Calling exception_type...")
            exception_type(
                model="model-name",
                original_exception=e,
                custom_llm_provider="bedrock"
            )
        
        error = exc_info.value
        print(f"Caught BadRequestError: {error}")
        print(f"Error status code: {error.status_code}")
        print(f"Error message: {str(error)}")
        
        assert error.status_code == 400
        assert "PDF files are not supported" in str(error)
        print("Assertions passed successfully!")

def test_error_mapping_413_error():
    """Test that 413 status code gets mapped correctly"""
    print("\n--- Running test_error_mapping_413_error ---")
    mock_exception = Mock()
    mock_exception.status_code = 413
    mock_exception.message = "File too large"
    
    print(f"Mock exception status code: {mock_exception.status_code}")
    print(f"Mock exception message: {mock_exception.message}")
    
    with pytest.raises(BadRequestError) as exc_info:
        exception_type(
            model="model-name",
            original_exception=mock_exception,
            custom_llm_provider="replicate"
        )
    
    error = exc_info.value
    print(f"Caught BadRequestError: {error}")
    print(f"Error status code: {error.status_code}")
    print(f"Error message: {str(error)}")
    
    assert error.status_code == 400
    assert "ReplicateException" in str(error)
    assert "File too large" in str(error)
    print("Assertions passed successfully!")

def test_validation_edge_cases():
    """Test edge cases in format validation"""
    print("\n--- Running test_validation_edge_cases ---")
    with patch('litellm.AmazonConverseConfig', return_value=MockAmazonConfig()):
        # Test empty mime type
        print("Testing empty mime type")
        with pytest.raises(ValueError) as exc_info:
            BedrockImageProcessor._validate_format("", "")
        print(f"Caught ValueError: {exc_info.value}")
        assert "Client error '400 Bad Request'" in str(exc_info.value)
        
        # Test invalid mime type
        print("Testing invalid mime type")
        with pytest.raises(ValueError) as exc_info:
            BedrockImageProcessor._validate_format("invalid/type", "format")
        print(f"Caught ValueError: {exc_info.value}")
        assert "Client error '400 Bad Request'" in str(exc_info.value)
        print("Assertions passed successfully!")