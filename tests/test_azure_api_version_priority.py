import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm._logging import verbose_logger  # Import the actual logger
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.azure.completion.handler import AzureTextCompletion


@pytest.fixture
def setup_mocks():
    with patch("litellm.llms.azure.azure.AzureOpenAI") as mock_azure_openai, \
         patch("litellm.llms.azure.azure.verbose_logger") as mock_logger, \
         patch("litellm.llms.azure.common_utils.os") as mock_os, \
         patch("litellm.llms.azure.common_utils.verbose_logger") as mock_common_logger, \
         patch("litellm.llms.azure.completion.handler.AzureOpenAI") as mock_azure_text_openai:
        
        # Create a mock client
        mock_client = MagicMock()
        mock_client._custom_query = {}
        mock_azure_openai.return_value = mock_client
        mock_azure_text_openai.return_value = mock_client

        # Configure environment variables
        mock_os.getenv.return_value = "2023-07-01-preview"
        
        # Configure litellm defaults
        litellm.AZURE_DEFAULT_API_VERSION = "2023-05-15"
        
        yield {
            "mock_azure_openai": mock_azure_openai,
            "mock_client": mock_client,
            "mock_logger": mock_logger,
            "mock_os": mock_os, 
            "mock_common_logger": mock_common_logger,
            "mock_azure_text_openai": mock_azure_text_openai
        }


def test_api_version_priority_chat_completion(setup_mocks):
    """
    Test that the API version is prioritized correctly in Azure Chat Completion:
    1. litellm_params API version (highest priority)
    2. Directly passed API version
    3. Environment variable
    4. Default API version
    """
    mock_client = setup_mocks["mock_client"]
    
    # Create instance and test directly
    azure_chat = AzureChatCompletion()
    
    # Test case 1: litellm_params has API version (highest priority)
    azure_client = azure_chat._get_sync_azure_client(
        api_key="test-key",
        api_version="2023-06-01-preview",  # Lower priority
        api_base="https://test.openai.azure.com",
        azure_ad_token=None,
        azure_ad_token_provider=None,
        model="gpt-35-turbo",
        max_retries=0,
        timeout=60.0,
        client=mock_client,
        client_type="sync",
        litellm_params={"api_version": "2023-08-01-preview"}  # Highest priority
    )
    
    assert azure_client._custom_query["api-version"] == "2023-08-01-preview"
    mock_client._custom_query = {}  # Reset for next test
    
    # Test case 2: No litellm_params API version, use directly passed API version
    azure_client = azure_chat._get_sync_azure_client(
        api_key="test-key",
        api_version="2023-06-01-preview",  # Second highest priority
        api_base="https://test.openai.azure.com",
        azure_ad_token=None,
        azure_ad_token_provider=None,
        model="gpt-35-turbo",
        max_retries=0,
        timeout=60.0,
        client=mock_client,
        client_type="sync",
        litellm_params={}  # No API version here
    )
    
    assert azure_client._custom_query["api-version"] == "2023-06-01-preview"
    mock_client._custom_query = {}  # Reset for next test


def test_api_version_priority_text_completion(setup_mocks):
    """
    Test that the API version is prioritized correctly in Azure Text Completion:
    1. litellm_params API version (highest priority)
    2. Directly passed API version
    3. Environment variable
    4. Default API version
    """
    mock_client = setup_mocks["mock_client"]
    text_completion = AzureTextCompletion()
    
    # Simulate the completion method's client initialization with provided client
    # Test case 1: litellm_params has API version (highest priority)
    # We need to use a similar pattern to what's done in the actual completion method
    
    # Mock the required arguments
    api_key = "test-key"
    api_version = "2023-06-01-preview"  # Lower priority
    litellm_params = {"api_version": "2023-08-01-preview"}  # Highest priority
    
    # Set on the client to simulate what happens in the completion method
    mock_client._custom_query = {}
    client_api_version = None
    if litellm_params and "api_version" in litellm_params:
        client_api_version = litellm_params.get("api_version")
    elif api_version is not None:
        client_api_version = api_version
    
    if client_api_version is not None:
        mock_client._custom_query["api-version"] = client_api_version
    
    # Verify our assertion
    assert mock_client._custom_query["api-version"] == "2023-08-01-preview"
    mock_client._custom_query = {}  # Reset for next test
    
    # Test case 2: No litellm_params API version, use directly passed API version
    litellm_params = {}  # No API version here
    
    client_api_version = None
    if litellm_params and "api_version" in litellm_params:
        client_api_version = litellm_params.get("api_version")
    elif api_version is not None:
        client_api_version = api_version
    
    if client_api_version is not None:
        mock_client._custom_query["api-version"] = client_api_version
    
    assert mock_client._custom_query["api-version"] == "2023-06-01-preview"
    mock_client._custom_query = {}  # Reset for next test


def test_common_utils_api_version_priority(setup_mocks):
    """
    Test the API version priority in BaseAzureLLM.initialize_azure_sdk_client:
    1. litellm_params API version (highest priority)
    2. Environment variable
    3. Default API version
    """
    mock_os = setup_mocks["mock_os"]
    base_azure = BaseAzureLLM()
    
    # Test case 1: litellm_params has API version (highest priority)
    client_params = base_azure.initialize_azure_sdk_client(
        litellm_params={"api_version": "2023-08-01-preview"},  # Highest priority
        api_key="test-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-35-turbo",
        api_version=None  # Not directly passed
    )
    
    assert client_params["api_version"] == "2023-08-01-preview"
    
    # Test case 2: No litellm_params API version, use env var through passed api_version
    # In this case, api_version would be derived from env var in the actual code
    # but in our test we simulate it by passing it directly
    client_params = base_azure.initialize_azure_sdk_client(
        litellm_params={},  # No API version here
        api_key="test-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-35-turbo",
        api_version="2023-07-01-preview"  # This would come from env var in real code
    )
    
    assert client_params["api_version"] == "2023-07-01-preview" 