import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.embedding.transformation import GithubCopilotEmbeddingConfig
from litellm.llms.github_copilot.common_utils import GetAPIKeyError

def test_github_copilot_embedding_config_validate_environment():
    """Test the GitHub Copilot embedding configuration environment validation."""
    config = GithubCopilotEmbeddingConfig()
    model = "github_copilot/text-embedding-3-small"

    validated_headers = config.validate_environment(
        headers={},
        model=model,
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="gh-access-token",
    )

    # api_key is used directly — no re-exchange in validate_environment
    assert validated_headers["Authorization"] == "Bearer gh-access-token"
    assert validated_headers["copilot-integration-id"] == "vscode-chat"
    assert validated_headers["editor-version"] == "vscode/1.95.0"
    assert "x-request-id" in validated_headers

    # Test with no api_key → immediate AuthenticationError
    with pytest.raises(AuthenticationError) as excinfo:
        config.validate_environment(
            headers={},
            model=model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
    assert "required" in str(excinfo.value).lower()

    # Auth failures now happen upstream (in _get_openai_compatible_provider_info
    # or main.py), not in validate_environment. No re-exchange test needed.

@patch("litellm.llms.github_copilot.embedding.transformation.Authenticator")
def test_github_copilot_embedding_config_get_complete_url(mock_authenticator_class):
    """Test the GitHub Copilot embedding configuration URL generation."""
    mock_auth_instance = MagicMock()
    mock_authenticator_class.return_value = mock_auth_instance

    config = GithubCopilotEmbeddingConfig()

    # No api_key → always default base
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.githubcopilot.com/embeddings"

    # api_key + authenticator returns custom base
    mock_auth_instance.get_api_base.return_value = "https://api.enterprise.githubcopilot.com"
    url = config.get_complete_url(
        api_base=None,
        api_key="gh-access-token",
        model="github_copilot/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.enterprise.githubcopilot.com/embeddings"

    # Explicit api_base always wins
    url = config.get_complete_url(
        api_base="https://custom.api.com",
        api_key=None,
        model="github_copilot/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.api.com/embeddings"

def test_github_copilot_embedding_config_transform_request():
    """Test the GitHub Copilot embedding request transformation."""
    config = GithubCopilotEmbeddingConfig()
    
    model = "github_copilot/text-embedding-3-small"
    input_data = ["hello world"]
    optional_params = {"user": "test-user"}
    headers = {}

    transformed_request = config.transform_embedding_request(
        model=model,
        input=input_data,
        optional_params=optional_params,
        headers=headers,
    )

    assert transformed_request["model"] == "text-embedding-3-small"
    assert transformed_request["input"] == input_data
    assert transformed_request["user"] == "test-user"

    # Test with string input
    input_str = "hello world"
    transformed_request_str = config.transform_embedding_request(
        model=model,
        input=input_str,
        optional_params=optional_params,
        headers=headers,
    )
    assert transformed_request_str["input"] == [input_str]

def test_github_copilot_embedding_config_transform_request_param_filtering():
    """Test the GitHub Copilot embedding request parameter filtering."""
    config = GithubCopilotEmbeddingConfig()
    
    # Test text-embedding-ada-002 
    model = "github_copilot/text-embedding-ada-002"
    input_data = ["hello"]
    optional_params = {"dimensions": 1536, "user": "test-user"}
    headers = {}

    transformed_request = config.transform_embedding_request(
        model=model,
        input=input_data,
        optional_params=optional_params,
        headers=headers,
    )

    assert transformed_request["model"] == "text-embedding-ada-002"
    assert transformed_request["dimensions"] == 1536
    assert transformed_request["user"] == "test-user"

    # Test text-embedding-3-small
    model = "github_copilot/text-embedding-3-small"
    optional_params = {"dimensions": 512, "user": "test-user"}

    transformed_request = config.transform_embedding_request(
        model=model,
        input=input_data,
        optional_params=optional_params,
        headers=headers,
    )

    assert transformed_request["model"] == "text-embedding-3-small"
    assert transformed_request["dimensions"] == 512
    assert transformed_request["user"] == "test-user"

def test_github_copilot_embedding_config_transform_response():
    """Test the GitHub Copilot embedding response transformation."""
    config = GithubCopilotEmbeddingConfig()
    from litellm.types.utils import EmbeddingResponse
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }
    mock_response.text = "mock response text"

    model_response = EmbeddingResponse()
    logging_obj = MagicMock()

    response = config.transform_embedding_response(
        model="github_copilot/text-embedding-3-small",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=logging_obj,
        api_key="test-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )

    # Verify logging
    logging_obj.post_call.assert_called_once()
    
    assert response is not None
    assert len(response.data) == 1
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert response.model == "text-embedding-3-small"
