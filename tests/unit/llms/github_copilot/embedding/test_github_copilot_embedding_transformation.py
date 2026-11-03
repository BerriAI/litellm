from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError


from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.embedding.transformation import (
    GithubCopilotEmbeddingConfig,
)
from litellm.llms.github_copilot.common_utils import GetAPIKeyError
from litellm.types.utils import EmbeddingResponse


def test_github_copilot_embedding_config_validate_environment():
    """Test the GitHub Copilot embedding configuration environment validation."""
    config = GithubCopilotEmbeddingConfig()

    # Mock the authenticator
    mock_api_key = "gh.test-key-123456789"
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = mock_api_key

    # Test with valid API key
    headers = {}
    model = "github_copilot/text-embedding-3-small"

    validated_headers = config.validate_environment(
        headers=headers,
        model=model,
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
    )

    assert validated_headers["Authorization"] == f"Bearer {mock_api_key}"
    assert validated_headers["copilot-integration-id"] == "vscode-chat"
    assert validated_headers["editor-version"] == "vscode/1.95.0"
    assert "x-request-id" in validated_headers

    # Test with authentication failure
    config.authenticator.get_api_key.side_effect = GetAPIKeyError(
        message="Failed to get API key",
        status_code=401,
    )

    with pytest.raises(AuthenticationError) as excinfo:
        config.validate_environment(
            headers={},
            model=model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )

    assert "Failed to get API key" in str(excinfo.value)


def test_github_copilot_embedding_config_get_complete_url():
    """Test the GitHub Copilot embedding configuration URL generation."""
    config = GithubCopilotEmbeddingConfig()
    config.authenticator = MagicMock()

    # Test with default API base
    config.authenticator.get_api_base.return_value = None
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.githubcopilot.com/embeddings"

    # Test with custom API base from authenticator
    config.authenticator.get_api_base.return_value = "https://api.enterprise.githubcopilot.com"
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.enterprise.githubcopilot.com/embeddings"

    # Test with custom API base from params
    config.authenticator.get_api_base.return_value = None
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
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
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


def _transform(raw_response: httpx.Response) -> EmbeddingResponse:
    return GithubCopilotEmbeddingConfig().transform_embedding_response(
        model="github_copilot/text-embedding-3-small",
        raw_response=raw_response,
        model_response=EmbeddingResponse(),
        logging_obj=MagicMock(),
        api_key="test-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )


def test_transform_embedding_response_keeps_the_openai_envelope():
    response = _transform(
        httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "unknown": "ignored",
            },
        )
    )

    assert response.model_dump() == {
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "object": "list",
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 5,
            "total_tokens": 5,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }


@pytest.mark.parametrize("body", [b"null", b"[]", b"7", b'"leaked payload text"', b'[{"data": "leaked payload text"}]'])
def test_transform_embedding_response_rejects_a_body_that_is_not_an_object(body: bytes):
    with pytest.raises(ValidationError) as exc_info:
        _transform(httpx.Response(200, content=body))

    assert "leaked payload text" not in str(exc_info.value)


def test_transform_embedding_response_object_without_data_is_an_invalid_response_object():
    with pytest.raises(Exception, match="Invalid response object"):
        _transform(httpx.Response(200, json={"model": "text-embedding-3-small"}))


def test_validate_environment_uses_per_user_session_and_skips_authenticator():
    """With a per-user session the Authorization header carries the caller's Copilot token
    and the shared Authenticator is never consulted."""
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    config = GithubCopilotEmbeddingConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = AssertionError("shared authenticator must not run")

    session = GithubCopilotUserSession(token="user-copilot-token", api_base="https://api.githubcopilot.com")
    headers = config.validate_environment(
        headers={},
        model="github_copilot/text-embedding-3-small",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"
    config.authenticator.get_api_key.assert_not_called()


def test_per_user_session_token_wins_over_caller_authorization():
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    config = GithubCopilotEmbeddingConfig()
    config.authenticator = MagicMock()
    session = GithubCopilotUserSession(token="user-copilot-token", api_base="https://api.githubcopilot.com")
    headers = config.validate_environment(
        headers={"Authorization": "Bearer caller-token"},
        model="github_copilot/text-embedding-3-small",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"


def test_get_complete_url_prefers_per_user_session_api_base():
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    config = GithubCopilotEmbeddingConfig()
    config.authenticator = MagicMock()
    session = GithubCopilotUserSession(token="t", api_base="https://tenant.githubcopilot.com")
    url = config.get_complete_url(
        api_base="https://attacker.example",
        api_key=None,
        model="m",
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )
    assert url == "https://tenant.githubcopilot.com/embeddings"
