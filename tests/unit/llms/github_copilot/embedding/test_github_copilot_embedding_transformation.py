from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.common_utils import DEFAULT_GITHUB_COPILOT_API_BASE, GetAPIKeyError
from litellm.llms.github_copilot.embedding.transformation import (
    GithubCopilotEmbeddingConfig,
)
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
    assert validated_headers["editor-version"] == "vscode/1.115.0"
    assert "x-request-id" not in validated_headers

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


@pytest.mark.parametrize(
    ("deployment_base", "environment_base", "expected_base"),
    [
        (None, None, DEFAULT_GITHUB_COPILOT_API_BASE),
        (None, "https://copilot.example.test/", "https://copilot.example.test"),
        (None, "http://localhost:8080/", "http://localhost:8080"),
        ("https://deployment.example.test/", "https://env.example.test", "https://deployment.example.test"),
        ("http://localhost:9090/", "https://env.example.test", "http://localhost:9090"),
        ("not-a-url", "https://env.example.test", "https://env.example.test"),
        (None, "not-a-url", DEFAULT_GITHUB_COPILOT_API_BASE),
    ],
)
def test_github_copilot_embedding_config_get_complete_url(
    monkeypatch: pytest.MonkeyPatch,
    deployment_base: str | None,
    environment_base: str | None,
    expected_base: str,
) -> None:
    if environment_base is None:
        monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    else:
        monkeypatch.setenv("GITHUB_COPILOT_API_BASE", environment_base)
    config: Final = GithubCopilotEmbeddingConfig()
    config.authenticator = MagicMock()

    url: Final = config.get_complete_url(
        api_base=deployment_base,
        api_key=None,
        model="github_copilot/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )

    assert url == f"{expected_base}/embeddings"
    assert config.authenticator.mock_calls == []


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

    session = GithubCopilotUserSession(token="user-copilot-token")
    headers = config.validate_environment(
        headers={},
        model="github_copilot/text-embedding-3-small",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"
    config.authenticator.get_api_key.assert_not_called()


@pytest.mark.parametrize("authorization_header", ["Authorization", "authorization", "AUTHORIZATION"])
def test_per_user_session_token_wins_over_caller_authorization(authorization_header: str):
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    config = GithubCopilotEmbeddingConfig()
    config.authenticator = MagicMock()
    session = GithubCopilotUserSession(token="user-copilot-token")
    headers = config.validate_environment(
        headers={authorization_header: "Bearer caller-token", "x-custom": "preserved"},
        model="github_copilot/text-embedding-3-small",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"
    assert {key: value for key, value in headers.items() if key.lower() == "authorization"} == {
        "Authorization": "Bearer user-copilot-token"
    }
    assert headers["x-custom"] == "preserved"
    assert config.authenticator.mock_calls == []


@pytest.mark.parametrize(
    ("deployment_base", "expected_base"),
    [(None, "http://localhost:8080"), ("https://deployment.example.test/", "https://deployment.example.test")],
)
def test_get_complete_url_uses_configured_base_with_per_user_session(
    monkeypatch: pytest.MonkeyPatch, deployment_base: str | None, expected_base: str
) -> None:
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    monkeypatch.setenv("GITHUB_COPILOT_API_BASE", "http://localhost:8080/")
    config: Final = GithubCopilotEmbeddingConfig()
    config.authenticator = MagicMock()
    session: Final = GithubCopilotUserSession(token="user-oauth-token")
    url: Final = config.get_complete_url(
        api_base=deployment_base,
        api_key=None,
        model="github_copilot/text-embedding-3-small",
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )

    assert url == f"{expected_base}/embeddings"
    assert config.authenticator.mock_calls == []
