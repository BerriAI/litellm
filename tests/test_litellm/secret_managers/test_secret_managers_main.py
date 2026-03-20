import logging
import os
from unittest.mock import Mock, patch

import pytest

from litellm.secret_managers.main import get_secret

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Mock HTTPHandler and oidc_cache
class MockHTTPHandler:
    def __init__(self, timeout):
        self.timeout = timeout
        self.status_code = 200
        self.text = "mocked_token"
        self.json_data = {"value": "mocked_token"}

    def get(self, url, params=None, headers=None):
        # Store params for audience verification
        self.last_params = params
        logger.debug(
            f"MockHTTPHandler.get called with url={url}, params={params}, headers={headers}"
        )
        mock_response = Mock()
        mock_response.status_code = self.status_code
        mock_response.text = self.text
        mock_response.json.return_value = self.json_data
        return mock_response


@pytest.fixture
def mock_oidc_cache():
    cache = Mock()
    cache.get_cache.return_value = None
    cache.set_cache = Mock()
    return cache


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {}, clear=True):
        yield os.environ


def test_oidc_google_success():
    """Test Google OIDC token fetch with mocked handler (no real network calls)."""
    secret_name = "oidc/google/[invalid url, do not cite]"
    mock_handler = MockHTTPHandler(timeout=600.0)
    mock_get_http_handler = Mock(return_value=mock_handler)
    mock_oidc_cache = Mock()
    mock_oidc_cache.get_cache.return_value = None

    with patch("litellm.secret_managers.main.oidc_cache", mock_oidc_cache):
        with patch(
            "litellm.secret_managers.main._get_oidc_http_handler",
            mock_get_http_handler,
        ):
            with patch(
                "litellm.secret_managers.main.HTTPHandler",
                side_effect=lambda timeout=None: mock_handler,
            ):
                result = get_secret(secret_name)

    assert result == "mocked_token"
    assert mock_handler.last_params == {"audience": "[invalid url, do not cite]"}
    mock_oidc_cache.set_cache.assert_called_once_with(
        key=secret_name, value="mocked_token", ttl=3540
    )


def test_oidc_google_cached():
    """Test Google OIDC uses cache and does not call HTTP (no real network calls)."""
    secret_name = "oidc/google/[invalid url, do not cite]"
    mock_get_http_handler = Mock()
    mock_oidc_cache = Mock()
    mock_oidc_cache.get_cache.return_value = "cached_token"

    with patch("litellm.secret_managers.main.oidc_cache", mock_oidc_cache):
        with patch(
            "litellm.secret_managers.main._get_oidc_http_handler",
            mock_get_http_handler,
        ):
            with patch(
                "litellm.secret_managers.main.HTTPHandler",
                Mock(side_effect=AssertionError("HTTPHandler should not be used")),
            ):
                result = get_secret(secret_name)

    assert result == "cached_token", f"Expected cached token, got {result}"
    mock_oidc_cache.get_cache.assert_called_with(key=secret_name)
    mock_get_http_handler.assert_not_called()


def test_oidc_google_failure():
    """Test Google OIDC raises when provider returns error (no real network calls)."""
    secret_name = "oidc/google/https://example.com/api"
    mock_handler = MockHTTPHandler(timeout=600.0)
    mock_handler.status_code = 400
    mock_get_http_handler = Mock(return_value=mock_handler)
    mock_oidc_cache = Mock()
    mock_oidc_cache.get_cache.return_value = None

    with patch("litellm.secret_managers.main.oidc_cache", mock_oidc_cache):
        with patch(
            "litellm.secret_managers.main._get_oidc_http_handler",
            mock_get_http_handler,
        ):
            with patch(
                "litellm.secret_managers.main.HTTPHandler",
                side_effect=lambda timeout=None: mock_handler,
            ):
                with pytest.raises(ValueError, match="Google OIDC provider failed"):
                    get_secret(secret_name)


def test_oidc_circleci_success(monkeypatch):
    monkeypatch.setenv("CIRCLE_OIDC_TOKEN", "circleci_token")

    secret_name = "oidc/circleci/test-audience"
    result = get_secret(secret_name)

    assert result == "circleci_token"


def test_oidc_circleci_failure(monkeypatch):
    monkeypatch.delenv("CIRCLE_OIDC_TOKEN", raising=False)
    secret_name = "oidc/circleci/test-audience"

    with pytest.raises(ValueError, match="CIRCLE_OIDC_TOKEN not found in environment"):
        get_secret(secret_name)


@patch("litellm.secret_managers.main.oidc_cache")
@patch("litellm.secret_managers.main._get_oidc_http_handler")
def test_oidc_github_success(mock_get_http_handler, mock_oidc_cache, mock_env):
    mock_env["ACTIONS_ID_TOKEN_REQUEST_URL"] = "https://github.com/token"
    mock_env["ACTIONS_ID_TOKEN_REQUEST_TOKEN"] = "github_token"
    mock_oidc_cache.get_cache.return_value = None
    mock_handler = MockHTTPHandler(timeout=600.0)
    mock_get_http_handler.return_value = mock_handler

    secret_name = "oidc/github/github-audience"
    result = get_secret(secret_name)

    assert result == "mocked_token", f"Expected token 'mocked_token', got {result}"
    assert mock_handler.last_params == {"audience": "github-audience"}
    logger.debug(f"set_cache call args: {mock_oidc_cache.set_cache.call_args}")
    mock_oidc_cache.set_cache.assert_called_once()
    mock_oidc_cache.set_cache.assert_called_with(
        key=secret_name, value="mocked_token", ttl=295
    )


def test_oidc_github_missing_env():
    secret_name = "oidc/github/github-audience"

    with pytest.raises(
        ValueError,
        match="ACTIONS_ID_TOKEN_REQUEST_URL or ACTIONS_ID_TOKEN_REQUEST_TOKEN not found in environment",
    ):
        get_secret(secret_name)


def test_oidc_azure_file_success(mock_env, tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("azure_token")
    mock_env["AZURE_FEDERATED_TOKEN_FILE"] = str(token_file)

    secret_name = "oidc/azure/azure-audience"
    result = get_secret(secret_name)    

    assert result == "azure_token"


@patch("litellm.secret_managers.main.get_azure_ad_token_provider")
def test_oidc_azure_ad_token_success(mock_get_azure_ad_token_provider, monkeypatch):
    # Force-unset so we always hit the Azure AD token provider path (CI may set AZURE_FEDERATED_TOKEN_FILE)
    monkeypatch.delenv("AZURE_FEDERATED_TOKEN_FILE", raising=False)

    # Mock the token provider function that gets returned and called
    mock_token_provider = Mock(return_value="azure_ad_token")
    mock_get_azure_ad_token_provider.return_value = mock_token_provider

    # Also mock the Azure Identity SDK to prevent any real Azure calls
    with patch("azure.identity.get_bearer_token_provider") as mock_bearer:
        mock_bearer.return_value = mock_token_provider

        secret_name = "oidc/azure/api://azure-audience"
        result = get_secret(secret_name)

        assert result == "azure_ad_token"
        mock_get_azure_ad_token_provider.assert_called_once_with(
            azure_scope="api://azure-audience"
        )
        mock_token_provider.assert_called_once_with()


def test_oidc_file_success(tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("file_token")

    secret_name = f"oidc/file/{token_file}"
    result = get_secret(secret_name)

    assert result == "file_token"


def test_oidc_env_success(mock_env):
    mock_env["CUSTOM_TOKEN"] = "env_token"

    secret_name = "oidc/env/CUSTOM_TOKEN"
    result = get_secret(secret_name)

    assert result == "env_token"


def test_oidc_env_path_success(mock_env, tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("env_path_token")
    mock_env["TOKEN_PATH"] = str(token_file)

    secret_name = "oidc/env_path/TOKEN_PATH"
    result = get_secret(secret_name)

    assert result == "env_path_token"


def test_unsupported_oidc_provider():
    secret_name = "oidc/unsupported/unsupported-audience"

    with pytest.raises(ValueError, match="Unsupported OIDC provider"):
        get_secret(secret_name)
