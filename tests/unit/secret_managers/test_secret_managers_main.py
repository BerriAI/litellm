import base64
import json
import logging
import os
import time
from unittest.mock import Mock, patch

import pytest

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.secret_managers.main import (
    get_secret,
    normalize_nonempty_secret_str,
    secret_manager_would_be_consulted,
)
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem

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


def _jwt_with_exp(exp: int) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


def test_oidc_google_cache_ttl_capped_by_token_exp():
    """A token the metadata server returns near its expiry must not be cached past
    its exp claim; the cached-entry TTL is exp - now - 60s, not the 59m default."""
    secret_name = "oidc/google/https://example.com/api"
    mock_handler = MockHTTPHandler(timeout=600.0)
    mock_handler.text = _jwt_with_exp(int(time.time()) + 300)
    mock_get_http_handler = Mock(return_value=mock_handler)
    mock_oidc_cache = Mock()
    mock_oidc_cache.get_cache.return_value = None

    with patch("litellm.secret_managers.main.oidc_cache", mock_oidc_cache):
        with patch(
            "litellm.secret_managers.main._get_oidc_http_handler",
            mock_get_http_handler,
        ):
            result = get_secret(secret_name)

    assert result == mock_handler.text
    mock_oidc_cache.set_cache.assert_called_once()
    ttl = mock_oidc_cache.set_cache.call_args.kwargs["ttl"]
    assert 0 < ttl <= 240


def test_oidc_google_expired_token_not_cached():
    """An already-expired token is returned (STS gives the authoritative error) but
    never cached, so the next call fetches a fresh token instead of replaying it."""
    secret_name = "oidc/google/https://example.com/api"
    mock_handler = MockHTTPHandler(timeout=600.0)
    mock_handler.text = _jwt_with_exp(int(time.time()) - 10)
    mock_get_http_handler = Mock(return_value=mock_handler)
    mock_oidc_cache = Mock()
    mock_oidc_cache.get_cache.return_value = None

    with patch("litellm.secret_managers.main.oidc_cache", mock_oidc_cache):
        with patch(
            "litellm.secret_managers.main._get_oidc_http_handler",
            mock_get_http_handler,
        ):
            result = get_secret(secret_name)

    assert result == mock_handler.text
    mock_oidc_cache.set_cache.assert_not_called()


def test_oidc_google_long_lived_token_still_capped_at_default_ttl():
    """A token expiring far in the future must not extend the cache past the
    59m policy ceiling; exp only ever shortens the TTL."""
    secret_name = "oidc/google/https://example.com/api"
    mock_handler = MockHTTPHandler(timeout=600.0)
    mock_handler.text = _jwt_with_exp(int(time.time()) + 7200)
    mock_get_http_handler = Mock(return_value=mock_handler)
    mock_oidc_cache = Mock()
    mock_oidc_cache.get_cache.return_value = None

    with patch("litellm.secret_managers.main.oidc_cache", mock_oidc_cache):
        with patch(
            "litellm.secret_managers.main._get_oidc_http_handler",
            mock_get_http_handler,
        ):
            result = get_secret(secret_name)

    assert result == mock_handler.text
    mock_oidc_cache.set_cache.assert_called_once_with(
        key=secret_name, value=mock_handler.text, ttl=3540
    )


def test_oidc_google_non_jwt_token_keeps_default_ttl():
    """A token without a readable exp claim falls back to the 59m default TTL."""
    secret_name = "oidc/google/https://example.com/api"
    mock_handler = MockHTTPHandler(timeout=600.0)
    mock_get_http_handler = Mock(return_value=mock_handler)
    mock_oidc_cache = Mock()
    mock_oidc_cache.get_cache.return_value = None

    with patch("litellm.secret_managers.main.oidc_cache", mock_oidc_cache):
        with patch(
            "litellm.secret_managers.main._get_oidc_http_handler",
            mock_get_http_handler,
        ):
            result = get_secret(secret_name)

    assert result == "mocked_token"
    mock_oidc_cache.set_cache.assert_called_once_with(
        key=secret_name, value="mocked_token", ttl=3540
    )


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


def test_oidc_file_success(tmp_path, monkeypatch):
    token_file = tmp_path / "token.txt"
    token_file.write_text("file_token")
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))

    secret_name = f"oidc/file/{token_file}"
    result = get_secret(secret_name)

    assert result == "file_token"


def test_oidc_file_rejects_path_outside_allowlist(tmp_path, monkeypatch):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("should_not_read")
    # Allowlist a different directory.
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(allowed_dir))

    with pytest.raises(ValueError, match="outside the allowed credential directories"):
        get_secret(f"oidc/file/{outside_file}")


def test_oidc_file_rejects_relative_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    with pytest.raises(ValueError, match="must be absolute"):
        get_secret("oidc/file/relative/path/token")


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   \t\n", None),
        ("abc", "abc"),
        ("  xyz  ", "xyz"),
    ],
)
def test_normalize_nonempty_secret_str(raw, expected):
    assert normalize_nonempty_secret_str(raw) == expected


class _SpySecretManager(CustomSecretManager):
    """Records every name the manager is actually asked for."""

    def __init__(self, asked):
        self.asked = asked

    def sync_read_secret(self, secret_name, optional_params=None, timeout=None, **kwargs):
        self.asked.append(secret_name)
        return "a-value"

    async def async_read_secret(self, secret_name, optional_params=None, timeout=None, **kwargs):
        self.asked.append(secret_name)
        return "a-value"


@pytest.mark.parametrize(
    ("access_mode", "hosted_keys", "secret_name", "expected"),
    [
        ("read_only", None, "ANY_NAME", True),
        ("read_only", ["ALLOWED"], "ALLOWED", True),
        ("read_only", ["ALLOWED"], "NOT_ALLOWED", False),
        ("read_and_write", ["ALLOWED"], "ALLOWED", True),
        ("write_only", None, "ANY_NAME", False),
        ("write_only", ["ALLOWED"], "ALLOWED", False),
    ],
)
def test_secret_manager_would_be_consulted_matches_get_secret(
    monkeypatch, access_mode, hosted_keys, secret_name, expected
):
    """The predicate must agree with what get_secret actually does, not with a reading of it.

    Callers use it to tell "the manager does not have this key" apart from "the manager was
    never asked", so a predicate that drifts from get_secret's gating makes them state a
    lookup that never happened.
    """
    asked = []
    monkeypatch.setattr(litellm, "secret_manager_client", _SpySecretManager(asked))
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(access_mode=access_mode, hosted_keys=hosted_keys),
    )
    monkeypatch.delenv(secret_name, raising=False)

    predicted = secret_manager_would_be_consulted(f"os.environ/{secret_name}")
    get_secret(f"os.environ/{secret_name}")

    assert {"predicted": predicted, "actually_consulted": bool(asked)} == {
        "predicted": expected,
        "actually_consulted": expected,
    }


def test_secret_manager_would_be_consulted_is_false_without_a_client(monkeypatch):
    monkeypatch.setattr(litellm, "secret_manager_client", None)

    assert secret_manager_would_be_consulted("os.environ/ANY_NAME") is False
