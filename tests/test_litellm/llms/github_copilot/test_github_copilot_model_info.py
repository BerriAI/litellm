from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import GetAPIKeyError
from litellm.llms.github_copilot.model_info import GithubCopilotModelInfo
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

API_BASE = "https://api.githubcopilot.com"

# Shape verified against a live Copilot /models response on 2026-09-24: top-level keys
# ["data", "object"], every entry carrying id, capabilities, model_picker_enabled, policy.
# model_picker_enabled is deliberately not a filter: Copilot clears it on callable models
# such as gpt-4o and text-embedding-3-small.
MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "gpt-4o", "capabilities": {"type": "chat"}, "model_picker_enabled": True},
        {"id": "claude-opus-5.5", "capabilities": {"type": "chat"}, "model_picker_enabled": True},
        {"id": "text-embedding-3-small", "capabilities": {"type": "embeddings"}, "model_picker_enabled": False},
    ],
}


class StubAuthenticator(Authenticator):
    """Authenticator with its token files pointed at a tmp path, injected instead of patched."""

    def __init__(self, token_file: str, api_key: str | None = "gh.test-key-123456789") -> None:
        self.api_key_file = token_file
        self.access_token_file = token_file
        self._api_key = api_key

    def get_api_key(self) -> str:
        if self._api_key is None:
            raise GetAPIKeyError("no cached credential")
        return self._api_key


@pytest.fixture
def cached_token(tmp_path):
    token = tmp_path / "api-key.json"
    token.write_text('{"token": "gh.test-key-123456789"}')
    return str(token)


@pytest.fixture
def missing_token(tmp_path):
    return str(tmp_path / "absent.json")


def _response(payload, status_code=200):
    request = httpx.Request("GET", f"{API_BASE}/models")
    return httpx.Response(status_code=status_code, json=payload, request=request)


def test_get_models_returns_every_prefixed_id(cached_token):
    model_info = GithubCopilotModelInfo(StubAuthenticator(cached_token))

    with patch.object(litellm.module_level_client, "get", return_value=_response(MODELS_PAYLOAD)) as mock_get:
        models = model_info.get_models(api_base=API_BASE)

    assert models == [
        "github_copilot/gpt-4o",
        "github_copilot/claude-opus-5.5",
        "github_copilot/text-embedding-3-small",
    ]
    assert mock_get.call_args.kwargs["url"] == f"{API_BASE}/models"


def test_get_models_sends_copilot_integration_headers(cached_token):
    model_info = GithubCopilotModelInfo(StubAuthenticator(cached_token))

    with patch.object(litellm.module_level_client, "get", return_value=_response(MODELS_PAYLOAD)) as mock_get:
        model_info.get_models(api_base=API_BASE)

    headers = mock_get.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer gh.test-key-123456789"
    assert headers["copilot-integration-id"] == "vscode-chat"


def test_get_models_respects_explicit_api_base_and_key(missing_token):
    model_info = GithubCopilotModelInfo(StubAuthenticator(missing_token, api_key=None))

    with patch.object(litellm.module_level_client, "get", return_value=_response(MODELS_PAYLOAD)) as mock_get:
        model_info.get_models(api_key="explicit-key", api_base="https://copilot.example.com")

    assert mock_get.call_args.kwargs["url"] == "https://copilot.example.com/models"
    assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer explicit-key"


def test_get_models_without_cached_credentials_never_starts_device_login(missing_token):
    """Listing must fail fast rather than block on the interactive device-code flow."""
    model_info = GithubCopilotModelInfo(StubAuthenticator(missing_token, api_key=None))

    with patch.object(Authenticator, "_login") as login:
        with pytest.raises(ValueError, match="not authenticated"):
            model_info.get_models(api_base=API_BASE)

    login.assert_not_called()


def test_get_models_raises_on_http_error(cached_token):
    model_info = GithubCopilotModelInfo(StubAuthenticator(cached_token))

    with patch.object(litellm.module_level_client, "get", return_value=_response({"error": "forbidden"}, 403)):
        with pytest.raises(Exception, match="Failed to fetch models from GitHub Copilot"):
            model_info.get_models(api_base=API_BASE)


def test_get_models_raises_on_unexpected_payload(cached_token):
    model_info = GithubCopilotModelInfo(StubAuthenticator(cached_token))

    with patch.object(litellm.module_level_client, "get", return_value=_response({"models": []})):
        with pytest.raises(ValidationError, match="data"):
            model_info.get_models(api_base=API_BASE)


def test_get_base_model_strips_prefix():
    assert GithubCopilotModelInfo.get_base_model("github_copilot/gpt-4o") == "gpt-4o"
    assert GithubCopilotModelInfo.get_base_model("gpt-4o") == "gpt-4o"


def test_provider_config_manager_returns_model_info():
    model_info = ProviderConfigManager.get_provider_model_info(model=None, provider=LlmProviders.GITHUB_COPILOT)
    assert isinstance(model_info, GithubCopilotModelInfo)
