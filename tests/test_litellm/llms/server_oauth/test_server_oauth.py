import base64
import json
import time
from unittest.mock import MagicMock, patch

from litellm.llms.claude_max.authenticator import ClaudeMaxAuthenticator
from litellm.llms.claude_max.chat.transformation import ClaudeMaxConfig
from litellm.llms.server_oauth_base import JsonOAuthTokenStore
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_json_store_restores_base64_env(tmp_path, monkeypatch):
    token_path = tmp_path / "claude-max.json"
    payload = {"access_token": "token", "expires_at": time.time() + 3600}
    monkeypatch.setenv("CLAUDE_MAX_OAUTH_FILE", str(token_path))
    monkeypatch.setenv("CLAUDE_MAX_OAUTH_JSON_B64", base64.b64encode(json.dumps(payload).encode()).decode())

    store = JsonOAuthTokenStore("claude-max", "CLAUDE_MAX")

    assert store.load()["access_token"] == "token"
    assert token_path.exists()


def test_claude_max_refresh_uses_anthropic_oauth_contract(tmp_path, monkeypatch):
    token_path = tmp_path / "claude-max.json"
    monkeypatch.setenv("CLAUDE_MAX_OAUTH_FILE", str(token_path))
    token_path.write_text(json.dumps({"access_token": "old", "refresh_token": "refresh", "expires_at": 1}))
    response = MagicMock()
    response.json.return_value = {"access_token": "new", "expires_in": 3600}
    response.raise_for_status.return_value = None
    client = MagicMock()
    client.post.return_value = response

    with patch("litellm.llms.server_oauth_base._get_httpx_client", return_value=client):
        assert ClaudeMaxAuthenticator().get_access_token() == "new"

    client.post.assert_called_once()
    _, kwargs = client.post.call_args
    assert kwargs["json"]["grant_type"] == "refresh_token"
    assert kwargs["json"]["client_id"] == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"


def test_provider_config_registration():
    claude_config = ProviderConfigManager.get_provider_chat_config("claude-opus-4-8", LlmProviders.CLAUDE_MAX)

    assert isinstance(claude_config, ClaudeMaxConfig)


def test_claude_max_default_headers():
    headers = ClaudeMaxConfig().get_default_headers()

    assert headers["x-app"] == "cli"
    assert headers["X-Stainless-Lang"] == "js"
