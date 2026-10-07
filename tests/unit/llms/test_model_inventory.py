from typing import Final

import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.model_inventory import model_inventory_api_base, model_inventory_identity


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("vercel_ai_gateway", "https://ai-gateway.vercel.sh/v1"),
        ("openai", None),
    ],
)
def test_inventory_connection_defaults_and_explicit_endpoints(provider, expected):
    assert model_inventory_api_base(provider, None) == expected
    assert model_inventory_api_base(provider, "https://configured.test/v1") == "https://configured.test/v1"
    assert model_inventory_identity(provider) is None


def test_native_inventory_registry_uses_server_owned_backend_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    trusted: Final = "https://server-owned.test/backend-api/codex"
    monkeypatch.setenv("CHATGPT_API_BASE", trusted)
    monkeypatch.setattr(Authenticator, "get_account_id", lambda self: "fixture-account")
    assert model_inventory_api_base("chatgpt", None) == trusted
    assert model_inventory_api_base("chatgpt", trusted + "/") == trusted
    assert model_inventory_identity("chatgpt") == "fixture-account"
    with pytest.raises(ValueError, match="server-configured API base"):
        model_inventory_api_base("chatgpt", "https://caller-owned.test/capture")
