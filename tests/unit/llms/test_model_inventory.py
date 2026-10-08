import json
from pathlib import Path
from typing import Final

import pytest

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


def test_native_inventory_registry_uses_server_owned_backend_and_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trusted: Final = "https://server-owned.test/backend-api/codex"
    monkeypatch.setenv("CHATGPT_API_BASE", trusted)
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("CHATGPT_AUTH_FILE", "auth.json")
    (tmp_path / "auth.json").write_text(json.dumps({"account_id": "fixture-account"}))
    assert model_inventory_api_base("chatgpt", None) == trusted
    assert model_inventory_api_base("chatgpt", trusted + "/") == trusted
    assert model_inventory_identity("chatgpt") == "fixture-account"
    with pytest.raises(ValueError, match="server-configured API base"):
        model_inventory_api_base("chatgpt", "https://caller-owned.test/capture")
