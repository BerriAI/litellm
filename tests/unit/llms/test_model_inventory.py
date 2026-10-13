import json
from pathlib import Path
from typing import Final

import httpx
import pytest

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.model_inventory import (
    get_provider_model_inventory,
    model_inventory_api_base,
    model_inventory_identity,
)
from litellm.llms.openrouter.responses.transformation import OpenRouterResponsesAPIConfig
from litellm.llms.vercel_ai_gateway.chat.transformation import VercelAIGatewayConfig
from litellm.types.proxy.model_inventory import SupplierModelInventory


@pytest.mark.parametrize("provider", ("openrouter", "vercel_ai_gateway"))
@pytest.mark.parametrize("source", ("explicit", "environment", "global", "default"))
def test_inventory_base_matches_native_resolution_and_preserves_paths(
    monkeypatch: pytest.MonkeyPatch, provider: str, source: str
) -> None:
    explicit: Final = "https://explicit.test/custom/v1/" if source == "explicit" else None
    environment: Final = "https://environment.test/custom/v1"
    global_base: Final = "https://global.test/custom/v1"
    monkeypatch.setattr(litellm, "api_base", global_base if source in ("explicit", "global") else None)
    variable: Final = "OPENROUTER_API_BASE" if provider == "openrouter" else "VERCEL_AI_GATEWAY_API_BASE"
    if source in ("explicit", "environment"):
        monkeypatch.setenv(variable, environment)
    else:
        monkeypatch.delenv(variable, raising=False)
    expected: Final = (
        explicit or OpenRouterResponsesAPIConfig().get_complete_url(None, {}).removesuffix("/responses")
        if provider == "openrouter"
        else VercelAIGatewayConfig().get_openai_compatible_provider_info(explicit, "fixture-key")[0]
    )
    assert model_inventory_api_base(provider, explicit) == expected


@pytest.mark.parametrize("provider", ("openrouter", "vercel_ai_gateway"))
async def test_native_base_resolution_does_not_borrow_ambient_discovery_credentials(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    base: Final = "https://supplier.test/custom/v1"
    variable: Final = "OPENROUTER" if provider == "openrouter" else "VERCEL_AI_GATEWAY"
    monkeypatch.setenv(f"{variable}_API_BASE", base)
    monkeypatch.setenv(f"{variable}_API_KEY", "ambient-provider-key")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "ambient-oidc")
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "api_key", "ambient-global-key")

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{base}/models"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"id": "org/backend"}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        inventory: Final = await get_provider_model_inventory(
            provider=provider, api_base=None, headers={}, client=handler, cache=InMemoryCache()
        )
    assert isinstance(inventory, SupplierModelInventory)
    assert frozenset(inventory.models) == {"org/backend"}


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
