import json
from pathlib import Path
from typing import Final

import httpx
import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.model_info import _ChatGPTModel, get_chatgpt_model_info
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


async def test_account_catalog_preserves_client_policy_separately_from_model_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    auth_file: Final = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps({"access_token": "local-token", "account_id": "local-account", "expires_at": 10**30})
    )
    card: Final = {
        "slug": "fixture-model",
        "context_window": 600,
        "max_context_window": 1200,
        "supported_reasoning_levels": [{"effort": "high"}, {"effort": "future-effort"}],
        "default_reasoning_level": "future-effort",
        "input_modalities": ["text", "image"],
        "supports_parallel_tool_calls": False,
    }

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/backend-api/codex/models"
        assert request.url.params.get("client_version")
        assert request.headers["authorization"] == "Bearer local-token"
        assert request.headers["chatgpt-account-id"] == "local-account"
        assert request.headers["accept"] == "application/json"
        return httpx.Response(200, json={"models": [card]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        cache: Final = InMemoryCache()
        result: Final = await get_chatgpt_model_info(
            model="fixture-model",
            api_base="https://account.test/backend-api/codex/",
            authenticator=Authenticator(),
            client=handler,
            cache=cache,
        )
        assert result == {
            "codex_context_window": card["context_window"],
            "codex_max_context_window": card["max_context_window"],
            "supports_parallel_function_calling": card["supports_parallel_tool_calls"],
            "supports_reasoning": True,
            "reasoning_effort_levels": tuple(level["effort"] for level in card["supported_reasoning_levels"]),
            "default_reasoning_effort": card["default_reasoning_level"],
            "supported_endpoints": ("/responses",),
            "supported_modalities": tuple(card["input_modalities"]),
        }
        assert (
            await get_chatgpt_model_info(
                model="not-selected",
                api_base="https://account.test/backend-api/codex/",
                authenticator=Authenticator(),
                client=handler,
                cache=cache,
            )
            == {}
        )


async def test_switching_accounts_invalidates_catalog_without_exposing_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    auth_file: Final = tmp_path / "auth.json"
    cache: Final = InMemoryCache()
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()

    def respond(request: httpx.Request) -> httpx.Response:
        capacity: Final = 1024 if request.headers["chatgpt-account-id"] == "first" else 2048
        return httpx.Response(200, json={"models": [{"slug": "fixture", "context_window": capacity}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        for account, capacity in (("first", 1024), ("second", 2048)):
            auth_file.write_text(
                json.dumps({"access_token": "local-token", "account_id": account, "expires_at": 10**30})
            )
            result: Final = await get_chatgpt_model_info(
                model="fixture",
                api_base="https://account.test/backend-api/codex",
                authenticator=Authenticator(),
                client=handler,
                cache=cache,
            )
            assert result["codex_context_window"] == capacity
            assert "local-token" not in json.dumps(dict(result))


async def test_discovery_without_authorization_never_starts_device_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    with pytest.raises(GetAccessTokenError, match="model discovery cannot start device login"):
        Authenticator().get_access_token(allow_device_login=False)

    def reject(request: httpx.Request) -> httpx.Response:
        pytest.fail("Discovery without credentials must not send any request")

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
        handler.client = client
        assert (
            await get_chatgpt_model_info(
                model="fixture",
                authenticator=Authenticator(),
                client=handler,
                cache=InMemoryCache(),
            )
            == {}
        )
    assert not (tmp_path / "auth.json").exists()


@pytest.mark.parametrize(("status", "body"), ((503, {}), (200, {"models": "invalid"})))
async def test_invalid_or_unavailable_catalog_preserves_fallback(
    status: int, body: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({"access_token": "local", "expires_at": 10**30}))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body))) as client:
        handler.client = client
        assert (
            await get_chatgpt_model_info(
                model="fixture",
                authenticator=Authenticator(),
                client=handler,
                cache=InMemoryCache(),
            )
            == {}
        )


@pytest.mark.parametrize(
    ("default", "multi_agent", "expected"),
    (("ultra", "high", "high"), ("ultra", None, "max"), ("persistent", None, "disabled")),
)
def test_codex_ui_efforts_are_not_advertised_as_api_efforts(
    default: str, multi_agent: str | None, expected: str
) -> None:
    metadata: Final = _ChatGPTModel.model_validate(
        {
            "slug": "fixture",
            "supported_reasoning_levels": [{"effort": effort} for effort in ("high", "max", "ultra", "persistent")],
            "default_reasoning_level": default,
            "multi_agent_reasoning_effort": multi_agent,
        }
    ).metadata()
    assert metadata["reasoning_effort_levels"] == ("high", "max", "disabled")
    assert metadata["default_reasoning_effort"] == expected
