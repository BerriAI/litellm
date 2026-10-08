import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="admin-user", user_role=LitellmUserRoles.PROXY_ADMIN)


def _request() -> MagicMock:
    request = MagicMock()
    request.base_url = "http://localhost:4000/"
    return request


@pytest.mark.asyncio
async def test_start_rejects_non_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException
    from litellm.proxy import proxy_server
    from litellm.proxy.moyai_endpoints import MoyaiConnectStartRequest, moyai_connect_start

    monkeypatch.setattr(proxy_server, "master_key", "sk-master")
    actor = UserAPIKeyAuth(user_id="member", user_role="internal_user")

    with pytest.raises(HTTPException) as exc:
        await moyai_connect_start(
            _request(), MoyaiConnectStartRequest(moyai_url="https://moyai.example.com", return_to="http://localhost:3000/ui/moyai"), actor
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_start_requires_master_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException
    from litellm.proxy import proxy_server
    from litellm.proxy.moyai_endpoints import MoyaiConnectStartRequest, moyai_connect_start

    monkeypatch.setattr(proxy_server, "master_key", None)

    with pytest.raises(HTTPException) as exc:
        await moyai_connect_start(
            _request(), MoyaiConnectStartRequest(moyai_url="https://moyai.example.com", return_to="http://localhost:3000/ui/moyai"), _admin()
        )
    assert exc.value.status_code == 400
    assert "LITELLM_MASTER_KEY" in exc.value.detail


@pytest.mark.asyncio
async def test_start_returns_connect_url_with_all_params(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.moyai_endpoints import MoyaiConnectStartRequest, moyai_connect_start
    from urllib.parse import parse_qs, urlparse

    monkeypatch.setattr(proxy_server, "master_key", "sk-master")

    response = await moyai_connect_start(
        _request(),
        MoyaiConnectStartRequest(moyai_url="https://moyai.example.com/", return_to="http://localhost:3000/ui/moyai"),
        _admin(),
    )

    parsed = urlparse(response.connect_url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://moyai.example.com/connect/litellm"
    params = parse_qs(parsed.query)
    assert params["gateway_url"] == ["http://localhost:4000"]
    assert params["return_to"] == ["http://localhost:3000/ui/moyai"]
    assert params["code"] and "." in params["code"][0]


async def _exchange_env(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server

    cache: dict = {}

    async def _get(key: str):
        return cache.get(key)

    async def _set(key: str, value, ttl=None):
        cache[key] = value

    monkeypatch.setattr(proxy_server, "master_key", "sk-master")
    monkeypatch.setattr(proxy_server, "llm_router", None)
    user_api_key_cache = SimpleNamespace(async_get_cache=AsyncMock(side_effect=_get), async_set_cache=AsyncMock(side_effect=_set))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", user_api_key_cache)

    prisma = MagicMock()
    prisma.db.litellm_uisettings.find_unique = AsyncMock(return_value=SimpleNamespace(ui_settings={}))
    persisted: dict = {}

    async def _upsert(where, data):
        persisted.update(json.loads(data["update"]["ui_settings"]))

    prisma.db.litellm_uisettings.upsert = AsyncMock(side_effect=_upsert)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    mint_calls: list = []

    async def _mint(request_type, **kwargs):
        mint_calls.append(kwargs)
        return {"key": "sk-new-virtual-key", **kwargs}

    import litellm.proxy.moyai_endpoints as m
    monkeypatch.setattr(m, "_moyai_key_alias", AsyncMock(return_value="moyai-moyai.example.com"))
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
        AsyncMock(side_effect=_mint),
    )
    return persisted, mint_calls


async def _exchange_call(code: str, moyai_url: str):
    from litellm.proxy.moyai_endpoints import MoyaiConnectExchangeRequest, moyai_connect_exchange

    return await moyai_connect_exchange(_request(), MoyaiConnectExchangeRequest(code=code, moyai_url=moyai_url))


@pytest.mark.asyncio
async def test_exchange_happy_path_mints_key_and_saves_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.moyai_endpoints import _sign_connect_code

    persisted, mint_calls = await _exchange_env(monkeypatch)
    code = _sign_connect_code("sk-master", "https://moyai.example.com", "admin-user")

    response = await _exchange_call(code, "https://moyai.example.com")

    assert response.api_key == "sk-new-virtual-key"
    assert response.key_alias == "moyai-moyai.example.com"
    assert response.api_base == "http://localhost:4000"
    assert persisted["moyai_url"] == "https://moyai.example.com"
    mint = mint_calls[0]
    assert "user_id" not in mint
    assert mint["allowed_routes"] == ["openai_routes", "anthropic_routes", "/model/info"]
    assert mint["metadata"] == {
        "created_via": "moyai_quick_connect",
        "moyai_url": "https://moyai.example.com",
        "connected_by": "admin-user",
    }


@pytest.mark.asyncio
async def test_exchange_without_database_fails_before_nonce(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException
    from litellm.proxy import proxy_server
    from litellm.proxy.moyai_endpoints import _sign_connect_code

    await _exchange_env(monkeypatch)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    set_cache = proxy_server.user_api_key_cache.async_set_cache
    code = _sign_connect_code("sk-master", "https://moyai.example.com", "admin-user")

    with pytest.raises(HTTPException) as exc:
        await _exchange_call(code, "https://moyai.example.com")
    assert exc.value.status_code == 400
    assert "database" in exc.value.detail
    set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_exchange_rejects_tampered_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException
    from litellm.proxy.moyai_endpoints import _sign_connect_code

    await _exchange_env(monkeypatch)
    code = _sign_connect_code("sk-master", "https://moyai.example.com", "admin-user")
    tampered = code[:-2] + ("aa" if not code.endswith("aa") else "bb")

    with pytest.raises(HTTPException) as exc:
        await _exchange_call(tampered, "https://moyai.example.com")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_exchange_rejects_expired_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import base64
    import hashlib
    import hmac as hmac_mod
    from fastapi import HTTPException
    from litellm.proxy.moyai_endpoints import _b64url, _master_key_hmac_key

    await _exchange_env(monkeypatch)
    payload = json.dumps(
        {"moyai_origin": "https://moyai.example.com", "user_id": "admin-user", "exp": int(time.time()) - 10, "nonce": "n"},
        separators=(",", ":"), sort_keys=True,
    ).encode()
    sig = hmac_mod.new(_master_key_hmac_key("sk-master"), payload, hashlib.sha256).digest()
    code = f"{_b64url(payload)}.{_b64url(sig)}"

    with pytest.raises(HTTPException) as exc:
        await _exchange_call(code, "https://moyai.example.com")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_exchange_rejects_origin_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException
    from litellm.proxy.moyai_endpoints import _sign_connect_code

    await _exchange_env(monkeypatch)
    code = _sign_connect_code("sk-master", "https://moyai.example.com", "admin-user")

    with pytest.raises(HTTPException) as exc:
        await _exchange_call(code, "https://evil.example.com")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_exchange_rejects_replayed_nonce(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException
    from litellm.proxy.moyai_endpoints import _sign_connect_code

    await _exchange_env(monkeypatch)
    code = _sign_connect_code("sk-master", "https://moyai.example.com", "admin-user")

    await _exchange_call(code, "https://moyai.example.com")
    with pytest.raises(HTTPException) as exc:
        await _exchange_call(code, "https://moyai.example.com")
    assert exc.value.status_code == 400
