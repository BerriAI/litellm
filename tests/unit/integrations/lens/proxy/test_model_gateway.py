import json
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt
import pytest
from pydantic import BaseModel, TypeAdapter
from starlette.exceptions import HTTPException

from litellm.integrations.lens.proxy.adapter import Connection, Identity
from litellm.integrations.lens.proxy.model_gateway import (
    GatewayStatus,
    SetupFailure,
    connect_models,
    ensure_model_key,
    key_request,
    model_key,
    setup,
)
from litellm.integrations.lens.remote import LensConnection
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import check_model_access_helper

CONNECTION: Final = Connection(
    LensConnection("https://lens.test", "lens-service-fixture-32-characters"),
    "lens-signing-fixture-32-characters",
)
ADMIN: Final = Identity(
    user_role=LitellmUserRoles.PROXY_ADMIN,
    user_id="admin",
    team_id=None,
    org_id=None,
    token=None,
    models=(),
    log_team_ids=(),
)
EMPTY: Final = GatewayStatus(configured=False, connected=False, supports_gateway_connect=True)
CONNECTED: Final = GatewayStatus(
    configured=True, connected=True, api_base="https://gateway.test/prefix", analysis_models=1
)


@pytest.mark.asyncio
async def test_connects_the_gateway_server_side_and_returns_only_public_status() -> None:
    key: Final = model_key(CONNECTION)
    ensure_key: Final = AsyncMock(return_value=True)

    def transport(request: httpx.Request) -> httpx.Response:
        token: Final = request.headers["authorization"].removeprefix("Bearer ")
        claims: Final = jwt.decode(
            token,
            CONNECTION.secret,
            algorithms=["HS256"],
            audience="litellm-lens",
            options={"verify_exp": False},
        )
        assert claims["identity"]["user_role"] == "proxy_admin"
        if request.method == "GET":
            return httpx.Response(200, json=EMPTY.model_dump())
        assert str(request.url) == "https://lens.test/lens/admin/gateway/connect"
        assert json.loads(request.content) == {"gateway_url": CONNECTED.api_base, "api_key": key}
        return httpx.Response(200, json={**CONNECTED.model_dump(), "api_key": key})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result: Final = await connect_models(
            CONNECTION, ADMIN, CONNECTED.api_base or "", client, ensure_key, 1_700_000_000
        )
    assert isinstance(result, GatewayStatus)
    assert result == CONNECTED
    assert key not in result.model_dump_json()
    ensure_key.assert_awaited_once_with(key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role", (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM)
)
async def test_non_admins_cannot_send_credentials_or_create_keys(role: LitellmUserRoles) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        raise AssertionError("An unauthorized setup reached Lens")

    ensure_key: Final = AsyncMock(return_value=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result: Final = await connect_models(
            CONNECTION,
            ADMIN.model_copy(update={"user_role": role}),
            "https://gateway.test",
            client,
            ensure_key,
            1_700_000_000,
        )
    assert result is SetupFailure.FORBIDDEN
    ensure_key.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    ((CONNECTED, CONNECTED), (GatewayStatus(configured=False, connected=False), SetupFailure.UNSUPPORTED)),
    ids=("existing_configuration", "older_lens"),
)
async def test_existing_connections_and_older_lens_versions_do_not_create_keys(
    status: GatewayStatus, expected: GatewayStatus | SetupFailure
) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json=status.model_dump())

    ensure_key: Final = AsyncMock(return_value=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result: Final = await connect_models(
            CONNECTION, ADMIN, "https://gateway.test", client, ensure_key, 1_700_000_000
        )
    assert result == expected
    ensure_key.assert_not_awaited()


@pytest.mark.asyncio
async def test_key_policy_failure_prevents_sending_credentials() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json=EMPTY.model_dump())

    ensure_key: Final = AsyncMock(return_value=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result: Final = await connect_models(
            CONNECTION, ADMIN, "https://gateway.test", client, ensure_key, 1_700_000_000
        )
    assert result is SetupFailure.KEY_POLICY


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (401, 403, 500, 302))
async def test_upstream_errors_never_return_private_response_details(status: int) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="private-credential")

    ensure_key: Final = AsyncMock(return_value=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result: Final = await connect_models(
            CONNECTION, ADMIN, "https://gateway.test", client, ensure_key, 1_700_000_000
        )
    assert result is SetupFailure.UNAVAILABLE
    ensure_key.assert_not_awaited()


def test_dedicated_key_preserves_route_restrictions() -> None:
    request: Final = key_request(model_key(CONNECTION))
    assert request.allowed_routes == [
        "/model_group/info",
        "/chat/completions",
        "/v1/chat/completions",
        "/embeddings",
        "/v1/embeddings",
        "/decisions",
        "/v1/decisions",
    ]
    assert request.user_id is None
    assert request.team_id is None


def test_dedicated_key_allows_models_added_after_setup() -> None:
    request: Final = key_request(model_key(CONNECTION))
    models: Final = TypeAdapter(list[str]).validate_python(request.models)
    assert check_model_access_helper("added-after-setup", None, models)


def test_separate_lens_services_do_not_share_model_keys() -> None:
    other: Final = Connection(LensConnection("https://other-lens.test", CONNECTION.remote.token), CONNECTION.secret)
    assert model_key(CONNECTION) != model_key(other)


@pytest.mark.asyncio
async def test_a_lens_only_admin_key_cannot_provision_model_access() -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, allowed_routes=["/lens/*"])
    assert await setup(CONNECTION, ADMIN, "https://gateway.test", auth) is SetupFailure.KEY_POLICY


@pytest.mark.asyncio
async def test_setup_reuses_a_key_committed_by_another_worker_during_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    database: Final = MagicMock()
    transaction: Final = database.tx.return_value.__aenter__.return_value
    transaction.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    transaction.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[])
    transaction.litellm_verificationtoken.find_many = AsyncMock(return_value=[object()])

    async def committed_snapshot(_query: str, _token: str, _alias: str, *, model: type[BaseModel]) -> BaseModel:
        return model.model_validate({"state": "active"})

    transaction.query_first = committed_snapshot
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", database)

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=EMPTY.model_dump())
        assert json.loads(request.content)["api_key"] == model_key(CONNECTION)
        return httpx.Response(200, json=CONNECTED.model_dump())

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        monkeypatch.setattr(
            "litellm.integrations.lens.remote.get_async_httpx_client",
            lambda *_args, **_kwargs: SimpleNamespace(client=client),
        )
        assert await setup(CONNECTION, ADMIN, "https://gateway.test", auth) == CONNECTED


@pytest.mark.asyncio
async def test_a_retired_model_key_is_never_reissued_on_reconnection() -> None:
    from litellm.integrations.lens.proxy.model_gateway import _ModelKeyState

    lookup: Final = AsyncMock(return_value=_ModelKeyState.RETIRED)
    generate: Final = AsyncMock()
    key: Final = model_key(CONNECTION)
    assert not await ensure_model_key(key, lookup, generate)
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_retries_reuse_an_active_key_without_creating_another() -> None:
    from litellm.integrations.lens.proxy.model_gateway import _ModelKeyState

    lookup: Final = AsyncMock(return_value=_ModelKeyState.ACTIVE)
    generate: Final = AsyncMock()
    assert await ensure_model_key(model_key(CONNECTION), lookup, generate)
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_provisioning_creates_the_restricted_key() -> None:
    from litellm.integrations.lens.proxy.model_gateway import _ModelKeyState

    lookup: Final = AsyncMock(return_value=_ModelKeyState.MISSING)
    generate: Final = AsyncMock()
    key: Final = model_key(CONNECTION)
    assert await ensure_model_key(key, lookup, generate)
    generate.assert_awaited_once_with(key_request(key))


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", (True, False))
async def test_a_persistence_conflict_retries_only_with_an_active_key(retired: bool) -> None:
    from litellm.integrations.lens.proxy.model_gateway import _ModelKeyState

    lookup: Final = AsyncMock(
        side_effect=(_ModelKeyState.MISSING, _ModelKeyState.RETIRED if retired else _ModelKeyState.ACTIVE)
    )
    generate: Final = AsyncMock(side_effect=HTTPException(409))
    assert await ensure_model_key(model_key(CONNECTION), lookup, generate) is not retired
