"""Moyai quick-connect endpoints.

`/moyai/connect/start` hands a proxy admin a signed, single-use code pointing
at their Moyai deployment. `/moyai/connect/exchange` trades that code for a
fresh virtual key and persists the deployment as the `moyai_url` UI setting.
The signed code is the credential for the exchange, so it must stay short
lived and single use.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final
from urllib.parse import urlencode, urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from litellm._internal_context import with_service_target
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
    UI_SETTINGS_CACHE_KEY,
    UI_SETTINGS_CACHE_TTL,
    _ui_settings_db,
    normalize_moyai_url,
)
from litellm.proxy.utils import CONFIG_PARAMS_TARGET
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.table_repositories import UISettingsRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter()

_MOYAI_CODE_TTL_SECONDS: Final = 600
_MOYAI_NONCE_CONFIG_PREFIX: Final = "moyai_connect_nonce:"
_MOYAI_CONNECT_EXCHANGE_ROUTE: Final = "/moyai/connect/exchange"


class MoyaiConnectStartRequest(BaseModel):
    moyai_url: str
    return_to: str


class MoyaiConnectStartResponse(BaseModel):
    connect_url: str


class MoyaiConnectExchangeRequest(BaseModel):
    code: str
    moyai_url: str


class MoyaiConnectExchangeResponse(BaseModel):
    api_key: str
    key_alias: str
    api_base: str


@dataclass(frozen=True, slots=True)
class _MoyaiConnectCode:
    moyai_origin: str
    nonce: str
    exp: int
    user_id: str | None


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _origin(url: str) -> str:
    parsed: Final = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _master_key_hmac_key(master_key: str) -> bytes:
    return hashlib.sha256(master_key.encode()).digest()


def _gateway_url(request: Request) -> str:
    if os.environ.get("PROXY_BASE_URL"):
        return os.environ["PROXY_BASE_URL"].rstrip("/")
    return str(request.base_url).rstrip("/")


_MOYAI_KEY_ALLOWED_ROUTES: Final = ["openai_routes", "anthropic_routes", "/model/info"]


def _sign_connect_code(master_key: str, moyai_url: str, user_id: str | None) -> str:
    payload: Final = json.dumps(
        {
            "moyai_origin": _origin(moyai_url),
            "user_id": user_id,
            "exp": int(time.time()) + _MOYAI_CODE_TTL_SECONDS,
            "nonce": secrets.token_urlsafe(16),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    signature: Final = hmac.new(_master_key_hmac_key(master_key), payload, hashlib.sha256).digest()
    return f"{_b64url(payload)}.{_b64url(signature)}"


def _decode_connect_code(master_key: str, code: str) -> _MoyaiConnectCode:
    try:
        payload_b64, signature_b64 = code.split(".", 1)
        payload_raw: Final = _b64url_decode(payload_b64)
        signature: Final = _b64url_decode(signature_b64)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")
    expected: Final = hmac.new(_master_key_hmac_key(master_key), payload_raw, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")
    try:
        payload: Final = json.loads(payload_raw)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")
    if not isinstance(payload, dict) or not isinstance(payload.get("exp"), int) or payload["exp"] < int(time.time()):
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")
    if not isinstance(payload.get("moyai_origin"), str) or not isinstance(payload.get("nonce"), str):
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")
    user_id: Final = payload.get("user_id")
    return _MoyaiConnectCode(
        moyai_origin=payload["moyai_origin"],
        nonce=payload["nonce"],
        exp=payload["exp"],
        user_id=user_id if isinstance(user_id, str) else None,
    )


@router.post(
    "/moyai/connect/start",
    response_model=MoyaiConnectStartResponse,
    tags=["moyai"],
)
async def moyai_connect_start(
    request: Request,
    body: MoyaiConnectStartRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> MoyaiConnectStartResponse:
    from litellm.proxy.proxy_server import master_key

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only proxy admins can connect Moyai")

    try:
        moyai_url: Final = normalize_moyai_url(body.moyai_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if moyai_url is None:
        raise HTTPException(status_code=400, detail="moyai_url is required")

    return_to_parsed: Final = urlparse(body.return_to)
    if return_to_parsed.scheme not in ("http", "https") or not return_to_parsed.netloc:
        raise HTTPException(status_code=400, detail="return_to must be an absolute http or https URL")

    if not master_key:
        raise HTTPException(
            status_code=400,
            detail="Moyai quick connect needs LITELLM_MASTER_KEY set on the proxy",
        )

    code: Final = _sign_connect_code(master_key, moyai_url, user_api_key_dict.user_id)
    connect_url: Final = f"{moyai_url}/connect/litellm?" + urlencode(
        {"gateway_url": _gateway_url(request), "code": code, "return_to": body.return_to}
    )
    return MoyaiConnectStartResponse(connect_url=connect_url)


async def _claim_connect_nonce(prisma_client: "PrismaClient", nonce: str, exp: int) -> None:
    from prisma.errors import UniqueViolationError

    try:
        await ConfigRepository(prisma_client, use_writer=True).table.create(
            data={
                "param_name": f"{_MOYAI_NONCE_CONFIG_PREFIX}{nonce}",
                "param_value": json.dumps({"exp": exp}),
            }
        )
    except UniqueViolationError:
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")


async def _moyai_key_alias(prisma_client: "PrismaClient", moyai_url: str) -> str:
    from litellm.repositories.verification_token_repository import (
        VerificationTokenRepository,
    )

    host: Final = urlparse(moyai_url).hostname or "deployment"
    alias: Final = f"moyai-{host}"
    rows: Final = await VerificationTokenRepository(prisma_client).find_many(where={"key_alias": alias}, take=1)
    if rows:
        return f"{alias}-{secrets.token_hex(2)}"
    return alias


@with_service_target(CONFIG_PARAMS_TARGET)
async def _persist_moyai_url(prisma_client: "PrismaClient", moyai_url: str) -> None:
    from litellm.proxy.proxy_server import user_api_key_cache

    db_existing: Final = await _ui_settings_db(UISettingsRepository(prisma_client)).find_unique(
        where={"id": "ui_settings"}
    )
    raw: Final = db_existing.ui_settings if db_existing else None
    existing: Final[Mapping[str, object]] = (json.loads(raw) if isinstance(raw, str) else dict(raw)) if raw else {}

    ui_settings: Final = {**existing, "moyai_url": moyai_url}
    await _ui_settings_db(UISettingsRepository(prisma_client)).upsert(
        where={"id": "ui_settings"},
        data={
            "create": {"id": "ui_settings", "ui_settings": json.dumps(ui_settings)},
            "update": {"ui_settings": json.dumps(ui_settings)},
        },
    )
    await user_api_key_cache.async_set_cache(key=UI_SETTINGS_CACHE_KEY, value=ui_settings, ttl=UI_SETTINGS_CACHE_TTL)


@router.post(
    _MOYAI_CONNECT_EXCHANGE_ROUTE,
    response_model=MoyaiConnectExchangeResponse,
    tags=["moyai"],
)
async def moyai_connect_exchange(request: Request, body: MoyaiConnectExchangeRequest) -> MoyaiConnectExchangeResponse:
    from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_helper_fn
    from litellm.proxy.proxy_server import llm_router, master_key, prisma_client

    if not master_key:
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")

    payload: Final = _decode_connect_code(master_key, body.code)

    try:
        moyai_url: Final = normalize_moyai_url(body.moyai_url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")
    if moyai_url is None or _origin(moyai_url) != payload.moyai_origin:
        raise HTTPException(status_code=400, detail="Invalid Moyai connect code")

    if prisma_client is None:
        raise HTTPException(status_code=400, detail="Moyai quick connect needs a database connected to the proxy")

    await _claim_connect_nonce(prisma_client, payload.nonce, payload.exp)

    alias: Final = await _moyai_key_alias(prisma_client, moyai_url)
    key_response: Final = await generate_key_helper_fn(
        request_type="key",
        key_alias=alias,
        allowed_routes=_MOYAI_KEY_ALLOWED_ROUTES,
        metadata={
            "created_via": "moyai_quick_connect",
            "moyai_url": moyai_url,
            "connected_by": payload.user_id,
        },
        table_name="key",
        llm_router=llm_router,
    )

    await _persist_moyai_url(prisma_client, moyai_url)

    return MoyaiConnectExchangeResponse(
        api_key=key_response["token"],
        key_alias=alias,
        api_base=_gateway_url(request),
    )
