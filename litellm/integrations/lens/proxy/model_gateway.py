import hashlib
import hmac
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import timedelta
from enum import Enum
from functools import partial
from typing import (  # noqa: TID251  # generated Prisma cursor type needs the typed repository interface
    TYPE_CHECKING,
    Final,
    cast,
)

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.exceptions import HTTPException

from litellm.integrations.lens.proxy.adapter import Connection, Identity
from litellm.integrations.lens.remote import bounded_response
from litellm.proxy._types import GenerateKeyRequest, LitellmUserRoles, ProxyException, SpecialModelNames, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient

MODEL_ROUTES: Final = (
    "/model_group/info",
    "/chat/completions",
    "/v1/chat/completions",
    "/embeddings",
    "/v1/embeddings",
    "/decisions",
    "/v1/decisions",
)


class GatewayStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    configured: bool
    connected: bool
    login_connected: bool = False
    supports_gateway_connect: bool = False
    api_base: str | None = None
    evaluation_models: int = 0
    analysis_models: int = 0
    error: str | None = None
    last_refreshed: str | None = None


class GatewaySetupRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SetupFailure(Enum):
    FORBIDDEN = "Only gateway administrators can connect models to Lens"
    UNAVAILABLE = "Lens could not be reached. Check the connection and try again"
    UNSUPPORTED = "Update Lens to connect this gateway automatically"
    KEY_POLICY = (
        "The gateway could not create the Lens key. "
        "Check key-generation permissions, policy, and whether the Lens key was revoked"
    )
    DATABASE = "Connect a database to LiteLLM before creating a Lens model key"


def model_key(connection: Connection) -> str:
    digest: Final = hmac.new(
        connection.secret.encode(),
        f"litellm.lens.models.v1\0{connection.remote.url}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"sk-lens-{digest}"


def key_request(key: str) -> GenerateKeyRequest:
    return GenerateKeyRequest.model_validate(
        {
            "key": key,
            "key_alias": f"lens-models-{hashlib.sha256(key.encode()).hexdigest()[:12]}",
            "models": [SpecialModelNames.all_proxy_models.value],
            "allowed_routes": MODEL_ROUTES,
            "user_id": None,
            "team_id": None,
            "metadata": {"purpose": "lens-model-gateway"},
        }
    )


async def connect_models(
    connection: Connection,
    identity: Identity,
    gateway_url: str,
    client: httpx.AsyncClient,
    ensure_key: Callable[[str], Awaitable[bool]],
    now: int,
) -> GatewayStatus | SetupFailure:
    if identity.user_role != LitellmUserRoles.PROXY_ADMIN:
        return SetupFailure.FORBIDDEN
    token: Final = connection.identity_token(identity, now)
    if token is None:
        return SetupFailure.FORBIDDEN
    headers: Final = {"authorization": f"Bearer {token}", "x-lens-contract": "1"}
    try:
        async with client.stream("GET", connection.remote.endpoint("/lens/admin/gateway"), headers=headers) as response:
            if response.status_code == 404:
                return SetupFailure.UNSUPPORTED
            if response.status_code != 200:
                return SetupFailure.UNAVAILABLE
            status: Final = GatewayStatus.model_validate_json(await bounded_response(response, 16 * 1024))
        if status.configured:
            return status
        if not status.supports_gateway_connect:
            return SetupFailure.UNSUPPORTED
        key: Final = model_key(connection)
        if not await ensure_key(key):
            return SetupFailure.KEY_POLICY
        async with client.stream(
            "POST",
            connection.remote.endpoint("/lens/admin/gateway/connect"),
            headers=headers,
            json={"gateway_url": gateway_url, "api_key": key},
        ) as response:
            if response.status_code == 404:
                return SetupFailure.UNSUPPORTED
            if response.status_code != 200:
                return SetupFailure.UNAVAILABLE
            return GatewayStatus.model_validate_json(await bounded_response(response, 16 * 1024))
    except (httpx.HTTPError, ValueError, RuntimeError):
        return SetupFailure.UNAVAILABLE


async def setup(
    connection: Connection, identity: Identity, gateway_url: str, auth: UserAPIKeyAuth
) -> GatewayStatus | SetupFailure:
    from litellm.proxy import proxy_server
    from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_fn

    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        return SetupFailure.FORBIDDEN
    try:
        RouteChecks.should_call_route("/key/generate", auth)
    except HTTPException:
        return SetupFailure.KEY_POLICY
    database: Final = proxy_server.prisma_client
    if database is None:
        return SetupFailure.DATABASE

    async def active(hashed: str) -> bool:
        async with database.tx() as transaction:
            return await transaction.litellm_verificationtoken.find_unique(where={"token": hashed}) is not None

    async def retired(hashed: str, alias: str) -> bool:
        async with database.tx() as transaction:
            deleted_table: Final = (
                cast(  # cast-ok: generated Prisma cursor type is absent; use the repository interface
                    TableActions[object], transaction.litellm_deletedverificationtoken
                )
            )
            deleted: Final = await deleted_table.find_many(where={"token": hashed}, take=1)
            return bool(deleted) or bool(
                await transaction.litellm_verificationtoken.find_many(where={"key_alias": alias}, take=1)
            )

    async def generate(request: GenerateKeyRequest) -> None:
        await generate_key_fn(
            data=request,
            user_api_key_dict=auth,
            litellm_changed_by=None,
            key_writer=partial(_persist_model_key, database),
        )

    async def ensure_key(key: str) -> bool:
        return await ensure_model_key(key, active, retired, generate)

    return await connect_models(
        connection, identity, gateway_url, connection.remote.control_client(), ensure_key, int(time.time())
    )


async def _persist_model_key(
    database: "PrismaClient", data: Mapping[str, object]
) -> "prisma_models.LiteLLM_VerificationToken":
    from litellm.proxy.management_endpoints.key_management_endpoints import KeyPersistenceError

    async with database.tx(timeout=timedelta(seconds=10)) as transaction:
        await transaction.query_raw(
            "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtext($1))", f"lens-model-key:{data['token']}"
        )
        keys: Final = cast(  # cast-ok: normalized insert fields use the repository's mapping interface
            "TableActions[prisma_models.LiteLLM_VerificationToken]", transaction.litellm_verificationtoken
        )
        existing: Final = await keys.find_unique(where={"token": data["token"]})
        if existing is not None:
            raise KeyPersistenceError(status_code=409, detail="The Lens key already exists")
        deleted_table: Final = cast(  # cast-ok: generated Prisma cursor type is absent; use the repository interface
            TableActions[object], transaction.litellm_deletedverificationtoken
        )
        deleted: Final = await deleted_table.find_many(where={"token": data["token"]}, take=1)
        rotated: Final = await keys.find_many(where={"key_alias": data["key_alias"]}, take=1)
        if deleted or rotated:
            raise KeyPersistenceError(status_code=409, detail=SetupFailure.KEY_POLICY.value)
        return await keys.create(data=data, include={"litellm_budget_table": True})


async def ensure_model_key(
    key: str,
    active: Callable[[str], Awaitable[bool]],
    retired: Callable[[str, str], Awaitable[bool]],
    generate: Callable[[GenerateKeyRequest], Awaitable[None]],
) -> bool:
    hashed: Final = hashlib.sha256(key.encode()).hexdigest()
    if await active(hashed):
        return True
    request: Final = key_request(key)
    if await retired(hashed, request.key_alias or ""):
        return False
    try:
        await generate(request)
    except (HTTPException, ProxyException, ValidationError):
        return await active(hashed)
    else:
        return True
