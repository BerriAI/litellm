import hashlib
import hmac
import time
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.integrations.lens.proxy.adapter import Connection, Identity
from litellm.integrations.lens.remote import bounded_response
from litellm.proxy._types import GenerateKeyRequest, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks

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
    NO_MODELS = "Add a model to this LiteLLM gateway before connecting Lens"
    KEY_POLICY = "The gateway could not create the Lens key. Check key-generation permissions and policy"
    DATABASE = "Connect a database to LiteLLM before creating a Lens model key"


def model_key(connection: Connection) -> str:
    digest: Final = hmac.new(
        connection.secret.encode(),
        f"litellm.lens.models.v1\0{connection.remote.url}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"sk-lens-{digest}"


def key_request(key: str, models: Sequence[str]) -> GenerateKeyRequest:
    return GenerateKeyRequest.model_validate(
        {
            "key": key,
            "key_alias": f"lens-models-{hashlib.sha256(key.encode()).hexdigest()[:12]}",
            "models": sorted(set(models)),
            "allowed_routes": MODEL_ROUTES,
            "metadata": {"purpose": "lens-model-gateway"},
        }
    )


async def connect_models(
    connection: Connection,
    identity: Identity,
    gateway_url: str,
    models: Sequence[str],
    client: httpx.AsyncClient,
    ensure_key: Callable[[str, Sequence[str]], Awaitable[bool]],
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
        if not models:
            return SetupFailure.NO_MODELS
        key: Final = model_key(connection)
        if not await ensure_key(key, models):
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
    from starlette.exceptions import HTTPException

    from litellm.proxy import proxy_server
    from litellm.proxy._types import ProxyException
    from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_fn
    from litellm.repositories.verification_token_repository import VerificationTokenRepository

    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        return SetupFailure.FORBIDDEN
    try:
        RouteChecks.should_call_route("/key/generate", auth)
    except HTTPException:
        return SetupFailure.KEY_POLICY
    database: Final = proxy_server.prisma_client
    router: Final = proxy_server.llm_router
    if database is None:
        return SetupFailure.DATABASE
    models: Final = tuple(router.registry().names()) if router else ()
    repository: Final = VerificationTokenRepository(database)

    async def ensure_key(key: str, models: Sequence[str]) -> bool:
        hashed: Final = hashlib.sha256(key.encode()).hexdigest()
        if await repository.find_for_auth(hashed, use_writer=True) is not None:
            return True
        try:
            await generate_key_fn(data=key_request(key, models), user_api_key_dict=auth, litellm_changed_by=None)
        except (HTTPException, ProxyException, ValidationError):
            return await repository.find_for_auth(hashed, use_writer=True) is not None
        else:
            return True

    return await connect_models(
        connection, identity, gateway_url, models, connection.remote.control_client(), ensure_key, int(time.time())
    )
