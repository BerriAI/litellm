from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Final, Literal

import jwt
from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.spend_tracking.charge_receipt_capture import (
    AuthenticatedBillingContext,
    BillingContext,
    ChargeCapture,
    ChargeReceiptRuntime,
)
from litellm.proxy.spend_tracking.charge_receipts import (
    ChargeReceiptSettings,
    ChargeReceiptStore,
    Digest,
    ReceiptQuery,
)
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

SUPPORTED_RECEIPT_ROUTES: Final = frozenset(
    {
        "/responses",
        "/v1/responses",
        "/responses/compact",
        "/v1/responses/compact",
        "/chat/completions",
        "/v1/chat/completions",
    }
)


class _SignedContext(LiteLLMBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    sub: str = Field(min_length=1, max_length=200)
    aud: Literal["litellm-charge-receipts"]
    iat: int
    exp: int
    gateway_scope: Digest
    key_hash: Digest
    body_sha256: Digest
    method: Literal["POST"]
    route: str = Field(min_length=1, max_length=200)


class _RequestFeatures(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    background: bool = False
    model: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True, slots=True)
class ChargeReceiptConfigurationError:
    reason: str


def writer_query(client: "PrismaClient") -> ReceiptQuery:
    writer: Final[object] = client.writer_db
    query: Final[object] = getattr(writer, "query_raw", None)
    if not isinstance(query, ReceiptQuery):
        raise TypeError("The writer database does not expose raw queries.")
    return query


def receipt_runtime(logging: object) -> ChargeReceiptRuntime | None:
    runtime: Final[object] = getattr(logging, "charge_receipts", None)
    return runtime if isinstance(runtime, ChargeReceiptRuntime) else None


async def configured_charge_receipts(
    client: "PrismaClient | None",
) -> ChargeReceiptRuntime | ChargeReceiptConfigurationError | None:
    from litellm.proxy.hooks.proxy_track_cost_callback import prepare_required_charge_counters
    from litellm.proxy.utils import ProxyUpdateSpend

    settings: Final = ChargeReceiptSettings()
    if not settings.enabled:
        return None
    error: Final = settings.configuration_error()
    if error is not None:
        return ChargeReceiptConfigurationError(error)
    if client is None or ProxyUpdateSpend.disable_spend_updates():
        return ChargeReceiptConfigurationError("Charge receipts require PostgreSQL and enabled spend updates.")
    runtime: Final = ChargeReceiptRuntime(
        settings,
        ChargeReceiptStore(writer_query(client)),
        before_commit=prepare_required_charge_counters,
    )
    return (
        runtime
        if await runtime.ready()
        else ChargeReceiptConfigurationError("The durable charge receipt schema or synchronous commit is unavailable.")
    )


async def authorize_charge_receipt(
    request: Request,
    auth: UserAPIKeyAuth,
    runtime: ChargeReceiptRuntime | None,
    request_data: object,
) -> None:
    if runtime is None:
        return
    key_hash: Final = required_receipt_key(auth, runtime)
    if key_hash is None:
        return
    if request.scope.get("litellm_websocket_auth"):
        raise HTTPException(400, "WebSocket inference does not support required charge receipts.")
    if request.method in {"GET", "HEAD"} and RouteChecks.is_info_route(request.url.path):
        return
    if request.method == "POST" and request.url.path == "/utils/token_counter":
        return
    if request.method != "POST" or request.url.path not in SUPPORTED_RECEIPT_ROUTES:
        raise HTTPException(400, "This route does not support required charge receipts.")
    token: Final = request.headers.get("x-litellm-billing-context", "")
    if not token or len(token) > 4096:
        raise HTTPException(400, "A signed billing context is required for this key.")
    try:
        features: Final = _RequestFeatures.model_validate(request_data)
        claims: Final = _SignedContext.model_validate(
            jwt.decode(
                token,
                runtime.settings.destination_secret.get_secret_value(),
                algorithms=["HS256"],
                audience="litellm-charge-receipts",
                leeway=30,
                options={
                    "require": [
                        "version",
                        "sub",
                        "aud",
                        "iat",
                        "exp",
                        "gateway_scope",
                        "key_hash",
                        "body_sha256",
                        "method",
                        "route",
                    ]
                },
            )
        )
    except (jwt.PyJWTError, ValidationError):
        raise HTTPException(400, "The billing context is invalid or expired.") from None
    if features.background:
        raise HTTPException(400, "Background inference does not support required charge receipts.")
    if (
        claims.key_hash != key_hash
        or claims.gateway_scope != runtime.settings.gateway_scope
        or claims.exp - claims.iat > 300
        or claims.exp <= claims.iat
        or claims.method != request.method
        or claims.route != request.url.path
        or claims.body_sha256 != sha256(await request.body()).hexdigest()
    ):
        raise HTTPException(400, "The billing context does not match this request.")
    if not await runtime.ready():
        raise HTTPException(503, "Durable charge accounting is temporarily unavailable.")
    request.state.charge_receipt_context = AuthenticatedBillingContext(  # rebind-ok: ASGI state carries authenticated authority into endpoint and stream tasks
        BillingContext(request_id=claims.sub), key_hash
    )


def required_receipt_key(auth: UserAPIKeyAuth, runtime: ChargeReceiptRuntime | None) -> str | None:
    if runtime is None or auth.token is None:
        return None
    return auth.token if auth.token in runtime.settings.key_hashes else None


def bind_charge_receipt(
    request: Request,
    logging_obj: object,
    runtime: ChargeReceiptRuntime | None,
) -> None:
    context: Final[object] = getattr(request.state, "charge_receipt_context", None)
    if not isinstance(context, AuthenticatedBillingContext):
        return
    if runtime is None or not isinstance(logging_obj, Logging):
        raise HTTPException(503, "This inference path cannot durably capture charges.")
    existing: Final = logging_charge_capture(logging_obj)
    if existing is not None and (existing.context != context.context or existing.key_hash != context.key_hash):
        raise HTTPException(503, "The inference billing context changed before dispatch.")
    capture: Final = existing if existing is not None else runtime.bind(context.context, context.key_hash)
    logging_obj.required_charge_capture = capture  # rebind-ok: install the SDK's mandatory capture hook
    request.state.required_charge_capture = capture  # rebind-ok: connect ASGI's response barrier to capture


def logging_charge_capture(logging_obj: object) -> ChargeCapture | None:
    if not isinstance(logging_obj, Logging):
        return None
    capture: Final = logging_obj.required_charge_capture
    return capture if isinstance(capture, ChargeCapture) else None


def request_charge_capture(request: Request | None) -> ChargeCapture | None:
    if request is None:
        return None
    capture: Final[object] = getattr(request.state, "required_charge_capture", None)
    return capture if isinstance(capture, ChargeCapture) else None


async def await_charge_receipt(request: Request | None) -> None:
    capture: Final = request_charge_capture(request)
    if (
        capture is None
        and request is not None
        and isinstance(getattr(request.state, "charge_receipt_context", None), AuthenticatedBillingContext)
    ):
        raise HTTPException(
            503,
            "This inference path did not capture its charge.",
            headers={"x-should-retry": "false"},
        )
    if capture is not None and not await capture.wait():
        raise HTTPException(
            503,
            "The final charge could not be durably recorded. Do not replay this request.",
            headers={"x-should-retry": "false"},
        )
