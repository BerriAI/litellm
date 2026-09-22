from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Set
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Final, Protocol, TypedDict, runtime_checkable
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Request
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from starlette.middleware import Middleware
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from typing_extensions import ReadOnly

from litellm.proxy._lazy_features import LazyFeatureMiddleware
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableRequestMetricsMiddleware
from litellm.proxy.middleware.budget_reservation_release_middleware import BudgetReservationReleaseMiddleware

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_RESPONSE_LIMIT: Final = 128 * 1024
_CREDENTIAL_HEADERS: Final = frozenset({"authorization", "x-litellm-api-key"})
_INTERNAL_MIDDLEWARE: Final[frozenset[object]] = frozenset(
    {BudgetReservationReleaseMiddleware, BillableRequestMetricsMiddleware}
)


class _RequestContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    app: FastAPI
    root_path: str = ""
    scheme: str = "http"
    http_version: str = "1.1"
    client: tuple[str, int] | None = None
    server: tuple[str, int | None] | None = None


class _AuthSettings(BaseModel):
    litellm_key_header_name: str | None = None


class _ProxySettings(BaseModel):
    general_settings: _AuthSettings


class _ErrorBody(TypedDict):
    error: ReadOnly[str]


@runtime_checkable
class _MiddlewareLayer(Protocol):
    @property
    def app(self) -> object: ...


@dataclass(frozen=True, slots=True)
class DispatchResult:
    status_code: int
    data: JsonValue


def _error(status_code: int, message: str) -> DispatchResult:
    data: Final[_ErrorBody] = {"error": message}
    return DispatchResult(status_code, _JSON.validate_python(data))


def _credential_headers(request: Request) -> tuple[tuple[bytes, bytes], ...]:
    from litellm.proxy import proxy_server

    configured: Final = _ProxySettings.model_validate(vars(proxy_server)).general_settings.litellm_key_header_name
    allowed: Final = _CREDENTIAL_HEADERS | frozenset((configured.lower(),)) if configured else _CREDENTIAL_HEADERS
    return tuple((name, value) for name, value in request.headers.raw if name.decode("latin-1").lower() in allowed)


def credential_fingerprint(request: Request) -> str | None:
    credentials: Final = tuple(
        sorted(((name.lower(), value) for name, value in _credential_headers(request)), key=lambda item: item[0])
    )
    if not credentials:
        return None
    encoded: Final = b"".join(
        len(name).to_bytes(4, "big") + name + len(value).to_bytes(4, "big") + value for name, value in credentials
    )
    return hashlib.sha256(encoded).hexdigest()


def credential_secrets(request: Request) -> tuple[str, ...]:
    values: Final = tuple(value.decode("latin-1") for _, value in _credential_headers(request))
    return tuple(
        dict.fromkeys(
            secret
            for value in values
            for secret in (value, value[7:].strip() if value[:7].lower() == "bearer " else value)
            if secret
        )
    )


def _child_scope(
    request: Request,
    context: _RequestContext,
    method: str,
    path: str,
    body: bytes,
    query: tuple[tuple[str, str], ...],
) -> Scope:
    root_path: Final = context.root_path.rstrip("/")
    full_path: Final = root_path + path
    return {  # mutable-ok: ASGI scope must let FastAPI stamp route and dependency state
        "type": "http",
        "asgi": MappingProxyType({"version": "3.0"}),
        "http_version": context.http_version,
        "method": method,
        "scheme": context.scheme,
        "path": full_path,
        "raw_path": full_path.encode("utf-8"),
        "query_string": urlencode(query).encode("ascii"),
        "root_path": root_path,
        "client": context.client,
        "server": context.server,
        "app": context.app,
        "state": {},  # mutable-ok: Starlette request.state writes into this request-owned mapping
        "headers": (
            *_credential_headers(request),
            *((name, value) for name, value in request.headers.raw if name.lower() == b"host"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ),
    }


@dataclass(slots=True)
class _Exchange:
    request_body: bytes
    status_code: int = 500
    response_body: bytes = b""
    too_large: bool = False
    request_sent: bool = False
    complete: asyncio.Event = field(default_factory=asyncio.Event)

    async def receive(self) -> Message:
        if not self.request_sent:
            self.request_sent = True
            return {  # mutable-ok: ASGI receive requires a mutable Message mapping
                "type": "http.request",
                "body": self.request_body,
                "more_body": False,
            }
        await self.complete.wait()
        return {"type": "http.disconnect"}  # mutable-ok: ASGI receive requires a mutable Message mapping

    async def send(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self.status_code = message["status"]
        if message["type"] != "http.response.body":
            return
        chunk: Final = TypeAdapter(bytes).validate_python(message.get("body", b""))
        if len(self.response_body) + len(chunk) > _RESPONSE_LIMIT:
            self.too_large = True
        if not self.too_large:
            self.response_body += chunk
        if not message.get("more_body", False):
            self.complete.set()

    def result(self) -> DispatchResult:
        if self.too_large:
            return _error(502, "The gateway response was too large. Use narrower filters.")
        if not self.complete.is_set() or 300 <= self.status_code < 400:
            return _error(502, "The gateway returned an unsupported response.")
        if not self.response_body and self.status_code == 204:
            return DispatchResult(204, None)
        try:
            data: Final = _JSON.validate_json(self.response_body)
        except ValidationError:
            return _error(502, "The gateway returned an unsupported response.")
        return DispatchResult(self.status_code, data)


async def _target_route(app: FastAPI, scope: Scope, path: str) -> APIRoute | None:
    exchange: Final = _Exchange(b"")
    await LazyFeatureMiddleware(_loaded, fastapi_app=app)(scope, exchange.receive, exchange.send)
    existing: Final = next(
        (route for route in app.router.routes if isinstance(route, APIRoute) and route.matches(scope)[0] is Match.FULL),
        None,
    )
    if existing is not None:
        return existing
    if path != "/chat/completions" or scope["method"] != "POST":
        return None
    from litellm.proxy.proxy_server import chat_completion

    return APIRoute(
        "/chat/completions",
        chat_completion,
        methods=["POST"],  # mutable-ok: APIRoute requires a list or set for methods
        dependencies=(Depends(user_api_key_auth),),
        dependency_overrides_provider=app,
    )


async def _loaded(scope: Scope, receive: Receive, send: Send) -> None:
    return


async def _invoke_route(scope: Scope, receive: Receive, send: Send) -> None:
    route: Final = scope.get("liteask_route")
    if not isinstance(route, APIRoute):
        return
    _, matched_scope = route.matches(scope)
    await route.handle(
        {**scope, **matched_scope},  # mutable-ok: ASGI route handling stamps the child scope
        receive,
        send,
    )


def _billing_owner(layer: object, remaining: int = 32) -> BillableRequestMetricsMiddleware | None:
    if isinstance(layer, BillableRequestMetricsMiddleware):
        return layer
    if remaining == 0 or not isinstance(layer, _MiddlewareLayer):
        return None
    return _billing_owner(layer.app, remaining - 1)


@lru_cache(maxsize=32)
def _internal_stack(app: FastAPI, billing: BillableRequestMetricsMiddleware | None) -> ASGIApp:
    internal: Final = FastAPI(
        middleware=tuple(
            Middleware(BillableRequestMetricsMiddleware, recorder=billing.recorder, sink=billing.sink)
            if entry.cls is BillableRequestMetricsMiddleware and billing is not None
            else entry
            for entry in app.user_middleware
            if entry.cls in _INTERNAL_MIDDLEWARE
        ),
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    internal.exception_handlers = app.exception_handlers
    internal.router.default = _invoke_route
    return internal.build_middleware_stack()


async def dispatch(
    request: Request,
    method: str,
    path: str,
    body: JsonValue = None,
    query: tuple[tuple[str, str], ...] = (),
    *,
    allowed_routes: Set[tuple[str, str]],
) -> DispatchResult:
    if (method, path) not in allowed_routes:
        return _error(403, "This operation is not available in LiteAsk.")
    try:
        context: Final = _RequestContext.model_validate(request.scope)
    except ValidationError:
        return _error(503, "The gateway is not ready for LiteAsk.")
    payload: Final = _JSON.dump_json(body) if body is not None else b""
    scope: Final = _child_scope(request, context, method, path, payload, query)
    route: Final = await _target_route(context.app, scope, path)
    if route is None:
        return _error(404, "This operation is not available on this gateway.")
    billing: Final = _billing_owner(context.app.middleware_stack)
    if billing is None and any(entry.cls is BillableRequestMetricsMiddleware for entry in context.app.user_middleware):
        return _error(503, "The gateway accounting service is not ready.")
    exchange: Final = _Exchange(payload)
    try:
        await _internal_stack(context.app, billing)(
            {**scope, "liteask_route": route},  # mutable-ok: private ASGI scope binds this call to its checked route
            exchange.receive,
            exchange.send,
        )
    except Exception:  # noqa: BLE001  # handlers render known failures; otherwise keep internal errors private
        if not exchange.complete.is_set():
            return _error(502, "The gateway operation could not be completed.")
    return exchange.result()
