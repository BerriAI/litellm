"""Per-session local model endpoint every CLI harness talks to.

The runtime inside the sandbox points its Anthropic / OpenAI base URL at this endpoint and
authenticates with a random per-session token. The endpoint either reverse-proxies to a LiteLLM
AI Gateway (gateway mode) or calls the LiteLLM SDK directly (SDK mode), and counts usage + cost.

starlette and uvicorn are optional: they are imported only when an endpoint starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    DEFAULT_POLLING_INTERVAL,
    HARNESS_ENDPOINT_HOST,
    HARNESS_ENDPOINT_REQUEST_TIMEOUT_SECONDS,
    HARNESS_ENDPOINT_STARTUP_TIMEOUT_SECONDS,
    HARNESS_PROCESS_KILL_GRACE_SECONDS,
    HARNESS_SESSION_TOKEN_BYTES,
)
from litellm.harness.context import GatewayTarget
from litellm.harness.errors import HarnessError, HarnessInstallFailed
from litellm.harness.types import Harness, Usage

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

MISSING_DEPS_MESSAGE = "litellm.harness needs starlette and uvicorn: pip install starlette uvicorn"

ROUTE_MESSAGES = "messages"
ROUTE_CHAT = "chat/completions"
ROUTE_RESPONSES = "responses"
POST_ROUTES = (ROUTE_MESSAGES, ROUTE_CHAT, ROUTE_RESPONSES)

HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
DROPPED_REQUEST_HEADERS = HOP_BY_HOP_HEADERS | {
    "authorization",
    "x-api-key",
    "accept-encoding",
}
DROPPED_RESPONSE_HEADERS = HOP_BY_HOP_HEADERS | {"content-encoding"}
COST_HEADER = "x-litellm-response-cost"
SSE_MEDIA_TYPE = "text/event-stream"


@dataclass(frozen=True)
class _ServerDeps:
    uvicorn: ModuleType
    applications: ModuleType
    routing: ModuleType
    responses: ModuleType


def _load_server_deps() -> _ServerDeps:
    """Import starlette + uvicorn on demand; they are not litellm dependencies."""
    try:
        import uvicorn
        from starlette import applications, responses, routing
    except ImportError as e:
        raise HarnessInstallFailed(MISSING_DEPS_MESSAGE) from e
    return _ServerDeps(
        uvicorn=uvicorn,
        applications=applications,
        routing=routing,
        responses=responses,
    )


@dataclass
class UsageTracker:
    """Running token + cost totals for one session."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    calls: int = 0

    def add(self, input_tokens: int = 0, output_tokens: int = 0, cost: float = 0.0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost += cost
        self.calls += 1

    def snapshot(self) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            calls=self.calls,
        )


# ---------------------------------------------------------------------------
# Usage parsing
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def usage_from_mapping(usage: Any) -> tuple[int, int]:
    """(input, output) from a usage dict using OpenAI or Anthropic/Responses field names."""
    if not isinstance(usage, Mapping):
        return 0, 0
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    return _as_int(input_tokens), _as_int(output_tokens)


def usage_from_body(body: Any) -> tuple[int, int]:
    """Usage from a non-streaming JSON response body."""
    if not isinstance(body, Mapping):
        return 0, 0
    if isinstance(body.get("usage"), Mapping):
        return usage_from_mapping(body["usage"])
    response = body.get("response")
    if isinstance(response, Mapping):
        return usage_from_mapping(response.get("usage"))
    return 0, 0


class SSEUsageParser:
    """Collects token usage from an SSE byte stream as it passes through."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self._buffer = b""

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            self._feed_line(line)

    def close(self) -> None:
        if self._buffer:
            self._feed_line(self._buffer)
            self._buffer = b""

    def _feed_line(self, line: bytes) -> None:
        text = line.strip()
        if not text.startswith(b"data:"):
            return
        payload = text[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            event = json.loads(payload)
        except ValueError:
            return
        if isinstance(event, Mapping):
            self.absorb(event)

    def absorb(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "message_start":
            self._absorb_message_start(event)
        elif event_type == "message_delta":
            self._absorb_message_delta(event)
        elif event_type == "response.completed":
            self._absorb_response_completed(event)
        elif isinstance(event.get("usage"), Mapping):
            self._set(*usage_from_mapping(event["usage"]))

    def _absorb_message_start(self, event: Mapping[str, Any]) -> None:
        message = event.get("message")
        if isinstance(message, Mapping):
            self._set(*usage_from_mapping(message.get("usage")))

    def _absorb_message_delta(self, event: Mapping[str, Any]) -> None:
        # message_delta output_tokens is cumulative for the whole message.
        self._set(*usage_from_mapping(event.get("usage")))

    def _absorb_response_completed(self, event: Mapping[str, Any]) -> None:
        response = event.get("response")
        if isinstance(response, Mapping):
            self._set(*usage_from_mapping(response.get("usage")))

    def _set(self, input_tokens: int, output_tokens: int) -> None:
        if input_tokens:
            self.input_tokens = input_tokens
        if output_tokens:
            self.output_tokens = output_tokens


# ---------------------------------------------------------------------------
# Cost + helpers
# ---------------------------------------------------------------------------


def compute_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    """Cost from LiteLLM's price map. Never raises; unknown models cost 0.0."""
    if not model or not (input_tokens or output_tokens):
        return 0.0
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception as e:  # accounting must never break a call
        verbose_logger.debug("harness endpoint: cost lookup failed for %s: %s", model, e)
        return 0.0


def header_cost(headers: Mapping[str, str]) -> float | None:
    raw = headers.get(COST_HEADER)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def hidden_cost(response: Any) -> float | None:
    hidden = getattr(response, "_hidden_params", None)
    if not isinstance(hidden, Mapping):
        return None
    try:
        cost = hidden.get("response_cost")
        return None if cost is None else float(cost)
    except (TypeError, ValueError):
        return None


def extract_token(headers: Mapping[str, str]) -> str | None:
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[len("bearer ") :].strip()
    return headers.get("x-api-key")


def gateway_headers(
    incoming: Mapping[str, str],
    gateway: GatewayTarget,
    harness: Harness,
    metadata: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Incoming headers minus hop-by-hop/auth/x-litellm-*, plus gateway auth, tags, metadata."""
    headers = {
        name: value
        for name, value in incoming.items()
        if name.lower() not in DROPPED_REQUEST_HEADERS and not name.lower().startswith("x-litellm-")
    }
    headers["authorization"] = f"Bearer {gateway.api_key}"
    headers["x-litellm-tags"] = f"harness,{harness.value}"
    if metadata:
        headers["x-litellm-spend-logs-metadata"] = json.dumps(dict(metadata), default=str)
    return headers


def response_headers(upstream: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in upstream.items() if name.lower() not in DROPPED_RESPONSE_HEADERS}


def sanitize(message: str, secret_values: tuple[str | None, ...]) -> str:
    for value in secret_values:
        if value:
            message = message.replace(value, "***")
    return message


def error_status(exc: BaseException) -> int:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status <= 599:
        return status
    return 500


def error_body(exc: BaseException, message: str) -> dict[str, Any]:
    return {"error": {"type": type(exc).__name__, "message": message}}


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, Mapping):
        return dict(obj)
    return obj


def encode_anthropic_chunk(chunk: Any) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, str):
        return chunk.encode()
    data = to_jsonable(chunk)
    event_type = data.get("type", "message") if isinstance(data, Mapping) else "message"
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def encode_chat_chunk(chunk: Any) -> bytes:
    if hasattr(chunk, "model_dump_json"):
        return f"data: {chunk.model_dump_json()}\n\n".encode()
    return f"data: {json.dumps(to_jsonable(chunk))}\n\n".encode()


def encode_responses_chunk(chunk: Any) -> bytes:
    data = to_jsonable(chunk)
    event_type = data.get("type", "message") if isinstance(data, Mapping) else "message"
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


STREAM_ENCODERS = {
    ROUTE_MESSAGES: encode_anthropic_chunk,
    ROUTE_CHAT: encode_chat_chunk,
    ROUTE_RESPONSES: encode_responses_chunk,
}
STREAM_TRAILERS = {ROUTE_CHAT: b"data: [DONE]\n\n"}


def route_of(path: str) -> str:
    stripped = path.strip("/")
    stripped = stripped.removeprefix("v1/")
    return stripped


def _noop() -> None:
    return None


# ---------------------------------------------------------------------------
# ModelEndpoint
# ---------------------------------------------------------------------------


class ModelEndpoint:
    """Local HTTP endpoint for one harness session. Use as an async context manager."""

    def __init__(
        self,
        harness: Harness,
        model: str | None,
        gateway: GatewayTarget | None,
        api_key: str | None = None,
        api_base: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.harness = harness
        self.model = model
        self.gateway = gateway
        self.api_key = api_key
        self.api_base = api_base
        self.metadata = dict(metadata) if metadata else {}
        self.token = secrets.token_urlsafe(HARNESS_SESSION_TOKEN_BYTES)
        self.usage = UsageTracker()
        self.port = 0
        self._transport = transport
        self._deps: _ServerDeps | None = None
        self._client: httpx.AsyncClient | None = None
        self._server: Any = None
        self._task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        return f"http://{HARNESS_ENDPOINT_HOST}:{self.port}"

    # -- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> ModelEndpoint:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def start(self) -> None:
        self._deps = _load_server_deps()
        if self.gateway is not None:
            self._client = httpx.AsyncClient(
                timeout=HARNESS_ENDPOINT_REQUEST_TIMEOUT_SECONDS,
                transport=self._transport,
            )
        self._server = self._build_server(self._deps)
        self._task = asyncio.create_task(self._server.serve())
        try:
            await asyncio.wait_for(self._wait_started(), HARNESS_ENDPOINT_STARTUP_TIMEOUT_SECONDS)
        except BaseException:
            await self.stop()
            raise
        self.port = self._server.servers[0].sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(BaseException):
                await self._task
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _wait_started(self) -> None:
        while not self._server.started:
            if self._task is not None and self._task.done():
                raise HarnessError("harness model endpoint failed to start")
            await asyncio.sleep(DEFAULT_POLLING_INTERVAL)

    def _build_server(self, deps: _ServerDeps) -> Any:
        config = deps.uvicorn.Config(
            self._build_app(deps),
            host=HARNESS_ENDPOINT_HOST,
            port=0,
            log_config=None,
            log_level="warning",
            access_log=False,
            lifespan="off",
            timeout_graceful_shutdown=HARNESS_PROCESS_KILL_GRACE_SECONDS,
        )
        server = deps.uvicorn.Server(config)
        # Never touch the host process's signal handlers.
        if hasattr(server, "capture_signals"):
            server.capture_signals = contextlib.nullcontext
        if hasattr(server, "install_signal_handlers"):
            server.install_signal_handlers = _noop
        return server

    def _build_app(self, deps: _ServerDeps) -> Any:
        Route = deps.routing.Route
        routes = []
        for prefix in ("", "/v1"):
            for route in POST_ROUTES:
                routes.append(Route(f"{prefix}/{route}", self._handle, methods=["POST"]))
            routes.append(Route(f"{prefix}/models", self._models, methods=["GET"]))
        return deps.applications.Starlette(routes=routes)

    # -- request handling ---------------------------------------------------

    @property
    def _responses(self) -> ModuleType:
        if self._deps is None:
            raise HarnessError("harness model endpoint is not started")
        return self._deps.responses

    def _authorized(self, request: Request) -> bool:
        token = extract_token(request.headers)
        return token is not None and secrets.compare_digest(token.encode(), self.token.encode())

    def _json(self, body: Any, status_code: int = 200) -> Response:
        return self._responses.JSONResponse(body, status_code=status_code)

    def _unauthorized(self) -> Response:
        return self._json(
            {"error": {"type": "authentication_error", "message": "invalid token"}},
            401,
        )

    def _error(self, exc: BaseException, status_code: int | None = None) -> Response:
        message = sanitize(str(exc), self._secrets())
        return self._json(error_body(exc, message), status_code or error_status(exc))

    def _secrets(self) -> tuple[str | None, ...]:
        gateway_key = self.gateway.api_key if self.gateway else None
        return (gateway_key, self.api_key, self.token)

    async def _models(self, request: Request) -> Response:
        if not self._authorized(request):
            return self._unauthorized()
        data = []
        if self.model:
            data.append(
                {
                    "id": self.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "litellm",
                }
            )
        return self._json({"object": "list", "data": data})

    async def _handle(self, request: Request) -> Response:
        if not self._authorized(request):
            return self._unauthorized()
        try:
            body = json.loads(await request.body())
        except ValueError as e:
            return self._error(e, 400)
        if not isinstance(body, dict):
            return self._error(ValueError("request body must be a JSON object"), 400)
        route = route_of(request.url.path)
        if self.gateway is not None:
            return await self._forward(request, route, body)
        return await self._call_sdk(route, body)

    def _cost_model(self, body: Mapping[str, Any]) -> str | None:
        model = self.model or body.get("model")
        return model if isinstance(model, str) else None

    def _record(
        self,
        model: str | None,
        input_tokens: int,
        output_tokens: int,
        cost: float | None,
    ) -> None:
        if cost is None:
            cost = compute_cost(model, input_tokens, output_tokens)
        self.usage.add(input_tokens, output_tokens, cost)

    # -- gateway mode -------------------------------------------------------

    async def _forward(self, request: Request, route: str, body: dict[str, Any]) -> Response:
        if self._client is None or self.gateway is None:
            raise HarnessError("gateway client is not started")
        if self.model:
            body = {**body, "model": self.model}
        upstream_request = self._client.build_request(
            "POST",
            f"{self.gateway.api_base}/v1/{route}",
            json=body,
            headers=gateway_headers(request.headers, self.gateway, self.harness, self.metadata),
        )
        try:
            upstream = await self._client.send(upstream_request, stream=True)
        except httpx.HTTPError as e:
            return self._error(e, 502)
        return self._responses.StreamingResponse(
            self._relay(upstream, self._cost_model(body)),
            status_code=upstream.status_code,
            headers=response_headers(upstream.headers),
        )

    async def _relay(self, upstream: httpx.Response, model: str | None) -> AsyncIterator[bytes]:
        is_sse = SSE_MEDIA_TYPE in upstream.headers.get("content-type", "")
        parser = SSEUsageParser()
        collected = bytearray()
        try:
            async for chunk in upstream.aiter_bytes():
                if is_sse:
                    parser.feed(chunk)
                else:
                    collected.extend(chunk)
                yield chunk
        finally:
            await upstream.aclose()
            if upstream.status_code < 400:
                self._record_relayed(upstream, model, parser, is_sse, bytes(collected))

    def _record_relayed(
        self,
        upstream: httpx.Response,
        model: str | None,
        parser: SSEUsageParser,
        is_sse: bool,
        collected: bytes,
    ) -> None:
        if is_sse:
            parser.close()
            tokens = (parser.input_tokens, parser.output_tokens)
        else:
            try:
                tokens = usage_from_body(json.loads(collected))
            except ValueError:
                tokens = (0, 0)
        self._record(model, tokens[0], tokens[1], header_cost(upstream.headers))

    # -- SDK mode -----------------------------------------------------------

    def _sdk_kwargs(self, body: Mapping[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {**body}
        if self.model:
            kwargs["model"] = self.model
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    async def _invoke_sdk(self, route: str, kwargs: dict[str, Any]) -> Any:
        if route == ROUTE_MESSAGES:
            return await litellm.anthropic.messages.acreate(**kwargs)
        if route == ROUTE_CHAT:
            if kwargs.get("stream"):
                stream_options = kwargs.get("stream_options") or {}
                kwargs["stream_options"] = {"include_usage": True, **stream_options}
            return await litellm.acompletion(**kwargs)
        return await litellm.aresponses(**kwargs)

    async def _call_sdk(self, route: str, body: Mapping[str, Any]) -> Response:
        kwargs = self._sdk_kwargs(body)
        model = self._cost_model(kwargs)
        try:
            response = await self._invoke_sdk(route, kwargs)
        except Exception as e:
            verbose_logger.debug("harness endpoint: SDK call failed: %s", type(e).__name__)
            return self._error(e)
        if kwargs.get("stream"):
            return self._responses.StreamingResponse(
                self._sdk_stream(route, response, model), media_type=SSE_MEDIA_TYPE
            )
        data = to_jsonable(response)
        input_tokens, output_tokens = usage_from_body(data)
        self._record(model, input_tokens, output_tokens, hidden_cost(response))
        return self._json(data)

    async def _sdk_stream(self, route: str, iterator: Any, model: str | None) -> AsyncIterator[bytes]:
        encode = STREAM_ENCODERS[route]
        parser = SSEUsageParser()
        try:
            async for chunk in iterator:
                encoded = encode(chunk)
                parser.feed(encoded)
                yield encoded
            trailer = STREAM_TRAILERS.get(route)
            if trailer:
                yield trailer
        except Exception as e:
            message = sanitize(str(e), self._secrets())
            yield f"event: error\ndata: {json.dumps(error_body(e, message))}\n\n".encode()
        finally:
            parser.close()
            self._record(model, parser.input_tokens, parser.output_tokens, None)
