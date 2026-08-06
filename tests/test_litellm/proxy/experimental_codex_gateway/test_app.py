import asyncio
import hashlib
import json
import os
import stat
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.experimental_codex_gateway.app import ReplayedReceive, create_gateway_app, map_responses_path
from litellm.proxy.experimental_codex_gateway.capture import TraceStore, replay_response_chunks
from litellm.proxy.experimental_codex_gateway.pipeline import RequestPipeline, StageResult
from litellm.proxy.experimental_codex_gateway.settings import GatewaySettings


class FakeInnerApp:
    def __init__(self, status: int = 200, chunks: tuple[bytes, ...] = (b"ok",)) -> None:
        self.status = status
        self.chunks = chunks
        self.scope: Scope | None = None
        self.body = b""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.scope = scope
        request = await receive()
        self.body = request.get("body", b"")
        await send(
            {
                "type": "http.response.start",
                "status": self.status,
                "headers": ((b"content-type", b"text/event-stream"), (b"retry-after", b"3")),
            }
        )
        for index, chunk in enumerate(self.chunks):
            await send({"type": "http.response.body", "body": chunk, "more_body": index < len(self.chunks) - 1})


class EchoInnerApp:
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = await receive()
        body = request.get("body", b"")
        await asyncio.sleep(0)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": ((b"content-type", b"application/json"),),
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


class FragmentedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.blocked = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"event: response.created\ndata: {}\n\n"
        self.blocked.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def aclose(self) -> None:
        self.closed.set()


@dataclass(frozen=True, slots=True)
class FakeReadinessProbe:
    result: bool

    async def reachable(self, url: str) -> bool:
        return self.result


@dataclass(frozen=True, slots=True)
class RaisingStage:
    def apply(self, body: bytes, content_type: str) -> StageResult:
        raise RuntimeError("stage failed")


@dataclass(frozen=True, slots=True)
class RaisingSecretStage:
    message: str

    def apply(self, body: bytes, content_type: str) -> StageResult:
        raise RuntimeError(self.message)


def _settings(tmp_path: Path, capture: bool = False) -> GatewaySettings:
    return GatewaySettings(
        local_key="local-gateway-key-123",
        capture_enabled=capture,
        trace_directory=tmp_path,
        max_trace_bytes=10_000,
        max_trace_storage_bytes=20_000,
        trace_retention_seconds=60,
    )


def test_map_responses_path_and_reject_traversal() -> None:
    assert map_responses_path("/v1/responses") == "/chatgpt/responses"
    assert map_responses_path("/v1/responses/resp_1/cancel") == "/chatgpt/responses/resp_1/cancel"
    assert map_responses_path("/v1/chat/completions") is None
    assert map_responses_path("/v1/responses/../secrets") is None
    assert map_responses_path("/v1/responses/%2e%2e/secrets", b"/v1/responses/%2e%2e/secrets") is None


@pytest.mark.asyncio
async def test_maps_request_and_preserves_body_query_and_stream_chunks(tmp_path: Path) -> None:
    body = b'{ "model": "gpt-5.3-codex", "stream": true }'
    chunks = (b"event: response.created\n", b"data: {}\n\n", b"event: response.completed\n\n")
    inner = FakeInnerApp(chunks=chunks)
    app = create_gateway_app(settings=_settings(tmp_path), inner_app=inner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post(
            "/v1/responses/resp_1/cancel?mode=exact",
            content=body,
            headers={
                "content-type": "application/json",
                "authorization": "Bearer oauth-secret",
                "chatgpt-account-id": "account-secret",
                "x-litellm-api-key": "local-gateway-key-123",
            },
        )
    assert response.status_code == 200
    assert response.content == b"".join(chunks)
    assert inner.body == body
    assert inner.scope is not None
    assert inner.scope["path"] == "/chatgpt/responses/resp_1/cancel"
    assert inner.scope["query_string"] == b"mode=exact"
    assert dict(inner.scope["headers"])[b"authorization"] == b"Bearer oauth-secret"
    assert dict(inner.scope["headers"])[b"chatgpt-account-id"] == b"account-secret"


@pytest.mark.parametrize("status_code", (401, 429, 503))
@pytest.mark.asyncio
async def test_real_chatgpt_route_relays_headroom_failures_without_gateway_credentials(
    tmp_path: Path, status_code: int
) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import app as litellm_app

    request_body = b'{ "model": "gpt-5.3-codex", "stream": false, "unknown": {"encrypted_reasoning":"opaque"} }'
    response_body = json.dumps({"error": {"status": status_code}}, separators=(",", ":")).encode()
    downstream_requests: list[httpx.Request] = []

    async def headroom_transport(request: httpx.Request) -> httpx.Response:
        downstream_requests.append(request)
        return httpx.Response(
            status_code=status_code,
            headers={"content-type": "application/json", "retry-after": "7"},
            content=response_body,
            request=request,
        )

    downstream_client = httpx.AsyncClient(transport=httpx.MockTransport(headroom_transport))
    app = create_gateway_app(settings=_settings(tmp_path), inner_app=litellm_app)
    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed")),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
            return_value=SimpleNamespace(client=downstream_client),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
            new=AsyncMock(side_effect=lambda **kwargs: kwargs["data"]),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_response_headers_hook",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new=AsyncMock(),
        ),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            response = await client.post(
                "/v1/responses?include=usage&include=trace&flag=&encoded=a%2Fb",
                content=request_body,
                headers={
                    "content-type": "application/json",
                    "authorization": "Bearer oauth-secret",
                    "chatgpt-account-id": "account-secret",
                    "cookie": "session=cookie-secret",
                    "x-litellm-api-key": "local-gateway-key-123",
                    "connection": "x-remove-me",
                    "x-remove-me": "hop-secret",
                    "content-length": "999",
                    "traceparent": "00-trace-parent",
                },
            )
    await downstream_client.aclose()

    assert response.status_code == status_code
    assert response.content == response_body
    assert response.headers["retry-after"] == "7"
    assert len(downstream_requests) == 1
    downstream = downstream_requests[0]
    assert str(downstream.url) == ("http://127.0.0.1:8787/v1/responses?include=usage&include=trace&flag=&encoded=a%2Fb")
    assert downstream.content == request_body
    assert downstream.headers["authorization"] == "Bearer oauth-secret"
    assert downstream.headers["chatgpt-account-id"] == "account-secret"
    assert downstream.headers["traceparent"] == "00-trace-parent"
    assert downstream.headers["content-length"] == str(len(request_body))
    assert "x-litellm-api-key" not in downstream.headers
    assert "cookie" not in downstream.headers
    assert "x-remove-me" not in downstream.headers


@pytest.mark.asyncio
async def test_real_chatgpt_route_preserves_fragmented_sse_events(tmp_path: Path) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import app as litellm_app

    request_body = b'{"model":"gpt-5.3-codex","stream":true}'
    chunks = (
        b'event: response.output_item.added\ndata: {"item":{"type":"function_call","call_id":"call_1"}}\n\n',
        b'event: response.reasoning.delta\ndata: {"encrypted_content":"opaque"}\n\n',
        b'event: vendor.unknown\ndata: {"future":true}\n\n',
    )
    downstream_requests: list[httpx.Request] = []

    async def headroom_transport(request: httpx.Request) -> httpx.Response:
        downstream_requests.append(request)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=FragmentedStream(chunks),
            request=request,
        )

    downstream_client = httpx.AsyncClient(transport=httpx.MockTransport(headroom_transport))
    app = create_gateway_app(settings=_settings(tmp_path), inner_app=litellm_app)
    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed")),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
            return_value=SimpleNamespace(client=downstream_client),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
            new=AsyncMock(side_effect=lambda **kwargs: kwargs["data"]),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_response_headers_hook",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new=AsyncMock(),
        ),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            response = await client.post(
                "/v1/responses",
                content=request_body,
                headers={
                    "content-type": "application/json",
                    "authorization": "Bearer oauth-secret",
                    "x-litellm-api-key": "local-gateway-key-123",
                },
            )
    await downstream_client.aclose()

    assert response.status_code == 200
    assert response.content == b"".join(chunks)
    assert len(downstream_requests) == 1
    assert downstream_requests[0].content == request_body


@pytest.mark.asyncio
async def test_client_disconnect_cancels_and_closes_in_flight_headroom_stream(tmp_path: Path) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import app as litellm_app

    request_body = b'{"model":"gpt-5.3-codex","stream":true}'
    downstream_stream = BlockingStream()

    async def headroom_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=downstream_stream,
            request=request,
        )

    downstream_client = httpx.AsyncClient(transport=httpx.MockTransport(headroom_transport))
    app = create_gateway_app(settings=_settings(tmp_path), inner_app=litellm_app)
    request_delivered = False
    sent_messages: list[Message] = []

    async def receive() -> Message:
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": request_body, "more_body": False}
        await downstream_stream.blocked.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent_messages.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "root_path": "",
        "headers": (
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer oauth-secret"),
            (b"x-litellm-api-key", b"local-gateway-key-123"),
        ),
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 4000),
    }
    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed")),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
            return_value=SimpleNamespace(client=downstream_client),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
            new=AsyncMock(side_effect=lambda **kwargs: kwargs["data"]),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_response_headers_hook",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new=AsyncMock(),
        ),
    ):
        await asyncio.wait_for(app(scope, receive, send), timeout=3)

    assert any(message["type"] == "http.response.start" for message in sent_messages)
    assert downstream_stream.cancelled.is_set()
    assert downstream_stream.closed.is_set()
    await downstream_client.aclose()


@pytest.mark.asyncio
async def test_operational_routes_and_local_key_auth(tmp_path: Path) -> None:
    app = create_gateway_app(
        settings=_settings(tmp_path),
        inner_app=FakeInnerApp(),
        readiness_probe=FakeReadinessProbe(result=False),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        health = await client.get("/healthz")
        ready = await client.get("/readyz")
        unauthorized_metrics = await client.get("/metrics")
        metrics = await client.get("/metrics", headers={"x-litellm-api-key": "local-gateway-key-123"})
    assert health.status_code == 204 and health.content == b""
    assert ready.status_code == 503 and ready.content == b""
    assert unauthorized_metrics.status_code == 401
    assert metrics.status_code == 200
    assert b"litellm_codex_gateway_requests_total" in metrics.content


@pytest.mark.asyncio
async def test_capture_redacts_credentials_across_chunks_and_replays_deterministically(tmp_path: Path) -> None:
    chunks = (b'data: {"token":"Bearer oauth-', b'secret","email":"person@example.com"}\n\n')
    inner = FakeInnerApp(status=429, chunks=chunks)
    settings = _settings(tmp_path, capture=True)
    app = create_gateway_app(settings=settings, inner_app=inner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post(
            "/v1/responses",
            content=json.dumps(
                {
                    "input": "contact person@example.com",
                    "authorization": "Bearer request-secret",
                    "local_key": settings.local_key,
                    "owner": "785d0158-2d4e-4ba4-b290-a15b042af9a0",
                }
            ),
            headers={
                "content-type": "application/json",
                "authorization": "Bearer oauth-secret",
                "chatgpt-account-id": "account-secret",
                "cookie": "session=cookie-secret",
                "x-litellm-api-key": settings.local_key,
                "traceparent": "Bearer trace-secret",
            },
        )
    assert response.status_code == 429
    trace_path = next(tmp_path.glob("*.json"))
    trace = trace_path.read_bytes()
    for secret in (
        b"oauth-secret",
        b"request-secret",
        b"account-secret",
        b"cookie-secret",
        settings.local_key.encode(),
        b"person@example.com",
        b"785d0158-2d4e-4ba4-b290-a15b042af9a0",
        b"trace-secret",
    ):
        assert secret not in trace
    replayed = replay_response_chunks(trace)
    assert replayed is not None
    assert b"".join(replayed) != b"".join(chunks)
    assert len(replayed) == len(chunks)
    exported = json.loads(trace)
    trace_id = exported["trace_id"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        unauthorized = await client.get(f"/debug/traces/{trace_id}/export")
        authorized = await client.get(
            f"/debug/traces/{trace_id}/export", headers={"x-litellm-api-key": settings.local_key}
        )
    assert unauthorized.status_code == 401
    assert authorized.content == trace


@pytest.mark.asyncio
async def test_seeded_secrets_absent_from_traces_metrics_logs_and_stage_exceptions(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings(tmp_path, capture=True)
    secrets = (
        "oauth-seeded-value",
        "sk-seeded-api-key",
        "account-seeded-value",
        "cookie-seeded-value",
        settings.local_key,
        "seeded.person@example.com",
        "+1 (415) 555-0182",
        "785d0158-2d4e-4ba4-b290-a15b042af9a0",
    )
    request_body = json.dumps(
        {
            "authorization": f"Bearer {secrets[0]}",
            "api_key": secrets[1],
            "account_id": secrets[2],
            "cookie": secrets[3],
            "local_key": secrets[4],
            "email": secrets[5],
            "phone": secrets[6],
            "owner": secrets[7],
        },
        separators=(",", ":"),
    ).encode()
    response_body = json.dumps(
        {
            "authorization": f"Bearer {secrets[0]}",
            "api_key": secrets[1],
            "account_id": secrets[2],
            "cookie": secrets[3],
        },
        separators=(",", ":"),
    ).encode()
    app = create_gateway_app(
        settings=settings,
        inner_app=FakeInnerApp(chunks=(response_body,)),
        pipeline=RequestPipeline((RaisingSecretStage("forced stage exception: " + " ".join(secrets)),)),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post(
            "/v1/responses",
            content=request_body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {secrets[0]}",
                "chatgpt-account-id": secrets[2],
                "cookie": f"session={secrets[3]}",
                "x-litellm-api-key": settings.local_key,
                "traceparent": f"Bearer {secrets[0]}",
            },
        )
        metrics = await client.get("/metrics", headers={"x-litellm-api-key": settings.local_key})

    captured = capsys.readouterr()
    traces = b"\n".join(path.read_bytes() for path in tmp_path.glob("*.json"))
    artifacts = b"\n".join(
        (traces, metrics.content, caplog.text.encode(), captured.out.encode(), captured.err.encode())
    )
    assert response.content == response_body
    assert b"litellm_codex_gateway_fail_open_total 1" in metrics.content
    assert traces
    for secret in secrets:
        assert secret.encode() not in artifacts


@pytest.mark.asyncio
async def test_concurrent_captures_remain_isolated_and_complete(tmp_path: Path) -> None:
    request_bodies = tuple(
        json.dumps({"index": index, "secret": f"Bearer concurrent-secret-{index}"}, separators=(",", ":")).encode()
        for index in range(16)
    )
    settings = GatewaySettings(
        local_key="local-gateway-key-123",
        capture_enabled=True,
        trace_directory=tmp_path,
        max_trace_bytes=10_000,
        max_trace_storage_bytes=100_000,
        trace_retention_seconds=60,
    )
    app = create_gateway_app(settings=settings, inner_app=EchoInnerApp())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    "/v1/responses",
                    content=body,
                    headers={"content-type": "application/json", "x-litellm-api-key": settings.local_key},
                )
                for body in request_bodies
            )
        )
        metrics = await client.get("/metrics", headers={"x-litellm-api-key": settings.local_key})

    traces = tuple(json.loads(path.read_bytes()) for path in tmp_path.glob("*.json"))
    expected_hashes = {hashlib.sha256(body).hexdigest() for body in request_bodies}
    captured_hashes = {trace["request"]["body_sha256"] for trace in traces}
    assert tuple(response.content for response in responses) == request_bodies
    assert len(traces) == len(request_bodies)
    assert len({trace["trace_id"] for trace in traces}) == len(request_bodies)
    assert captured_hashes == expected_hashes
    assert all("concurrent-secret" not in json.dumps(trace) for trace in traces)
    assert b"litellm_codex_gateway_requests_total 16" in metrics.content
    assert b"litellm_codex_gateway_capture_drops_total 0" in metrics.content


@pytest.mark.asyncio
async def test_capture_redacts_escaped_json_secrets_and_rejects_malformed_export(tmp_path: Path) -> None:
    inner = FakeInnerApp()
    settings = _settings(tmp_path, capture=True)
    app = create_gateway_app(settings=settings, inner_app=inner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post(
            "/v1/responses",
            content=b'{"note":"Bearer escaped\\u002dsecret","password":"quoted\\u002dsecret"}',
            headers={"content-type": "application/json", "x-litellm-api-key": settings.local_key},
        )
        trace_path = next(tmp_path.glob("*.json"))
        trace = trace_path.read_bytes()
        trace_id = json.loads(trace)["trace_id"]
        trace_path.write_bytes(b'{"schema":"litellm-codex-gateway.trace.v1","trace_id":')
        malformed = await client.get(
            f"/debug/traces/{trace_id}/export", headers={"x-litellm-api-key": settings.local_key}
        )

    assert response.status_code == 200
    assert b"escaped-secret" not in trace
    assert b"quoted-secret" not in trace
    assert malformed.status_code == 404


@pytest.mark.asyncio
async def test_stage_exception_fails_open_with_original_body(tmp_path: Path) -> None:
    body = b'{"unknown":{"encrypted_reasoning":"opaque"}}'
    inner = FakeInnerApp()
    app = create_gateway_app(
        settings=_settings(tmp_path),
        inner_app=inner,
        pipeline=RequestPipeline((RaisingStage(),)),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post(
            "/v1/responses",
            content=body,
            headers={"content-type": "application/json", "x-litellm-api-key": "local-gateway-key-123"},
        )
        metrics = await client.get("/metrics", headers={"x-litellm-api-key": "local-gateway-key-123"})
    assert response.status_code == 200
    assert inner.body == body
    assert b"litellm_codex_gateway_fail_open_total 1" in metrics.content


@pytest.mark.asyncio
async def test_oversized_capture_drops_trace_without_changing_response(tmp_path: Path) -> None:
    chunks = (b"z" * 500,)
    inner = FakeInnerApp(chunks=chunks)
    settings = GatewaySettings(
        local_key="local-gateway-key-123",
        capture_enabled=True,
        trace_directory=tmp_path,
        max_trace_bytes=100,
        max_trace_storage_bytes=1_000,
        trace_retention_seconds=60,
    )
    app = create_gateway_app(settings=settings, inner_app=inner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post(
            "/v1/responses",
            content=b'{"input":"unchanged"}',
            headers={"content-type": "application/json", "x-litellm-api-key": settings.local_key},
        )
        metrics = await client.get("/metrics", headers={"x-litellm-api-key": settings.local_key})
    assert response.content == b"".join(chunks)
    assert not tuple(tmp_path.glob("*.json"))
    assert b"litellm_codex_gateway_capture_drops_total 1" in metrics.content


@pytest.mark.asyncio
async def test_capture_omits_truncated_request_and_response_content(tmp_path: Path) -> None:
    secret = b"boundary-secret-value"
    body = (b"x" * 3_300) + secret + (b"y" * 800)
    chunks = ((b"z" * 3_300) + secret + (b"w" * 800),)
    inner = FakeInnerApp(chunks=chunks)
    settings = _settings(tmp_path, capture=True)
    app = create_gateway_app(settings=settings, inner_app=inner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post(
            "/v1/responses",
            content=body,
            headers={"content-type": "application/octet-stream", "x-litellm-api-key": settings.local_key},
        )

    assert response.content == b"".join(chunks)
    trace = json.loads(next(tmp_path.glob("*.json")).read_bytes())
    assert trace["request"]["body"] == {"omitted": True, "truncated": True}
    assert trace["response"]["chunks"] == []
    assert trace["response"]["truncated"] is True
    assert secret not in json.dumps(trace).encode()


@pytest.mark.asyncio
async def test_malformed_non_utf8_request_fails_open_and_captures_redacted_bytes(tmp_path: Path) -> None:
    body = b'\xff\xfe{"authorization":"Bearer malformed-secret"'
    inner = FakeInnerApp()
    settings = _settings(tmp_path, capture=True)
    app = create_gateway_app(settings=settings, inner_app=inner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post(
            "/v1/responses",
            content=body,
            headers={"content-type": "application/json", "x-litellm-api-key": settings.local_key},
        )
        metrics = await client.get("/metrics", headers={"x-litellm-api-key": settings.local_key})

    trace = json.loads(next(tmp_path.glob("*.json")).read_bytes())
    assert response.status_code == 200
    assert inner.body == body
    assert trace["pipeline"]["outcome"] == "failed"
    assert b"malformed-secret" not in json.dumps(trace).encode()
    assert b"litellm_codex_gateway_fail_open_total 1" in metrics.content


@pytest.mark.asyncio
async def test_replayed_receive_propagates_disconnect_after_body() -> None:
    async def disconnected_receive() -> Message:
        return {"type": "http.disconnect"}

    receive = ReplayedReceive(b'{"input":"preserved"}', False, disconnected_receive)
    assert await receive() == {"type": "http.request", "body": b'{"input":"preserved"}', "more_body": False}
    assert await receive() == {"type": "http.disconnect"}


def test_trace_store_evicts_oldest_files(tmp_path: Path) -> None:
    store = TraceStore(directory=tmp_path, max_trace_bytes=1_000, max_storage_bytes=180, retention_seconds=60)
    first = {"schema": "litellm-codex-gateway.trace.v1", "trace_id": "a" * 32, "payload": "x" * 30}
    second = {"schema": "litellm-codex-gateway.trace.v1", "trace_id": "b" * 32, "payload": "y" * 30}
    assert store.write(first)
    assert store.write(second)
    assert store.read("a" * 32) is None
    assert store.read("b" * 32) is not None


def test_trace_store_enforces_permissions_and_retention(tmp_path: Path) -> None:
    trace_directory = tmp_path / "world-readable"
    trace_directory.mkdir(mode=0o755)
    os.chmod(trace_directory, 0o755)
    store = TraceStore(directory=trace_directory, max_trace_bytes=1_000, max_storage_bytes=1_000, retention_seconds=10)
    trace_id = "c" * 32
    assert store.write({"schema": "litellm-codex-gateway.trace.v1", "trace_id": trace_id, "payload": "safe"})
    trace_path = trace_directory / f"{trace_id}.json"
    assert stat.S_IMODE(trace_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600
    expired = time.time() - 11
    os.utime(trace_path, (expired, expired))
    store.evict()
    assert store.read(trace_id) is None
