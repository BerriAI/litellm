import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from queue import SimpleQueue
from threading import Event
from typing import Final

import httpx
import pytest
import pytest_asyncio
import respx
from fastapi import FastAPI
from starlette.types import Message, Receive, Scope, Send

from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy._types import UserAPIKeyAuth
from tests._master_key import MASTER_KEY

pytestmark: Final = pytest.mark.usefixtures("httpx_transport")

_CREDENTIALS: Final = {"authorization": "Bearer synthetic-subscription", "x-litellm-api-key": MASTER_KEY}
_BASE: Final = "https://chatgpt.com/backend-api/codex"


class HeaderObserver(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.observed: Final[SimpleQueue[tuple[str, str]]] = SimpleQueue()

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
        logging_obj: Final = data.get("litellm_logging_obj")
        if isinstance(logging_obj, Logging):
            logging_obj.add_dynamic_callback(self)
        return data

    def log_pre_api_call(self, model: object, messages: object, kwargs: object) -> None:
        self.observed.put(("pre", repr(kwargs)))

    async def async_post_call_response_headers_hook(
        self,
        data: object,
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> dict[str, str]:
        self.observed.put(("headers", repr(request_headers)))
        if isinstance(response, httpx.Response):
            self.observed.put(("attached-request", repr(response.request.headers)))
        return {"x-test-headers-observed": "1"}


headers_observer: Final = HeaderObserver()


@pytest_asyncio.fixture
async def codex_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    from litellm.proxy.proxy_server import app, initialize

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config: Final = tmp_path / "proxy.yaml"
    config.write_text(
        json.dumps(
            {
                "general_settings": {"master_key": MASTER_KEY},
                "litellm_settings": {"model_alias_map": {"synthetic-native": "must-not-rewrite"}},
            }
        )
    )
    await initialize(config=str(config))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        yield client
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    finally:
        await GLOBAL_LOGGING_WORKER.stop()


@pytest.mark.asyncio
async def test_native_catalog_and_query_with_separate_credentials(codex_gateway: httpx.AsyncClient) -> None:
    catalog: Final = {"models": [{"slug": "synthetic-native-model", "custom_field": {"effort": ["medium"]}}]}
    with respx.mock() as upstream:
        observed: Final = upstream.get(f"{_BASE}/models", params={"client_version": "synthetic-version"}).respond(
            json=catalog, headers={"x-request-id": "catalog-request", "x-codex-primary-used-percent": "7"}
        )
        response: Final = await codex_gateway.get(
            "/backend-api/codex/models?client_version=synthetic-version",
            headers={**_CREDENTIALS, "chatgpt-account-id": "synthetic-account", "cookie": "session=synthetic-cookie"},
        )
    assert response.status_code == 200, response.text
    assert response.json() == catalog
    assert response.headers["x-request-id"] == "catalog-request"
    assert response.headers["x-codex-primary-used-percent"] == "7"
    assert observed.calls.last.request.headers["authorization"] == _CREDENTIALS["authorization"]
    assert observed.calls.last.request.headers["chatgpt-account-id"] == "synthetic-account"
    assert MASTER_KEY not in str(observed.calls.last.request.headers)
    assert "cookie" not in observed.calls.last.request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"authorization": _CREDENTIALS["authorization"]},
        {"x-litellm-api-key": MASTER_KEY},
        {**_CREDENTIALS, "x-litellm-api-key": "synthetic-invalid-key"},
        {**_CREDENTIALS, "authorization": f"Bearer {MASTER_KEY}"},
    ],
)
async def test_missing_invalid_or_shared_credentials_never_contact_provider(
    codex_gateway: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=False) as upstream:
        provider: Final = upstream.get(f"{_BASE}/models").respond(json={"models": []})
        response: Final = await codex_gateway.get("/backend-api/codex/models", headers=headers)
    assert response.is_client_error, response.text
    assert not provider.called


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [307, 401, 429])
async def test_status_body_and_headers_are_relayed_without_redirects(
    codex_gateway: httpx.AsyncClient, status: int
) -> None:
    payload: Final = {"error": {"message": "synthetic-provider-error", "type": "synthetic-error"}}
    with respx.mock(assert_all_called=False) as upstream:
        observed: Final = upstream.post(f"{_BASE}/responses").respond(
            status, json=payload, headers={"location": "https://untrusted.example/collect", "retry-after": "3"}
        )
        redirect: Final = upstream.post("https://untrusted.example/collect").respond(200)
        response: Final = await codex_gateway.post(
            "/backend-api/codex/responses", headers=_CREDENTIALS, json={"model": "synthetic-native", "input": "hello"}
        )
    assert observed.called
    assert not redirect.called
    assert response.status_code == status, response.text
    assert response.json() == payload
    assert response.headers["retry-after"] == "3"


@pytest.mark.asyncio
async def test_native_envelope_is_not_interpreted_as_relay_configuration(
    codex_gateway: httpx.AsyncClient,
) -> None:
    payload: Final = {
        "model": "synthetic-native",
        "input": "hello",
        "custom_body": {"target": "https://untrusted.example/collect"},
        "query_params": {"target": "https://untrusted.example/collect"},
    }
    with respx.mock() as upstream:
        observed: Final = upstream.post(f"{_BASE}/responses").respond(429, json={"error": {"message": "synthetic"}})
        response: Final = await codex_gateway.post(
            "/backend-api/codex/responses",
            json=payload,
            headers=_CREDENTIALS,
        )
    assert response.status_code == 429, response.text
    assert json.loads(observed.calls.last.request.content) == payload
    assert "content-encoding" not in observed.calls.last.request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"{", b"[]"])
async def test_invalid_json_object_never_contacts_provider(codex_gateway: httpx.AsyncClient, body: bytes) -> None:
    with respx.mock(assert_all_called=False) as upstream:
        provider: Final = upstream.post(f"{_BASE}/responses").respond(json={})
        response: Final = await codex_gateway.post(
            "/backend-api/codex/responses",
            content=body,
            headers={**_CREDENTIALS, "content-type": "application/json"},
        )
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "A JSON object is required"}
    assert not provider.called


@pytest.mark.asyncio
@pytest.mark.parametrize("encodings", [("zstd",), ("gzip",), ("br",), ("",), ("identity", "zstd")])
@pytest.mark.parametrize("authenticated", [False, True])
async def test_encoded_requests_are_rejected_before_reading_or_authentication(
    codex_gateway: httpx.AsyncClient, encodings: tuple[str, ...], authenticated: bool
) -> None:
    body: Final = ObservedStream()
    with respx.mock(assert_all_called=False) as upstream:
        provider: Final = upstream.post(f"{_BASE}/responses").respond(json={})
        response: Final = await codex_gateway.post(
            "/backend-api/codex/responses",
            content=body,
            headers=(
                *(_CREDENTIALS.items() if authenticated else ()),
                *(("content-encoding", encoding) for encoding in encodings),
            ),
        )
    assert response.status_code == 415, response.text
    assert "enable_request_compression = false" in response.json()["detail"]
    assert not body.started.is_set()
    assert not provider.called


@pytest.mark.asyncio
async def test_gateway_credentials_in_aliases_and_x_pass_headers_cannot_escape(
    codex_gateway: httpx.AsyncClient,
) -> None:
    with respx.mock() as upstream:
        observed: Final = upstream.get(f"{_BASE}/models").respond(json={"models": []})
        response: Final = await codex_gateway.get(
            "/backend-api/codex/models",
            headers={
                **_CREDENTIALS,
                "x-alias": f"Bearer {MASTER_KEY}",
                "x-api-key": MASTER_KEY,
                "x-pass-authorization": f"Bearer {MASTER_KEY}",
                "x-pass-chatgpt-account-id": "cannot-replace-account",
                "chatgpt-account-id": "real-account",
            },
        )
    assert response.status_code == 200, response.text
    assert MASTER_KEY not in str(observed.calls.last.request.headers)
    assert observed.calls.last.request.headers["chatgpt-account-id"] == "real-account"


@pytest.mark.asyncio
async def test_stream_keeps_native_events_ids_usage_and_quota_headers(codex_gateway: httpx.AsyncClient) -> None:
    terminal: Final = {
        "type": "response.completed",
        "response": {
            "id": "resp_litellm_test_native",
            "object": "response",
            "created_at": 1786374786,
            "model": "litellm-test-native-unpriced",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        },
    }
    events: Final = (
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hello"}\n\n'
        f"event: response.completed\ndata: {json.dumps(terminal)}\n\n"
    ).encode()
    with respx.mock() as upstream:
        observed: Final = upstream.post(f"{_BASE}/responses").respond(
            content=events,
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "native-stream",
                "x-codex-primary-used-percent": "8",
            },
        )
        response: Final = await codex_gateway.post(
            "/backend-api/codex/responses",
            headers=_CREDENTIALS,
            json={"model": "litellm-test-native-unpriced", "input": "hello", "stream": True},
        )
    assert response.status_code == 200, response.text
    assert response.content == events
    assert response.headers["x-request-id"] == "native-stream"
    assert response.headers["x-codex-primary-used-percent"] == "8"
    assert json.loads(observed.calls.last.request.content)["model"] == terminal["response"]["model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["catalog", "stream", "failure"])
async def test_custom_auth_and_callback_copies_cannot_expose_credentials(
    codex_gateway: httpx.AsyncClient, tmp_path: Path, mode: str
) -> None:
    from litellm.proxy.proxy_server import initialize

    config: Final = tmp_path / "custom-auth.yaml"
    config.write_text(
        json.dumps(
            {
                "general_settings": {"master_key": MASTER_KEY, "litellm_key_header_name": "x-gateway-key"},
                "litellm_settings": {"callbacks": [f"{__name__}.headers_observer"]},
            }
        )
    )
    await initialize(config=str(config))
    credentials: Final = {"authorization": _CREDENTIALS["authorization"], "x-gateway-key": f"Bearer {MASTER_KEY}"}
    with respx.mock() as upstream:
        provider: Final = upstream.route(host="chatgpt.com").respond(
            status_code=429 if mode == "failure" else 200,
            content=b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n' if mode == "stream" else b"{}",
            headers={"content-type": "text/event-stream" if mode == "stream" else "application/json"},
        )
        response: Final = await codex_gateway.request(
            "GET" if mode == "catalog" else "POST",
            "/backend-api/codex/models" if mode == "catalog" else "/backend-api/codex/responses",
            headers=credentials,
            json=None
            if mode == "catalog"
            else {"model": "synthetic-native", "stream": mode == "stream", "input": "hello"},
        )
    observations: Final = tuple(
        headers_observer.observed.get_nowait() for _ in range(headers_observer.observed.qsize())
    )
    assert response.status_code == (429 if mode == "failure" else 200), response.text
    assert response.headers["x-test-headers-observed"] == "1"
    assert {stage for stage, _ in observations} == {"pre", "headers", "attached-request"}
    assert "synthetic-subscription" not in repr(observations)
    assert MASTER_KEY not in repr(observations)
    assert provider.calls.last.request.headers["authorization"] == credentials["authorization"]
    assert MASTER_KEY not in str(provider.calls.last.request.headers)


class ObservedStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started: Final = Event()
        self.closed: Final = Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        yield b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_upstream_is_closed_if_downstream_header_delivery_fails_before_iteration(
    codex_gateway: httpx.AsyncClient,
) -> None:
    from litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints import router

    isolated: Final = FastAPI()
    isolated.include_router(router)

    async def broken_downstream(scope: Scope, receive: Receive, send: Send) -> None:
        async def fail_start(message: Message) -> None:
            if message["type"] == "http.response.start":
                raise OSError("synthetic downstream header failure")
            await send(message)

        await isolated(scope, receive, fail_start)

    stream: Final = ObservedStream()
    with respx.mock() as upstream:
        upstream.post(f"{_BASE}/responses").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=broken_downstream), base_url="http://gateway"
        ) as client:
            with pytest.raises(OSError, match="synthetic downstream header failure"):
                await client.post(
                    "/backend-api/codex/responses",
                    headers=_CREDENTIALS,
                    json={"model": "synthetic-native", "input": "hello", "stream": True},
                )
    assert not stream.started.is_set()
    assert stream.closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/backend-api/codex/login", "/backend-api/codex/responses/arbitrary-id"])
async def test_only_native_allowlisted_endpoints_exist(codex_gateway: httpx.AsyncClient, path: str) -> None:
    with respx.mock(assert_all_called=False) as upstream:
        provider: Final = upstream.route().respond(200)
        response: Final = await codex_gateway.post(path, headers=_CREDENTIALS, json={})
    assert response.status_code == 404
    assert not provider.called
