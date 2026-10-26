import asyncio
import json
from collections.abc import Iterator
from datetime import datetime
from typing import Final

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy import proxy_server
from tests._master_key import MASTER_KEY

_BASE: Final = "https://chatgpt.com/backend-api/codex"
_TOKEN: Final = "test-subscription-token"
_HEADERS: Final = {"authorization": f"Bearer {_TOKEN}", "x-litellm-api-key": MASTER_KEY}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, httpx_transport: None) -> Iterator[TestClient]:
    monkeypatch.setattr(proxy_server, "master_key", MASTER_KEY)
    yield TestClient(proxy_server.app)


def test_native_catalog_preserves_account_and_filters_gateway_credentials(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    catalog: Final = {"models": [{"slug": "subscription-model", "supported_reasoning_levels": ["high"]}]}
    upstream: Final = respx_mock.get(f"{_BASE}/models", params={"client_version": "1.0.0"}).respond(json=catalog)
    response: Final = client.get(
        "/backend-api/codex/models?client_version=1.0.0",
        headers={
            **_HEADERS,
            "chatgpt-account-id": "account-a",
            "cookie": "session=test-cookie",
            "x-pass-authorization": "Bearer test-injected-token",
            "x-pass-x-litellm-api-key": "test-injected-key",
        },
    )
    assert response.status_code == 200
    assert response.json() == catalog
    assert upstream.calls.last.request.headers["authorization"] == f"Bearer {_TOKEN}"
    assert upstream.calls.last.request.headers["chatgpt-account-id"] == "account-a"
    assert "x-litellm-api-key" not in upstream.calls.last.request.headers
    assert "cookie" not in upstream.calls.last.request.headers


@pytest.mark.parametrize("account", ["account-a", "account-b"])
def test_native_responses_preserves_body_and_stream(
    client: TestClient, respx_mock: respx.MockRouter, account: str
) -> None:
    body: Final = {
        "model": "subscription-model",
        "input": [{"role": "user", "content": "Hello"}],
        "reasoning": {"effort": "high"},
        "store": False,
        "stream": True,
    }
    stream: Final = (
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
    )
    upstream: Final = respx_mock.post(f"{_BASE}/responses").respond(
        content=stream, headers={"content-type": "text/event-stream"}
    )
    response: Final = client.post(
        "/backend-api/codex/responses",
        json=body,
        headers={**_HEADERS, "authorization": f"Bearer test-{account}", "chatgpt-account-id": account},
    )
    assert response.status_code == 200
    assert response.content == stream
    assert response.headers["content-type"].startswith("text/event-stream")
    assert json.loads(upstream.calls.last.request.content) == body
    assert upstream.calls.last.request.headers["authorization"] == f"Bearer test-{account}"
    assert upstream.calls.last.request.headers["chatgpt-account-id"] == account


@pytest.mark.parametrize("gateway_key", [None, "sk-invalid-test-key"])
def test_gateway_auth_rejects_before_upstream(
    client: TestClient, respx_mock: respx.MockRouter, gateway_key: str | None
) -> None:
    headers: Final = {
        "authorization": f"Bearer {_TOKEN}",
        **({"x-litellm-api-key": gateway_key} if gateway_key else {}),
    }
    response: Final = client.get("/backend-api/codex/models", headers=headers)
    assert response.status_code in (400, 401, 403)
    assert not respx_mock.calls


@pytest.mark.parametrize("authorization", ["", "Basic test", "Bearer ", f"Bearer {MASTER_KEY}"])
def test_subscription_credential_required(client: TestClient, respx_mock: respx.MockRouter, authorization: str) -> None:
    response: Final = client.get("/backend-api/codex/models", headers={**_HEADERS, "authorization": authorization})
    assert response.status_code == 401
    assert not respx_mock.calls


@pytest.mark.parametrize("status", [401, 403, 429])
def test_provider_errors_do_not_fall_back(client: TestClient, respx_mock: respx.MockRouter, status: int) -> None:
    upstream: Final = respx_mock.post(f"{_BASE}/responses").respond(
        status, json={"error": {"message": "subscription unavailable"}}, headers={"retry-after": "60"}
    )
    response: Final = client.post(
        "/backend-api/codex/responses", headers=_HEADERS, json={"model": "subscription-model", "stream": True}
    )
    assert response.status_code == status
    assert upstream.call_count == 1


@pytest.mark.parametrize("method,path", [("POST", "models"), ("GET", "responses"), ("GET", "deviceauth")])
def test_only_native_inference_routes_are_forwarded(
    client: TestClient, respx_mock: respx.MockRouter, method: str, path: str
) -> None:
    response: Final = client.request(method, f"/backend-api/codex/{path}", headers=_HEADERS)
    assert response.status_code in (405, 422)
    assert not respx_mock.calls


class UsageRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self.events.put(kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_usage_logged_without_credentials(
    monkeypatch: pytest.MonkeyPatch, httpx_transport: None, respx_mock: respx.MockRouter, stream: bool
) -> None:
    monkeypatch.setattr(proxy_server, "master_key", MASTER_KEY)
    recorder: Final = UsageRecorder()
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    litellm.register_model(
        {
            "chatgpt/subscription-model": {
                "litellm_provider": "chatgpt",
                "mode": "responses",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }
        }
    )
    result: Final = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": "subscription-model",
        "output": [],
        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
    }
    event: Final = {"type": "response.completed", "response": result}
    response_body: Final = (
        httpx.Response(
            200,
            content=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )
        if stream
        else httpx.Response(200, json=result)
    )
    respx_mock.post(f"{_BASE}/responses").mock(return_value=response_body)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://test") as caller:
        response: Final = await caller.post(
            "/backend-api/codex/responses",
            json={"model": "subscription-model", "stream": stream, "input": "Hello"},
            headers={**_HEADERS, "x-pass-authorization": "Bearer test-hidden-token", "cookie": "test-hidden-cookie"},
        )
        assert response.status_code == 200
        logged: Final = await asyncio.wait_for(recorder.events.get(), timeout=5)
    payload: Final = logged["standard_logging_object"]
    assert isinstance(payload, dict)
    assert payload["prompt_tokens"] == 12
    assert payload["completion_tokens"] == 3
    serialized: Final = json.dumps(logged, default=str)
    for secret in (_TOKEN, MASTER_KEY, "test-hidden-token", "test-hidden-cookie"):
        assert secret not in serialized
