import asyncio
import base64
import json
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Final

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.chatgpt.common_utils import OPENAI_AUTH_CLAIM_KEY
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.codex_model_catalog import bundled_codex_models
from litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints import codex_shared_cache
from litellm.proxy.utils import hash_token
from litellm.responses.litellm_completion_transformation.reasoning_items import (
    encode_thinking_blocks,
    mint_reasoning_item_id,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from tests._master_key import MASTER_KEY

_BASE: Final = "https://chatgpt.com/backend-api/codex"
_TOKEN: Final = "test-subscription-token"
_HEADERS: Final = {"authorization": f"Bearer {_TOKEN}", "x-litellm-api-key": MASTER_KEY}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, httpx_transport: None) -> Iterator[TestClient]:
    monkeypatch.setattr(proxy_server, "master_key", MASTER_KEY)
    shared_cache: Final = DualCache()
    monkeypatch.setitem(proxy_server.app.dependency_overrides, codex_shared_cache, lambda: shared_cache)
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


_PROVIDER_BASE: Final = "https://open-weights.test/v1"
_CHATGPT_JWT: Final = (
    ".".join(
        base64.urlsafe_b64encode(json.dumps(part).encode()).decode().rstrip("=")
        for part in ({"alg": "none"}, {OPENAI_AUTH_CLAIM_KEY: {"chatgpt_account_id": "account-a"}})
    )
    + "."
)
_PROVIDER_KEY: Final = "test-provider-key"
_BUNDLED_SLUG: Final = next(iter(bundled_codex_models()))
_PRICED_CHATGPT_SLUG: Final = "priced-subscription-model"
_UPSTREAM_CATALOG: Final = {
    "models": [
        {"slug": "subscription-model", "priority": 0, "supported_reasoning_levels": ["high"]},
        {"slug": "shared-name", "priority": 7, "visibility": "list"},
        {"slug": "subscription-hidden", "priority": 3, "visibility": "hide"},
    ]
}


def _open_weights(model_name: str, mode: str | None = None, upstream: str | None = None) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"fireworks_ai/{upstream or model_name}",
            "api_base": _PROVIDER_BASE,
            "api_key": _PROVIDER_KEY,
        },
        "model_info": {"mode": mode} if mode else {},
    }


@pytest.fixture
def mixed_router(monkeypatch: pytest.MonkeyPatch) -> Router:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-*",
                "litellm_params": {"model": "chatgpt/gpt-*"},
                "model_info": {"access_groups": ["chatgpt-native"]},
            },
            {
                "model_name": "codex-*",
                "litellm_params": {"model": "chatgpt/codex-*"},
                "model_info": {"access_groups": ["chatgpt-native"]},
            },
            {"model_name": "chatgpt-subscription", "litellm_params": {"model": "chatgpt/subscription-model"}},
            _open_weights("open-weights"),
            _open_weights("shared-name"),
            _open_weights(_BUNDLED_SLUG),
            _open_weights(_PRICED_CHATGPT_SLUG),
            _open_weights("open-embedder", mode="embedding"),
            _open_weights("*"),
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    unpriced_bundled: Final = {
        name: price for name, price in litellm.model_cost.items() if name != f"chatgpt/{_BUNDLED_SLUG}"
    }
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {**unpriced_bundled, f"chatgpt/{_PRICED_CHATGPT_SLUG}": {"litellm_provider": "chatgpt", "mode": "responses"}},
    )
    return router


def test_native_catalog_lists_the_keys_other_chat_models_after_chatgpts(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router
) -> None:
    respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG, headers={"etag": '"upstream-v1"'})
    response: Final = client.get("/backend-api/codex/models?client_version=1.0.0", headers=_HEADERS)
    assert response.status_code == 200
    models: Final = response.json()["models"]
    assert models[: len(_UPSTREAM_CATALOG["models"])] == _UPSTREAM_CATALOG["models"]
    added: Final = models[len(_UPSTREAM_CATALOG["models"]) :]
    assert [model["slug"] for model in added] == ["open-weights"]
    assert added[0]["visibility"] == "list"
    assert added[0]["priority"] > max(model["priority"] for model in _UPSTREAM_CATALOG["models"])
    assert "etag" not in response.headers


@pytest.mark.parametrize(
    "upstream",
    [
        pytest.param(httpx.Response(503, json={"detail": "busy"}, headers={"etag": '"e"'}), id="non-200"),
        pytest.param(httpx.Response(200, content=b"<html>login</html>", headers={"etag": '"e"'}), id="not json"),
        pytest.param(httpx.Response(200, json={"models": "none"}, headers={"etag": '"e"'}), id="no model list"),
    ],
)
def test_native_catalog_relays_an_unmergeable_upstream_answer_unchanged(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, upstream: httpx.Response
) -> None:
    respx_mock.get(f"{_BASE}/models").mock(return_value=upstream)
    response: Final = client.get("/backend-api/codex/models?client_version=1.0.0", headers=_HEADERS)
    assert response.status_code == upstream.status_code
    assert response.content == upstream.content
    assert response.headers["etag"] == '"e"'


def _provider_stream(text: str) -> bytes:
    completed: Final = {
        "id": "resp_open_weights",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": "open-weights",
        "output": [
            {
                "type": "message",
                "id": "msg_open_weights",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**completed, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_open_weights",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.completed", "sequence_number": 2, "response": completed},
    )
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def _event_types(stream: bytes) -> list[str]:
    payloads: Final = [line.removeprefix(b"data: ") for line in stream.splitlines() if line.startswith(b"data: {")]
    return [json.loads(payload)["type"] for payload in payloads]


class PreCallRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.requests: Final[list[tuple[str | None, str]]] = []
        self.raw_header_names: Final[list[frozenset[str]]] = []

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: object, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
        self.requests.append((user_api_key_dict.request_route, repr(data)))
        secret_fields: Final = data.get("secret_fields")
        raw_headers: Final = secret_fields.get("raw_headers") if isinstance(secret_fields, dict) else None
        self.raw_header_names.append(frozenset(raw_headers) if isinstance(raw_headers, dict) else frozenset())
        return data


def _open_weights_provider(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{_PROVIDER_BASE}/responses").mock(
        side_effect=lambda _: httpx.Response(
            200, content=_provider_stream("Hi"), headers={"content-type": "text/event-stream"}
        )
    )


def _chatgpt_responses(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{_BASE}/responses").respond(
        content=b"event: response.completed\ndata: {}\n\n",
        headers={"content-type": "text/event-stream", "x-models-etag": '"upstream-v2"'},
    )


def test_litellm_model_on_native_route_streams_like_v1_without_chatgpt_credentials(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = PreCallRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    chatgpt: Final = respx_mock.post(f"{_BASE}/responses").respond(json={})
    provider: Final = respx_mock.post(f"{_PROVIDER_BASE}/responses").mock(
        side_effect=lambda _: httpx.Response(
            200, content=_provider_stream("Hi"), headers={"content-type": "text/event-stream"}
        )
    )
    body: Final = {"model": "open-weights", "input": "Say hi", "stream": True, "store": False}
    native: Final = client.post(
        "/backend-api/codex/responses",
        json=body,
        headers={**_HEADERS, "authorization": f"Bearer {_CHATGPT_JWT}", "chatgpt-account-id": "account-a"},
    )
    v1: Final = client.post("/v1/responses", json=body, headers={"authorization": f"Bearer {MASTER_KEY}"})
    assert native.status_code == v1.status_code == 200
    assert native.headers["content-type"].startswith("text/event-stream")
    assert _event_types(native.content) == _event_types(v1.content)
    assert "response.completed" in _event_types(native.content)
    assert not chatgpt.called
    provider_headers: Final = provider.calls[0].request.headers
    assert provider_headers["authorization"] == f"Bearer {_PROVIDER_KEY}"
    assert "chatgpt-account-id" not in provider_headers
    assert all(_CHATGPT_JWT not in value for value in provider_headers.values())
    (native_route, native_call), (v1_route, _) = recorder.requests
    assert native_route == v1_route == "/v1/responses"
    assert _CHATGPT_JWT not in native_call
    assert "account-a" not in native_call


def test_litellm_model_on_native_route_keeps_chatgpt_credentials_out_of_hook_raw_headers(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = PreCallRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    provider: Final = _open_weights_provider(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "open-weights", "input": "Say hi", "stream": True, "store": False},
        headers={**_HEADERS, "authorization": f"Bearer {_CHATGPT_JWT}", "chatgpt-account-id": "account-a"},
    )
    assert response.status_code == 200
    assert provider.called
    (raw_header_names,) = recorder.raw_header_names
    assert "x-litellm-api-key" in raw_header_names
    assert not raw_header_names & {"authorization", "chatgpt-account-id"}, sorted(raw_header_names)


@pytest.mark.parametrize(
    ("model", "sent", "expected"),
    [
        pytest.param("open-weights", "none", None, id="picker fills in none for a model without a default effort"),
        pytest.param("open-weights", "high", "high", id="effort set for a model without a default effort"),
        pytest.param("coding-model", "none", "none", id="none on a model whose entry names a default effort"),
    ],
)
def test_litellm_model_on_native_route_gets_the_reasoning_effort_codex_m_would_send(
    client: TestClient,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    sent: str,
    expected: str | None,
) -> None:
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(model_list=[_open_weights("open-weights"), _open_weights("coding-model", upstream="gpt-5.5")]),
    )
    provider: Final = respx_mock.post(f"{_PROVIDER_BASE}/responses").mock(
        side_effect=lambda _: httpx.Response(
            200, content=_provider_stream("Hi"), headers={"content-type": "text/event-stream"}
        )
    )
    body: Final = {"model": model, "input": "Say hi", "stream": True, "store": False}
    expected_reasoning: Final = {"summary": "auto"} if expected is None else {"effort": expected, "summary": "auto"}
    native: Final = client.post(
        "/backend-api/codex/responses",
        json={**body, "reasoning": {"effort": sent, "summary": "auto"}},
        headers={**_HEADERS, "authorization": f"Bearer {_CHATGPT_JWT}"},
    )
    v1: Final = client.post(
        "/v1/responses",
        json={**body, "reasoning": expected_reasoning},
        headers={"authorization": f"Bearer {MASTER_KEY}"},
    )
    assert native.status_code == v1.status_code == 200
    native_request, v1_request = (json.loads(call.request.content) for call in provider.calls)
    assert native_request == v1_request


@pytest.mark.parametrize(
    "model", ["gpt-unlisted", "chatgpt-subscription", _BUNDLED_SLUG, _PRICED_CHATGPT_SLUG, "newly-released"]
)
def test_chatgpt_model_keeps_passthrough_without_other_providers_reasoning(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, model: str
) -> None:
    provider: Final = respx_mock.post(f"{_PROVIDER_BASE}/responses").respond(json={})
    chatgpt: Final = respx_mock.post(f"{_BASE}/responses").respond(
        content=b"event: response.completed\ndata: {}\n\n",
        headers={"content-type": "text/event-stream", "x-models-etag": '"upstream-v2"'},
    )
    user_turn: Final = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
    chatgpt_reasoning: Final = {
        "type": "reasoning",
        "id": "rs_upstream",
        "summary": [],
        "encrypted_content": "opaque-upstream-blob",
    }
    assistant_turn: Final = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}
    other_providers_reasoning: Final = [
        {"type": "reasoning", "id": mint_reasoning_item_id(), "summary": []},
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": encode_thinking_blocks([{"type": "thinking", "thinking": "t", "signature": "s"}]),
        },
        {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "t"}]},
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id("blob", "deployment"),
        },
    ]
    body: Final = {
        "model": model,
        "input": [user_turn, *other_providers_reasoning, chatgpt_reasoning, assistant_turn],
        "stream": True,
    }
    response: Final = client.post("/backend-api/codex/responses", json=body, headers=_HEADERS)
    assert response.status_code == 200
    assert not provider.called
    assert json.loads(chatgpt.calls.last.request.content) == {
        **body,
        "input": [user_turn, chatgpt_reasoning, assistant_turn],
    }
    assert "x-models-etag" not in response.headers


@pytest.fixture
def restricted_key(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    monkeypatch.setattr(proxy_server, "prisma_client", "test-database")

    def mint(models: list[str], allowed_routes: tuple[str, ...] = ()) -> str:
        key: Final = f"sk-test-{'-'.join((*models, *allowed_routes))}"
        proxy_server.user_api_key_cache.set_cache(
            key=hash_token(key),
            value=UserAPIKeyAuth(token=hash_token(key), models=models, allowed_routes=list(allowed_routes)),
        )
        return key

    return mint


@pytest.mark.parametrize(
    ("models", "added"),
    [
        pytest.param(["chatgpt-native", "open-weights"], ["open-weights"], id="key with a LiteLLM model"),
        pytest.param(["chatgpt-native"], [], id="key with ChatGPT models only"),
    ],
)
def test_native_catalog_adds_only_the_litellm_models_the_key_can_call(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    restricted_key: Callable[..., str],
    models: list[str],
    added: list[str],
) -> None:
    respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG, headers={"etag": '"upstream-v1"'})
    response: Final = client.get(
        "/backend-api/codex/models?client_version=1.0.0",
        headers={**_HEADERS, "x-litellm-api-key": restricted_key(models)},
    )
    assert response.status_code == 200
    assert [model["slug"] for model in response.json()["models"]] == [
        *(model["slug"] for model in _UPSTREAM_CATALOG["models"]),
        *added,
    ]
    assert ("etag" in response.headers) is not added


@pytest.mark.parametrize(
    ("models", "model", "allowed"),
    [
        pytest.param(["chatgpt-native", "open-weights"], "gpt-unlisted", True, id="chatgpt access group"),
        pytest.param(["chatgpt-subscription", "open-weights"], "gpt-unlisted", False, id="single subscription model"),
        pytest.param(["chatgpt-native"], "open-weights", False, id="litellm model outside the key"),
    ],
)
def test_native_route_serves_only_the_models_the_key_allows(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    restricted_key: Callable[..., str],
    models: list[str],
    model: str,
    allowed: bool,
) -> None:
    provider: Final = respx_mock.post(f"{_PROVIDER_BASE}/responses").respond(json={})
    chatgpt: Final = respx_mock.post(f"{_BASE}/responses").respond(
        content=b"event: response.completed\ndata: {}\n\n", headers={"content-type": "text/event-stream"}
    )
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": model, "input": "Hello", "stream": True},
        headers={**_HEADERS, "x-litellm-api-key": restricted_key(models)},
    )
    assert (response.status_code == 200) is allowed
    assert chatgpt.called is allowed
    assert not provider.called


_NATIVE_ROUTES: Final = ("/backend-api/codex/models", "/backend-api/codex/responses")


@pytest.mark.parametrize(
    ("allowed_routes", "added"),
    [
        pytest.param(_NATIVE_ROUTES, [], id="key limited to the native routes"),
        pytest.param((*_NATIVE_ROUTES, "/v1/responses"), ["open-weights"], id="key that may call /v1/responses"),
    ],
)
def test_native_catalog_adds_litellm_models_only_for_a_key_that_may_call_v1_responses(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    restricted_key: Callable[..., str],
    allowed_routes: tuple[str, ...],
    added: list[str],
) -> None:
    respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG)
    response: Final = client.get(
        "/backend-api/codex/models?client_version=1.0.0",
        headers={**_HEADERS, "x-litellm-api-key": restricted_key(["chatgpt-native", "open-weights"], allowed_routes)},
    )
    assert response.status_code == 200, response.text
    assert [model["slug"] for model in response.json()["models"]] == [
        *(model["slug"] for model in _UPSTREAM_CATALOG["models"]),
        *added,
    ]


@pytest.mark.parametrize(
    ("allowed_routes", "through_litellm"),
    [
        pytest.param(_NATIVE_ROUTES, False, id="key limited to the native routes"),
        pytest.param((*_NATIVE_ROUTES, "/v1/responses"), True, id="key that may call /v1/responses"),
    ],
)
def test_native_responses_reach_litellm_models_only_for_a_key_that_may_call_v1_responses(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    restricted_key: Callable[..., str],
    allowed_routes: tuple[str, ...],
    through_litellm: bool,
) -> None:
    provider: Final = _open_weights_provider(respx_mock)
    chatgpt: Final = _chatgpt_responses(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "open-weights", "input": "Hello", "stream": True},
        headers={**_HEADERS, "x-litellm-api-key": restricted_key(["chatgpt-native", "open-weights"], allowed_routes)},
    )
    assert response.status_code == 200, response.text
    assert provider.called is through_litellm
    assert chatgpt.called is not through_litellm


_USER_TURN: Final = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
_ASSISTANT_TURN: Final = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}
_FIREWORKS_REASONING: Final = {
    "type": "reasoning",
    "id": f"rs_{'0f' * 16}",
    "summary": [{"type": "summary_text", "text": "Thinking it over"}],
}


@pytest.mark.parametrize(
    ("item", "store", "forwarded"),
    [
        pytest.param(_FIREWORKS_REASONING, False, False, id="another provider's summary-only item"),
        pytest.param({**_FIREWORKS_REASONING, "encrypted_content": None}, None, False, id="null blob, store unset"),
        pytest.param(
            {"type": "reasoning", "id": "rs_upstream", "summary": [], "encrypted_content": "gAAAAB-test-blob"},
            False,
            True,
            id="ChatGPT's own item",
        ),
        pytest.param({**_FIREWORKS_REASONING, "id": None}, False, True, id="summary-only item without an id"),
        pytest.param(_FIREWORKS_REASONING, True, True, id="stored conversation"),
    ],
)
def test_chatgpt_passthrough_drops_reasoning_ids_chatgpt_never_persisted(
    client: TestClient,
    respx_mock: respx.MockRouter,
    item: dict[str, object],
    store: bool | None,
    forwarded: bool,
) -> None:
    chatgpt: Final = _chatgpt_responses(respx_mock)
    body: Final = {
        "model": "subscription-model",
        "input": [_USER_TURN, item, _ASSISTANT_TURN],
        "stream": True,
        **({} if store is None else {"store": store}),
    }
    response: Final = client.post("/backend-api/codex/responses", json=body, headers=_HEADERS)
    assert response.status_code == 200
    assert json.loads(chatgpt.calls.last.request.content)["input"] == (
        [_USER_TURN, item, _ASSISTANT_TURN] if forwarded else [_USER_TURN, _ASSISTANT_TURN]
    )


_CODEX_USER_AGENT: Final = "codex_cli_rs/0.165.0 (Ubuntu 24.4.0; x86_64) xterm-256color"
_ACCOUNT_HEADERS: Final = {
    **_HEADERS,
    "authorization": f"Bearer {_CHATGPT_JWT}",
    "chatgpt-account-id": "account-a",
    "originator": "codex_cli_rs",
    "user-agent": _CODEX_USER_AGENT,
}


def test_exact_deployment_named_like_an_account_slug_stays_on_the_subscription_after_the_catalog_read(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router
) -> None:
    catalog: Final = respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG)
    provider: Final = _open_weights_provider(respx_mock)
    chatgpt: Final = _chatgpt_responses(respx_mock)
    assert client.get("/backend-api/codex/models?client_version=0.165.0", headers=_ACCOUNT_HEADERS).status_code == 200
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "shared-name", "input": "Hello", "stream": True},
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200
    assert chatgpt.called
    assert not provider.called
    assert catalog.call_count == 1


@pytest.mark.parametrize(
    ("model", "user_agent", "catalog_answer", "subscription"),
    [
        pytest.param("shared-name", _CODEX_USER_AGENT, httpx.Response(200, json=_UPSTREAM_CATALOG), True, id="listed"),
        pytest.param("shared-name", _CODEX_USER_AGENT, httpx.Response(503), False, id="catalog unavailable"),
        pytest.param(
            "shared-name", "codex_cli_rs", httpx.Response(200, json=_UPSTREAM_CATALOG), False, id="no version"
        ),
        pytest.param(
            "open-weights", _CODEX_USER_AGENT, httpx.Response(200, json=_UPSTREAM_CATALOG), False, id="unlisted"
        ),
    ],
)
def test_native_responses_ask_chatgpt_which_models_the_account_lists_on_a_cache_miss(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    model: str,
    user_agent: str,
    catalog_answer: httpx.Response,
    subscription: bool,
) -> None:
    catalog: Final = respx_mock.get(f"{_BASE}/models").mock(return_value=catalog_answer)
    provider: Final = _open_weights_provider(respx_mock)
    chatgpt: Final = _chatgpt_responses(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": model, "input": "Hello", "stream": True},
        headers={**_ACCOUNT_HEADERS, "user-agent": user_agent},
    )
    assert response.status_code == 200
    assert chatgpt.called is subscription
    assert provider.called is not subscription
    assert catalog.called is (user_agent == _CODEX_USER_AGENT)


def test_account_catalog_lookup_sends_the_callers_chatgpt_login_and_codex_version(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router
) -> None:
    catalog: Final = respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG)
    _chatgpt_responses(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "shared-name", "input": "Hello", "stream": True},
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200
    fetched: Final = catalog.calls.last.request
    assert fetched.url.params["client_version"] == "0.165.0"
    assert {name: fetched.headers[name] for name in ("authorization", "chatgpt-account-id", "originator")} == {
        "authorization": f"Bearer {_CHATGPT_JWT}",
        "chatgpt-account-id": "account-a",
        "originator": "codex_cli_rs",
    }
    assert fetched.headers["user-agent"] == _CODEX_USER_AGENT
    assert "x-litellm-api-key" not in fetched.headers


@pytest.mark.parametrize(
    ("models", "etag_kept"),
    [
        pytest.param(["chatgpt-native", "open-weights"], False, id="catalog with LiteLLM models"),
        pytest.param(["chatgpt-native"], True, id="catalog of ChatGPT models only"),
    ],
)
def test_native_responses_keep_the_models_etag_only_for_a_key_whose_catalog_was_not_merged(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    restricted_key: Callable[..., str],
    models: list[str],
    etag_kept: bool,
) -> None:
    respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG)
    _chatgpt_responses(respx_mock)
    headers: Final = {**_HEADERS, "x-litellm-api-key": restricted_key(models)}
    assert client.get("/backend-api/codex/models?client_version=1.0.0", headers=headers).status_code == 200
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "gpt-unlisted", "input": "Hello", "stream": True},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers.get("x-models-etag") == ('"upstream-v2"' if etag_kept else None)
