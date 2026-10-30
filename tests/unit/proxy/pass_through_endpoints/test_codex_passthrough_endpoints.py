import asyncio
import base64
import json
from collections.abc import Callable, Iterator, Sequence
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
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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
_CODEX_USER_AGENT: Final = "codex_cli_rs/0.165.0 (Ubuntu 24.4.0; x86_64) xterm-256color"
_ACCOUNT_HEADERS: Final = {
    **_HEADERS,
    "authorization": f"Bearer {_CHATGPT_JWT}",
    "chatgpt-account-id": "account-a",
    "originator": "codex_cli_rs",
    "user-agent": _CODEX_USER_AGENT,
}
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
                "model_name": "chatgpt/gpt-*",
                "litellm_params": {"model": "chatgpt/gpt-*"},
                "model_info": {"access_groups": ["chatgpt-native"]},
            },
            {
                "model_name": "chatgpt/codex-*",
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


@pytest.mark.parametrize(
    ("client_version", "expected_efforts"),
    [
        pytest.param("0.165.0", ["low", "max"], id="a codex that parses custom efforts gets them all"),
        pytest.param("0.137.0", ["low"], id="an older codex gets only the efforts it parses"),
    ],
)
def test_native_catalog_offers_the_declared_reasoning_efforts_the_requesting_codex_parses(
    client: TestClient,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    client_version: str,
    expected_efforts: list[str],
) -> None:
    declared: Final = {
        **_open_weights("open-weights"),
        "model_info": {"supports_reasoning": True, "reasoning_effort_levels": ["low", "max"]},
    }
    monkeypatch.setattr(proxy_server, "llm_router", Router(model_list=[declared]))
    respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG, headers={"etag": '"upstream-v1"'})
    response: Final = client.get(f"/backend-api/codex/models?client_version={client_version}", headers=_HEADERS)
    assert response.status_code == 200
    (added,) = response.json()["models"][len(_UPSTREAM_CATALOG["models"]) :]
    assert [level["effort"] for level in added["supported_reasoning_levels"]] == expected_efforts


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


def _provider_completed(text: str) -> dict[str, object]:
    return {
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


def _provider_stream(text: str) -> bytes:
    completed: Final = _provider_completed(text)
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


def _provider_reply(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream") is not True:
        return httpx.Response(200, json=_provider_completed("Hi"))
    return httpx.Response(200, content=_provider_stream("Hi"), headers={"content-type": "text/event-stream"})


def _open_weights_provider(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{_PROVIDER_BASE}/responses").mock(side_effect=_provider_reply)


_UPSTREAM_MODELS_ETAG: Final = '"upstream-v2"'


def _chatgpt_responses(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{_BASE}/responses").respond(
        content=b"event: response.completed\ndata: {}\n\n",
        headers={"content-type": "text/event-stream", "x-models-etag": _UPSTREAM_MODELS_ETAG},
    )


def _account_catalog(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG)


def test_litellm_model_on_native_route_streams_like_v1_without_chatgpt_credentials(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = PreCallRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    _account_catalog(respx_mock)
    chatgpt: Final = respx_mock.post(f"{_BASE}/responses").respond(json={})
    provider: Final = respx_mock.post(f"{_PROVIDER_BASE}/responses").mock(
        side_effect=lambda _: httpx.Response(
            200, content=_provider_stream("Hi"), headers={"content-type": "text/event-stream"}
        )
    )
    body: Final = {"model": "open-weights", "input": "Say hi", "stream": True, "store": False}
    native: Final = client.post("/backend-api/codex/responses", json=body, headers=_ACCOUNT_HEADERS)
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
    _account_catalog(respx_mock)
    provider: Final = _open_weights_provider(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "open-weights", "input": "Say hi", "stream": True, "store": False},
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200
    assert provider.called
    (raw_header_names,) = recorder.raw_header_names
    assert "x-litellm-api-key" in raw_header_names
    assert not raw_header_names & {"authorization", "chatgpt-account-id"}, sorted(raw_header_names)


@pytest.mark.parametrize(
    ("model", "sent", "drop_params", "expected"),
    [
        pytest.param("open-weights", "none", None, "none", id="none without drop_params"),
        pytest.param("open-weights", "none", "global", None, id="none under litellm_settings drop_params"),
        pytest.param("open-weights", "none", "deployment", None, id="none under the deployment's drop_params"),
        pytest.param("open-weights", "none", "request", None, id="none under the request's drop_params"),
        pytest.param("open-weights", "high", "global", "high", id="effort set for a model without a default effort"),
        pytest.param("coding-model", "none", "global", "none", id="none on a model whose entry names a default effort"),
        pytest.param("api-model", "none", "global", "none", id="none on a model that supports it"),
        pytest.param("glm-model", "none", "global", None, id="none on a model that does not support it"),
        pytest.param("declared-model", "none", "global", "none", id="none on a deployment whose model_info allows it"),
    ],
)
def test_litellm_model_on_native_route_gets_the_reasoning_effort_codex_m_would_send(
    client: TestClient,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    sent: str,
    drop_params: str | None,
    expected: str | None,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", drop_params == "global")
    open_weights: Final = _open_weights("open-weights")
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    **open_weights,
                    "litellm_params": {**open_weights["litellm_params"], "drop_params": drop_params == "deployment"},
                },
                _open_weights("coding-model", upstream="gpt-5.5"),
                _open_weights("glm-model", upstream="glm-5p3"),
                {**_open_weights("declared-model"), "model_info": {"supports_none_reasoning_effort": True}},
                {
                    "model_name": "api-model",
                    "litellm_params": {"model": "openai/gpt-5.4", "api_base": _PROVIDER_BASE, "api_key": _PROVIDER_KEY},
                },
            ]
        ),
    )
    _account_catalog(respx_mock)
    provider: Final = respx_mock.post(f"{_PROVIDER_BASE}/responses").mock(
        side_effect=lambda _: httpx.Response(
            200, content=_provider_stream("Hi"), headers={"content-type": "text/event-stream"}
        )
    )
    body: Final = {
        "model": model,
        "input": "Say hi",
        "stream": True,
        "store": False,
        **({"drop_params": True} if drop_params == "request" else {}),
    }
    expected_reasoning: Final = {"summary": "auto"} if expected is None else {"effort": expected, "summary": "auto"}
    native: Final = client.post(
        "/backend-api/codex/responses",
        json={**body, "reasoning": {"effort": sent, "summary": "auto"}},
        headers=_ACCOUNT_HEADERS,
    )
    v1: Final = client.post(
        "/v1/responses",
        json={**body, "reasoning": expected_reasoning},
        headers={"authorization": f"Bearer {MASTER_KEY}"},
    )
    assert native.status_code == v1.status_code == 200
    native_request, v1_request = (json.loads(call.request.content) for call in provider.calls)
    assert native_request == v1_request
    assert native_request["reasoning"].get("effort") == expected


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
    assert (response.headers["etag"] == '"upstream-v1"') is not added


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
    assert [json.loads(call.request.content)["model"] for call in chatgpt.calls] == ([model] if allowed else [])


def test_v1_request_for_a_gpt_name_still_reaches_the_catch_all_with_the_access_group_recipe(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router
) -> None:
    provider: Final = _open_weights_provider(respx_mock)
    chatgpt: Final = _chatgpt_responses(respx_mock)
    response: Final = client.post(
        "/v1/responses",
        json={"model": "gpt-unlisted", "input": "Hello", "stream": True},
        headers={"authorization": f"Bearer {MASTER_KEY}"},
    )
    assert response.status_code == 200, response.text
    assert provider.called
    assert not chatgpt.called


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
    _account_catalog(respx_mock)
    provider: Final = _open_weights_provider(respx_mock)
    chatgpt: Final = _chatgpt_responses(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "open-weights", "input": "Hello", "stream": True},
        headers={
            **_ACCOUNT_HEADERS,
            "x-litellm-api-key": restricted_key(["chatgpt-native", "open-weights"], allowed_routes),
        },
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


_LISTED: Final = httpx.Response(200, json=_UPSTREAM_CATALOG)


@pytest.mark.parametrize(
    ("model", "user_agent", "catalog_answer", "subscription"),
    [
        pytest.param("shared-name", _CODEX_USER_AGENT, _LISTED, True, id="listed"),
        pytest.param("open-weights", _CODEX_USER_AGENT, _LISTED, False, id="unlisted"),
        pytest.param("open-weights", _CODEX_USER_AGENT, httpx.Response(503), True, id="catalog unavailable"),
        pytest.param("open-weights", _CODEX_USER_AGENT, httpx.ReadTimeout("slow"), True, id="catalog timeout"),
        pytest.param("open-weights", "codex_cli_rs", _LISTED, True, id="no version"),
        pytest.param("open-weights", "curl/8.5.0", _LISTED, True, id="user agent of another client"),
    ],
)
def test_native_responses_ask_chatgpt_which_models_the_account_lists_on_a_cache_miss(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    model: str,
    user_agent: str,
    catalog_answer: httpx.Response | httpx.TimeoutException,
    subscription: bool,
) -> None:
    catalog: Final = respx_mock.get(f"{_BASE}/models").mock(side_effect=[catalog_answer])
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


def test_a_failed_account_catalog_read_keeps_the_next_turns_on_the_subscription_without_asking_again(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router
) -> None:
    catalog: Final = respx_mock.get(f"{_BASE}/models").respond(503)
    provider: Final = _open_weights_provider(respx_mock)
    chatgpt: Final = _chatgpt_responses(respx_mock)
    responses: Final = tuple(
        client.post(
            "/backend-api/codex/responses",
            json={"model": "open-weights", "input": "Hello", "stream": True},
            headers=_ACCOUNT_HEADERS,
        )
        for _ in range(2)
    )
    assert [response.status_code for response in responses] == [200, 200]
    assert chatgpt.call_count == 2
    assert not provider.called
    assert catalog.call_count == 1


def test_a_failed_account_catalog_read_with_another_login_leaves_the_accounts_own_list_reachable(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router
) -> None:
    own_login: Final = f"Bearer {_CHATGPT_JWT}"
    catalog: Final = respx_mock.get(f"{_BASE}/models").mock(
        side_effect=lambda request: (
            httpx.Response(200, json=_UPSTREAM_CATALOG)
            if request.headers["authorization"] == own_login
            else httpx.Response(401)
        )
    )
    provider: Final = _open_weights_provider(respx_mock)
    chatgpt: Final = _chatgpt_responses(respx_mock)
    turn: Final = {"model": "open-weights", "input": "Hello", "stream": True}
    other_login: Final = client.post(
        "/backend-api/codex/responses",
        json=turn,
        headers={**_ACCOUNT_HEADERS, "authorization": "Bearer test-someone-elses-login"},
    )
    own: Final = client.post("/backend-api/codex/responses", json=turn, headers=_ACCOUNT_HEADERS)
    assert other_login.status_code == own.status_code == 200
    assert chatgpt.call_count == 1
    assert provider.call_count == 1
    assert [call.request.headers["authorization"] for call in catalog.calls] == [
        "Bearer test-someone-elses-login",
        own_login,
    ]


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


def _models_etags(client: TestClient, headers: dict[str, str]) -> tuple[str | None, str | None]:
    catalog: Final = client.get("/backend-api/codex/models?client_version=0.165.0", headers=headers)
    responses: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "gpt-unlisted", "input": "Hello", "stream": True},
        headers=headers,
    )
    assert catalog.status_code == responses.status_code == 200, (catalog.text, responses.text)
    return catalog.headers.get("etag"), responses.headers.get("x-models-etag")


@pytest.mark.parametrize(
    ("models", "merged"),
    [
        pytest.param(["chatgpt-native", "open-weights"], True, id="catalog with LiteLLM models"),
        pytest.param(["chatgpt-native"], False, id="catalog of ChatGPT models only"),
    ],
)
def test_native_catalog_and_responses_carry_the_same_models_etag(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    restricted_key: Callable[..., str],
    models: list[str],
    merged: bool,
) -> None:
    respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG, headers={"etag": _UPSTREAM_MODELS_ETAG})
    _chatgpt_responses(respx_mock)
    catalog_etag, responses_etag = _models_etags(client, {**_HEADERS, "x-litellm-api-key": restricted_key(models)})
    assert catalog_etag is not None
    assert catalog_etag == responses_etag
    assert (catalog_etag != _UPSTREAM_MODELS_ETAG) is merged


def test_native_models_etag_is_the_same_on_replicas_with_empty_caches(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(proxy_server.app.dependency_overrides, codex_shared_cache, lambda: DualCache())
    respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG, headers={"etag": _UPSTREAM_MODELS_ETAG})
    _chatgpt_responses(respx_mock)
    first_replica: Final = _models_etags(client, _HEADERS)
    second_replica: Final = _models_etags(client, _HEADERS)
    assert first_replica[0] not in (None, _UPSTREAM_MODELS_ETAG)
    assert len({*first_replica, *second_replica}) == 1


def test_native_models_etag_changes_with_the_keys_litellm_models_and_with_upstreams_etag(
    client: TestClient, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_weights: Final = _open_weights("open-weights")
    wider_window: Final = {**open_weights, "model_info": {"max_input_tokens": 64000}}

    def catalog_etag(deployment: dict[str, object], upstream_etag: str) -> str | None:
        monkeypatch.setattr(proxy_server, "llm_router", Router(model_list=[deployment]))
        respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG, headers={"etag": upstream_etag})
        return client.get("/backend-api/codex/models?client_version=0.165.0", headers=_HEADERS).headers.get("etag")

    baseline: Final = catalog_etag(open_weights, '"upstream-v1"')
    assert baseline == catalog_etag(open_weights, '"upstream-v1"')
    assert (
        len({baseline, catalog_etag(wider_window, '"upstream-v1"'), catalog_etag(open_weights, '"upstream-v3"')}) == 3
    )
    assert baseline not in (None, '"upstream-v1"')


class FailingModelListing(CustomLogger):
    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        raise RuntimeError("model listing backend unavailable")


def test_native_catalog_relays_chatgpts_answer_when_listing_the_keys_models_fails(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [FailingModelListing()])
    respx_mock.get(f"{_BASE}/models").respond(json=_UPSTREAM_CATALOG, headers={"etag": '"upstream-v1"'})
    response: Final = client.get("/backend-api/codex/models?client_version=0.165.0", headers=_HEADERS)
    assert response.status_code == 200
    assert response.json() == _UPSTREAM_CATALOG
    assert response.headers["etag"] == '"upstream-v1"'


def test_native_responses_keep_chatgpts_models_etag_when_listing_the_keys_models_fails(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [FailingModelListing()])
    chatgpt: Final = _chatgpt_responses(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "gpt-unlisted", "input": "Hello", "stream": True},
        headers=_HEADERS,
    )
    assert response.status_code == 200
    assert chatgpt.called
    assert response.headers["x-models-etag"] == _UPSTREAM_MODELS_ETAG


_OPENAI_BASE: Final = "https://openai-api.test/v1"
_OPENAI_BLOB: Final = "gAAAAB-openai-reasoning-blob"
_OPENAI_REASONING: Final = {"type": "reasoning", "id": "rs_openai", "summary": [], "encrypted_content": _OPENAI_BLOB}


def _openai_reply(request: httpx.Request) -> httpx.Response:
    completed: Final = {
        "id": "resp_openai",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": "gpt-5.6",
        "output": [
            _OPENAI_REASONING,
            {
                "type": "message",
                "id": "msg_openai",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
            },
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }
    if json.loads(request.content).get("stream") is not True:
        return httpx.Response(200, json=completed)
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**completed, "status": "in_progress", "output": []},
        },
        {"type": "response.output_item.done", "sequence_number": 1, "output_index": 0, "item": _OPENAI_REASONING},
        {"type": "response.completed", "sequence_number": 2, "response": completed},
    )
    return httpx.Response(
        200,
        content="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode(),
        headers={"content-type": "text/event-stream"},
    )


def _reasoning_item_of(response: httpx.Response, stream: bool) -> dict[str, object]:
    if not stream:
        return response.json()["output"][0]
    payloads: Final = (
        json.loads(line.removeprefix(b"data: "))
        for line in response.content.splitlines()
        if line.startswith(b"data: {")
    )
    return next(event["item"] for event in payloads if event["type"] == "response.output_item.done")


@pytest.mark.parametrize("stream", [False, True])
def test_openai_reasoning_from_the_native_route_is_dropped_for_chatgpt_and_unwrapped_for_litellm(
    client: TestClient, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "api-model",
                    "litellm_params": {
                        "model": "openai/gpt-5.6",
                        "api_base": _OPENAI_BASE,
                        "api_key": "test-openai-key",
                    },
                }
            ]
        ),
    )
    _account_catalog(respx_mock)
    provider: Final = respx_mock.post(f"{_OPENAI_BASE}/responses").mock(side_effect=_openai_reply)
    chatgpt: Final = _chatgpt_responses(respx_mock)
    turn: Final = {"model": "api-model", "input": [_USER_TURN], "stream": stream, "store": False}
    first: Final = client.post("/backend-api/codex/responses", json=turn, headers=_ACCOUNT_HEADERS)
    assert first.status_code == 200, first.text
    reasoning: Final = _reasoning_item_of(first, stream)
    model_id, blob = ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(
        str(reasoning["encrypted_content"])
    )
    assert (model_id is not None, blob) == (True, _OPENAI_BLOB)
    history: Final = [_USER_TURN, reasoning, _ASSISTANT_TURN, _USER_TURN]
    to_chatgpt: Final = client.post(
        "/backend-api/codex/responses",
        json={**turn, "model": "subscription-model", "input": history, "stream": True},
        headers=_ACCOUNT_HEADERS,
    )
    back_to_litellm: Final = client.post(
        "/backend-api/codex/responses", json={**turn, "input": history}, headers=_ACCOUNT_HEADERS
    )
    v1: Final = client.post(
        "/v1/responses", json={**turn, "stream": False}, headers={"authorization": f"Bearer {MASTER_KEY}"}
    )
    assert to_chatgpt.status_code == back_to_litellm.status_code == v1.status_code == 200
    assert json.loads(chatgpt.calls.last.request.content)["input"] == [_USER_TURN, _ASSISTANT_TURN, _USER_TURN]
    replayed: Final = json.loads(provider.calls[1].request.content)["input"][1]
    assert (replayed["id"], replayed["encrypted_content"]) == ("rs_openai", _OPENAI_BLOB)
    assert v1.json()["output"][0]["encrypted_content"] == _OPENAI_BLOB


_XAI_BASE: Final = "https://xai-api.test/v1"
_LUNA_ID: Final = "oai-luna-deployment"
_LUNA_BLOB: Final = "gAAAAB-luna-reasoning-blob"
_LUNA_REASONING: Final = {
    "type": "reasoning",
    "summary": [],
    "encrypted_content": ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id(_LUNA_BLOB, _LUNA_ID),
}
_CHATGPT_REASONING: Final = {"type": "reasoning", "summary": [], "encrypted_content": "gAAAAB-chatgpt-reasoning-blob"}


@pytest.fixture
def xai_fireworks_and_openai_router(monkeypatch: pytest.MonkeyPatch) -> Router:
    router: Final = Router(
        model_list=[
            {
                "model_name": "grok-4.7",
                "litellm_params": {"model": "xai/grok-4.7", "api_base": _XAI_BASE, "api_key": "test-xai-key"},
            },
            _open_weights("open-weights"),
            {
                "model_name": "oai-luna",
                "litellm_params": {
                    "model": "openai/gpt-6-luna",
                    "api_base": _OPENAI_BASE,
                    "api_key": "test-openai-key",
                },
                "model_info": {"id": _LUNA_ID},
            },
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    return router


@pytest.mark.parametrize(
    ("model", "replayed", "expected"),
    [
        pytest.param("open-weights", _CHATGPT_REASONING, None, id="chatgpt reasoning to fireworks is dropped"),
        pytest.param("open-weights", _LUNA_REASONING, None, id="another deployment's reasoning to fireworks is dropped"),
        pytest.param("oai-luna", _LUNA_REASONING, _LUNA_BLOB, id="the minting deployment's reasoning is kept"),
    ],
)
def test_litellm_model_on_native_route_gets_only_the_reasoning_it_can_decrypt(
    client: TestClient,
    respx_mock: respx.MockRouter,
    xai_fireworks_and_openai_router: Router,
    model: str,
    replayed: dict[str, object],
    expected: str | None,
) -> None:
    _account_catalog(respx_mock)
    fireworks: Final = _open_weights_provider(respx_mock)
    luna: Final = respx_mock.post(f"{_OPENAI_BASE}/responses").mock(side_effect=_provider_reply)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={
            "model": model,
            "input": [_USER_TURN, replayed, _ASSISTANT_TURN, _USER_TURN],
            "stream": False,
            "store": False,
        },
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200, response.text
    provider: Final = fireworks if model == "open-weights" else luna
    sent: Final = json.loads(provider.calls.last.request.content)["input"]
    assert [item.get("encrypted_content") for item in sent if item.get("type") == "reasoning"] == (
        [] if expected is None else [expected]
    )


_WHOLE_FIREWORKS_REASONING: Final = {**_FIREWORKS_REASONING, "status": "completed"}
_SUMMARY_ONLY_REASONING: Final = {key: value for key, value in _FIREWORKS_REASONING.items() if key != "id"}
_OTHER_DEPLOYMENTS_BLOB: Final = ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id(
    "gAAAAB-other-deployment-blob", "other-deployment"
)


@pytest.mark.parametrize(
    ("model", "replayed", "store", "expected"),
    [
        pytest.param(
            "oai-luna", _WHOLE_FIREWORKS_REASONING, False, [_SUMMARY_ONLY_REASONING], id="whole fireworks item to openai"
        ),
        pytest.param(
            "oai-luna",
            {**_WHOLE_FIREWORKS_REASONING, "encrypted_content": _OTHER_DEPLOYMENTS_BLOB},
            False,
            [_SUMMARY_ONLY_REASONING],
            id="whole item with another deployment's blob to openai",
        ),
        pytest.param(
            "oai-luna",
            {**_SUMMARY_ONLY_REASONING, "content": [], "encrypted_content": _OTHER_DEPLOYMENTS_BLOB},
            False,
            [{**_SUMMARY_ONLY_REASONING, "content": []}],
            id="codex's replay of another deployment's item to openai",
        ),
        pytest.param(
            "oai-luna",
            {**_SUMMARY_ONLY_REASONING, "content": None, "encrypted_content": None},
            False,
            [_SUMMARY_ONLY_REASONING],
            id="codex's replay of a blobless item to openai",
        ),
        pytest.param(
            "oai-luna",
            {"type": "reasoning", "id": "rs_unreadable", "summary": []},
            False,
            [],
            id="an unresolvable id with nothing readable is dropped",
        ),
        pytest.param(
            "open-weights",
            {**_SUMMARY_ONLY_REASONING, "id": "rs_openai"},
            False,
            [_SUMMARY_ONLY_REASONING],
            id="openai summary-only item to fireworks",
        ),
        pytest.param(
            "oai-luna",
            {**_LUNA_REASONING, "id": "rs_luna", "content": []},
            False,
            [{**_LUNA_REASONING, "id": "rs_luna", "content": [], "encrypted_content": _LUNA_BLOB}],
            id="the minting deployment gets its item whole",
        ),
        pytest.param(
            "oai-luna", _FIREWORKS_REASONING, True, [_FIREWORKS_REASONING], id="a stored conversation keeps the id"
        ),
    ],
)
def test_reasoning_the_target_cannot_resolve_reaches_it_without_its_id(
    client: TestClient,
    respx_mock: respx.MockRouter,
    xai_fireworks_and_openai_router: Router,
    model: str,
    replayed: dict[str, object],
    store: bool,
    expected: list[dict[str, object]],
) -> None:
    _account_catalog(respx_mock)
    fireworks: Final = _open_weights_provider(respx_mock)
    luna: Final = respx_mock.post(f"{_OPENAI_BASE}/responses").mock(side_effect=_provider_reply)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={
            "model": model,
            "input": [_USER_TURN, replayed, _ASSISTANT_TURN, _USER_TURN],
            "stream": False,
            "store": store,
        },
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200, response.text
    sent: Final = json.loads((fireworks if model == "open-weights" else luna).calls.last.request.content)["input"]
    assert [item for item in sent if item.get("type") == "reasoning"] == expected


def test_reasoning_the_chat_bridge_target_cannot_resolve_reaches_it_as_its_summary(
    client: TestClient, respx_mock: respx.MockRouter, xai_fireworks_and_openai_router: Router
) -> None:
    _account_catalog(respx_mock)
    chat_endpoint: Final = respx_mock.post(f"{_XAI_BASE}/chat/completions").mock(side_effect=_xai_search_call_reply)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={
            "model": "grok-4.7",
            "input": [_USER_TURN, _WHOLE_FIREWORKS_REASONING, _ASSISTANT_TURN, _USER_TURN],
            "stream": False,
            "store": False,
        },
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200, response.text
    messages: Final = json.loads(chat_endpoint.calls.last.request.content)["messages"]
    assert [message.get("reasoning_content") for message in messages if message["role"] == "assistant"] == [
        "Thinking it over"
    ]


def _namespace(name: str, *members: str) -> dict[str, object]:
    return {
        "type": "namespace",
        "name": name,
        "description": "",
        "tools": [{"type": "function", "name": member, "parameters": {"type": "object"}} for member in members],
    }


_EXEC_TOOL: Final = {"type": "function", "name": "exec_command", "parameters": {"type": "object"}}
_ADDITIONAL_TOOLS: Final = {
    "type": "additional_tools",
    "role": "developer",
    "tools": [_namespace("collaboration", "followup_task"), _namespace("mcp__docs__", "search")],
}


@pytest.mark.parametrize(
    ("model", "expected_input", "expected_tool_names"),
    [
        pytest.param(
            "open-weights",
            [_USER_TURN],
            ["exec_command", "collaboration", "mcp__docs__"],
            id="fireworks gets the item's tools as top-level tools",
        ),
        pytest.param("oai-luna", [_ADDITIONAL_TOOLS, _USER_TURN], ["exec_command"], id="the openai api keeps the item"),
    ],
)
def test_codex_additional_tools_item_reaches_each_provider_in_a_shape_it_accepts(
    client: TestClient,
    respx_mock: respx.MockRouter,
    xai_fireworks_and_openai_router: Router,
    model: str,
    expected_input: list[dict[str, object]],
    expected_tool_names: list[str],
) -> None:
    _account_catalog(respx_mock)
    fireworks: Final = _open_weights_provider(respx_mock)
    luna: Final = respx_mock.post(f"{_OPENAI_BASE}/responses").mock(side_effect=_provider_reply)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={
            "model": model,
            "input": [_ADDITIONAL_TOOLS, _USER_TURN],
            "tools": [_EXEC_TOOL],
            "stream": False,
            "store": False,
        },
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200, response.text
    sent: Final = json.loads((fireworks if model == "open-weights" else luna).calls.last.request.content)
    assert sent["input"] == expected_input
    assert [tool["name"] for tool in sent["tools"]] == expected_tool_names


_ANTHROPIC_BASE: Final = "https://anthropic-api.test"


def test_codex_namespace_tool_called_through_the_chat_bridge_comes_back_with_its_namespace(
    client: TestClient, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "claude-sonnet-5-5",
                    "litellm_params": {
                        "model": "anthropic/claude-sonnet-5-5",
                        "api_base": _ANTHROPIC_BASE,
                        "api_key": "test-anthropic-key",
                    },
                }
            ]
        ),
    )
    _account_catalog(respx_mock)
    anthropic: Final = respx_mock.post(f"{_ANTHROPIC_BASE}/v1/messages").respond(
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [
                {"type": "tool_use", "id": "toolu_test", "name": "mcp__docs____search", "input": {"query": "proxy"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    )
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={
            "model": "claude-sonnet-5-5",
            "input": [_ADDITIONAL_TOOLS, _USER_TURN],
            "tools": [_EXEC_TOOL],
            "stream": False,
            "store": False,
        },
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert anthropic.called
    (call,) = (item for item in response.json()["output"] if item["type"] == "function_call")
    assert (call["namespace"], call["name"]) == ("mcp__docs__", "search")


_CLOCK_NAMESPACE: Final = _namespace("clock", "sleep")
_CLOCK_TOOLS: Final = {"type": "additional_tools", "role": "developer", "tools": [_CLOCK_NAMESPACE]}
def test_codex_namespace_tool_reaches_fireworks_as_a_namespace(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router
) -> None:
    _account_catalog(respx_mock)
    provider: Final = _open_weights_provider(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={"model": "open-weights", "input": [_CLOCK_TOOLS, _USER_TURN], "stream": False, "store": False},
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert json.loads(provider.calls.last.request.content)["tools"] == [_CLOCK_NAMESPACE]



@pytest.mark.parametrize(
    ("drop_params", "expected_tools"),
    [
        pytest.param({}, [_CLOCK_NAMESPACE], id="codex's drop_params default merges the copies"),
        pytest.param({"drop_params": False}, [_CLOCK_NAMESPACE, _CLOCK_NAMESPACE], id="the caller's off sends each"),
    ],
)
def test_codex_tool_list_repeated_on_every_turn_reaches_fireworks_merged_only_under_drop_params(
    client: TestClient,
    respx_mock: respx.MockRouter,
    mixed_router: Router,
    drop_params: dict[str, bool],
    expected_tools: list[dict[str, object]],
) -> None:
    _account_catalog(respx_mock)
    provider: Final = _open_weights_provider(respx_mock)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={
            "model": "open-weights",
            "input": [_CLOCK_TOOLS, _USER_TURN, _CLOCK_TOOLS],
            "stream": False,
            "store": False,
            **drop_params,
        },
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert json.loads(provider.calls.last.request.content)["tools"] == expected_tools

_CUSTOM_EXEC_TOOL: Final = {"type": "custom", "name": "exec", "description": "Run a command", "format": {"type": "text"}}
_XAI_SEARCH_CALL: Final = {
    "id": "call_search",
    "type": "function",
    "function": {"name": "mcp__docs____search", "arguments": '{"query":"proxy"}'},
}
_XAI_USAGE: Final = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def _xai_chat_chunk(choice: dict[str, object], **fields: object) -> str:
    chunk: Final = {
        "id": "chatcmpl-xai",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "grok-4.7",
        "choices": [{"index": 0, **choice}],
        **fields,
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _xai_search_call_reply(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream") is not True:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-xai",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "grok-4.7",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None, "tool_calls": [_XAI_SEARCH_CALL]},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": _XAI_USAGE,
            },
        )
    stream: Final = (
        _xai_chat_chunk(
            {"delta": {"role": "assistant", "tool_calls": [{"index": 0, **_XAI_SEARCH_CALL}]}, "finish_reason": None}
        )
        + _xai_chat_chunk({"delta": {}, "finish_reason": "tool_calls"}, usage=_XAI_USAGE)
        + "data: [DONE]\n\n"
    )
    return httpx.Response(200, content=stream.encode(), headers={"content-type": "text/event-stream"})


def _data_lines(stream: bytes) -> list[bytes]:
    return [line.removeprefix(b"data: ") for line in stream.splitlines() if line.startswith(b"data: {")]


def _function_calls_of(response: httpx.Response, stream: bool) -> list[dict[str, object]]:
    items: Final = (
        response.json()["output"]
        if not stream
        else [
            event["item"]
            for event in map(json.loads, _data_lines(response.content))
            if event["type"] == "response.output_item.done"
        ]
    )
    return [item for item in items if item["type"] == "function_call"]


@pytest.mark.parametrize("stream", [False, True])
def test_codex_turn_for_xai_goes_through_chat_completions_and_its_namespaced_call_comes_back_whole(
    client: TestClient, respx_mock: respx.MockRouter, xai_fireworks_and_openai_router: Router, stream: bool
) -> None:
    _account_catalog(respx_mock)
    responses_endpoint: Final = respx_mock.post(f"{_XAI_BASE}/responses").respond(json=_provider_completed("Hi"))
    chat_endpoint: Final = respx_mock.post(f"{_XAI_BASE}/chat/completions").mock(side_effect=_xai_search_call_reply)
    response: Final = client.post(
        "/backend-api/codex/responses",
        json={
            "model": "grok-4.7",
            "input": [_ADDITIONAL_TOOLS, _USER_TURN],
            "tools": [_EXEC_TOOL, _CUSTOM_EXEC_TOOL],
            "stream": stream,
            "store": False,
        },
        headers=_ACCOUNT_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert not responses_endpoint.called
    sent_tools: Final = json.loads(chat_endpoint.calls.last.request.content)["tools"]
    assert [(tool["type"], tool["function"]["name"]) for tool in sent_tools] == [
        ("function", "exec_command"),
        ("function", "exec"),
        ("function", "collaboration__followup_task"),
        ("function", "mcp__docs____search"),
    ]
    (call,) = _function_calls_of(response, stream)
    assert (call["name"], call["namespace"], call["call_id"]) == ("search", "mcp__docs__", "call_search")


def test_encrypted_reasoning_stays_only_when_every_deployment_the_router_may_pick_can_decrypt_it(
    client: TestClient, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    second_base: Final = "https://second-openai-org.test/v1"

    def replayed_blobs(optional_pre_call_checks: list[str]) -> list[object]:
        monkeypatch.setattr(
            proxy_server,
            "llm_router",
            Router(
                model_list=[
                    {
                        "model_name": "oai-pair",
                        "litellm_params": {"model": "openai/gpt-6-luna", "api_base": base, "api_key": "test-key"},
                        "model_info": {"id": model_id},
                    }
                    for base, model_id in ((_OPENAI_BASE, _LUNA_ID), (second_base, "second-org-deployment"))
                ],
                optional_pre_call_checks=optional_pre_call_checks,
            ),
        )
        providers: Final = tuple(
            respx_mock.post(f"{base}/responses").mock(side_effect=_provider_reply)
            for base in (_OPENAI_BASE, second_base)
        )
        response: Final = client.post(
            "/backend-api/codex/responses",
            json={"model": "oai-pair", "input": [_USER_TURN, _LUNA_REASONING, _USER_TURN], "store": False},
            headers=_ACCOUNT_HEADERS,
        )
        assert response.status_code == 200, response.text
        (sent,) = (json.loads(call.request.content)["input"] for provider in providers for call in provider.calls)
        return [item.get("encrypted_content") for item in sent if item.get("type") == "reasoning"]

    _account_catalog(respx_mock)
    assert replayed_blobs([]) == []
    respx_mock.reset()
    _account_catalog(respx_mock)
    assert replayed_blobs(["encrypted_content_affinity"]) == [_LUNA_BLOB]


class ModelListingCounter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.listings: Final[list[tuple[str, ...]]] = []

    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        self.listings.append(tuple(model_names))
        return model_names


def test_jwt_caller_builds_the_models_etag_once_across_chatgpt_turns(
    client: TestClient, respx_mock: respx.MockRouter, mixed_router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter: Final = ModelListingCounter()
    monkeypatch.setattr(litellm, "callbacks", [counter])
    monkeypatch.setitem(
        proxy_server.app.dependency_overrides,
        user_api_key_auth,
        lambda: UserAPIKeyAuth(api_key=None, user_id="jwt-user", user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    _chatgpt_responses(respx_mock)
    turns: Final = tuple(
        client.post(
            "/backend-api/codex/responses",
            json={"model": "gpt-unlisted", "input": "Hello", "stream": True},
            headers={**_HEADERS, "x-litellm-api-key": "test-jwt-credential"},
        )
        for _ in range(2)
    )
    assert [turn.status_code for turn in turns] == [200, 200]
    assert len(counter.listings) == 1
    first_etag, second_etag = (turn.headers["x-models-etag"] for turn in turns)
    assert first_etag == second_etag != _UPSTREAM_MODELS_ETAG
