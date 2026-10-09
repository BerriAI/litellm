from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

import pytest
import respx
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import Match

import litellm
from litellm.proxy._lazy_features import LAZY_FEATURES, LazyFeature, attach_lazy_features
from litellm.proxy.decisions_endpoints.endpoints import decisions, systemone
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import SafeRouteAdder
from litellm.proxy.proxy_server import (
    app,
    cleanup_router_config_variables,
    initialize,
)

_INPUT_TOKENS: Final[int] = 367
_OUTPUT_TOKENS: Final[int] = 3
_RESPONSE: Final[Mapping[str, object]] = {
    "model": "pplx-decider-v1-27b",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
    },
    "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS},
}
_STRANDS_RESPONSE: Final[Mapping[str, object]] = {
    "model": "strands-decider-2B-hobson-v19",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
    },
    "usage": {"input_tokens": 216, "output_tokens": 3},
    "latency_ms": 3722.17,
}
_REQUEST: Final[Mapping[str, object]] = {
    "model": "decider",
    "state": {"source": "proxy-test"},
    "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://fake-openai.example")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    cleanup_router_config_variables()
    config_path: Final = Path(__file__).parents[1] / "test_configs" / "test_config_no_auth.yaml"
    asyncio.run(initialize(config=str(config_path), debug=True))
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "decider",
                    "litellm_params": {
                        "model": "perplexity/pplx-decider-v1-27b",
                        "api_key": "test-key",
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield TestClient(app)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.parametrize("endpoint", ("/v1/systemone", "/systemone"))
def test_proxy_decisions_route_returns_answers_and_cost(
    client: TestClient,
    respx_mock: respx.MockRouter,
    endpoint: str,
) -> None:
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post(endpoint, json=_REQUEST)

    assert response.status_code == 200, response.text
    assert response.json()["answers"] == _RESPONSE["answers"]
    assert "_hidden_params" not in response.json()
    perplexity_cost: Final = litellm.model_cost["perplexity/pplx-decider-v1-27b"]
    expected_cost: Final = _INPUT_TOKENS * float(perplexity_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        perplexity_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected_cost)
    assert upstream.called
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "pplx-decider-v1-27b",
        "state": {"source": "proxy-test"},
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert upstream.calls[0].request.headers["authorization"] == "Bearer test-key"


def test_proxy_decisions_dispatches_typesafe_deployment(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "jev",
                "litellm_params": {
                    "model": "typesafe/jev-latest",
                    "api_key": "k",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    upstream: Final = respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(json=_RESPONSE)

    response: Final = client.post(
        "/v1/systemone",
        json={
            "model": "jev",
            "state": {"source": "proxy-test"},
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["answers"] == _RESPONSE["answers"]
    assert upstream.called
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "jev-latest",
        "state": {"source": "proxy-test"},
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }


def test_proxy_decisions_sends_the_env_key_to_the_deployment_api_base(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "server-key")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "decider",
                "litellm_params": {
                    "model": "perplexity/pplx-decider-v1-27b",
                    "api_base": "https://egress.example/perplexity",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    upstream: Final = respx_mock.post("https://egress.example/perplexity/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post("/v1/systemone", json=_REQUEST)

    assert response.status_code == 200, response.text
    assert upstream.call_count == 1
    assert upstream.calls[0].request.headers["authorization"] == "Bearer server-key"


def test_proxy_decisions_unknown_model_is_a_client_error(
    client: TestClient,
    respx_mock: respx.MockRouter,
) -> None:
    response: Final = client.post(
        "/v1/systemone",
        json={
            "model": "missing-model",
            "state": "review",
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
    )

    assert 400 <= response.status_code < 500, response.text
    assert len(respx_mock.calls) == 0


@pytest.mark.parametrize(
    "request_body",
    (
        {
            "model": "decider",
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
        {
            "model": "decider",
            "state": {"source": "proxy-test"},
        },
    ),
    ids=("missing_state", "missing_questions"),
)
def test_proxy_decisions_missing_required_field_is_a_client_error(
    client: TestClient,
    respx_mock: respx.MockRouter,
    request_body: Mapping[str, object],
) -> None:
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post("/v1/systemone", json=request_body)

    assert response.status_code == 400, response.text
    assert not upstream.called


def test_proxy_decisions_dispatches_strands_decider(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "strands",
                "litellm_params": {
                    "model": "strands_decider/strands-decider-2B-hobson-v19",
                    "api_base": "https://strands.example",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    upstream: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    response: Final = client.post(
        "/v1/systemone",
        json={
            "model": "strands",
            "state": {"source": "proxy-test"},
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["answers"] == _STRANDS_RESPONSE["answers"]
    assert upstream.called
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "strands-decider-2B-hobson-v19",
        "state": {"source": "proxy-test"},
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert "authorization" not in upstream.calls[0].request.headers


def test_proxy_decisions_without_model_uses_the_proxy_default_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_model", "decider")
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post(
        "/v1/systemone", json={key: value for key, value in _REQUEST.items() if key != "model"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["answers"] == _RESPONSE["answers"]
    assert upstream.called
    assert json.loads(upstream.calls[0].request.content)["model"] == "pplx-decider-v1-27b"


_OPENAI_FORMAT_REQUEST: Final[Mapping[str, object]] = {
    "model": "decider",
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "The package arrived with a broken screen."},
                {"type": "input_text", "text": "I want a refund."},
            ],
        },
        {"role": "user", "content": "Order 1234."},
    ],
    "questions": [
        {"type": "predicate", "name": "damaged", "instructions": "Does the customer report a damaged item?"},
        {
            "type": "choice",
            "instructions": "Should we refund?",
            "choices": [{"value": True, "description": "Refund now"}, {"value": "escalate"}],
        },
        {
            "type": "score",
            "name": "severity",
            "instructions": "How severe is the issue?",
            "levels": [{"label": "minor"}, {"label": "major", "description": "Product unusable"}],
        },
        {"type": "predicate", "name": "fraud", "instructions": "Is this fraud?"},
    ],
    "safety_identifier": "end-user-1",
}
_SYSTEMONE_ANSWERS_FOR_OPENAI_REQUEST: Final[Mapping[str, object]] = {
    "model": "pplx-decider-v1-27b",
    "answers": {
        "0": {"type": "noul", "noul": 0.95},
        "1": {"type": "choice", "choice": "true", "confidence": 0.8, "probabilities": {"true": 0.9, "escalate": 0.1}},
        "2": {
            "type": "score",
            "score": 0.7,
            "confidence": 0.6,
            "legend": {"0": "minor", "1": "major: Product unusable"},
            "probabilities": {"0": 0.3, "1": 0.7},
        },
    },
    "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS},
}

_OPENAI_FORMAT_ANSWERS: Final = [
    {"type": "predicate", "name": "damaged", "probability": 0.95},
    {
        "type": "choice",
        "name": None,
        "choice": True,
        "probabilities": [{"value": True, "probability": 0.9}, {"value": "escalate", "probability": 0.1}],
        "confidence": 0.8,
    },
    {
        "type": "score",
        "name": "severity",
        "score": 0.7,
        "probabilities": [
            {"value": 0, "label": "minor", "probability": 0.3},
            {"value": 1, "label": "major", "probability": 0.7},
        ],
        "confidence": 0.6,
    },
    {"type": "refusal", "name": "fraud"},
]


@pytest.mark.parametrize("endpoint", ("/v1/decisions", "/decisions"))
def test_openai_format_decisions_translate_through_systemone(
    client: TestClient,
    respx_mock: respx.MockRouter,
    endpoint: str,
) -> None:
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        json=_SYSTEMONE_ANSWERS_FOR_OPENAI_REQUEST
    )

    response: Final = client.post(endpoint, json=_OPENAI_FORMAT_REQUEST)

    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "pplx-decider-v1-27b",
        "state": "The package arrived with a broken screen.\n\nI want a refund.\n\nOrder 1234.",
        "questions": {
            "0": {"type": "noul", "instructions": "Does the customer report a damaged item?"},
            "1": {
                "type": "choice",
                "instructions": "Should we refund?",
                "criteria": {"true": "Refund now", "escalate": None},
            },
            "2": {
                "type": "score",
                "instructions": "How severe is the issue?",
                "criteria": ["minor", "major: Product unusable"],
            },
            "3": {"type": "noul", "instructions": "Is this fraud?"},
        },
    }
    body: Final = response.json()
    assert body["model"] == _SYSTEMONE_ANSWERS_FOR_OPENAI_REQUEST["model"]
    assert body["answers"] == _OPENAI_FORMAT_ANSWERS
    assert body["usage"] == {
        "input_tokens": _INPUT_TOKENS,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": _OUTPUT_TOKENS,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
    }
    assert float(response.headers["x-litellm-response-cost"]) > 0


@pytest.mark.parametrize(
    ("endpoint", "request_body", "upstream_response"),
    (
        ("/v1/systemone", _REQUEST, _RESPONSE),
        ("/v1/decisions", _OPENAI_FORMAT_REQUEST, _SYSTEMONE_ANSWERS_FOR_OPENAI_REQUEST),
    ),
    ids=("systemone", "openai_format"),
)
def test_decisions_return_guardrail_information_when_requested(
    client: TestClient,
    respx_mock: respx.MockRouter,
    endpoint: str,
    request_body: Mapping[str, object],
    upstream_response: Mapping[str, object],
) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=upstream_response)

    response: Final = client.post(endpoint, json={**request_body, "include_guardrail_response": True})

    assert response.status_code == 200, response.text
    assert response.json()["guardrail_information"] == []


@pytest.mark.parametrize(
    "request_body",
    (
        _REQUEST,
        {
            "model": "decider",
            "input": "review",
            "questions": [
                {
                    "type": "choice",
                    "instructions": "Pick one",
                    "choices": [{"value": True}, {"value": "true"}],
                }
            ],
        },
        {
            "model": "decider",
            "input": [
                {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}
            ],
            "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
        },
    ),
    ids=("systemone_body", "colliding_choice_values", "image_input"),
)
def test_openai_format_decisions_rejects_bodies_it_cannot_translate(
    client: TestClient,
    respx_mock: respx.MockRouter,
    request_body: Mapping[str, object],
) -> None:
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post("/v1/decisions", json=request_body)

    assert response.status_code == 400, response.text
    assert not upstream.called


@pytest.mark.parametrize("endpoint", ("/v1/systemone", "/v1/decisions"))
@pytest.mark.parametrize("raw_body", (b"", b"{not json"), ids=("empty", "malformed"))
def test_a_body_that_is_not_json_is_a_client_error(
    client: TestClient,
    respx_mock: respx.MockRouter,
    endpoint: str,
    raw_body: bytes,
) -> None:
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = client.post(endpoint, content=raw_body, headers={"Content-Type": "application/json"})

    assert response.status_code == 400, response.text
    assert not upstream.called


def test_openai_format_decisions_reach_an_openai_deployment_unchanged_including_images(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[{"model_name": "decider", "litellm_params": {"model": "openai/gpt-6-luna", "api_key": "k"}}]
        ),
    )
    image_message: Final = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "low"}],
    }
    request_body: Final = {**_OPENAI_FORMAT_REQUEST, "input": [*_OPENAI_FORMAT_REQUEST["input"], image_message]}
    cached_tokens: Final = 128
    upstream_usage: Final = {
        "input_tokens": _INPUT_TOKENS,
        "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
        "output_tokens": _OUTPUT_TOKENS,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
    }
    upstream: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(
        json={"model": "gpt-6-luna", "answers": _OPENAI_FORMAT_ANSWERS, "usage": upstream_usage}
    )

    response: Final = client.post("/v1/decisions", json=request_body)

    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "gpt-6-luna",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "The package arrived with a broken screen."},
                    {"type": "input_text", "text": "I want a refund."},
                ],
            },
            {"type": "message", "role": "user", "content": "Order 1234."},
            image_message,
        ],
        "questions": _OPENAI_FORMAT_REQUEST["questions"],
        "safety_identifier": "end-user-1",
    }
    body: Final = response.json()
    assert body["answers"] == _OPENAI_FORMAT_ANSWERS
    assert body["usage"] == upstream_usage
    luna_cost: Final = litellm.model_cost["gpt-6-luna"]
    expected_cost: Final = (
        (_INPUT_TOKENS - cached_tokens) * float(luna_cost["input_cost_per_token"])
        + cached_tokens * float(luna_cost["cache_read_input_token_cost"])
        + _OUTPUT_TOKENS * float(luna_cost["output_cost_per_token"])
    )
    assert expected_cost > 0
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected_cost)


@pytest.mark.parametrize("safety_identifier", (7, ["end-user-1"]), ids=("numeric", "list"))
@pytest.mark.parametrize("deployment_drops_params", (False, True), ids=("strict", "drop_params"))
def test_a_non_string_safety_identifier_is_refused_unless_the_deployment_drops_params(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    safety_identifier: object,
    deployment_drops_params: bool,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "decider",
                    "litellm_params": {
                        "model": "openai/gpt-6-luna",
                        "api_key": "k",
                        "drop_params": deployment_drops_params,
                    },
                }
            ]
        ),
    )
    upstream: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(
        json={
            "model": "gpt-6-luna",
            "answers": _OPENAI_FORMAT_ANSWERS[:1],
            "usage": {
                "input_tokens": _INPUT_TOKENS,
                "output_tokens": _OUTPUT_TOKENS,
                "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
            },
        }
    )
    request_body: Final = {
        "model": "decider",
        "input": "The package arrived with a broken screen.",
        "questions": _OPENAI_FORMAT_REQUEST["questions"][:1],
        "safety_identifier": safety_identifier,
    }

    response: Final = client.post("/v1/decisions", json=request_body)

    if not deployment_drops_params:
        assert response.status_code == 400, response.text
        assert "safety_identifier" in response.json()["error"]["message"]
        assert not upstream.called
        return
    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "gpt-6-luna",
        "input": "The package arrived with a broken screen.",
        "questions": _OPENAI_FORMAT_REQUEST["questions"][:1],
    }


def _decisions_feature() -> LazyFeature:
    return next(feature for feature in LAZY_FEATURES if feature.name == "decisions")


def _serving_endpoint(bare: FastAPI, path: str) -> object:
    scope: Final = {"type": "http", "method": "POST", "path": path, "root_path": "", "query_string": b"", "headers": ()}
    return next(
        route.endpoint for route in bare.routes if isinstance(route, APIRoute) and route.matches(scope)[0] is Match.FULL
    )


def test_a_config_pass_through_at_v1_decisions_keeps_its_route_and_the_native_api_serves_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_DISABLE_LAZY_ROUTES", raising=False)

    async def pass_through() -> dict[str, str]:
        return {"served_by": "pass-through"}

    bare: Final = FastAPI()
    attach_lazy_features(bare, (_decisions_feature(),))
    SafeRouteAdder.add_api_route_if_not_exists(bare, "/v1/decisions", pass_through, ["POST"])
    with TestClient(bare) as client:
        assert client.post("/v1/decisions", json={"model": "gpt-6-luna"}).json() == {"served_by": "pass-through"}
    assert _serving_endpoint(bare, "/v1/decisions") is pass_through
    assert _serving_endpoint(bare, "/decisions") is decisions
    assert _serving_endpoint(bare, "/v1/systemone") is systemone
    assert _serving_endpoint(bare, "/systemone") is systemone


def test_with_lazy_routes_disabled_a_config_pass_through_at_v1_decisions_still_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_DISABLE_LAZY_ROUTES", "true")

    async def pass_through() -> dict[str, str]:
        return {"served_by": "pass-through"}

    @asynccontextmanager
    async def loads_the_config(app_: FastAPI) -> AsyncGenerator[None]:
        assert SafeRouteAdder.add_api_route_if_not_exists(app_, "/v1/decisions", pass_through, ["POST"]), (
            "the native route registered at startup must not block the config pass-through"
        )
        yield

    bare: Final = FastAPI(lifespan=loads_the_config)
    attach_lazy_features(bare, (_decisions_feature(),))
    with TestClient(bare) as client:
        assert client.post("/v1/decisions", json={"model": "gpt-6-luna"}).json() == {"served_by": "pass-through"}
    assert _serving_endpoint(bare, "/v1/decisions") is pass_through
    assert _serving_endpoint(bare, "/decisions") is decisions
