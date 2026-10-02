from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
import respx
from fastapi.testclient import TestClient

import litellm
from litellm.proxy.proxy_server import (
    app,
    cleanup_router_config_variables,
    initialize,
)

_RESPONSE: Final[Mapping[str, object]] = {
    "model": "pplx-decider-v1-27b",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
    },
    "usage": {"input_tokens": 367, "output_tokens": 3},
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


@pytest.mark.parametrize("endpoint", ("/v1/decisions", "/decisions"))
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
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(367 * 4e-8)
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
        "/v1/decisions",
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


def test_proxy_decisions_unknown_model_is_a_client_error(
    client: TestClient,
    respx_mock: respx.MockRouter,
) -> None:
    response: Final = client.post(
        "/v1/decisions",
        json={
            "model": "missing-model",
            "state": "review",
            "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        },
    )

    assert 400 <= response.status_code < 500, response.text
    assert len(respx_mock.calls) == 0
