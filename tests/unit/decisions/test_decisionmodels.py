from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

import pytest
import respx

import litellm
from litellm.types.decisions import NoulAnswer

_RESPONSE: Final[Mapping[str, object]] = {
    "model": "s1-fast",
    "answers": {"is_defect": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 41, "output_tokens": 1},
}


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.parametrize("model", ("s1-fast", "s1-pro", "reax-1", "reax-1.3"))
def test_decisionmodels_prices_are_input_only(model: str) -> None:
    entry: Final = litellm.model_cost[f"decisionmodels/{model}"]

    assert entry["litellm_provider"] == "decisionmodels"
    assert entry["mode"] == "evaluation"
    assert entry["max_input_tokens"] > 0
    assert entry["input_cost_per_token"] > 0
    assert entry["output_cost_per_token"] == 0


def test_decisionmodels_sync_call_uses_default_base_and_environment_key(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("DECISIONMODELS_API_KEY", "environment-key")
    monkeypatch.delenv("DECISIONMODELS_API_BASE", raising=False)
    upstream: Final = respx_mock.post("https://api.decisionmodels.io/v1/systemone").respond(json=_RESPONSE)
    response: Final = litellm.systemone(
        model="decisionmodels/s1-fast",
        state="damaged order",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )
    assert response.answers["is_defect"] == NoulAnswer(type="noul", noul=0.9)
    assert upstream.calls[0].request.headers["authorization"] == "Bearer environment-key"
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "s1-fast",
        "state": "damaged order",
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }


def test_decisionmodels_environment_base_and_explicit_key_win(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("DECISIONMODELS_API_BASE", "https://decisionmodels.example/")
    monkeypatch.setenv("DECISIONMODELS_API_KEY", "environment-key")
    upstream: Final = respx_mock.post("https://decisionmodels.example/v1/systemone").respond(json=_RESPONSE)
    litellm.systemone(
        model="decisionmodels/s1-pro",
        state={"message": "damaged order"},
        api_key="explicit-key",
        questions={"is_defect": {"type": "noul", "criteria": {"true": "damaged", "false": "intact"}}},
    )
    assert upstream.calls[0].request.headers["authorization"] == "Bearer explicit-key"
    assert json.loads(upstream.calls[0].request.content)["model"] == "s1-pro"


@pytest.mark.parametrize(
    ("status", "exception"),
    (
        (401, litellm.AuthenticationError),
        (402, litellm.PaymentRequiredError),
        (422, litellm.BadRequestError),
        (429, litellm.RateLimitError),
    ),
)
def test_decisionmodels_upstream_failures_preserve_status_and_message(
    status: int,
    exception: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("DECISIONMODELS_API_BASE", raising=False)
    upstream: Final = respx_mock.post("https://api.decisionmodels.io/v1/systemone").respond(
        status,
        json={"detail": {"error_type": "test_error", "message": "decision rejected"}},
        headers={"retry-after": "1"},
    )
    with pytest.raises(exception, match="decision rejected") as error:
        litellm.systemone(
            model="decisionmodels/s1-fast",
            state="text",
            api_key="test-key",
            questions={"a": {"type": "noul", "instructions": "Is it urgent?"}},
            num_retries=0,
        )
    assert error.value.status_code == status
    assert upstream.call_count == 1
