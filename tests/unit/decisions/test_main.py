from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.decisions import (
    ChoiceAnswer,
    DecisionsResponse,
    DecisionsUsage,
    NoulAnswer,
    ScoreAnswer,
)

_QUESTIONS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "is_defect": {"type": "noul", "instructions": "Is this a defect?", "provider_field": "kept"},
        "sentiment": {"type": "choice", "criteria": {"positive": None, "negative": "unhappy"}},
        "severity": {"type": "score", "criteria": ["none", "low", "high"]},
    }
)
_RESPONSE: Final[Mapping[str, object]] = {
    "model": "jev-1.13",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
        "sentiment": {
            "type": "choice",
            "choice": "positive",
            "confidence": 0.8,
            "probabilities": {"positive": 0.8, "negative": 0.2},
        },
        "severity": {
            "type": "score",
            "score": 1,
            "confidence": 0.7,
            "legend": {"0": "none", "1": "low", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
        },
    },
    "usage": {"input_tokens": 367, "output_tokens": 3},
}
_PROVIDERS: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "perplexity",
        "perplexity/pplx-decider-v1-27b",
        "https://api.perplexity.ai/v1/decisions",
        "pplx-decider-v1-27b",
    ),
    ("typesafe", "typesafe/jev-1.13", "https://api.typesafe.ai/v1/systemone", "jev-1.13"),
    (
        "openrouter",
        "openrouter/typesafe/jev-1.13",
        "https://openrouter.ai/api/alpha/decisions",
        "typesafe/jev-1.13",
    ),
)


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.standard_logging_object: Mapping[str, object] | None = None

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            self.standard_logging_object = standard_logging_object


async def _drain_logging_worker() -> None:
    await asyncio.sleep(0)
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "model", "url", "upstream_model"), _PROVIDERS)
async def test_adecisions_sends_the_provider_wire_contract(
    provider: str,
    model: str,
    url: str,
    upstream_model: str,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(url).respond(json=_RESPONSE)

    response: Final = await litellm.adecisions(
        model=model,
        state={"source": "unit-test"},
        questions=_QUESTIONS,
        api_key="caller-key",
        extra_headers={
            "x-request-tag": "decisions-test",
            "AUTHORIZATION": "attacker-key",
            "Content-Type": "text/plain",
        },
        internal_kwarg="must-not-leak",
    )

    assert route.called
    assert len(respx_mock.calls) == 1
    request: Final = respx_mock.calls[0].request
    assert request.headers["authorization"] == "Bearer caller-key"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-request-tag"] == "decisions-test"
    assert json.loads(request.content) == {
        "model": upstream_model,
        "state": {"source": "unit-test"},
        "questions": {
            "is_defect": {
                "type": "noul",
                "instructions": "Is this a defect?",
                "provider_field": "kept",
            },
            "sentiment": {"type": "choice", "criteria": {"positive": None, "negative": "unhappy"}},
            "severity": {"type": "score", "criteria": ["none", "low", "high"]},
        },
    }
    assert isinstance(response.answers["is_defect"], NoulAnswer)
    assert isinstance(response.answers["sentiment"], ChoiceAnswer)
    assert isinstance(response.answers["severity"], ScoreAnswer)
    assert response._hidden_params["custom_llm_provider"] == provider


def test_decisions_uses_the_same_wire_contract_for_sync_calls(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = litellm.decisions(
        model="perplexity/pplx-decider-v1-27b",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        api_key="caller-key",
    )

    assert route.called
    assert response.model == "jev-1.13"


def test_openrouter_response_keeps_provider_fields(respx_mock: respx.MockRouter) -> None:
    payload: Final = {
        **_RESPONSE,
        "id": "decision-1",
        "provider": "typesafe",
        "usage": {**_RESPONSE["usage"], "cost": 0.25},
    }
    respx_mock.post("https://openrouter.ai/api/alpha/decisions").respond(json=payload)

    response: Final = litellm.decisions(
        model="openrouter/typesafe/jev-1.13",
        state="review",
        questions=_QUESTIONS,
        api_key="caller-key",
    )

    assert response.model_extra["id"] == "decision-1"
    assert response.model_extra["provider"] == "typesafe"
    assert response.usage is not None
    assert response.usage.model_extra["cost"] == 0.25


def test_decisions_cost_uses_litellm_token_pricing() -> None:
    response: Final = DecisionsResponse(
        model="pplx-decider-v1-27b",
        answers={},
        usage=DecisionsUsage(input_tokens=367, output_tokens=3),
    )
    response._hidden_params = {
        "model": "perplexity/pplx-decider-v1-27b",
        "custom_llm_provider": "perplexity",
    }

    cost: Final = litellm.completion_cost(completion_response=response)

    assert cost == pytest.approx(367 * 4e-8)


@pytest.mark.asyncio
async def test_decisions_cost_is_in_standard_logging_object(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)
    recording_logger: Final = _RecordingLogger()
    original_callbacks: Final = litellm.callbacks
    litellm.callbacks = [recording_logger]

    try:
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )
        await _drain_logging_worker()
    finally:
        litellm.callbacks = original_callbacks

    assert recording_logger.standard_logging_object is not None
    assert recording_logger.standard_logging_object["response_cost"] == pytest.approx(367 * 4e-8)
    assert recording_logger.standard_logging_object["prompt_tokens"] == 367
    assert recording_logger.standard_logging_object["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_unknown_provider_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Supported providers"):
        await litellm.adecisions(
            model="unknown/jev-1.13",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_empty_custom_provider_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Supported providers"):
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
            custom_llm_provider="",
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_invalid_question_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"sentiment": {"type": "choice"}},
            api_key="caller-key",
        )

    assert len(respx_mock.calls) == 0


def test_upstream_bad_request_maps_to_litellm_error(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        status_code=400,
        json={"error": {"message": "invalid decision"}},
    )

    with pytest.raises(litellm.BadRequestError):
        litellm.decisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )


def test_server_key_is_not_sent_to_an_untrusted_api_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "server-key")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    with pytest.raises(litellm.BadRequestError, match="caller-supplied api_base"):
        litellm.decisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_base="https://untrusted.example/decisions",
        )

    assert len(respx_mock.calls) == 0
