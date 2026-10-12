from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
from collections.abc import Mapping
from concurrent.futures import wait
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

import litellm
from litellm.constants import AWS_SIGNING_MAX_THREADS
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.bedrock.base_aws_llm import AWS_SIGNING_EXECUTOR
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.decisions import (
    ChoiceAnswer,
    DecisionsResponse,
    DecisionsUsage,
    NoulAnswer,
    OpenAIDecisionResponse,
    ScoreAnswer,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

_QUESTIONS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "is_defect": {"type": "noul", "instructions": "Is this a defect?", "provider_field": "kept"},
        "sentiment": {"type": "choice", "criteria": {"positive": None, "negative": "unhappy"}},
        "severity": {"type": "score", "criteria": ["none", "low", "high"]},
    }
)
_INPUT_TOKENS: Final[int] = 367
_OUTPUT_TOKENS: Final[int] = 3
_CACHED_TOKENS: Final[int] = 256
_CACHE_WRITE_TOKENS: Final[int] = 64
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
    "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS},
}
_STRANDS_RESPONSE: Final[Mapping[str, object]] = {
    "model": "strands-decider-2B-hobson-v19",
    "answers": {
        "severity": {
            "type": "score",
            "score": 1,
            "confidence": 0.7,
            "legend": {"0": "none", "1": "low", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
        }
    },
    "usage": {"input_tokens": 216, "output_tokens": 3},
    "latency_ms": 3722.17,
}
_DATABRICKS_ENDPOINT: Final = "databricks-openjev-qwen35-4b"
_DATABRICKS_ENDPOINT_RESPONSE: Final[Mapping[str, object]] = {
    "model": "/mosaicml/local_model",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
        "sentiment": {
            "type": "choice",
            "choice": "positive",
            "confidence": 0.8,
            "probabilities": {"positive": 0.8, "negative": 0.2},
        },
    },
    "usage": {"input_tokens": 214, "output_tokens": 0},
}
_DATABRICKS_ROUTE: Final = "https://workspace.example/api/2.0/ai-functions/ai-decide"
_DATABRICKS_RESPONSE: Final[Mapping[str, object]] = {
    "response": {
        "answers": {
            "is_defect": {"type": "noul", "probability": 0.9},
            "sentiment": {
                "type": "choice",
                "choice": "positive",
                "confidence": 0.8,
                "probabilities": {"positive": 0.8, "negative": 0.2},
            },
        }
    },
    "metadata": {"version": "1.0"},
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

_OPENAI_RESPONSE: Final[Mapping[str, object]] = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "is_defect", "probability": 0.9},
        {"type": "refusal", "name": "sentiment"},
    ],
    "usage": {
        "input_tokens": _INPUT_TOKENS,
        "input_tokens_details": {"cached_tokens": _CACHED_TOKENS, "cache_write_tokens": _CACHE_WRITE_TOKENS},
        "output_tokens": _OUTPUT_TOKENS,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
    },
}


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


_GPT_6_LUNA_DECISIONS_INPUT_RATE: Final = 1e-07
_MICROSOFT_DECISION_1_DECISIONS_INPUT_RATE: Final = 7e-08
_DECISIONS_PRICING_FIELDS: Final[dict[str, dict[str, object]]] = {
    "gpt-6-luna": {
        "input_cost_per_token_decisions": _GPT_6_LUNA_DECISIONS_INPUT_RATE,
        "cache_read_input_token_cost_decisions": _GPT_6_LUNA_DECISIONS_INPUT_RATE,
        "cache_creation_input_token_cost_decisions": _GPT_6_LUNA_DECISIONS_INPUT_RATE,
        "output_cost_per_token_decisions": 0.0,
    },
    "azure_ai/Microsoft-Decision-1": {"input_cost_per_token_decisions": _MICROSOFT_DECISION_1_DECISIONS_INPUT_RATE},
}


@pytest.fixture
def decisions_pricing_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, fields in _DECISIONS_PRICING_FIELDS.items():
        monkeypatch.setitem(litellm.model_cost, key, {**litellm.model_cost[key], **fields})


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "model", "url", "upstream_model"), _PROVIDERS)
async def test_adecisions_sends_the_provider_wire_contract(
    provider: str,
    model: str,
    url: str,
    upstream_model: str,
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(url).respond(json=_RESPONSE)

    response: Final = await litellm.asystemone(
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


@pytest.mark.asyncio
async def test_router_dispatches_typesafe_decisions_without_api_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_BASE", raising=False)
    provider_resolution: Final = litellm.get_llm_provider("typesafe/jev-latest")

    assert provider_resolution[:2] == ("jev-latest", "typesafe")

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
    upstream: Final = respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(json=_RESPONSE)

    response: Final = await router.asystemone(
        model="jev",
        state="router-test",
        questions={
            "sentiment": {
                "type": "choice",
                "criteria": {"positive": None, "negative": "unhappy"},
            }
        },
    )

    assert upstream.called
    assert len(respx_mock.calls) == 1
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "jev-latest",
        "state": "router-test",
        "questions": {
            "sentiment": {
                "type": "choice",
                "criteria": {"positive": None, "negative": "unhappy"},
            }
        },
    }
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer k"
    assert isinstance(response.answers["sentiment"], ChoiceAnswer)
    assert response.answers["sentiment"].choice == "positive"


def test_decisions_uses_the_same_wire_contract_for_sync_calls(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = litellm.systemone(
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

    response: Final = litellm.systemone(
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
        usage=DecisionsUsage(input_tokens=_INPUT_TOKENS, output_tokens=_OUTPUT_TOKENS),
    )
    response.set_hidden_params({"model": "perplexity/pplx-decider-v1-27b", "custom_llm_provider": "perplexity"})

    cost: Final = litellm.completion_cost(completion_response=response)
    perplexity_cost: Final = litellm.model_cost["perplexity/pplx-decider-v1-27b"]
    expected_cost: Final = _INPUT_TOKENS * float(perplexity_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        perplexity_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert cost == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_openai_decisions_uses_evaluation_pricing_and_explicit_base_model(
    respx_mock: respx.MockRouter,
    decisions_pricing_fields: None,
) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    response: Final = await litellm.adecisions(
        model="openai/gpt-6-luna",
        input="The package arrived broken.",
        questions=[{"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"}],
        api_key="caller-key",
        api_base="https://api.openai.com/v1",
    )
    chat_cost: Mapping[str, object] = litellm.model_cost["gpt-6-luna"]
    expected_chat_cost: Final = (
        (_INPUT_TOKENS - _CACHED_TOKENS - _CACHE_WRITE_TOKENS) * float(chat_cost["input_cost_per_token"])
        + _CACHED_TOKENS * float(chat_cost["cache_read_input_token_cost"])
        + _CACHE_WRITE_TOKENS * float(chat_cost["cache_creation_input_token_cost"])
        + _OUTPUT_TOKENS * float(chat_cost["output_cost_per_token"])
    )

    assert route.called
    assert response._hidden_params["model"] == "openai/gpt-6-luna"
    assert response._hidden_params["provider_response_model"] == "openai/gpt-6-luna"
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        _INPUT_TOKENS * _GPT_6_LUNA_DECISIONS_INPUT_RATE
    )
    assert litellm.completion_cost(completion_response=response, call_type="acompletion") == pytest.approx(
        expected_chat_cost
    )


@pytest.mark.asyncio
async def test_systemone_uses_evaluation_pricing(respx_mock: respx.MockRouter, decisions_pricing_fields: None) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    response: Final = await litellm.asystemone(
        model="openai/gpt-6-luna",
        state="The package arrived broken.",
        questions=_QUESTIONS,
        api_key="caller-key",
        api_base="https://api.openai.com/v1",
    )

    assert route.called
    assert response._hidden_params["model"] == "openai/gpt-6-luna"
    assert response._hidden_params["provider_response_model"] == "openai/gpt-6-luna"
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        _INPUT_TOKENS * _GPT_6_LUNA_DECISIONS_INPUT_RATE
    )


@pytest.mark.asyncio
async def test_systemone_falls_back_to_plain_pricing_without_decisions_row(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post("https://decisions.example/v1/decisions").respond(json=_RESPONSE)
    response: Final = await litellm.asystemone(
        model="perplexity/sonar",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        api_key="caller-key",
        api_base="https://decisions.example/v1",
    )
    plain_cost: Mapping[str, object] = litellm.model_cost["perplexity/sonar"]

    assert route.called
    assert response._hidden_params["provider_response_model"] == "perplexity/sonar"
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        _INPUT_TOKENS * float(plain_cost["input_cost_per_token"])
        + _OUTPUT_TOKENS * float(plain_cost["output_cost_per_token"])
    )


@pytest.mark.parametrize(
    "response",
    (
        DecisionsResponse(
            model="jev-latest",
            answers={},
            usage=DecisionsUsage(
                input_tokens=_INPUT_TOKENS,
                output_tokens=_OUTPUT_TOKENS,
                cached_tokens=_CACHED_TOKENS,
                cache_write_tokens=_CACHE_WRITE_TOKENS,
            ),
        ),
        OpenAIDecisionResponse.model_validate(_OPENAI_RESPONSE),
    ),
    ids=("systemone", "openai"),
)
def test_custom_token_pricing_bills_cached_decisions_input_tokens_once(
    response: DecisionsResponse | OpenAIDecisionResponse,
) -> None:
    cost: Final = litellm.completion_cost(
        completion_response=response,
        model="gpt-6-luna",
        custom_llm_provider="openai",
        custom_cost_per_token={
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
            "cache_read_input_token_cost": 0.1,
            "cache_creation_input_token_cost": 1.25,
        },
    )

    assert cost == pytest.approx(
        (_INPUT_TOKENS - _CACHED_TOKENS - _CACHE_WRITE_TOKENS) * 1.0
        + _CACHED_TOKENS * 0.1
        + _CACHE_WRITE_TOKENS * 1.25
        + _OUTPUT_TOKENS * 2.0
    )


def test_decisions_response_hidden_params_getter_preserves_mutable_identity() -> None:
    response: Final = DecisionsResponse(model="decider", answers={}, usage=None)

    assert response.hidden_params is response._hidden_params

    response.hidden_params["mutation"] = "visible"
    assert response._hidden_params["mutation"] == "visible"


@pytest.mark.asyncio
async def test_decisions_cost_is_in_standard_logging_object(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)
    recording_logger: Final = _RecordingLogger()
    original_callbacks: Final = litellm.callbacks
    litellm.callbacks = [recording_logger]

    try:
        await litellm.asystemone(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )
        await _drain_logging_worker()
    finally:
        litellm.callbacks = original_callbacks

    assert recording_logger.standard_logging_object is not None
    perplexity_cost: Final = litellm.model_cost["perplexity/pplx-decider-v1-27b"]
    expected_cost: Final = _INPUT_TOKENS * float(perplexity_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        perplexity_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert recording_logger.standard_logging_object["response_cost"] == pytest.approx(expected_cost)
    assert recording_logger.standard_logging_object["prompt_tokens"] == _INPUT_TOKENS
    assert recording_logger.standard_logging_object["completion_tokens"] == _OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_unknown_provider_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
        await litellm.asystemone(
            model="unknown/jev-1.13",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_provider_without_decisions_support_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match=r"Unknown Decisions provider 'anthropic'\. Supported providers"):
        await litellm.asystemone(
            model="anthropic/claude-sonnet-4-5",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_empty_custom_provider_falls_back_to_the_model_prefix(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = await litellm.asystemone(
        model="perplexity/pplx-decider-v1-27b",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        api_key="caller-key",
        custom_llm_provider="",
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content)["model"] == "pplx-decider-v1-27b"
    assert response._hidden_params["custom_llm_provider"] == "perplexity"


@pytest.mark.asyncio
async def test_upstream_reply_without_answers_is_a_server_error(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        json={"model": "pplx-decider-v1-27b", "usage": {"input_tokens": 10, "output_tokens": 0}}
    )

    with pytest.raises(litellm.InternalServerError, match="unexpected response"):
        await litellm.asystemone(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )


@pytest.mark.asyncio
async def test_router_sends_the_openrouter_deployment_key_when_the_provider_is_already_resolved(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "jev",
                "litellm_params": {
                    "model": "openrouter/typesafe/jev-1.13",
                    "api_key": "deployment-key",
                    "api_base": "https://egress.example/openrouter",
                },
            }
        ]
    )
    upstream: Final = respx_mock.post("https://egress.example/openrouter/alpha/decisions").respond(json=_RESPONSE)

    await router.asystemone(
        model="jev",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert upstream.called
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer deployment-key"
    assert json.loads(respx_mock.calls[0].request.content)["model"] == "typesafe/jev-1.13"


@pytest.mark.asyncio
async def test_invalid_question_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        await litellm.asystemone(
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
        litellm.systemone(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )


def test_unreachable_upstream_maps_to_a_connection_error(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").mock(
        side_effect=httpx.ConnectError("Cannot connect to host api.perplexity.ai:443")
    )

    with pytest.raises(litellm.APIConnectionError, match="PerplexityException - Cannot connect to host"):
        litellm.systemone(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )


def test_server_key_is_sent_to_an_explicit_api_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "server-key")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    route: Final = respx_mock.post("https://egress.example/perplexity/v1/decisions").respond(json=_RESPONSE)

    litellm.systemone(
        model="perplexity/pplx-decider-v1-27b",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        api_base="https://egress.example/perplexity",
    )

    assert route.call_count == 1
    assert route.calls[0].request.headers["authorization"] == "Bearer server-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("cloudflare/clef", "cloudflare/@cf/cloudflare/clef"))
@pytest.mark.parametrize("wrapped", (False, True))
async def test_cloudflare_clef_resolves_model_and_response_envelope(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    model: str,
    wrapped: bool,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    response_body: Final[Mapping[str, object]] = (
        {"result": _RESPONSE, "success": True, "errors": [], "messages": []} if wrapped else _RESPONSE
    )
    route: Final = respx_mock.post(
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"
    ).respond(json=response_body)

    response: Final = await litellm.asystemone(
        model=model,
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called
    request: Final = respx_mock.calls[0].request
    assert request.headers["authorization"] == "Bearer cloudflare-key"
    assert json.loads(request.content) == {
        "model": "clef",
        "state": "review",
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert response.answers == {"is_defect": NoulAnswer(type="noul", noul=0.9)}
    assert response._hidden_params["model"] == "cloudflare/@cf/cloudflare/clef"


@pytest.mark.asyncio
async def test_cloudflare_clef_flash_uses_flash_endpoint_and_request_model(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    route: Final = respx_mock.post(
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash"
    ).respond(json=_RESPONSE)

    await litellm.asystemone(
        model="cloudflare/clef-flash",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content)["model"] == "clef-flash"


@pytest.mark.asyncio
async def test_cloudflare_api_base_from_env_uses_workers_ai_run_path(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_BASE", "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    route: Final = respx_mock.post(
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"
    ).respond(json=_RESPONSE)

    await litellm.asystemone(
        model="cloudflare/clef",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called


@pytest.mark.asyncio
async def test_cloudflare_requires_account_id_or_api_base_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")

    with pytest.raises(litellm.BadRequestError, match="Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID"):
        await litellm.asystemone(
            model="cloudflare/clef",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_cloudflare_clef_cost_uses_the_model_cost_map(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    respx_mock.post("https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef").respond(
        json=_RESPONSE
    )

    response: Final = await litellm.asystemone(
        model="cloudflare/clef",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    cost: Final = litellm.completion_cost(completion_response=response)
    clef_cost: Final = litellm.model_cost["cloudflare/@cf/cloudflare/clef"]
    expected_cost: Final = _INPUT_TOKENS * float(clef_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        clef_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert cost == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_strands_decider_requires_api_base_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)

    with pytest.raises(litellm.BadRequestError, match="api_base is required"):
        await litellm.asystemone(
            model="strands_decider/strands-decider-2B-hobson-v19",
            state="review",
            questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_strands_decider_without_key_preserves_response_extras(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    response: Final = await litellm.asystemone(
        model="strands_decider/strands-decider-2B-hobson-v19",
        state="review",
        questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
        api_base="https://strands.example",
    )

    assert route.called
    assert "authorization" not in respx_mock.calls[0].request.headers
    assert response.model_extra["latency_ms"] == _STRANDS_RESPONSE["latency_ms"]
    severity: Final = response.answers["severity"]
    assert isinstance(severity, ScoreAnswer)
    assert severity.legend == {"0": "none", "1": "low", "2": "high"}


@pytest.mark.asyncio
async def test_strands_decider_plain_api_base_answers_while_every_aws_signing_thread_is_busy(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)
    release: Final = threading.Event()
    busy: Final = tuple(AWS_SIGNING_EXECUTOR.submit(release.wait) for _ in range(AWS_SIGNING_MAX_THREADS))

    try:
        response: Final = await asyncio.wait_for(
            litellm.asystemone(
                model="strands_decider/strands-decider-2B-hobson-v19",
                state="review",
                questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
                api_base="https://strands.example",
            ),
            timeout=5,
        )
    finally:
        release.set()
        wait(busy)

    assert isinstance(response.answers["severity"], ScoreAnswer)


@pytest.mark.asyncio
async def test_strands_decider_uses_key_from_matching_environment_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("STRANDS_DECIDER_API_BASE", "https://strands.example")
    monkeypatch.setenv("STRANDS_DECIDER_API_KEY", "strands-key")
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    await litellm.asystemone(
        model="strands_decider/strands-decider-2B-hobson-v19",
        state="review",
        questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
        api_base="https://strands.example",
    )

    assert route.called
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer strands-key"


@pytest.mark.asyncio
async def test_strands_decider_provider_resolution_and_router_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    provider_resolution: Final = litellm.get_llm_provider("strands_decider/strands-decider-2B-hobson-v19")
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
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    response: Final = await router.asystemone(
        model="strands",
        state="review",
        questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
    )

    assert provider_resolution[:2] == ("strands-decider-2B-hobson-v19", "strands_decider")
    assert route.called
    assert response.model == _STRANDS_RESPONSE["model"]


_RUNTIME_ARN: Final = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/qa_decider-AbC123xyz0"
_RUNTIME_URL: Final = (
    "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
    "arn%3Aaws%3Abedrock-agentcore%3Aus-east-1%3A123456789012%3Aruntime%2Fqa_decider-AbC123xyz0/invocations"
)
_RUNTIME_CREDENTIALS: Final = Credentials(access_key="AKIDEXAMPLE", secret_key="example-secret")
_SESSION_HEADER: Final = "x-amzn-bedrock-agentcore-runtime-session-id"


def _sigv4_verifies(request: httpx.Request) -> bool:
    authorization: Final = request.headers["authorization"]
    signed_names: Final = authorization.split("SignedHeaders=")[1].split(",")[0].split(";")
    replayed: Final = AWSRequest(
        method=request.method,
        url=str(request.url),
        data=request.content,
        headers={name: request.headers[name] for name in signed_names},
    )
    replayed.context["timestamp"] = request.headers["x-amz-date"]
    auth: Final = SigV4Auth(_RUNTIME_CREDENTIALS, "bedrock-agentcore", "us-east-1")
    string_to_sign: Final = auth.string_to_sign(replayed, auth.canonical_request(replayed))
    return authorization.endswith(f"Signature={auth.signature(string_to_sign, replayed)}")


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.asyncio
async def test_strands_decider_runtime_arn_sends_a_sigv4_signed_invoke_agent_runtime_call(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    use_async: bool,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    route: Final = respx_mock.post(_RUNTIME_URL).respond(json=_STRANDS_RESPONSE)
    call_args: Final[Mapping[str, object]] = {
        "model": "strands_decider/strands-decider-2B-hobson-v19",
        "state": "review",
        "questions": {"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
        "api_base": _RUNTIME_ARN,
        "aws_access_key_id": _RUNTIME_CREDENTIALS.access_key,
        "aws_secret_access_key": _RUNTIME_CREDENTIALS.secret_key,
    }

    response: Final = await litellm.asystemone(**call_args) if use_async else litellm.systemone(**call_args)

    assert route.call_count == 1
    sent: Final = route.calls[0].request
    assert _sigv4_verifies(sent)
    assert json.loads(sent.content)["model"] == "strands-decider-2B-hobson-v19"
    assert sent.headers[_SESSION_HEADER].startswith("litellm-decider-")
    assert response.model == _STRANDS_RESPONSE["model"]


@pytest.mark.asyncio
async def test_strands_decider_runtime_arn_router_deployment_answers_openai_format_calls(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "strands",
                "litellm_params": {
                    "model": "strands_decider/strands-decider-2B-hobson-v19",
                    "api_base": _RUNTIME_ARN,
                    "aws_access_key_id": _RUNTIME_CREDENTIALS.access_key,
                    "aws_secret_access_key": _RUNTIME_CREDENTIALS.secret_key,
                    "aws_region_name": "eu-west-1",
                    "agentcore_runtime_session_id": "pooled-session-0123456789abcdef0123456",
                },
            }
        ]
    )
    route: Final = respx_mock.post(_RUNTIME_URL).respond(json=_STRANDS_RESPONSE)

    response: Final = await router.adecisions(
        model="strands",
        input="review",
        extra_headers={_SESSION_HEADER: "caller-session-0123456789abcdef01234567"},
        questions=[
            {
                "type": "score",
                "name": "severity",
                "instructions": "How bad?",
                "levels": [{"label": "none"}, {"label": "low"}, {"label": "high"}],
            }
        ],
    )

    sent: Final = route.calls[0].request
    assert _sigv4_verifies(sent)
    assert sent.headers[_SESSION_HEADER] == "pooled-session-0123456789abcdef0123456"
    assert isinstance(response, OpenAIDecisionResponse)


@pytest.mark.asyncio
async def test_strands_decider_runtime_arn_with_api_key_sends_the_bearer_token_unsigned(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    route: Final = respx_mock.post(_RUNTIME_URL).respond(json=_STRANDS_RESPONSE)

    await litellm.asystemone(
        model="strands_decider/strands-decider-2B-hobson-v19",
        state="review",
        questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
        api_base=_RUNTIME_ARN,
        api_key="runtime-jwt",
    )

    sent: Final = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer runtime-jwt"
    assert "x-amz-date" not in sent.headers
    assert _SESSION_HEADER in sent.headers


@pytest.mark.parametrize(
    ("code", "error_class"),
    [
        ("bad_request", litellm.BadRequestError),
        ("loading", litellm.ServiceUnavailableError),
        ("inference_error", litellm.InternalServerError),
    ],
)
@pytest.mark.asyncio
async def test_strands_decider_runtime_error_envelope_raises_instead_of_answering(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    code: str,
    error_class: type[Exception],
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    respx_mock.post(_RUNTIME_URL).respond(json={"error": {"code": code, "message": "at most 16 questions"}})

    with pytest.raises(error_class, match="at most 16 questions"):
        await litellm.asystemone(
            model="strands_decider/strands-decider-2B-hobson-v19",
            state="review",
            questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
            api_base=_RUNTIME_ARN,
            aws_access_key_id=_RUNTIME_CREDENTIALS.access_key,
            aws_secret_access_key=_RUNTIME_CREDENTIALS.secret_key,
        )


@pytest.mark.asyncio
async def test_strands_decider_plain_api_base_error_envelope_stays_an_unexpected_response(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    respx_mock.post("https://strands.example/v1/systemone").respond(
        json={"error": {"code": "bad_request", "message": "at most 16 questions"}}
    )

    with pytest.raises(litellm.InternalServerError, match="returned an unexpected response"):
        await litellm.asystemone(
            model="strands_decider/strands-decider-2B-hobson-v19",
            state="review",
            questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
            api_base="https://strands.example",
        )


_VLLM_RESPONSE: Final[Mapping[str, object]] = {
    "id": "systemone-4af3d2c1",
    "object": "structured_decision",
    "created": 1760012345,
    "model": "Qwen/Qwen3-0.6B",
    "answers": {
        "is_defect": {
            "type": "choice",
            "choice": "yes",
            "confidence": 0.91,
            "probabilities": {"yes": 0.91, "no": 0.09},
        }
    },
    "usage": {"input_tokens": 154, "output_tokens": 2},
    "diagnostics": {"engine": "vllm", "logprobs_mode": "raw_logprobs"},
}


@pytest.mark.asyncio
async def test_hosted_vllm_requires_api_base_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)

    with pytest.raises(litellm.BadRequestError, match="api_base is required"):
        await litellm.asystemone(
            model="hosted_vllm/Qwen/Qwen3-0.6B",
            state="review",
            questions={"is_defect": {"type": "choice", "criteria": {"yes": None, "no": None}}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_hosted_vllm_without_key_sends_no_authorization_and_preserves_response_extras(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    route: Final = respx_mock.post("http://vllm.local:8000/v1/systemone").respond(json=_VLLM_RESPONSE)

    response: Final = await litellm.asystemone(
        model="hosted_vllm/Qwen/Qwen3-0.6B",
        state="The package arrived broken.",
        questions={"is_defect": {"type": "choice", "criteria": {"yes": None, "no": None}}},
        api_base="http://vllm.local:8000",
    )

    assert route.called
    assert "authorization" not in respx_mock.calls[0].request.headers
    assert response.model_extra["id"] == _VLLM_RESPONSE["id"]
    assert response.model_extra["object"] == _VLLM_RESPONSE["object"]
    assert response.model_extra["diagnostics"] == _VLLM_RESPONSE["diagnostics"]


@pytest.mark.asyncio
async def test_hosted_vllm_uses_key_and_base_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://vllm.local:8000/v1")
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "vllm-key")
    route: Final = respx_mock.post("http://vllm.local:8000/v1/systemone").respond(json=_VLLM_RESPONSE)

    await litellm.asystemone(
        model="hosted_vllm/Qwen/Qwen3-0.6B",
        state="The package arrived broken.",
        questions={"is_defect": {"type": "choice", "criteria": {"yes": None, "no": None}}},
    )

    assert route.called
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer vllm-key"


@pytest.mark.asyncio
async def test_hosted_vllm_api_base_with_v1_suffix_posts_to_v1_systemone(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    route: Final = respx_mock.post("http://vllm.local:8000/v1/systemone").respond(json=_VLLM_RESPONSE)

    await litellm.asystemone(
        model="hosted_vllm/Qwen/Qwen3-0.6B",
        state="The package arrived broken.",
        questions={"is_defect": {"type": "choice", "criteria": {"yes": None, "no": None}}},
        api_base="http://vllm.local:8000/v1",
    )

    assert route.called


@pytest.mark.asyncio
async def test_hosted_vllm_choice_question_goes_out_as_a_jev_choice_with_the_served_model_name(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    route: Final = respx_mock.post("http://vllm.local:8000/v1/systemone").respond(json=_VLLM_RESPONSE)

    await litellm.asystemone(
        model="hosted_vllm/Qwen/Qwen3-0.6B",
        state="The package arrived broken.",
        questions={
            "is_defect": {
                "type": "choice",
                "instructions": "Is this a defect?",
                "criteria": {"yes": "defective", "no": "intact"},
            }
        },
        api_base="http://vllm.local:8000",
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "Qwen/Qwen3-0.6B",
        "state": "The package arrived broken.",
        "questions": {
            "is_defect": {
                "type": "choice",
                "instructions": "Is this a defect?",
                "criteria": {"yes": "defective", "no": "intact"},
            }
        },
    }


@pytest.mark.asyncio
async def test_databricks_requires_api_base_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("DATABRICKS_API_BASE", raising=False)
    monkeypatch.setenv("DATABRICKS_API_KEY", "dapi-key")

    with pytest.raises(litellm.BadRequestError, match="DATABRICKS_API_BASE"):
        await litellm.asystemone(
            model="databricks/ai_decide",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_name", ["serving-endpoints/openjev", "openjev?x=1", "openjev#frag", "..", "open jev"]
)
async def test_databricks_rejects_a_serving_endpoint_name_that_would_rewrite_the_url(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    endpoint_name: str,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", "https://workspace.example/serving-endpoints")
    monkeypatch.setenv("DATABRICKS_API_KEY", "dapi-key")

    with pytest.raises(litellm.BadRequestError, match="bare serving endpoint name"):
        await litellm.asystemone(
            model=f"databricks/{endpoint_name}",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_databricks_requires_a_key_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", "https://workspace.example")
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    with pytest.raises(litellm.AuthenticationError, match="Missing API key for Decisions provider 'databricks'"):
        await litellm.asystemone(
            model="databricks/ai_decide",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_base", ["https://workspace.example/serving-endpoints", "https://workspace.example/serving-endpoints/"]
)
async def test_databricks_posts_the_jev_body_to_the_serving_endpoint_invocations_route(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    api_base: str,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", api_base)
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-token")
    route: Final = respx_mock.post(
        f"https://workspace.example/serving-endpoints/{_DATABRICKS_ENDPOINT}/invocations"
    ).respond(json=_DATABRICKS_ENDPOINT_RESPONSE)
    questions: Final = {
        "is_defect": {"type": "noul", "instructions": "Is this a defect?"},
        "sentiment": {"type": "choice", "criteria": {"positive": None, "negative": "unhappy"}},
    }

    response: Final = await litellm.asystemone(
        model=f"databricks/{_DATABRICKS_ENDPOINT}",
        state={"ticket": "export hangs"},
        questions=questions,
    )

    assert route.called
    sent: Final = respx_mock.calls[0].request
    assert sent.headers["authorization"] == "Bearer dapi-token"
    assert json.loads(sent.content) == {
        "model": _DATABRICKS_ENDPOINT,
        "state": {"ticket": "export hangs"},
        "questions": questions,
    }
    assert response.model == "/mosaicml/local_model"
    assert isinstance(response.answers["is_defect"], NoulAnswer)
    assert isinstance(response.answers["sentiment"], ChoiceAnswer)
    assert response.hidden_params["custom_llm_provider"] == "databricks"
    assert response.hidden_params["model"] == f"databricks/{_DATABRICKS_ENDPOINT}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_base",
    [
        "https://workspace.example",
        "https://workspace.example/",
        "https://workspace.example/serving-endpoints",
        "https://workspace.example/serving-endpoints/",
    ],
)
async def test_databricks_posts_the_systemone_body_without_model_to_the_ai_decide_route(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    api_base: str,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", api_base)
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-token")
    route: Final = respx_mock.post(_DATABRICKS_ROUTE).respond(json=_DATABRICKS_RESPONSE)
    questions: Final = {
        "is_defect": {"type": "noul", "instructions": "Is this a defect?"},
        "sentiment": {"type": "choice", "criteria": {"positive": None, "negative": "unhappy"}},
    }

    response: Final = await litellm.adecisions(
        model="databricks/ai_decide",
        state={"ticket": "export hangs"},
        questions=questions,
    )

    assert route.called
    sent: Final = respx_mock.calls[0].request
    assert sent.headers["authorization"] == "Bearer dapi-token"
    assert json.loads(sent.content) == {"state": {"ticket": "export hangs"}, "questions": questions}
    assert response.model is None
    assert response.answers["is_defect"] == NoulAnswer(type="noul", noul=0.9)
    assert isinstance(response.answers["sentiment"], ChoiceAnswer)
    assert response.model_extra == {"metadata": {"version": "1.0"}}
    assert response.hidden_params["custom_llm_provider"] == "databricks"
    assert response.hidden_params["model"] == "databricks/ai_decide"


@pytest.mark.asyncio
async def test_databricks_router_deployment_sends_its_own_key_to_its_own_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", "https://other-workspace.example")
    monkeypatch.setenv("DATABRICKS_API_KEY", "env-key-stays-home")
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "decider",
                "litellm_params": {
                    "model": "databricks/ai_decide",
                    "api_base": "https://workspace.example",
                    "api_key": "deployment-key",
                },
            }
        ]
    )
    route: Final = respx_mock.post(_DATABRICKS_ROUTE).respond(json=_DATABRICKS_RESPONSE)

    response: Final = await router.asystemone(
        model="decider",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer deployment-key"
    assert response.answers["is_defect"] == NoulAnswer(type="noul", noul=0.9)


_PROVIDERS_WHOSE_CHAT_MODELS_ALSO_DECIDE: Final = frozenset(
    {LlmProviders.OPENAI.value, LlmProviders.OPENROUTER.value, LlmProviders.HOSTED_VLLM.value}
)
_DEDICATED_DECIDER_PROVIDERS: Final[tuple[str, ...]] = tuple(
    sorted(
        provider.value
        for provider in LlmProviders
        if provider.value not in _PROVIDERS_WHOSE_CHAT_MODELS_ALSO_DECIDE
        and ProviderConfigManager.get_provider_decisions_config(model="", provider=provider) is not None
    )
)


@pytest.mark.parametrize("provider", _DEDICATED_DECIDER_PROVIDERS)
def test_every_dedicated_decider_provider_ships_an_evaluation_mode_cost_map_entry(provider: str) -> None:
    evaluation_entries: Final = tuple(
        name
        for name, info in litellm.model_cost.items()
        if isinstance(info, Mapping) and info.get("litellm_provider") == provider and info.get("mode") == "evaluation"
    )

    assert evaluation_entries, f"a {provider} decider would be health-checked as chat without a mode: evaluation entry"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_base", "url"),
    (
        (None, "https://api.openai.com/v1/decisions"),
        ("https://gateway.example/v1", "https://gateway.example/v1/decisions"),
    ),
)
async def test_openai_decisions_translate_systemone_to_the_openai_wire_contract_and_back(
    api_base: str | None,
    url: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    decisions_pricing_fields: None,
) -> None:
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    route: Final = respx_mock.post(url).respond(json=_OPENAI_RESPONSE)

    response: Final = await litellm.asystemone(
        model="openai/gpt-6-luna",
        state="The package arrived broken.",
        questions={
            "is_defect": {"type": "noul", "instructions": "Is this a defect?"},
            "sentiment": {
                "type": "choice",
                "instructions": "How does the customer feel?",
                "criteria": {"positive": None, "negative": "unhappy"},
            },
        },
        api_key="caller-key",
        api_base=api_base,
    )

    assert route.called
    request: Final = respx_mock.calls[0].request
    assert request.headers["authorization"] == "Bearer caller-key"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "input": "The package arrived broken.",
        "questions": [
            {"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"},
            {
                "type": "choice",
                "name": "sentiment",
                "instructions": "How does the customer feel?",
                "choices": [{"value": "positive"}, {"value": "negative", "description": "unhappy"}],
            },
        ],
    }
    assert response.answers == {"is_defect": NoulAnswer(type="noul", noul=0.9)}
    assert response.hidden_params["custom_llm_provider"] == "openai"
    expected_cost: Final = _INPUT_TOKENS * _GPT_6_LUNA_DECISIONS_INPUT_RATE
    assert expected_cost > 0
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("settings", "env", "url", "authorization"),
    (
        ({"openai_key": "sdk-key"}, {}, "https://api.openai.com/v1/decisions", "Bearer sdk-key"),
        (
            {"api_key": "global-key", "openai_key": "sdk-key"},
            {"OPENAI_API_KEY": "env-key"},
            "https://api.openai.com/v1/decisions",
            "Bearer global-key",
        ),
        (
            {},
            {"OPENAI_API_KEY": "env-key", "OPENAI_API_BASE": "https://legacy.example/v1"},
            "https://legacy.example/v1/decisions",
            "Bearer env-key",
        ),
        (
            {"api_base": "https://sdk.example"},
            {"OPENAI_API_KEY": "env-key", "OPENAI_BASE_URL": "https://env.example"},
            "https://sdk.example/v1/decisions",
            "Bearer env-key",
        ),
    ),
    ids=("openai_key", "api_key_before_env", "openai_api_base_env", "api_base_before_env"),
)
def test_openai_decisions_use_the_same_settings_as_other_openai_calls(
    settings: Mapping[str, str],
    env: Mapping[str, str],
    url: str,
    authorization: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    for name in ("api_key", "openai_key", "api_base"):
        monkeypatch.setattr(litellm, name, settings.get(name))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    route: Final = respx_mock.post(url).respond(json=_OPENAI_RESPONSE)

    litellm.systemone(
        model="openai/gpt-6-luna",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.call_count == 1
    assert route.calls[0].request.headers["authorization"] == authorization


@pytest.mark.asyncio
async def test_openai_format_calls_to_a_systemone_provider_get_openai_format_answers(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(json=_RESPONSE)

    response: Final = await litellm.adecisions(
        model="typesafe/jev-1.13",
        input="review",
        questions=[
            {"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"},
            {
                "type": "choice",
                "name": "sentiment",
                "instructions": "Tone?",
                "choices": [{"value": "positive"}, {"value": "negative"}],
            },
        ],
        api_key="caller-key",
    )

    assert route.called
    assert tuple(json.loads(respx_mock.calls[0].request.content)["questions"]) == ("is_defect", "sentiment")
    assert isinstance(response, OpenAIDecisionResponse)
    assert [answer.model_dump(mode="json") for answer in response.answers] == [
        {"type": "predicate", "name": "is_defect", "probability": 0.9},
        {
            "type": "choice",
            "name": "sentiment",
            "choice": "positive",
            "probabilities": [{"value": "positive", "probability": 0.8}, {"value": "negative", "probability": 0.2}],
            "confidence": 0.8,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "request_kwargs", "message"),
    (
        (
            "openai/gpt-6-luna",
            {
                "input": "review",
                "images": ["data:image/png;base64,AA=="],
                "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
            },
            "send images as input_image content parts",
        ),
        (
            "openai/gpt-6-luna",
            {
                "input": "review",
                "images": ["data:image/png;base64,AA=="],
                "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
                "drop_params": True,
            },
            "send images as input_image content parts",
        ),
    ),
    ids=("images_and_input", "images_and_input_under_drop_params"),
)
async def test_requests_a_provider_cannot_serve_are_rejected_before_http(
    respx_mock: respx.MockRouter,
    model: str,
    request_kwargs: Mapping[str, object],
    message: str,
) -> None:
    with pytest.raises(litellm.BadRequestError, match=message):
        await litellm.adecisions(model=model, api_key="caller-key", **request_kwargs)

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_openrouter_decisions_uses_provider_reported_cost_without_cost_map(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(litellm.model_cost, "openrouter/typesafe/jev-1.13", raising=False)
    cost: Final = 1.1802e-5
    route: Final = respx_mock.post("https://openrouter.ai/api/alpha/decisions").respond(
        json={
            "model": "typesafe/jev-1.13",
            "answers": _RESPONSE["answers"],
            "usage": {
                "input_tokens": _INPUT_TOKENS,
                "output_tokens": _OUTPUT_TOKENS,
                "cost": cost,
            },
        }
    )

    response: Final = await litellm.asystemone(
        model="openrouter/typesafe/jev-1.13",
        state={"source": "unit-test"},
        questions=_QUESTIONS,
        api_key="caller-key",
    )

    assert route.called
    assert get_response_cost_from_hidden_params(response.hidden_params) == cost


_SAFETY_IDENTIFIER: Final = "end-user-7"
_PREDICATE_QUESTIONS: Final[tuple[Mapping[str, object], ...]] = (
    {"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"},
)
_PREDICATE_REQUESTS: Final[tuple[tuple[str, Mapping[str, object]], ...]] = (
    ("decisions", MappingProxyType({"input": "review", "questions": _PREDICATE_QUESTIONS})),
    (
        "systemone",
        MappingProxyType(
            {"state": "review", "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}}}
        ),
    ),
)
_PREDICATE_REQUEST_IDS: Final = ("decisions", "systemone")


@pytest.mark.parametrize(("api", "request_kwargs"), _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
def test_safety_identifier_is_refused_before_http_when_the_provider_cannot_take_it(
    api: str, request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    with pytest.raises(litellm.UnsupportedParamsError, match=r"safety_identifier.*drop_params") as caught:
        getattr(litellm, api)(
            model="perplexity/pplx-decider-v1-27b",
            safety_identifier=_SAFETY_IDENTIFIER,
            api_key="caller-key",
            **request_kwargs,
        )

    assert caught.value.status_code == 400
    assert not route.called


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("call", "global"))
async def test_safety_identifier_is_dropped_from_the_wire_under_drop_params(
    scope: str, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", scope == "global")
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = await litellm.adecisions(
        model="perplexity/pplx-decider-v1-27b",
        input="review",
        questions=_PREDICATE_QUESTIONS,
        safety_identifier=_SAFETY_IDENTIFIER,
        api_key="caller-key",
        **({"drop_params": True} if scope == "call" else {}),
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "pplx-decider-v1-27b",
        "state": "review",
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert isinstance(response, OpenAIDecisionResponse)
    assert [answer.model_dump(mode="json") for answer in response.answers] == [
        {"type": "predicate", "name": "is_defect", "probability": 0.9}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("api", "request_kwargs"), _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
@pytest.mark.parametrize("drop_params", (False, True), ids=("strict", "drop_params"))
async def test_safety_identifier_reaches_openai_whether_or_not_params_are_dropped(
    drop_params: bool,
    api: str,
    request_kwargs: Mapping[str, object],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", drop_params)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)

    await getattr(litellm, f"a{api}")(
        model="openai/gpt-6-luna",
        safety_identifier=_SAFETY_IDENTIFIER,
        api_key="caller-key",
        **request_kwargs,
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "gpt-6-luna",
        "input": "review",
        "questions": [{"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"}],
        "safety_identifier": _SAFETY_IDENTIFIER,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("api", "request_kwargs"), _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
async def test_non_string_safety_identifier_is_rejected_before_http(
    api: str, request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    invalid_safety_identifier: Final[Mapping[str, object]] = MappingProxyType({"safety_identifier": 7})

    with pytest.raises(litellm.BadRequestError, match=r"(?s)Invalid Decisions request.*safety_identifier"):
        await getattr(litellm, f"a{api}")(
            model="openai/gpt-6-luna", api_key="caller-key", **request_kwargs, **invalid_safety_identifier
        )

    assert not route.called


@pytest.mark.asyncio
@pytest.mark.parametrize(("api", "request_kwargs"), _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
async def test_non_string_safety_identifier_is_dropped_from_the_wire_under_drop_params(
    api: str, request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    dropped_safety_identifier: Final[Mapping[str, object]] = MappingProxyType(
        {"safety_identifier": 7, "drop_params": True}
    )

    await getattr(litellm, f"a{api}")(
        model="openai/gpt-6-luna", api_key="caller-key", **request_kwargs, **dropped_safety_identifier
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "gpt-6-luna",
        "input": "review",
        "questions": [{"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"}],
    }


_CLOUDFLARE_RUN_URL: Final = "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare"
_PERPLEXITY_DECISIONS_URL: Final = "https://api.perplexity.ai/v1/decisions"
_PNG_DATA_URL: Final = "data:image/png;base64,AA=="
_REMOTE_IMAGE_URL: Final = "https://images.example/screen.png"
_SCREEN_TEXT: Final = "Is the screen cracked?"
_SYSTEMONE_PREDICATE: Final[Mapping[str, object]] = MappingProxyType(
    {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}}
)


def _screen_messages(image_url: str) -> tuple[Mapping[str, object], ...]:
    return (
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": _SCREEN_TEXT},
                {"type": "input_image", "image_url": image_url, "detail": "high"},
            ],
        },
    )


def _image_requests(image_url: str) -> tuple[tuple[str, Mapping[str, object]], ...]:
    return (
        ("decisions", MappingProxyType({"input": _screen_messages(image_url), "questions": _PREDICATE_QUESTIONS})),
        (
            "systemone",
            MappingProxyType({"state": _SCREEN_TEXT, "questions": _SYSTEMONE_PREDICATE, "images": [image_url]}),
        ),
    )


@pytest.fixture
def cloudflare_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_account")
@pytest.mark.parametrize("model", ("clef", "clef-flash"))
async def test_input_images_reach_cloudflare_clef_as_its_images_field(model: str, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/{model}").respond(json=_RESPONSE)

    response: Final = await litellm.adecisions(
        model=f"cloudflare/{model}",
        input=(
            *_screen_messages(_PNG_DATA_URL),
            {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/webp;base64,BB=="}]},
            {"role": "user", "content": "Order 1234."},
        ),
        questions=_PREDICATE_QUESTIONS,
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": model,
        "state": f"{_SCREEN_TEXT}\n\nOrder 1234.",
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        "images": [_PNG_DATA_URL, "data:image/webp;base64,BB=="],
    }
    assert isinstance(response, OpenAIDecisionResponse)


@pytest.mark.usefixtures("cloudflare_account")
def test_systemone_images_reach_cloudflare_clef_in_the_callers_form(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/clef").respond(json=_RESPONSE)

    response: Final = litellm.systemone(
        model="cloudflare/clef",
        state={"ticket": 1234},
        questions=_SYSTEMONE_PREDICATE,
        images=[_PNG_DATA_URL, {"content_type": "image/webp", "base64": "BB=="}],
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": "clef",
        "state": {"ticket": 1234},
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        "images": [_PNG_DATA_URL, {"content_type": "image/webp", "base64": "BB=="}],
    }
    assert isinstance(response, DecisionsResponse)


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_account")
@pytest.mark.parametrize(("api", "request_kwargs"), _image_requests(_REMOTE_IMAGE_URL), ids=_PREDICATE_REQUEST_IDS)
async def test_remote_images_are_fetched_off_the_event_loop_and_inlined_for_cloudflare(
    api: str, request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, async_only_image_fetch
) -> None:
    route: Final = respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/clef").respond(json=_RESPONSE)

    await getattr(litellm, f"a{api}")(model="cloudflare/clef", **request_kwargs)

    assert async_only_image_fetch.fetched == [_REMOTE_IMAGE_URL]
    assert json.loads(route.calls[0].request.content)["images"] == [async_only_image_fetch.data_url]


@pytest.mark.usefixtures("cloudflare_account")
@pytest.mark.parametrize(("api", "request_kwargs"), _image_requests(_REMOTE_IMAGE_URL), ids=_PREDICATE_REQUEST_IDS)
def test_sync_decisions_fetch_and_inline_remote_images_for_cloudflare(
    api: str, request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    png: Final = b"\x89PNG\r\n\x1a\n"
    image_route: Final = respx_mock.get(_REMOTE_IMAGE_URL).respond(content=png, headers={"content-type": "image/png"})
    route: Final = respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/clef").respond(json=_RESPONSE)

    getattr(litellm, api)(model="cloudflare/clef", **request_kwargs)

    assert image_route.call_count == 1
    assert json.loads(route.calls[0].request.content)["images"] == [
        f"data:image/png;base64,{base64.b64encode(png).decode()}"
    ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_account")
@pytest.mark.parametrize("sync", (False, True), ids=("async", "sync"))
async def test_remote_image_downloads_use_the_callers_timeout(
    sync: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(litellm, "module_level_aclient", AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()))
    image_route: Final = respx_mock.get(_REMOTE_IMAGE_URL).respond(
        content=b"\x89PNG\r\n\x1a\n", headers={"content-type": "image/png"}
    )
    respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/clef").respond(json=_RESPONSE)
    request_kwargs: Final = {**_image_requests(_REMOTE_IMAGE_URL)[1][1], "model": "cloudflare/clef", "timeout": 7.0}

    if sync:
        litellm.systemone(**request_kwargs)
    else:
        await litellm.asystemone(**request_kwargs)

    assert image_route.calls.last.request.extensions["timeout"] == httpx.Timeout(7.0).as_dict()


@pytest.mark.asyncio
async def test_remote_input_images_reach_openai_unfetched(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, async_only_image_fetch
) -> None:
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)

    await litellm.adecisions(
        model="openai/gpt-6-luna",
        input=_screen_messages(_REMOTE_IMAGE_URL),
        questions=_PREDICATE_QUESTIONS,
        api_key="caller-key",
    )

    assert json.loads(route.calls[0].request.content)["input"][0]["content"][1]["image_url"] == _REMOTE_IMAGE_URL
    assert async_only_image_fetch.fetched == []


_IMAGE_REFUSALS: Final[tuple[tuple[str, Mapping[str, object], str], ...]] = (
    ("decisions", _image_requests(_PNG_DATA_URL)[0][1], "input_image"),
    ("systemone", _image_requests(_PNG_DATA_URL)[1][1], "images"),
)


@pytest.mark.parametrize(("api", "request_kwargs", "param"), _IMAGE_REFUSALS, ids=_PREDICATE_REQUEST_IDS)
def test_images_are_refused_before_http_when_the_provider_cannot_take_them(
    api: str,
    request_kwargs: Mapping[str, object],
    param: str,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post(_PERPLEXITY_DECISIONS_URL).respond(json=_RESPONSE)

    with pytest.raises(litellm.UnsupportedParamsError, match=rf"\['{param}'\].*drop_params") as caught:
        getattr(litellm, api)(model="perplexity/pplx-decider-v1-27b", api_key="caller-key", **request_kwargs)

    assert caught.value.status_code == 400
    assert not route.called


@pytest.mark.asyncio
@pytest.mark.parametrize(("api", "request_kwargs"), _image_requests(_PNG_DATA_URL), ids=_PREDICATE_REQUEST_IDS)
async def test_images_are_dropped_from_the_wire_under_drop_params(
    api: str, request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post(_PERPLEXITY_DECISIONS_URL).respond(json=_RESPONSE)

    await getattr(litellm, f"a{api}")(
        model="perplexity/pplx-decider-v1-27b", api_key="caller-key", drop_params=True, **request_kwargs
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": "pplx-decider-v1-27b",
        "state": _SCREEN_TEXT,
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }


_MALFORMED_IMAGE_ENTRIES: Final[tuple[Mapping[str, object], ...]] = (
    {"url": _PNG_DATA_URL},
    {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
    {"content_type": "image/png", "base64": "AA==", "detail": "high"},
)
_MALFORMED_IMAGE_ENTRY_IDS: Final = ("url_object", "chat_image_url", "extra_key")
_MALFORMED_IMAGES: Final[tuple[object, ...]] = (_PNG_DATA_URL, *([entry] for entry in _MALFORMED_IMAGE_ENTRIES))
_MALFORMED_IMAGE_IDS: Final = ("bare_string", *_MALFORMED_IMAGE_ENTRY_IDS)


@pytest.mark.usefixtures("cloudflare_account")
@pytest.mark.parametrize("images", _MALFORMED_IMAGES, ids=_MALFORMED_IMAGE_IDS)
def test_malformed_images_are_rejected_before_http(
    images: object, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/clef").respond(json=_RESPONSE)

    with pytest.raises(litellm.BadRequestError, match=r"(?s)Invalid Decisions request.*images"):
        litellm.systemone(model="cloudflare/clef", state=_SCREEN_TEXT, questions=_SYSTEMONE_PREDICATE, images=images)

    assert not route.called


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_account")
@pytest.mark.parametrize("images", _MALFORMED_IMAGES, ids=_MALFORMED_IMAGE_IDS)
async def test_malformed_images_are_dropped_from_the_wire_under_drop_params(
    images: object, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/clef").respond(json=_RESPONSE)

    await litellm.asystemone(
        model="cloudflare/clef", state=_SCREEN_TEXT, questions=_SYSTEMONE_PREDICATE, images=images, drop_params=True
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": "clef",
        "state": _SCREEN_TEXT,
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_account")
@pytest.mark.parametrize("entry", _MALFORMED_IMAGE_ENTRIES, ids=_MALFORMED_IMAGE_ENTRY_IDS)
async def test_a_malformed_image_under_drop_params_keeps_the_well_formed_ones(
    entry: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/clef").respond(json=_RESPONSE)

    await litellm.asystemone(
        model="cloudflare/clef",
        state=_SCREEN_TEXT,
        questions=_SYSTEMONE_PREDICATE,
        images=[entry, _PNG_DATA_URL, {"content_type": "image/png", "base64": "AA=="}],
        drop_params=True,
    )

    assert json.loads(route.calls[0].request.content)["images"] == [
        _PNG_DATA_URL,
        {"content_type": "image/png", "base64": "AA=="},
    ]


@pytest.mark.asyncio
async def test_empty_images_next_to_input_are_ignored(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)

    await litellm.adecisions(
        model="openai/gpt-6-luna",
        api_key="caller-key",
        input="review",
        questions=_PREDICATE_QUESTIONS,
        images=[],
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": "gpt-6-luna",
        "input": "review",
        "questions": [{"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"}],
    }


@pytest.mark.asyncio
async def test_dropping_the_only_input_images_is_rejected_before_http(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post(_PERPLEXITY_DECISIONS_URL).respond(json=_RESPONSE)

    with pytest.raises(litellm.BadRequestError, match=r"\['input_image'\].*leaves no input") as caught:
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b",
            api_key="caller-key",
            input=[{"role": "user", "content": [{"type": "input_image", "image_url": _PNG_DATA_URL}]}],
            questions=_PREDICATE_QUESTIONS,
            drop_params=True,
        )

    assert caught.value.status_code == 400
    assert not route.called


@pytest.mark.usefixtures("cloudflare_account")
def test_remote_image_urls_with_an_uppercase_scheme_are_inlined_for_cloudflare(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    png: Final = b"\x89PNG\r\n\x1a\n"
    image_route: Final = respx_mock.get(_REMOTE_IMAGE_URL).respond(content=png, headers={"content-type": "image/png"})
    route: Final = respx_mock.post(f"{_CLOUDFLARE_RUN_URL}/clef").respond(json=_RESPONSE)

    litellm.systemone(
        model="cloudflare/clef",
        state=_SCREEN_TEXT,
        questions=_SYSTEMONE_PREDICATE,
        images=[_REMOTE_IMAGE_URL.replace("https://", "HTTPS://")],
    )

    assert image_route.call_count == 1
    assert json.loads(route.calls[0].request.content)["images"] == [
        f"data:image/png;base64,{base64.b64encode(png).decode()}"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_base",
    (
        "https://res.services.ai.azure.com",
        "https://res.services.ai.azure.com/",
        "https://res.services.ai.azure.com/models",
        "https://res.services.ai.azure.com/openai/v1",
        "https://res.services.ai.azure.com/api/projects/proj",
        "https://res.services.ai.azure.com/api/projects/proj/openai/v1",
        "https://res.services.ai.azure.com/api/projects/proj/models",
        "https://res.services.ai.azure.com/models?api-version=2024-05-01-preview",
    ),
)
async def test_azure_ai_decision_posts_deployment_to_foundry_systemone_route(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    api_base: str,
) -> None:
    monkeypatch.delenv("AZURE_AI_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    route: Final = respx_mock.post("https://res.services.ai.azure.com/providers/microsoft/v1/systemone").respond(
        json={**_RESPONSE, "model": "microsoft-decision-1"}
    )

    response: Final = await litellm.asystemone(
        model="azure_ai/decision-1",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        api_base=api_base,
        api_key="foundry-key",
    )

    assert route.called
    request: Final = respx_mock.calls[0].request
    assert request.headers["authorization"] == "Bearer foundry-key"
    assert "api-key" not in request.headers
    assert json.loads(request.content) == {
        "model": "decision-1",
        "state": "review",
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert response.answers == {"is_defect": NoulAnswer(type="noul", noul=0.9)}
    assert response._hidden_params["model"] == "azure_ai/decision-1"


@pytest.mark.asyncio
async def test_azure_ai_systemone_uses_its_decisions_pricing_fields(
    respx_mock: respx.MockRouter, decisions_pricing_fields: None
) -> None:
    route: Final = respx_mock.post("https://res.services.ai.azure.com/providers/microsoft/v1/systemone").respond(
        json={**_RESPONSE, "model": "microsoft-decision-1"}
    )
    response: Final = await litellm.asystemone(
        model="azure_ai/Microsoft-Decision-1",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        api_base="https://res.services.ai.azure.com",
        api_key="foundry-key",
    )

    assert route.called
    assert response._hidden_params["provider_response_model"] == "azure_ai/Microsoft-Decision-1"
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        _INPUT_TOKENS * _MICROSOFT_DECISION_1_DECISIONS_INPUT_RATE
    )


@pytest.mark.asyncio
async def test_azure_ai_decision_reads_foundry_env_and_keeps_gateway_path_prefix(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://gateway.example.com/foundry/models")
    monkeypatch.setenv("AZURE_AI_API_KEY", "env-foundry-key")
    route: Final = respx_mock.post("https://gateway.example.com/foundry/providers/microsoft/v1/systemone").respond(
        json=_RESPONSE
    )

    await litellm.asystemone(
        model="azure_ai/decision-1",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer env-foundry-key"


@pytest.mark.asyncio
async def test_azure_ai_decision_requires_api_base_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("AZURE_AI_API_BASE", raising=False)
    monkeypatch.setenv("AZURE_AI_API_KEY", "foundry-key")

    with pytest.raises(litellm.BadRequestError, match="Missing AZURE_AI_API_BASE"):
        await litellm.asystemone(
            model="azure_ai/decision-1",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0


_CLEF_OMNI_UPSTREAM: Final = "https://example.com/ai/run/@cf/cloudflare/clef-omni"
_CLEF_OMNI_RESPONSE: Final[Mapping[str, object]] = {
    "success": True,
    "result": {
        "model": "clef-omni",
        "answers": {"heard": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 100, "output_tokens": 0},
    },
}
_MEDIA_REQUESTS: Final[tuple[tuple[str, Mapping[str, object]], ...]] = (
    (
        "decisions",
        MappingProxyType(
            {
                "input": "Evaluate the clips",
                "questions": [{"type": "predicate", "name": "heard", "instructions": "Is there speech?"}],
            }
        ),
    ),
    (
        "systemone",
        MappingProxyType(
            {
                "state": "Evaluate the clips",
                "questions": {"heard": {"type": "noul", "instructions": "Is there speech?"}},
            }
        ),
    ),
)


@pytest.mark.parametrize("async_mode", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize(("api", "request_kwargs"), _MEDIA_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
@pytest.mark.parametrize("model", ("cloudflare/clef-omni", "cloudflare/@cf/cloudflare/clef-omni"))
@pytest.mark.asyncio
async def test_clef_omni_media_reaches_upstream_in_both_formats(
    respx_mock: respx.MockRouter, async_mode: bool, api: str, request_kwargs: Mapping[str, object], model: str
) -> None:
    audio: Final = [
        "data:audio/wav;base64,YQ==",
        {"content_type": "audio/mpeg", "base64": "Yg=="},
        {"content_type": "audio/webm;codecs=opus", "base64": "ZQ=="},
    ]
    videos: Final = ["data:video/mp4;base64,Yw==", {"content_type": "video/webm;codecs=vp9", "base64": "ZA=="}]
    route: Final = respx_mock.post(_CLEF_OMNI_UPSTREAM).respond(200, json=dict(_CLEF_OMNI_RESPONSE))
    call: Final = getattr(litellm, f"a{api}" if async_mode else api)

    response: Final = (
        await call(
            model=model, api_base="https://example.com", api_key="test", audio=audio, videos=videos, **request_kwargs
        )
        if async_mode
        else call(
            model=model, api_base="https://example.com", api_key="test", audio=audio, videos=videos, **request_kwargs
        )
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": "clef-omni",
        "state": "Evaluate the clips",
        "questions": {"heard": {"type": "noul", "instructions": "Is there speech?"}},
        "audio": audio,
        "videos": videos,
    }
    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert len(route.calls) == 1


@pytest.mark.parametrize(
    "model",
    (
        "cloudflare/clef",
        "cloudflare/clef-flash",
        "typesafe/jev-latest",
        "openai/decision",
        "fireworks_ai/accounts/fireworks/routers/rune",
    ),
)
@pytest.mark.parametrize(
    "media",
    (
        {"audio": ["data:audio/wav;base64,YQ=="]},
        {"videos": ["data:video/mp4;base64,YQ=="]},
        {"audio": ["data:audio/wav;base64,YQ=="], "videos": ["data:video/mp4;base64,YQ=="]},
    ),
    ids=("audio", "videos", "both"),
)
@pytest.mark.asyncio
async def test_clips_for_a_model_without_media_support_are_refused_before_http(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, model: str, media: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    named_params: Final = str(list(media))

    with pytest.raises(
        litellm.UnsupportedParamsError, match=rf"does not support parameters: {re.escape(named_params)}, .*drop_params"
    ):
        await litellm.asystemone(
            model=model,
            state="test",
            questions=_QUESTIONS,
            api_base="https://example.com",
            api_key="test",
            **media,
        )
    assert len(respx_mock.calls) == 0


@pytest.mark.parametrize(
    "clips",
    (
        {"audio": ["data:audio/wav;base64,YQ=="], "videos": ["data:video/mp4;base64,YQ=="]},
        {"audio": ["https://example.com/clip.wav"], "videos": [{"content_type": "video/mp4"}]},
    ),
    ids=("well_formed", "malformed"),
)
@pytest.mark.parametrize(("api", "request_kwargs"), _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
@pytest.mark.asyncio
async def test_clips_are_dropped_from_the_wire_for_a_model_without_media_support_under_drop_params(
    api: str,
    request_kwargs: Mapping[str, object],
    clips: Mapping[str, object],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)
    call: Final = getattr(litellm, f"a{api}")

    await call(model="perplexity/pplx-decider-v1-27b", api_key="caller-key", **request_kwargs)
    await call(
        model="perplexity/pplx-decider-v1-27b", api_key="caller-key", drop_params=True, **request_kwargs, **clips
    )

    assert len(route.calls) == 2
    text_only_body: Final = json.loads(route.calls[0].request.content)
    assert "audio" not in text_only_body and "videos" not in text_only_body
    assert json.loads(route.calls[1].request.content) == text_only_body


@pytest.mark.parametrize(
    "media",
    (
        {"audio": ["https://example.com/clip.wav"]},
        {"videos": ["https://example.com/clip.mp4"]},
        {"audio": [{"content_type": "audio/wav"}]},
        {"videos": [{"content_type": "video/mp4"}]},
        {"audio": [{"content_type": "video/mp4", "base64": "YQ=="}]},
        {"videos": ["data:audio/wav;base64,YQ=="]},
        {"audio": ["data:audio/webm;codecs=opus;base64,YQ=="]},
        {"videos": ["data:video/webm;codecs=vp9;base64,YQ=="]},
        {"audio": ["data:audio/wav;base64,YQ=="] * 5},
        {"videos": ["data:video/mp4;base64,YQ=="] * 3},
    ),
)
@pytest.mark.asyncio
async def test_systemone_validates_media_before_http(respx_mock: respx.MockRouter, media: Mapping[str, object]) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        await litellm.asystemone(
            model="cloudflare/clef-omni",
            state="test",
            questions=_QUESTIONS,
            api_base="https://example.com",
            api_key="test",
            **media,
        )
    assert len(respx_mock.calls) == 0


@pytest.mark.parametrize(
    "media", ({"audio": [], "videos": []}, {"audio": None, "videos": None}, {"audio": None, "videos": []}), ids=repr
)
@pytest.mark.asyncio
async def test_empty_or_null_media_keeps_text_only_request_unchanged(
    respx_mock: respx.MockRouter, media: Mapping[str, object]
) -> None:
    route: Final = respx_mock.post("https://example.com/v1/systemone").respond(200, json=dict(_RESPONSE))
    await litellm.asystemone(
        model="typesafe/jev-latest",
        state="test",
        questions=_QUESTIONS,
        api_base="https://example.com",
        api_key="test",
        **media,
    )
    assert json.loads(route.calls[0].request.content) == {
        "model": "jev-latest",
        "state": "test",
        "questions": _QUESTIONS,
    }


@pytest.mark.asyncio
async def test_a_malformed_clip_for_a_model_that_takes_media_is_refused_even_under_drop_params(
    respx_mock: respx.MockRouter,
) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        await litellm.asystemone(
            model="cloudflare/clef-omni",
            state="test",
            questions=_QUESTIONS,
            api_base="https://example.com",
            api_key="test",
            audio=["https://example.com/clip.wav"],
            drop_params=True,
        )
    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("evaluation", "decisions"))
async def test_media_support_is_read_per_kind_from_the_cost_map(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None, mode: str
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setitem(
        litellm.model_cost,
        "typesafe/jev-hears",
        {"litellm_provider": "typesafe", "mode": mode, "supports_audio_input": True},
    )
    route: Final = respx_mock.post("https://example.com/v1/systemone").respond(200, json=dict(_RESPONSE))
    audio: Final = ["data:audio/wav;base64,YQ=="]

    await litellm.asystemone(
        model="typesafe/jev-hears",
        state="test",
        questions=_QUESTIONS,
        api_base="https://example.com",
        api_key="test",
        audio=audio,
    )
    with pytest.raises(litellm.UnsupportedParamsError, match=re.escape("does not support parameters: ['videos']")):
        await litellm.asystemone(
            model="typesafe/jev-hears",
            state="test",
            questions=_QUESTIONS,
            api_base="https://example.com",
            api_key="test",
            audio=audio,
            videos=["data:video/mp4;base64,YQ=="],
        )

    assert len(route.calls) == 1
    assert json.loads(route.calls[0].request.content) == {
        "model": "jev-hears",
        "state": "test",
        "questions": _QUESTIONS,
        "audio": audio,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "url", "cost_map_entry"),
    (
        (
            "typesafe/jev-chatty",
            "https://example.com/v1/systemone",
            {
                "litellm_provider": "typesafe",
                "mode": "chat",
                "supports_audio_input": True,
                "supports_video_input": True,
            },
        ),
        ("openai/gpt-audio", "https://example.com/v1/decisions", None),
    ),
)
async def test_a_chat_models_media_flags_do_not_count_for_decisions(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    local_model_cost_map: None,
    model: str,
    url: str,
    cost_map_entry: Mapping[str, object] | None,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    if cost_map_entry is not None:
        monkeypatch.setitem(litellm.model_cost, model, cost_map_entry)
    assert litellm.supports_audio_input(model) is True
    route: Final = respx_mock.post(url).respond(200, json=dict(_RESPONSE if "systemone" in url else _OPENAI_RESPONSE))
    questions: Final = {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}}
    audio: Final = ["data:audio/wav;base64,YQ=="]
    videos: Final = ["data:video/mp4;base64,YQ=="]

    with pytest.raises(
        litellm.UnsupportedParamsError, match=re.escape("does not support parameters: ['audio', 'videos']")
    ):
        await litellm.asystemone(
            model=model,
            state="test",
            questions=questions,
            api_base="https://example.com",
            api_key="test",
            audio=audio,
            videos=videos,
        )
    assert not route.calls

    await litellm.asystemone(
        model=model,
        state="test",
        questions=questions,
        api_base="https://example.com",
        api_key="test",
        audio=audio,
        videos=videos,
        drop_params=True,
    )
    sent: Final = json.loads(route.calls[0].request.content)
    assert "audio" not in sent
    assert "videos" not in sent
    assert sent["model"] == model.split("/", 1)[1]


def test_positional_arguments_keep_the_order_the_sdk_had_before_media(respx_mock: respx.MockRouter) -> None:
    decisions_route: Final = respx_mock.post("https://example.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    systemone_route: Final = respx_mock.post("https://example.com/v1/systemone").respond(json=_RESPONSE)
    openai_questions: Final = [{"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"}]
    systemone_questions: Final = {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}}

    litellm.decisions("openai/gpt-6-luna", "review", openai_questions, None, "positional-key", "https://example.com")
    litellm.systemone("typesafe/jev-1.13", "review", systemone_questions, "positional-key", "https://example.com")

    assert decisions_route.calls[0].request.headers["authorization"] == "Bearer positional-key"
    assert systemone_route.calls[0].request.headers["authorization"] == "Bearer positional-key"
    for sent in (decisions_route.calls[0].request.content, systemone_route.calls[0].request.content):
        assert "audio" not in json.loads(sent)


_FIREWORKS_URL: Final = "https://api.fireworks.ai/inference/v1/systemone"
_FIREWORKS_MODEL: Final = "fireworks_ai/accounts/fireworks/routers/rune"
_FIREWORKS_RESPONSE: Final[Mapping[str, object]] = {
    "model": "accounts/fireworks/routers/rune",
    "answers": {"is_defect": {"type": "noul", "noul": 0.9902915235185259}},
    "usage": {"input_tokens": 300, "cached_input_tokens": 200, "output_tokens": 0},
}


@pytest.mark.asyncio
async def test_fireworks_ai_posts_the_jev_body_to_systemone_with_the_account_model_name(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("FIREWORKS_API_BASE", raising=False)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("FIREWORKS_AI_API_KEY", "fireworks-key")
    route: Final = respx_mock.post(_FIREWORKS_URL).respond(json=_FIREWORKS_RESPONSE)

    response: Final = await litellm.asystemone(
        model=_FIREWORKS_MODEL,
        state="The package arrived broken.",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer fireworks-key"
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "accounts/fireworks/routers/rune",
        "state": "The package arrived broken.",
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert response.model == _FIREWORKS_RESPONSE["model"]
    assert response.answers["is_defect"] == NoulAnswer(type="noul", noul=0.9902915235185259)


@pytest.mark.asyncio
async def test_fireworks_ai_bills_cached_input_tokens_at_the_cache_read_rate(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("FIREWORKS_API_BASE", raising=False)
    respx_mock.post(_FIREWORKS_URL).respond(json=_FIREWORKS_RESPONSE)
    prices: Final = litellm.model_cost[_FIREWORKS_MODEL]
    uncached: Final = 100 * float(prices["input_cost_per_token"])
    cached: Final = 200 * float(prices["cache_read_input_token_cost"])
    recording_logger: Final = _RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recording_logger])

    response: Final = await litellm.adecisions(
        model=_FIREWORKS_MODEL,
        input="The package arrived broken.",
        questions=[{"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"}],
        api_key="caller-key",
    )
    await _drain_logging_worker()

    assert cached > 0
    assert response.usage.input_tokens_details.cached_tokens == 200
    assert recording_logger.standard_logging_object is not None
    assert recording_logger.standard_logging_object["response_cost"] == pytest.approx(uncached + cached)
    assert recording_logger.standard_logging_object["response_cost"] < 300 * float(prices["input_cost_per_token"])
