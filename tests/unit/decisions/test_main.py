from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
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

    response: Final = await router.adecisions(
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
        await litellm.adecisions(
            model="unknown/jev-1.13",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_provider_without_decisions_support_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match=r"Unknown Decisions provider 'anthropic'\. Supported providers"):
        await litellm.adecisions(
            model="anthropic/claude-sonnet-4-5",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_empty_custom_provider_falls_back_to_the_model_prefix(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = await litellm.adecisions(
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
        await litellm.adecisions(
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

    await router.adecisions(
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


def test_unreachable_upstream_maps_to_a_connection_error(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").mock(
        side_effect=httpx.ConnectError("Cannot connect to host api.perplexity.ai:443")
    )

    with pytest.raises(litellm.APIConnectionError, match="PerplexityException - Cannot connect to host"):
        litellm.decisions(
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

    litellm.decisions(
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

    response: Final = await litellm.adecisions(
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

    await litellm.adecisions(
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

    await litellm.adecisions(
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
        await litellm.adecisions(
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

    response: Final = await litellm.adecisions(
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
        await litellm.adecisions(
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

    response: Final = await litellm.adecisions(
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
async def test_strands_decider_uses_key_from_matching_environment_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("STRANDS_DECIDER_API_BASE", "https://strands.example")
    monkeypatch.setenv("STRANDS_DECIDER_API_KEY", "strands-key")
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    await litellm.adecisions(
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

    response: Final = await router.adecisions(
        model="strands",
        state="review",
        questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
    )

    assert provider_resolution[:2] == ("strands-decider-2B-hobson-v19", "strands_decider")
    assert route.called
    assert response.model == _STRANDS_RESPONSE["model"]


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
        await litellm.adecisions(
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

    response: Final = await litellm.adecisions(
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

    await litellm.adecisions(
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

    await litellm.adecisions(
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

    await litellm.adecisions(
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
        await litellm.adecisions(
            model="databricks/ai_decide",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["databricks-openjev-qwen35-4b", "serving-endpoints/ai_decide", "ai-decide"])
async def test_databricks_rejects_a_model_other_than_ai_decide_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    model: str,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", "https://workspace.example")
    monkeypatch.setenv("DATABRICKS_API_KEY", "dapi-key")

    with pytest.raises(litellm.BadRequestError, match="'ai_decide'"):
        await litellm.adecisions(
            model=f"databricks/{model}",
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
        await litellm.adecisions(
            model="databricks/ai_decide",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0


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

    response: Final = await router.adecisions(
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
) -> None:
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    route: Final = respx_mock.post(url).respond(json=_OPENAI_RESPONSE)

    response: Final = await litellm.adecisions(
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
    luna_cost: Final = litellm.model_cost["gpt-6-luna"]
    expected_cost: Final = (
        (_INPUT_TOKENS - _CACHED_TOKENS - _CACHE_WRITE_TOKENS) * float(luna_cost["input_cost_per_token"])
        + _CACHED_TOKENS * float(luna_cost["cache_read_input_token_cost"])
        + _CACHE_WRITE_TOKENS * float(luna_cost["cache_creation_input_token_cost"])
        + _OUTPUT_TOKENS * float(luna_cost["output_cost_per_token"])
    )
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

    litellm.decisions(
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
            "typesafe/jev-1.13",
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}],
                    }
                ],
                "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
            },
            "cannot serve this request",
        ),
        (
            "openai/gpt-6-luna",
            {
                "state": "review",
                "input": "review",
                "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
            },
            "not both",
        ),
    ),
    ids=("image_to_systemone_provider", "state_and_input"),
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

    response: Final = await litellm.adecisions(
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
_PREDICATE_REQUESTS: Final[tuple[Mapping[str, object], ...]] = (
    MappingProxyType({"input": "review", "questions": _PREDICATE_QUESTIONS}),
    MappingProxyType(
        {"state": "review", "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}}}
    ),
)
_PREDICATE_REQUEST_IDS: Final = ("input_format", "state_format")


@pytest.mark.parametrize("request_kwargs", _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
def test_safety_identifier_is_refused_before_http_when_the_provider_cannot_take_it(
    request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    with pytest.raises(litellm.UnsupportedParamsError, match=r"safety_identifier.*drop_params") as caught:
        litellm.decisions(
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
@pytest.mark.parametrize("request_kwargs", _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
@pytest.mark.parametrize("drop_params", (False, True), ids=("strict", "drop_params"))
async def test_safety_identifier_reaches_openai_whether_or_not_params_are_dropped(
    drop_params: bool,
    request_kwargs: Mapping[str, object],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", drop_params)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)

    await litellm.adecisions(
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
@pytest.mark.parametrize("request_kwargs", _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
async def test_non_string_safety_identifier_is_rejected_before_http(
    request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    invalid_safety_identifier: Final[Mapping[str, object]] = MappingProxyType({"safety_identifier": 7})

    with pytest.raises(litellm.BadRequestError, match=r"(?s)Invalid Decisions request.*safety_identifier"):
        await litellm.adecisions(
            model="openai/gpt-6-luna", api_key="caller-key", **request_kwargs, **invalid_safety_identifier
        )

    assert not route.called


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", _PREDICATE_REQUESTS, ids=_PREDICATE_REQUEST_IDS)
async def test_non_string_safety_identifier_is_dropped_from_the_wire_under_drop_params(
    request_kwargs: Mapping[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    dropped_safety_identifier: Final[Mapping[str, object]] = MappingProxyType(
        {"safety_identifier": 7, "drop_params": True}
    )

    await litellm.adecisions(
        model="openai/gpt-6-luna", api_key="caller-key", **request_kwargs, **dropped_safety_identifier
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "gpt-6-luna",
        "input": "review",
        "questions": [{"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"}],
    }


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

    response: Final = await litellm.adecisions(
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
async def test_azure_ai_decision_reads_foundry_env_and_keeps_gateway_path_prefix(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://gateway.example.com/foundry/models")
    monkeypatch.setenv("AZURE_AI_API_KEY", "env-foundry-key")
    route: Final = respx_mock.post("https://gateway.example.com/foundry/providers/microsoft/v1/systemone").respond(
        json=_RESPONSE
    )

    await litellm.adecisions(
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
        await litellm.adecisions(
            model="azure_ai/decision-1",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0
