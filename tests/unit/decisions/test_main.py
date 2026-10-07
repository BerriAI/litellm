from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Final

import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.decisions import (
    ChoiceAnswer,
    DecisionInputMessage,
    DecisionsInputTokensDetails,
    DecisionsOutputTokensDetails,
    DecisionsResponse,
    DecisionsUsage,
    PredicateAnswer,
    ScoreAnswer,
)

_INPUT: Final = "The export job hangs at 99% and never finishes"
_QUESTIONS: Final[Sequence[Mapping[str, object]]] = (
    {"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"},
    {
        "type": "choice",
        "name": "sentiment",
        "instructions": "How does the customer feel?",
        "choices": [{"value": "positive"}, {"value": "negative", "description": "unhappy"}],
    },
    {
        "type": "score",
        "name": "severity",
        "instructions": "How severe is it?",
        "levels": [{"label": "none"}, {"label": "low"}, {"label": "high", "description": "blocks users"}],
    },
)
_SYSTEM_ONE_QUESTIONS: Final[Mapping[str, object]] = {
    "is_defect": {"type": "noul", "instructions": "Is this a defect?"},
    "sentiment": {
        "type": "choice",
        "instructions": "How does the customer feel?",
        "criteria": {"positive": None, "negative": "unhappy"},
    },
    "severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["none", "low", "blocks users"]},
}
_INPUT_TOKENS: Final[int] = 367
_OUTPUT_TOKENS: Final[int] = 3
_SYSTEM_ONE_RESPONSE: Final[Mapping[str, object]] = {
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
_EXPECTED_ANSWERS: Final[Sequence[Mapping[str, object]]] = (
    {"type": "predicate", "name": "is_defect", "probability": 0.9},
    {
        "type": "choice",
        "name": "sentiment",
        "choice": "positive",
        "probabilities": [{"value": "positive", "probability": 0.8}, {"value": "negative", "probability": 0.2}],
        "confidence": 0.8,
    },
    {
        "type": "score",
        "name": "severity",
        "score": 1.0,
        "probabilities": [
            {"value": 0, "label": "none", "probability": 0.1},
            {"value": 1, "label": "low", "probability": 0.8},
            {"value": 2, "label": "high", "probability": 0.1},
        ],
        "confidence": 0.7,
    },
)
_EXPECTED_USAGE: Final[Mapping[str, object]] = {
    "input_tokens": _INPUT_TOKENS,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": _OUTPUT_TOKENS,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
}
_ZERO_USAGE: Final = DecisionsUsage(
    input_tokens=0,
    input_tokens_details=DecisionsInputTokensDetails(cached_tokens=0, cache_write_tokens=0),
    output_tokens=0,
    output_tokens_details=DecisionsOutputTokensDetails(reasoning_tokens=0),
    total_tokens=0,
)
_OPENAI_RESPONSE: Final[Mapping[str, object]] = {
    "model": "gpt-6-luna",
    "answers": [*_EXPECTED_ANSWERS, {"type": "refusal", "name": None}],
    "usage": _EXPECTED_USAGE,
}
_PROVIDERS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("perplexity", "perplexity/pplx-decider-v1-27b", "https://api.perplexity.ai/v1/decisions", "pplx-decider-v1-27b"),
    ("typesafe", "typesafe/jev-1.13", "https://api.typesafe.ai/v1/systemone", "jev-1.13"),
    ("openrouter", "openrouter/typesafe/jev-1.13", "https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13"),
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


def _predicate(name: str = "is_defect") -> list[Mapping[str, object]]:
    return [{"type": "predicate", "name": name, "instructions": "Is this a defect?"}]


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "model", "url", "upstream_model"), _PROVIDERS)
async def test_adecisions_translates_to_the_system_one_wire_contract(
    provider: str,
    model: str,
    url: str,
    upstream_model: str,
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(url).respond(json=_SYSTEM_ONE_RESPONSE)

    response: Final = await litellm.adecisions(
        model=model,
        input=_INPUT,
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
        "state": _INPUT,
        "questions": _SYSTEM_ONE_QUESTIONS,
    }
    assert response.model_dump(mode="json") == {
        "model": "jev-1.13",
        "answers": list(_EXPECTED_ANSWERS),
        "usage": _EXPECTED_USAGE,
    }
    assert isinstance(response.answers[0], PredicateAnswer)
    assert isinstance(response.answers[1], ChoiceAnswer)
    assert isinstance(response.answers[2], ScoreAnswer)
    assert response._hidden_params["custom_llm_provider"] == provider


@pytest.mark.asyncio
async def test_openai_decisions_are_forwarded_without_translation(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    image_input: Final = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": _INPUT},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"},
            ],
        }
    ]

    response: Final = await litellm.adecisions(
        model="openai/gpt-6-luna",
        input=image_input,
        questions=[*_QUESTIONS, {"type": "choice", "instructions": "Refund?", "choices": [{"value": True}]}],
        safety_identifier="user-123",
        api_key="caller-key",
    )

    assert route.called
    request: Final = respx_mock.calls[0].request
    assert request.headers["authorization"] == "Bearer caller-key"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "input": image_input,
        "questions": [*_QUESTIONS, {"type": "choice", "instructions": "Refund?", "choices": [{"value": True}]}],
        "safety_identifier": "user-123",
    }
    assert response.model_dump(mode="json") == _OPENAI_RESPONSE
    assert response._hidden_params["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_openai_decisions_fall_back_to_the_global_openai_key(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx_mock.post("https://api.openai.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", "global-openai-key")

    await litellm.adecisions(model="openai/gpt-6-luna", input=_INPUT, questions=_predicate())

    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer global-openai-key"


@pytest.mark.asyncio
async def test_openai_decisions_honor_the_global_api_base(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    route: Final = respx_mock.post("https://gateway.example.com/v1/decisions").respond(json=_OPENAI_RESPONSE)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", "https://gateway.example.com/v1")

    await litellm.adecisions(model="openai/gpt-6-luna", input=_INPUT, questions=_predicate(), api_key="caller-key")

    assert route.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "input_value", "questions"),
    (
        (
            "input_image",
            [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}],
            _predicate(),
        ),
        ("boolean choice", _INPUT, [{"type": "choice", "instructions": "Refund?", "choices": [{"value": True}]}]),
        ("unique name", _INPUT, [*_predicate(), *_predicate()]),
        (
            "repeated choice",
            _INPUT,
            [{"type": "choice", "instructions": "Refund?", "choices": [{"value": "yes"}, {"value": "yes"}]}],
        ),
    ),
)
async def test_system_one_providers_refuse_what_they_cannot_express_before_http(
    label: str,
    input_value: object,
    questions: Sequence[Mapping[str, object]],
    respx_mock: respx.MockRouter,
) -> None:
    with pytest.raises(litellm.BadRequestError, match=label):
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b", input=input_value, questions=questions, api_key="caller-key"
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_positional_system_one_keys_never_shadow_a_supplied_name(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        json={
            "model": "pplx-decider-v1-27b",
            "answers": {"_q0": {"type": "noul", "noul": 0.25}, "q0": {"type": "noul", "noul": 0.75}},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )

    response: Final = await litellm.adecisions(
        model="perplexity/pplx-decider-v1-27b",
        input=_INPUT,
        questions=[{"type": "predicate", "instructions": "Is this a defect?"}, *_predicate("q0")],
        api_key="caller-key",
    )

    assert route.called
    assert tuple(json.loads(respx_mock.calls[0].request.content)["questions"]) == ("_q0", "q0")
    assert [answer.model_dump(mode="json") for answer in response.answers] == [
        {"type": "predicate", "name": None, "probability": 0.25},
        {"type": "predicate", "name": "q0", "probability": 0.75},
    ]


@pytest.mark.asyncio
async def test_unnamed_questions_get_positional_system_one_keys(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        json={
            "model": "pplx-decider-v1-27b",
            "answers": {"q0": {"type": "noul", "noul": 0.25}, "q1": {"type": "noul", "noul": 0.75}},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )
    user_messages: Final = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
    ]

    response: Final = await litellm.adecisions(
        model="perplexity/pplx-decider-v1-27b",
        input=user_messages,
        questions=[
            {"type": "predicate", "instructions": "Is this a defect?"},
            {"type": "predicate", "instructions": "Is this urgent?"},
        ],
        api_key="caller-key",
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "pplx-decider-v1-27b",
        "state": "first\nsecond",
        "questions": {
            "q0": {"type": "noul", "instructions": "Is this a defect?"},
            "q1": {"type": "noul", "instructions": "Is this urgent?"},
        },
    }
    assert [answer.model_dump(mode="json") for answer in response.answers] == [
        {"type": "predicate", "name": None, "probability": 0.25},
        {"type": "predicate", "name": None, "probability": 0.75},
    ]


@pytest.mark.asyncio
async def test_system_one_reply_missing_an_answer_is_a_server_error(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        json={"model": "pplx-decider-v1-27b", "answers": {}, "usage": {"input_tokens": 10, "output_tokens": 0}}
    )

    with pytest.raises(litellm.InternalServerError, match="no predicate answer for question 'is_defect'"):
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b", input=_INPUT, questions=_predicate(), api_key="caller-key"
        )


@pytest.mark.asyncio
async def test_router_dispatches_typesafe_decisions_without_api_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_BASE", raising=False)
    assert litellm.get_llm_provider("typesafe/jev-latest")[:2] == ("jev-latest", "typesafe")
    router: Final = litellm.Router(
        model_list=[{"model_name": "jev", "litellm_params": {"model": "typesafe/jev-latest", "api_key": "k"}}]
    )
    upstream: Final = respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(json=_SYSTEM_ONE_RESPONSE)

    response: Final = await router.adecisions(model="jev", input=_INPUT, questions=_QUESTIONS)

    assert upstream.called
    assert len(respx_mock.calls) == 1
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "jev-latest",
        "state": _INPUT,
        "questions": _SYSTEM_ONE_QUESTIONS,
    }
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer k"
    sentiment: Final = response.answers[1]
    assert isinstance(sentiment, ChoiceAnswer)
    assert sentiment.choice == "positive"


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
    upstream: Final = respx_mock.post("https://egress.example/openrouter/alpha/decisions").respond(
        json=_SYSTEM_ONE_RESPONSE
    )

    await router.adecisions(model="jev", input=_INPUT, questions=_predicate())

    assert upstream.called
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer deployment-key"
    assert json.loads(respx_mock.calls[0].request.content)["model"] == "typesafe/jev-1.13"


def test_decisions_uses_the_same_wire_contract_for_sync_calls(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_SYSTEM_ONE_RESPONSE)

    response: Final = litellm.decisions(
        model="perplexity/pplx-decider-v1-27b", input=_INPUT, questions=_QUESTIONS, api_key="caller-key"
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content)["questions"] == _SYSTEM_ONE_QUESTIONS
    assert response.model_dump(mode="json") == {
        "model": "jev-1.13",
        "answers": list(_EXPECTED_ANSWERS),
        "usage": _EXPECTED_USAGE,
    }


def test_system_one_provider_extras_are_not_part_of_the_response(respx_mock: respx.MockRouter) -> None:
    payload: Final = {
        **_SYSTEM_ONE_RESPONSE,
        "id": "decision-1",
        "provider": "typesafe",
        "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS, "cost": 0.25},
    }
    respx_mock.post("https://openrouter.ai/api/alpha/decisions").respond(json=payload)

    response: Final = litellm.decisions(
        model="openrouter/typesafe/jev-1.13", input=_INPUT, questions=_QUESTIONS, api_key="caller-key"
    )

    assert response.model_extra == {}
    assert response.usage.model_dump(mode="json") == _EXPECTED_USAGE


def test_decisions_cost_uses_litellm_token_pricing() -> None:
    response: Final = DecisionsResponse(
        model="pplx-decider-v1-27b",
        answers=[],
        usage=DecisionsUsage(
            input_tokens=_INPUT_TOKENS,
            input_tokens_details=DecisionsInputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens=_OUTPUT_TOKENS,
            output_tokens_details=DecisionsOutputTokensDetails(reasoning_tokens=0),
            total_tokens=_INPUT_TOKENS + _OUTPUT_TOKENS,
        ),
    )
    response.set_hidden_params({"model": "perplexity/pplx-decider-v1-27b", "custom_llm_provider": "perplexity"})

    cost: Final = litellm.completion_cost(completion_response=response)
    perplexity_cost: Final = litellm.model_cost["perplexity/pplx-decider-v1-27b"]
    expected_cost: Final = _INPUT_TOKENS * float(perplexity_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        perplexity_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert cost == pytest.approx(expected_cost)


def test_decisions_response_hidden_params_getter_preserves_mutable_identity() -> None:
    response: Final = DecisionsResponse(model="decider", answers=(), usage=_ZERO_USAGE)

    assert response.hidden_params is response._hidden_params

    response.hidden_params["mutation"] = "visible"
    assert response._hidden_params["mutation"] == "visible"


@pytest.mark.asyncio
async def test_decisions_cost_is_in_standard_logging_object(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_SYSTEM_ONE_RESPONSE)
    recording_logger: Final = _RecordingLogger()
    original_callbacks: Final = litellm.callbacks
    litellm.callbacks = [recording_logger]

    try:
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b", input=_INPUT, questions=_predicate(), api_key="caller-key"
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
    assert recording_logger.standard_logging_object["messages"] == [{"role": "user", "content": _INPUT}]


@pytest.mark.asyncio
async def test_decisions_log_typed_input_messages(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_SYSTEM_ONE_RESPONSE)
    recording_logger: Final = _RecordingLogger()
    original_callbacks: Final = litellm.callbacks
    litellm.callbacks = [recording_logger]

    try:
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b",
            input=[DecisionInputMessage(role="user", content=_INPUT)],
            questions=_predicate(),
            api_key="caller-key",
        )
        await _drain_logging_worker()
    finally:
        litellm.callbacks = original_callbacks

    assert route.called
    assert recording_logger.standard_logging_object is not None
    logged_messages: Final = recording_logger.standard_logging_object["messages"]
    assert json.loads(logged_messages[0]["content"]) == [{"role": "user", "content": _INPUT}]


@pytest.mark.asyncio
async def test_unknown_provider_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
        await litellm.adecisions(model="unknown/jev-1.13", input=_INPUT, questions=_predicate(), api_key="caller-key")

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_provider_without_decisions_support_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match=r"Unknown Decisions provider 'anthropic'\. Supported providers"):
        await litellm.adecisions(
            model="anthropic/claude-sonnet-4-5", input=_INPUT, questions=_predicate(), api_key="caller-key"
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_empty_custom_provider_falls_back_to_the_model_prefix(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_SYSTEM_ONE_RESPONSE)

    response: Final = await litellm.adecisions(
        model="perplexity/pplx-decider-v1-27b",
        input=_INPUT,
        questions=_predicate(),
        api_key="caller-key",
        custom_llm_provider="",
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content)["model"] == "pplx-decider-v1-27b"
    assert response._hidden_params["custom_llm_provider"] == "perplexity"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "questions",
    (
        [{"type": "choice", "name": "sentiment", "instructions": "How?"}],
        [{"type": "noul", "name": "is_defect", "instructions": "Is this a defect?"}],
        {"is_defect": {"type": "predicate", "instructions": "Is this a defect?"}},
        [],
    ),
    ids=("choice without choices", "jev question type", "questions map", "no questions"),
)
async def test_invalid_questions_are_rejected_before_http(questions: object, respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b", input=_INPUT, questions=questions, api_key="caller-key"
        )

    assert len(respx_mock.calls) == 0


def test_upstream_bad_request_maps_to_litellm_error(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        status_code=400, json={"error": {"message": "invalid decision"}}
    )

    with pytest.raises(litellm.BadRequestError):
        litellm.decisions(
            model="perplexity/pplx-decider-v1-27b", input=_INPUT, questions=_predicate(), api_key="caller-key"
        )


def test_server_key_is_sent_to_an_explicit_api_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "server-key")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    route: Final = respx_mock.post("https://egress.example/perplexity/v1/decisions").respond(json=_SYSTEM_ONE_RESPONSE)

    litellm.decisions(
        model="perplexity/pplx-decider-v1-27b",
        input=_INPUT,
        questions=_predicate(),
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
        {"result": _SYSTEM_ONE_RESPONSE, "success": True, "errors": [], "messages": []}
        if wrapped
        else _SYSTEM_ONE_RESPONSE
    )
    route: Final = respx_mock.post(
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"
    ).respond(json=response_body)

    response: Final = await litellm.adecisions(model=model, input=_INPUT, questions=_QUESTIONS)

    assert route.called
    request: Final = respx_mock.calls[0].request
    assert request.headers["authorization"] == "Bearer cloudflare-key"
    assert json.loads(request.content) == {"model": "clef", "state": _INPUT, "questions": _SYSTEM_ONE_QUESTIONS}
    assert [answer.model_dump(mode="json") for answer in response.answers] == list(_EXPECTED_ANSWERS)
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
    ).respond(json=_SYSTEM_ONE_RESPONSE)

    await litellm.adecisions(model="cloudflare/clef-flash", input=_INPUT, questions=_QUESTIONS)

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
    ).respond(json=_SYSTEM_ONE_RESPONSE)

    await litellm.adecisions(model="cloudflare/clef", input=_INPUT, questions=_QUESTIONS)

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
        await litellm.adecisions(model="cloudflare/clef", input=_INPUT, questions=_QUESTIONS)

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
        json=_SYSTEM_ONE_RESPONSE
    )

    response: Final = await litellm.adecisions(model="cloudflare/clef", input=_INPUT, questions=_QUESTIONS)

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
            model="strands_decider/strands-decider-2B-hobson-v19", input=_INPUT, questions=_QUESTIONS
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_strands_decider_without_key_sends_no_authorization(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(
        json={**_SYSTEM_ONE_RESPONSE, "model": "strands-decider-2B-hobson-v19", "latency_ms": 140.03}
    )

    response: Final = await litellm.adecisions(
        model="strands_decider/strands-decider-2B-hobson-v19",
        input=_INPUT,
        questions=_QUESTIONS,
        api_base="https://strands.example",
    )

    assert route.called
    assert "authorization" not in respx_mock.calls[0].request.headers
    assert response.model == "strands-decider-2B-hobson-v19"
    severity: Final = response.answers[2]
    assert isinstance(severity, ScoreAnswer)
    assert [probability.label for probability in severity.probabilities] == ["none", "low", "high"]


@pytest.mark.asyncio
async def test_strands_decider_uses_key_from_matching_environment_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("STRANDS_DECIDER_API_BASE", "https://strands.example")
    monkeypatch.setenv("STRANDS_DECIDER_API_KEY", "strands-key")
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_SYSTEM_ONE_RESPONSE)

    await litellm.adecisions(
        model="strands_decider/strands-decider-2B-hobson-v19",
        input=_INPUT,
        questions=_QUESTIONS,
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
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_SYSTEM_ONE_RESPONSE)

    response: Final = await router.adecisions(model="strands", input=_INPUT, questions=_QUESTIONS)

    assert provider_resolution[:2] == ("strands-decider-2B-hobson-v19", "strands_decider")
    assert route.called
    assert response.model == "jev-1.13"
