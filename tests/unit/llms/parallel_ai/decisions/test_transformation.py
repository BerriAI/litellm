import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.base_llm.decisions.transformation import (
    DecisionsResponseValidationError,
    ir_to_systemone_response,
    systemone_request_to_ir,
)
from litellm.llms.openai.decisions.transformation import ir_to_openai_response, openai_request_to_ir
from litellm.llms.parallel_ai.decisions.transformation import ParallelAIDecisionsConfig
from litellm.types.decisions import (
    DecisionsIRRefusal,
    DecisionsRequestBody,
    OpenAIDecisionRequestBody,
    UnsupportedDecisionsRequest,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, Rules, function_setup

QUESTIONS: Final = {
    "available": {"type": "noul", "instructions": "Is this product available?"},
    "tier": {"type": "choice", "instructions": "Which tier is offered?", "criteria": {"free": "Free", "paid": "Paid"}},
    "fit": {"type": "score", "instructions": "How well does it fit?", "criteria": ["Poor", "Good", "Excellent"]},
    "unknown": {"type": "noul", "instructions": "Is there enough evidence?"},
}
REQUEST: Final = systemone_request_to_ir(
    DecisionsRequestBody.model_validate({"state": {"product": "Example"}, "questions": QUESTIONS})
)
PAYLOAD: Final = {
    "mode": "fast",
    "answers": {
        "available": {"type": "noul", "noul": 0.9},
        "tier": {"type": "choice", "choice": "paid", "probabilities": {"free": 0.2, "paid": 0.8}},
        "fit": {
            "type": "score",
            "score": 1.7,
            "legend": {"0": "Poor", "1": "Good", "2": "Excellent"},
            "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
        },
        "unknown": {"type": "refusal"},
    },
    "usage": {"input_tokens": 4212, "output_tokens": 0},
    "citations": [{"url": "https://example.com/product", "title": "Product", "excerpts": ["Available now"]}],
    "decision_id": "decisions_example",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("surface", ["decisions", "systemone"])
async def test_sdk_sends_parallel_wire_format_and_preserves_evidence(
    use_async: bool, surface: str, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    route: Final = respx_mock.post("https://api.parallel.ai/v1/decisions").respond(json=PAYLOAD)
    connection: Final = {"model": "parallel_ai/decision", "api_key": "test-key", "timeout": 5, "num_retries": 0}
    fields: Final = (
        {"state": "Example", "questions": QUESTIONS}
        if surface == "systemone"
        else {
            "input": "Example",
            "questions": [{"type": "predicate", "name": "available", "instructions": "Available?"}],
        }
    )
    sync_call: Final = litellm.systemone if surface == "systemone" else litellm.decisions
    async_call: Final = litellm.asystemone if surface == "systemone" else litellm.adecisions
    response: Final = await async_call(**connection, **fields) if use_async else sync_call(**connection, **fields)
    sent: Final = route.calls.last.request
    body: Final = json.loads(sent.content)
    assert body["mode"] == "fast"
    assert "model" not in body
    assert body["state"] == "Example"
    assert sent.headers["x-api-key"] == "test-key"
    assert "authorization" not in sent.headers
    assert response.model == "decision"
    assert response.model_dump()["citations"] == PAYLOAD["citations"]
    assert response.model_dump()["decision_id"] == PAYLOAD["decision_id"]
    await GLOBAL_LOGGING_WORKER.clear_queue()


def test_provider_routes_systemone_questions_to_parallel_without_a_model_field() -> None:
    config: Final = ProviderConfigManager.get_provider_decisions_config("decision", LlmProviders.PARALLEL_AI)
    assert config is not None
    assert config.transform_decisions_request("decision", REQUEST, "parallel_ai") == {
        "mode": "fast",
        "state": {"product": "Example"},
        "questions": QUESTIONS,
    }
    assert config.get_complete_url("https://gateway.example/v1/", "decision") == "https://gateway.example/v1/decisions"


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/", "/v1/decisions", "/v1/decisions/", "/v1/search", "/v1/search/"])
def test_base_carrying_a_parallel_route_reaches_the_decisions_endpoint(suffix: str) -> None:
    complete_url: Final = ParallelAIDecisionsConfig().get_complete_url(f"https://api.parallel.ai{suffix}", "decision")
    assert complete_url == "https://api.parallel.ai/v1/decisions"


def test_auth_replaces_conflicting_headers_without_forwarding_bearer_credentials() -> None:
    headers: Final = ParallelAIDecisionsConfig().validate_environment(
        {"Authorization": "Bearer old", "X-API-Key": "old", "CONTENT-TYPE": "text/plain", "X-Trace": "test"},
        "decision",
        "test-key",
        {},
    )
    assert headers == {"x-api-key": "test-key", "Content-Type": "application/json", "X-Trace": "test"}


def test_openai_predicate_is_translated_to_a_named_noul_question() -> None:
    request: Final = openai_request_to_ir(
        OpenAIDecisionRequestBody.model_validate(
            {
                "input": "Example product",
                "questions": [{"type": "predicate", "name": "available", "instructions": "Available?"}],
            }
        )
    )
    assert ParallelAIDecisionsConfig().transform_decisions_request("decision", request, "parallel_ai") == {
        "mode": "fast",
        "state": "Example product",
        "questions": {"available": {"type": "noul", "instructions": "Available?"}},
    }


@pytest.mark.parametrize("surface", ["systemone", "decisions"])
def test_all_answers_usage_and_evidence_survive_response_translation(surface: str) -> None:
    response: Final = ParallelAIDecisionsConfig().parse_response(PAYLOAD, REQUEST)
    assert isinstance(response.answers[3], DecisionsIRRefusal)
    formatted: Final = (
        ir_to_systemone_response(response, REQUEST)
        if surface == "systemone"
        else ir_to_openai_response(
            response, REQUEST, "parallel_ai/decision", ParallelAIDecisionsConfig.openai_evidence_fields
        )
    )
    result: Final = formatted.model_dump(mode="json")
    assert result["citations"] == PAYLOAD["citations"]
    assert result["decision_id"] == PAYLOAD["decision_id"]
    assert result["mode"] == "fast"
    assert result["usage"]["input_tokens"] == 4212
    assert result["usage"]["output_tokens"] == 0
    if surface == "systemone":
        assert result["answers"]["available"]["noul"] == 0.9
        assert result["answers"]["tier"]["choice"] == "paid"
        assert result["answers"]["fit"]["score"] == 1.7
        assert "unknown" not in result["answers"]
    else:
        assert result["answers"][3]["type"] == "refusal"


def test_rejects_more_than_one_hundred_questions() -> None:
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": "Example",
                "questions": {str(index): QUESTIONS["available"] for index in range(101)},
            }
        )
    )
    assert isinstance(
        ParallelAIDecisionsConfig().transform_decisions_request("decision", request, "parallel_ai"),
        UnsupportedDecisionsRequest,
    )


@pytest.mark.parametrize("model", ["fast", "unknown"])
def test_rejects_unregistered_models_instead_of_forwarding_them_as_modes(model: str) -> None:
    assert isinstance(
        ParallelAIDecisionsConfig().transform_decisions_request(model, REQUEST, "parallel_ai"),
        UnsupportedDecisionsRequest,
    )


@pytest.mark.parametrize("confidence", [None, 0.8])
@pytest.mark.parametrize("surface", ["systemone", "decisions"])
def test_optional_confidence_is_preserved_without_inventing_a_value(confidence: float | None, surface: str) -> None:
    payload: Final = {
        **PAYLOAD,
        "answers": {name: {**answer, "confidence": confidence} for name, answer in PAYLOAD["answers"].items()},
    }
    response: Final = ParallelAIDecisionsConfig().parse_response(payload, REQUEST)
    formatted: Final = (
        ir_to_systemone_response(response, REQUEST)
        if surface == "systemone"
        else ir_to_openai_response(
            response, REQUEST, "parallel_ai/decision", ParallelAIDecisionsConfig.openai_evidence_fields
        )
    )
    answers: Final = formatted.model_dump(mode="json")["answers"]
    values: Final = answers.values() if isinstance(answers, dict) else answers
    assert [answer["confidence"] for answer in values if answer["type"] in {"choice", "score"}] == [confidence] * 2


def test_provider_extras_cannot_replace_normalized_response_fields() -> None:
    response: Final = ParallelAIDecisionsConfig().parse_response(PAYLOAD, REQUEST)
    with_collision: Final = replace(response, extra={**response.extra, "model": "wrong", "answers": [], "usage": {}})
    formatted: Final = ir_to_openai_response(
        with_collision, REQUEST, "parallel_ai/decision", ParallelAIDecisionsConfig.openai_evidence_fields
    )
    assert formatted.model == "decision"
    assert len(formatted.answers) == 4
    assert formatted.usage.input_tokens == 4212
    assert formatted.model_dump()["citations"] == PAYLOAD["citations"]


def test_undocumented_parallel_fields_stay_off_the_openai_response() -> None:
    response: Final = ParallelAIDecisionsConfig().parse_response({**PAYLOAD, "internal_trace_id": "trace"}, REQUEST)
    formatted: Final = ir_to_openai_response(
        response, REQUEST, "parallel_ai/decision", ParallelAIDecisionsConfig.openai_evidence_fields
    )
    assert "internal_trace_id" not in formatted.model_dump()


@pytest.mark.parametrize("tokens", [0, 100, 32000])
@pytest.mark.parametrize("call_type", ["decisions", "adecisions", "systemone", "asystemone"])
def test_request_pricing_is_independent_of_token_count(tokens: int, call_type: str) -> None:
    rate: Final = litellm.model_cost["parallel_ai/decision"]["input_cost_per_query"]
    cost: Final = litellm.cost_per_token(
        model="parallel_ai/decision",
        prompt_tokens=tokens,
        completion_tokens=0,
        call_type=call_type,
    )
    assert cost == (rate, 0.0)


@pytest.mark.asyncio
async def test_deployment_query_price_replaces_the_catalog_price(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post("https://api.parallel.ai/v1/decisions").respond(json=PAYLOAD)
    price: Final = 25 * litellm.model_cost["parallel_ai/decision"]["input_cost_per_query"]
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "webjev",
                "litellm_params": {
                    "model": "parallel_ai/decision",
                    "api_key": "test-key",
                    "input_cost_per_query": price,
                },
            }
        ]
    )
    fields: Final = {
        "model": "webjev",
        "input": "Example",
        "questions": [{"type": "predicate", "name": "available", "instructions": "Available?"}],
        "litellm_call_id": "query-price-call",
    }
    logging_obj, data = function_setup("adecisions", Rules(), datetime.now(), **fields)
    response: Final = await router.adecisions(**data, litellm_logging_obj=logging_obj)
    assert response._hidden_params["response_cost"] == pytest.approx(price)
    await GLOBAL_LOGGING_WORKER.clear_queue()


@pytest.mark.parametrize(
    "payload", [{"answers": {}}, {"mode": "fast", "answers": {"available": {"type": "noul", "noul": "invalid"}}}]
)
def test_malformed_provider_responses_raise_a_provider_error(payload: Mapping[str, object]) -> None:
    with pytest.raises(DecisionsResponseValidationError):
        ParallelAIDecisionsConfig().transform_decisions_response(
            "decision", "parallel_ai", httpx.Response(200, json=payload), REQUEST
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("surface", ["decisions", "systemone"])
@pytest.mark.parametrize("key_env", ["PARALLEL_AI_API_KEY", "PARALLEL_API_KEY"])
async def test_untrusted_base_cannot_receive_environment_key(
    use_async: bool, surface: str, key_env: str, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)
    monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.setenv(key_env, "server-secret")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://attacker.example/v1/decisions").respond(json=PAYLOAD)
    connection: Final = {"model": "parallel_ai/decision", "api_base": "https://attacker.example", "num_retries": 0}
    fields: Final = (
        {"state": "Example", "questions": QUESTIONS}
        if surface == "systemone"
        else {
            "input": "Example",
            "questions": [{"type": "predicate", "name": "available", "instructions": "Available?"}],
        }
    )
    sync_call: Final = litellm.systemone if surface == "systemone" else litellm.decisions
    async_call: Final = litellm.asystemone if surface == "systemone" else litellm.adecisions

    async def call() -> object:
        return await async_call(**connection, **fields) if use_async else sync_call(**connection, **fields)

    with pytest.raises(litellm.BadRequestError, match="explicit api_key"):
        await call()
    assert not route.called
    await GLOBAL_LOGGING_WORKER.clear_queue()


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.parallel.ai.attacker.example",
        "https://api.parallel.ai@attacker.example",
        "http://api.parallel.ai",
        "https://api.parallel.ai:8443",
        "https://attacker.example/api.parallel.ai",
        "https://api.parallel.ai/other-tenant",
        "https://api.parallel.ai/other-tenant/v1/search",
        "https://attacker.example/v1/search",
        "https://attacker.example/v1/decisions",
    ],
)
def test_environment_key_rejects_untrusted_destination_variants(api_base: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)
    monkeypatch.setenv("PARALLEL_AI_API_KEY", "server-secret")
    with pytest.raises(ValueError, match="explicit api_key"):
        ParallelAIDecisionsConfig().resolve_connection(api_base, None)


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "https://api.parallel.ai",
        "https://API.PARALLEL.AI:443/v1/",
        "https://api.parallel.ai/v1/decisions",
        "https://api.parallel.ai/v1/search",
    ],
)
def test_default_destination_retains_environment_key(api_base: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)
    monkeypatch.setenv("PARALLEL_AI_API_KEY", "server-secret")
    resolved_base, key = ParallelAIDecisionsConfig().resolve_connection(api_base, None)
    assert resolved_base == (api_base or "https://api.parallel.ai")
    assert key == "server-secret"


@pytest.mark.parametrize(
    "api_base", [None, "https://gateway.example/parallel/v1/", "https://gateway.example/parallel/v1/search"]
)
def test_operator_base_retains_environment_key(api_base: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARALLEL_AI_API_BASE", "https://gateway.example/parallel")
    monkeypatch.setenv("PARALLEL_AI_API_KEY", "server-secret")
    resolved_base, key = ParallelAIDecisionsConfig().resolve_connection(api_base, None)
    assert resolved_base == (api_base or "https://gateway.example/parallel")
    assert key == "server-secret"


@pytest.mark.parametrize("api_key", [None, "caller-key"])
@pytest.mark.parametrize("api_base", ["https://api.parallel.ai\t", "https://api.parallel.ai\r\nHost: attacker.example"])
def test_malformed_destination_is_a_bad_request(api_base: str, api_key: str | None) -> None:
    with pytest.raises(litellm.BadRequestError):
        litellm.decisions(
            model="parallel_ai/decision",
            api_base=api_base,
            api_key=api_key,
            input="Example",
            questions=[{"type": "predicate", "name": "available", "instructions": "Available?"}],
            num_retries=0,
        )


def test_custom_destination_uses_explicit_key_instead_of_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARALLEL_AI_API_KEY", "server-secret")
    assert ParallelAIDecisionsConfig().resolve_connection("https://gateway.example", "caller-key") == (
        "https://gateway.example",
        "caller-key",
    )
