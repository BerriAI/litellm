import math
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse
from pydantic import JsonValue, TypeAdapter

import litellm

_API_KEY: Final = "synthetic-decisions-key"
_SAFETY_IDENTIFIER: Final = "end-user-7"
_DROPPING_DEPLOYMENT: Final = "decisions-under-litellm-settings-drop-params"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_INPUT: Final = "Ticket (billing): The export job hangs at 99%"
_FOLLOW_UP: Final = "Customer: still stuck after retrying"
_QUESTIONS: Final[list[JsonValue]] = [
    {"type": "predicate", "name": "defect", "instructions": "Is this a defect?"},
    {
        "type": "choice",
        "name": "severity",
        "instructions": "How severe is it?",
        "choices": [{"value": "low", "description": "cosmetic"}, {"value": "high", "description": "blocks users"}],
    },
    {
        "type": "score",
        "name": "confidence",
        "instructions": "How sure are you?",
        "levels": [{"label": "unsure"}, {"label": "sure"}],
    },
]
_SYSTEM_ONE_QUESTIONS: Final[dict[str, JsonValue]] = {
    "defect": {"type": "noul", "instructions": "Is this a defect?"},
    "severity": {
        "type": "choice",
        "instructions": "How severe is it?",
        "criteria": {"low": "cosmetic", "high": "blocks users"},
    },
    "confidence": {"type": "score", "instructions": "How sure are you?", "criteria": ["unsure", "sure"]},
}
_SYSTEM_ONE_ANSWERS: Final[dict[str, JsonValue]] = {
    "defect": {"type": "noul", "noul": 0.93},
    "severity": {"type": "choice", "choice": "high", "confidence": 0.8, "probabilities": {"low": 0.2, "high": 0.8}},
    "confidence": {
        "type": "score",
        "score": 1.0,
        "confidence": 0.7,
        "legend": {"0": "unsure", "1": "sure"},
        "probabilities": {"0": 0.3, "1": 0.7},
    },
}
_ANSWERS: Final[list[JsonValue]] = [
    {"type": "predicate", "name": "defect", "probability": 0.93},
    {
        "type": "choice",
        "name": "severity",
        "choice": "high",
        "probabilities": [{"value": "low", "probability": 0.2}, {"value": "high", "probability": 0.8}],
        "confidence": 0.8,
    },
    {
        "type": "score",
        "name": "confidence",
        "score": 1.0,
        "probabilities": [
            {"value": 0, "label": "unsure", "probability": 0.3},
            {"value": 1, "label": "sure", "probability": 0.7},
        ],
        "confidence": 0.7,
    },
]
_INPUT_TOKENS: Final = 367
_OUTPUT_TOKENS: Final = 3
_SYSTEM_ONE_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS}
_USAGE: Final[dict[str, JsonValue]] = {
    "input_tokens": _INPUT_TOKENS,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": _OUTPUT_TOKENS,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
}
_SDK_QUESTIONS: Final = TypeAdapter(list[dict[str, object]]).validate_python(_QUESTIONS)
_SPEND_QUERY: Final = (
    "SELECT spend, status, call_type, model_group, custom_llm_provider, api_base, prompt_tokens, completion_tokens, "
    'request_tags FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
)


@dataclass(frozen=True, slots=True)
class _Provider:
    name: str
    model: str
    path: str
    body_model: str
    api_key: str | None
    wraps_result: bool
    cost_map_key: str | None
    speaks_openai: bool = False

    def upstream_body(self) -> dict[str, JsonValue]:
        if self.speaks_openai:
            return {"model": self.body_model, "input": _INPUT, "questions": _QUESTIONS}
        return {"model": self.body_model, "state": _INPUT, "questions": _SYSTEM_ONE_QUESTIONS}

    def upstream_reply(self) -> dict[str, JsonValue]:
        if self.speaks_openai:
            return {"model": self.body_model, "answers": _ANSWERS, "usage": _USAGE}
        answer: Final[dict[str, JsonValue]] = {
            "model": self.body_model,
            "answers": _SYSTEM_ONE_ANSWERS,
            "usage": _SYSTEM_ONE_USAGE,
        }
        return {"result": answer, "success": True} if self.wraps_result else answer

    def litellm_response(self) -> dict[str, JsonValue]:
        return {"model": self.body_model, "answers": _ANSWERS, "usage": _USAGE}


_PROVIDERS: Final = (
    _Provider(
        "perplexity",
        "perplexity/pplx-decider-v1-27b",
        "/v1/decisions",
        "pplx-decider-v1-27b",
        _API_KEY,
        False,
        "perplexity/pplx-decider-v1-27b",
    ),
    _Provider("typesafe", "typesafe/jev-1.13.0", "/v1/systemone", "jev-1.13.0", _API_KEY, False, "typesafe/jev-1.13.0"),
    _Provider(
        "openrouter",
        "openrouter/typesafe/jev-1.13",
        "/alpha/decisions",
        "typesafe/jev-1.13",
        _API_KEY,
        False,
        "openrouter/typesafe/jev-1.13",
    ),
    _Provider(
        "strands_decider", "strands_decider/systemone-decider", "/v1/systemone", "systemone-decider", None, False, None
    ),
    _Provider(
        "cloudflare",
        "cloudflare/clef",
        "/ai/run/@cf/cloudflare/clef",
        "clef",
        _API_KEY,
        True,
        "cloudflare/@cf/cloudflare/clef",
    ),
    _Provider("hosted_vllm", "hosted_vllm/Qwen/Qwen3-0.6B", "/v1/systemone", "Qwen/Qwen3-0.6B", None, False, None),
    _Provider(
        "databricks",
        "databricks/databricks-openjev-qwen35-4b",
        "/databricks-openjev-qwen35-4b/invocations",
        "databricks-openjev-qwen35-4b",
        _API_KEY,
        False,
        None,
    ),
    _Provider(
        "azure_ai", "azure_ai/decision-1", "/providers/microsoft/v1/systemone", "decision-1", _API_KEY, False, None
    ),
    _Provider("openai", "openai/gpt-6-luna", "/v1/decisions", "gpt-6-luna", _API_KEY, False, "gpt-6-luna", True),
)
_SYSTEM_ONE_PROVIDERS: Final = tuple(provider for provider in _PROVIDERS if not provider.speaks_openai)
_PERPLEXITY: Final = _PROVIDERS[0]
_TYPESAFE: Final = _PROVIDERS[1]
_OPENAI: Final = next(provider for provider in _PROVIDERS if provider.speaks_openai)
_PREDICATE: Final[dict[str, JsonValue]] = {"type": "predicate", "name": "q", "instructions": "Is it?"}
_INVALID_BODIES: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
    ("missing questions", {"input": _INPUT}),
    ("missing input", {"questions": _QUESTIONS}),
    ("numeric input", {"input": 5, "questions": _QUESTIONS}),
    ("assistant message input", {"input": [{"role": "assistant", "content": "hi"}], "questions": _QUESTIONS}),
    ("empty questions", {"input": _INPUT, "questions": []}),
    ("questions as a map", {"input": _INPUT, "questions": {"q": _PREDICATE}}),
    ("predicate without instructions", {"input": _INPUT, "questions": [{"type": "predicate", "name": "q"}]}),
    (
        "choice with one choice",
        {"input": _INPUT, "questions": [{**_PREDICATE, "type": "choice", "choices": [{"value": "only"}]}]},
    ),
    (
        "score with one level",
        {"input": _INPUT, "questions": [{**_PREDICATE, "type": "score", "levels": [{"label": "only"}]}]},
    ),
    ("unknown question type", {"input": _INPUT, "questions": [{**_PREDICATE, "type": "ranking"}]}),
    ("numeric safety_identifier", {"input": _INPUT, "questions": _QUESTIONS, "safety_identifier": 7}),
    ("list safety_identifier", {"input": _INPUT, "questions": _QUESTIONS, "safety_identifier": [_SAFETY_IDENTIFIER]}),
)
_IMAGE_INPUT: Final[list[JsonValue]] = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": _INPUT},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"},
        ],
    }
]
_OPENAI_IMAGE_MESSAGES: Final[list[JsonValue]] = [
    {
        "role": "user",
        "type": "message",
        "content": [
            {"type": "input_text", "text": _INPUT},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"},
        ],
    }
]
_MESSAGE_LIST_INPUT: Final[list[JsonValue]] = [
    {"role": "user", "content": _INPUT},
    {"role": "user", "content": [{"type": "input_text", "text": _FOLLOW_UP}]},
]


def _response_cost(response: httpx.Response) -> float:
    return float(response.headers["x-litellm-response-cost"]) if "x-litellm-response-cost" in response.headers else 0.0


def _provider_id(provider: _Provider) -> str:
    return provider.name


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def _expected_spend(cost_map_key: str | None) -> float:
    if cost_map_key is None:
        return 0.0
    cost_map: Final = _JSON_OBJECT.validate_json(Path("model_prices_and_context_window.json").read_bytes())
    prices: Final = object_value(cost_map[cost_map_key])
    return _INPUT_TOKENS * _number(prices["input_cost_per_token"]) + _OUTPUT_TOKENS * _number(
        prices["output_cost_per_token"]
    )


def _register(scenario: Scenario, body: dict[str, JsonValue], *, status: int = 200) -> ScenarioHandle:
    handle: Final = register_scenario(
        f"decisions-{uuid.uuid4().hex[:12]}", JsonResponse(content_type="application/json", body=body, status=status)
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _deployment(scenario: Scenario, handle: ScenarioHandle, provider: _Provider, *, drop_params: bool = False) -> str:
    if drop_params:
        return scenario.model(
            model=provider.model, api_base=handle.api_base(), api_key=provider.api_key, drop_params=True
        )
    return scenario.model(model=provider.model, api_base=handle.api_base(), api_key=provider.api_key)


def _decide(gateway: Gateway, model: str, *, route: str = "/v1/decisions", **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", route, {"model": model, "input": _INPUT, "questions": _QUESTIONS, **extra})


def _decide_in_system_one_format(gateway: Gateway, model: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/systemone", {"model": model, "state": _INPUT, "questions": _SYSTEM_ONE_QUESTIONS, **extra}
    )


def _observed_requests(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        observations: Final = _JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert isinstance(observations, list), observations
        return tuple(map(object_value, observations))


def _calls_to(requests: Sequence[dict[str, JsonValue]], handle: ScenarioHandle) -> list[dict[str, JsonValue]]:
    return [request for request in requests if string_value(request["path"]).startswith(f"/{handle.scenario_id}/")]


def _upstream_calls(gateway: Gateway, handle: ScenarioHandle) -> list[dict[str, JsonValue]]:
    return _calls_to(_observed_requests(gateway), handle)


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _assert_refused_as_invalid(gateway: Gateway, model: str, label: str, body: dict[str, JsonValue]) -> None:
    response: Final = gateway.request("POST", "/v1/decisions", {"model": model, **body})
    assert response.status_code == 400, (label, response.text)
    assert "Invalid Decisions request" in response.text, (label, response.text)


def _assert_refused_for_safety_identifier(response: httpx.Response) -> None:
    assert response.status_code == 400, response.text
    assert "safety_identifier" in response.text and "drop_params" in response.text, response.text


def _drop_params_config(directory: Path, api_base: str) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "litellm_settings": {**object_value(base["litellm_settings"]), "drop_params": True},
        "model_list": [
            {
                "model_name": _DROPPING_DEPLOYMENT,
                "litellm_params": {"model": _PERPLEXITY.model, "api_base": api_base, "api_key": _API_KEY},
            }
        ],
    }
    path: Final = directory / "decisions-drop-params.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("provider", _PROVIDERS, ids=_provider_id)
def test_each_provider_gets_its_own_path_key_and_body_and_is_billed_from_the_cost_map(
    gateway: Gateway, provider: _Provider
) -> None:
    expected_spend: Final = _expected_spend(provider.cost_map_key)
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, provider.upstream_reply())
        model: Final = _deployment(scenario, handle, provider)
        response: Final = _decide(gateway, model)
        assert response.status_code == 200, response.text
        assert response.json() == provider.litellm_response()
        assert response.headers["x-litellm-model-group"] == model
        assert math.isclose(_response_cost(response), expected_spend, rel_tol=1e-9)
        (call,) = _upstream_calls(gateway, handle)
        assert call["path"] == f"/{handle.scenario_id}{provider.path}"
        assert call["authorization"] == (f"Bearer {provider.api_key}" if provider.api_key else "")
        assert call["body"] == provider.upstream_body()
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (
            row["status"],
            row["call_type"],
            row["custom_llm_provider"],
            row["model_group"],
            row["api_base"],
            row["prompt_tokens"],
            row["completion_tokens"],
        ) == (
            "success",
            "adecisions",
            provider.name,
            model,
            f"{handle.api_base()}{provider.path}",
            _INPUT_TOKENS,
            _OUTPUT_TOKENS,
        )
        assert math.isclose(_number(row["spend"]), expected_spend, rel_tol=1e-9), row


def test_the_unversioned_alias_serves_the_same_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        response: Final = _decide(gateway, model, route="/decisions")
        assert response.status_code == 200, response.text
        assert response.json() == _PERPLEXITY.litellm_response()
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == _PERPLEXITY.upstream_body()


def test_repeated_identical_requests_each_reach_the_upstream_and_are_each_billed(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        responses: Final = tuple(_decide(gateway, model) for _ in range(2))
        assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
        call_ids: Final = tuple(response.headers["x-litellm-call-id"] for response in responses)
        assert len(set(call_ids)) == 2, call_ids
        assert len(_upstream_calls(gateway, handle)) == 2
        for call_id in call_ids:
            assert _spend_row(call_id)["status"] == "success"


async def test_sdk_sync_and_async_clients_send_the_same_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _TYPESAFE.upstream_reply())
        synchronous: Final = litellm.decisions(
            model=_TYPESAFE.model, input=_INPUT, questions=_SDK_QUESTIONS, api_base=handle.api_base(), api_key=_API_KEY
        )
        asynchronous: Final = await litellm.adecisions(
            model=_TYPESAFE.model, input=_INPUT, questions=_SDK_QUESTIONS, api_base=handle.api_base(), api_key=_API_KEY
        )
        for response in (synchronous, asynchronous):
            assert response.model_dump(mode="json") == _TYPESAFE.litellm_response()
        calls: Final = _upstream_calls(gateway, handle)
        assert len(calls) == 2, calls
        for call in calls:
            assert call["path"] == f"/{handle.scenario_id}{_TYPESAFE.path}"
            assert call["authorization"] == f"Bearer {_API_KEY}"
            assert call["body"] == _TYPESAFE.upstream_body()


def test_gateway_only_fields_stay_at_the_gateway_and_tags_reach_the_spend_log(gateway: Gateway) -> None:
    tag: Final = f"decisions-openai-format-{uuid.uuid4().hex[:8]}"
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        response: Final = _decide(
            gateway, model, user="auditor", num_retries=0, temperature=0.2, metadata={"tags": [tag]}
        )
        assert response.status_code == 200, response.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == _PERPLEXITY.upstream_body()
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        tags: Final = row["request_tags"]
        assert isinstance(tags, list) and tag in tags, row


def test_invalid_bodies_are_refused_at_the_gateway_without_an_upstream_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        for label, body in _INVALID_BODIES:
            _assert_refused_as_invalid(gateway, model, label, body)
        assert _upstream_calls(gateway, handle) == []


@pytest.mark.parametrize("provider", _SYSTEM_ONE_PROVIDERS, ids=_provider_id)
def test_system_one_providers_refuse_images_at_the_gateway_without_an_upstream_call(
    gateway: Gateway, provider: _Provider
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, provider.upstream_reply())
        model: Final = _deployment(scenario, handle, provider)
        response: Final = _decide(gateway, model, input=_IMAGE_INPUT)
        assert response.status_code == 400, response.text
        assert "input_image" in response.text
        assert _upstream_calls(gateway, handle) == []


def test_openai_forwards_image_input_in_its_own_message_shape(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _OPENAI.upstream_reply())
        model: Final = _deployment(scenario, handle, _OPENAI)
        response: Final = _decide(gateway, model, input=_IMAGE_INPUT)
        assert response.status_code == 200, response.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == {"model": _OPENAI.body_model, "input": _OPENAI_IMAGE_MESSAGES, "questions": _QUESTIONS}


def test_a_message_list_input_reaches_a_system_one_provider_as_its_flattened_text(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        response: Final = _decide(gateway, model, input=_MESSAGE_LIST_INPUT)
        assert response.status_code == 200, response.text
        assert response.json() == _PERPLEXITY.litellm_response()
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == {
            "model": _PERPLEXITY.body_model,
            "state": f"{_INPUT}\n\n{_FOLLOW_UP}",
            "questions": _SYSTEM_ONE_QUESTIONS,
        }
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], row["prompt_tokens"], row["completion_tokens"]) == (
            "success",
            "adecisions",
            _INPUT_TOKENS,
            _OUTPUT_TOKENS,
        )


@pytest.mark.parametrize("provider", _SYSTEM_ONE_PROVIDERS, ids=_provider_id)
def test_safety_identifier_is_refused_by_system_one_providers_unless_the_deployment_drops_params(
    gateway: Gateway, provider: _Provider
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, provider.upstream_reply())
        strict: Final = _deployment(scenario, handle, provider)
        refused: Final = _decide(gateway, strict, safety_identifier=_SAFETY_IDENTIFIER)
        _assert_refused_for_safety_identifier(refused)
        assert _upstream_calls(gateway, handle) == []

        dropping: Final = _deployment(scenario, handle, provider, drop_params=True)
        accepted: Final = _decide(gateway, dropping, safety_identifier=_SAFETY_IDENTIFIER)
        assert accepted.status_code == 200, accepted.text
        assert accepted.json() == provider.litellm_response()
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == provider.upstream_body()
        row: Final = _spend_row(accepted.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], row["model_group"]) == ("success", "adecisions", dropping)


@pytest.mark.parametrize("value", ("", "x" * 5000), ids=("empty", "5kb"))
def test_every_string_safety_identifier_is_refused_or_dropped_like_the_usual_one(gateway: Gateway, value: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        strict: Final = _deployment(scenario, handle, _PERPLEXITY)
        _assert_refused_for_safety_identifier(_decide(gateway, strict, safety_identifier=value))
        dropping: Final = _deployment(scenario, handle, _PERPLEXITY, drop_params=True)
        accepted: Final = _decide(gateway, dropping, safety_identifier=value)
        assert accepted.status_code == 200, accepted.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == _PERPLEXITY.upstream_body()


def test_a_request_body_drop_params_drops_the_safety_identifier_like_chat(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        strict: Final = _deployment(scenario, handle, _PERPLEXITY)
        response: Final = _decide(gateway, strict, safety_identifier=_SAFETY_IDENTIFIER, drop_params=True)
        assert response.status_code == 200, response.text
        assert response.json() == _PERPLEXITY.litellm_response()
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == _PERPLEXITY.upstream_body()


def test_openai_keeps_the_safety_identifier_on_the_wire_without_drop_params(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _OPENAI.upstream_reply())
        model: Final = _deployment(scenario, handle, _OPENAI)
        response: Final = _decide(gateway, model, safety_identifier=_SAFETY_IDENTIFIER)
        assert response.status_code == 200, response.text
        assert response.json() == _OPENAI.litellm_response()
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == {**_OPENAI.upstream_body(), "safety_identifier": _SAFETY_IDENTIFIER}


def test_a_system_one_format_safety_identifier_is_refused_unless_the_deployment_drops_params(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        strict: Final = _deployment(scenario, handle, _PERPLEXITY)
        refused: Final = _decide_in_system_one_format(gateway, strict, safety_identifier=_SAFETY_IDENTIFIER)
        _assert_refused_for_safety_identifier(refused)
        assert _upstream_calls(gateway, handle) == []

        dropping: Final = _deployment(scenario, handle, _PERPLEXITY, drop_params=True)
        accepted: Final = _decide_in_system_one_format(gateway, dropping, safety_identifier=_SAFETY_IDENTIFIER)
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["answers"] == _SYSTEM_ONE_ANSWERS
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == _PERPLEXITY.upstream_body()


def test_openai_keeps_a_system_one_format_safety_identifier_on_the_wire(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _OPENAI.upstream_reply())
        model: Final = _deployment(scenario, handle, _OPENAI)
        response: Final = _decide_in_system_one_format(gateway, model, safety_identifier=_SAFETY_IDENTIFIER)
        assert response.status_code == 200, response.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == {**_OPENAI.upstream_body(), "safety_identifier": _SAFETY_IDENTIFIER}


@pytest.mark.parametrize("decide", (_decide, _decide_in_system_one_format), ids=("openai_format", "system_one_format"))
@pytest.mark.parametrize("value", (7, [_SAFETY_IDENTIFIER]), ids=("numeric", "list"))
def test_a_non_string_safety_identifier_is_refused_as_invalid_unless_the_deployment_drops_params(
    gateway: Gateway, value: JsonValue, decide: Callable[..., httpx.Response]
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _OPENAI.upstream_reply())
        strict: Final = _deployment(scenario, handle, _OPENAI)
        refused: Final = decide(gateway, strict, safety_identifier=value)
        assert refused.status_code == 400, refused.text
        assert "Invalid Decisions request" in refused.text, refused.text
        assert _upstream_calls(gateway, handle) == []

        dropping: Final = _deployment(scenario, handle, _OPENAI, drop_params=True)
        accepted: Final = decide(gateway, dropping, safety_identifier=value)
        assert accepted.status_code == 200, accepted.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == _OPENAI.upstream_body()


def test_litellm_settings_drop_params_drops_the_safety_identifier_for_a_strict_deployment(
    gateway: Gateway, tmp_path: Path
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _PERPLEXITY.upstream_reply())
        config: Final = _drop_params_config(tmp_path, handle.api_base())
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
            response: Final = _decide(owned.gateway, _DROPPING_DEPLOYMENT, safety_identifier=_SAFETY_IDENTIFIER)
        assert response.status_code == 200, response.text
        assert response.json() == _PERPLEXITY.litellm_response()
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == _PERPLEXITY.upstream_body()
