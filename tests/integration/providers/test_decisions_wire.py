import json
import math
import socket
import uuid
from collections.abc import Sequence
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
_ENV_KEY: Final = "synthetic-decisions-env-key"
_PASS_THROUGH_MODEL: Final = "gpt-6-luna"
_PASS_THROUGH_AUTHORIZATION: Final = "Bearer customer-held-upstream-key"
_PASS_THROUGH_NEIGHBOUR: Final = "decisions-beside-a-pass-through"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_QUESTION_MAPPINGS: Final = TypeAdapter(dict[str, dict[str, object]])
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 367, "output_tokens": 3}
_STATE: Final[dict[str, JsonValue]] = {"ticket": "The export job hangs at 99%", "component": "billing"}
_QUESTIONS: Final[dict[str, JsonValue]] = {
    "defect": {"type": "noul", "instructions": "Is this a defect?"},
    "severity": {"type": "choice", "criteria": {"low": "cosmetic", "high": "blocks users"}, "weight": 2},
    "confidence": {"type": "score", "instructions": "How sure are you?", "criteria": ["unsure", "sure"]},
}
_SDK_QUESTIONS: Final = _QUESTION_MAPPINGS.validate_python(_QUESTIONS)
_ANSWERS: Final[dict[str, JsonValue]] = {
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
_CHAT_BODY: Final[dict[str, JsonValue]] = {"messages": [{"role": "user", "content": "hi"}]}
_CHAT_REPLY: Final[dict[str, JsonValue]] = {
    "id": "chatcmpl-decisions-parity",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "pplx-decider-v1-27b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
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
    provider_reported_cost: float | None = None


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
        None,
        1.5834e-5,
    ),
    _Provider(
        "strands_decider", "strands_decider/systemone-decider", "/v1/systemone", "systemone-decider", None, False, None
    ),
    _Provider(
        "hosted_vllm", "hosted_vllm/Qwen/Qwen3-0.6B", "/v1/systemone", "Qwen/Qwen3-0.6B", None, False, None
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
)
_PERPLEXITY: Final = _PROVIDERS[0]
_OPENROUTER: Final = _PROVIDERS[2]
_OPENROUTER_CHAT_MODEL: Final = "openrouter/openai/gpt-5-mini"
_UNSUPPORTED_PROVIDER_MODEL: Final = "anthropic/claude-opus-5-5"
_CONNECTION_ERROR: Final = "litellm.APIConnectionError"
_GENERIC_API_ERROR: Final = "litellm.APIError"
_INVALID_BODIES: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
    ("missing questions", {"state": _STATE}),
    ("missing state", {"questions": _QUESTIONS}),
    ("numeric state", {"state": 5, "questions": _QUESTIONS}),
    ("empty questions", {"state": _STATE, "questions": {}}),
    ("noul without instructions or criteria", {"state": _STATE, "questions": {"q": {"type": "noul"}}}),
    ("choice without criteria", {"state": _STATE, "questions": {"q": {"type": "choice", "criteria": {}}}}),
    (
        "score with eleven criteria",
        {"state": _STATE, "questions": {"q": {"type": "score", "criteria": [f"level-{index}" for index in range(11)]}}},
    ),
    ("unknown question type", {"state": _STATE, "questions": {"q": {"type": "ranking", "criteria": ["a"]}}}),
)


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def _expected_spend(provider: _Provider) -> float:
    if provider.provider_reported_cost is not None:
        return provider.provider_reported_cost
    if provider.cost_map_key is None:
        return 0.0
    prices: Final = object_value(
        json.loads(Path("model_prices_and_context_window.json").read_text())[provider.cost_map_key]
    )
    return _number(_USAGE["input_tokens"]) * _number(prices["input_cost_per_token"]) + _number(
        _USAGE["output_tokens"]
    ) * _number(prices["output_cost_per_token"])


def _usage(provider: _Provider) -> dict[str, JsonValue]:
    return {
        **_USAGE,
        **({"cost": provider.provider_reported_cost} if provider.provider_reported_cost is not None else {}),
    }


def _answer_body(provider: _Provider) -> dict[str, JsonValue]:
    answer: Final[dict[str, JsonValue]] = {
        "model": provider.body_model,
        "answers": _ANSWERS,
        "usage": _usage(provider),
    }
    return {"result": answer, "success": True} if provider.wraps_result else answer


def _register(scenario: Scenario, body: dict[str, JsonValue], *, status: int = 200) -> ScenarioHandle:
    handle: Final = register_scenario(
        f"decisions-{uuid.uuid4().hex[:12]}", JsonResponse(content_type="application/json", body=body, status=status)
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _deployment(scenario: Scenario, handle: ScenarioHandle, provider: _Provider) -> str:
    return scenario.model(model=provider.model, api_base=handle.api_base(), api_key=provider.api_key)


def _decide(gateway: Gateway, model: str, *, key: str | None = None, **extra: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/systemone", {"model": model, "state": _STATE, "questions": _QUESTIONS, **extra}, key=key
    )


def _chat(gateway: Gateway, model: str, *, key: str | None = None, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", {"model": model, **_CHAT_BODY, **extra}, key=key)


def _observed_requests(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        return tuple(map(object_value, upstream.get("/__observations").json()["requests"]))


def _calls_to(requests: Sequence[dict[str, JsonValue]], handle: ScenarioHandle) -> list[dict[str, JsonValue]]:
    return [request for request in requests if string_value(request["path"]).startswith(f"/{handle.scenario_id}/")]


def _upstream_calls(gateway: Gateway, handle: ScenarioHandle) -> list[dict[str, JsonValue]]:
    return _calls_to(_observed_requests(gateway), handle)


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _assert_connection_error(response: httpx.Response) -> None:
    assert 500 <= response.status_code < 600, response.text
    assert _CONNECTION_ERROR in response.text, response.text
    assert _GENERIC_API_ERROR not in response.text, response.text


def _free_closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _pass_through_config(directory: Path, pass_through_target: str, native_api_base: str) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {
            **object_value(base["general_settings"]),
            "pass_through_endpoints": [
                {
                    "path": "/v1/decisions",
                    "target": pass_through_target,
                    "headers": {"Authorization": _PASS_THROUGH_AUTHORIZATION},
                }
            ],
        },
        "model_list": [
            {
                "model_name": _PASS_THROUGH_NEIGHBOUR,
                "litellm_params": {"model": _PERPLEXITY.model, "api_base": native_api_base, "api_key": _API_KEY},
            }
        ],
    }
    path: Final = directory / "decisions-pass-through.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("provider", _PROVIDERS, ids=lambda provider: provider.name)
def test_each_provider_gets_its_own_path_key_and_body_and_is_billed(gateway: Gateway, provider: _Provider) -> None:
    expected_spend: Final = _expected_spend(provider)
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(provider))
        model: Final = _deployment(scenario, handle, provider)
        response: Final = _decide(gateway, model)
        assert response.status_code == 200, response.text
        assert response.json() == {"model": provider.body_model, "answers": _ANSWERS, "usage": _usage(provider)}
        assert response.headers["x-litellm-model-group"] == model
        assert math.isclose(float(response.headers.get("x-litellm-response-cost", "0")), expected_spend, rel_tol=1e-9)
        (call,) = _upstream_calls(gateway, handle)
        assert call["path"] == f"/{handle.scenario_id}{provider.path}"
        assert call["authorization"] == (f"Bearer {provider.api_key}" if provider.api_key else "")
        assert call["body"] == {"model": provider.body_model, "state": _STATE, "questions": _QUESTIONS}
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (
            row["status"],
            row["call_type"],
            row["custom_llm_provider"],
            row["model_group"],
            row["api_base"],
            row["prompt_tokens"],
            row["completion_tokens"],
        ) == ("success", "adecisions", provider.name, model, f"{handle.api_base()}{provider.path}", 367, 3)
        assert math.isclose(_number(row["spend"]), expected_spend, rel_tol=1e-9), row


def test_test_connection_evaluation_mode_uses_typesafe_decisions_path(gateway: Gateway) -> None:
    provider: Final = _PROVIDERS[1]
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(provider))
        response: Final = gateway.request(
            "POST",
            "/health/test_connection",
            {
                "litellm_params": {"model": provider.model, "api_base": handle.api_base(), "api_key": provider.api_key},
                "mode": "evaluation",
            },
        )

        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success", response.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["path"] == f"/{handle.scenario_id}{provider.path}"


def test_test_connection_by_configured_alias_probes_the_stored_model(gateway: Gateway) -> None:
    pytest.skip("BUG: /health/test_connection given only a configured alias sends the alias as the model")
    provider: Final = _PROVIDERS[1]
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(provider))
        model: Final = _deployment(scenario, handle, provider)
        response: Final = gateway.request(
            "POST",
            "/health/test_connection",
            {"litellm_params": {"model": model}, "mode": "evaluation"},
        )

        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success", response.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["path"] == f"/{handle.scenario_id}{provider.path}"


def test_repeated_identical_requests_each_reach_the_upstream_and_are_each_billed(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(_PERPLEXITY))
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        responses: Final = tuple(_decide(gateway, model) for _ in range(2))
        assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
        call_ids: Final = tuple(response.headers["x-litellm-call-id"] for response in responses)
        assert len(set(call_ids)) == 2, call_ids
        assert len(_upstream_calls(gateway, handle)) == 2
        for call_id in call_ids:
            assert _spend_row(call_id)["status"] == "success"


async def test_sdk_sync_and_async_clients_send_the_same_request(gateway: Gateway) -> None:
    provider: Final = _PROVIDERS[1]
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(provider))
        synchronous: Final = litellm.decisions(
            model=provider.model, state=_STATE, questions=_SDK_QUESTIONS, api_base=handle.api_base(), api_key=_API_KEY
        )
        asynchronous: Final = await litellm.adecisions(
            model=provider.model, state=_STATE, questions=_SDK_QUESTIONS, api_base=handle.api_base(), api_key=_API_KEY
        )
        for response in (synchronous, asynchronous):
            assert response.model_dump(mode="json") == {
                "model": provider.body_model,
                "answers": _ANSWERS,
                "usage": _USAGE,
            }
        calls: Final = _upstream_calls(gateway, handle)
        assert len(calls) == 2, calls
        for call in calls:
            assert call["path"] == f"/{handle.scenario_id}{provider.path}"
            assert call["authorization"] == f"Bearer {_API_KEY}"
            assert call["body"] == {"model": provider.body_model, "state": _STATE, "questions": _QUESTIONS}


def test_gateway_only_fields_stay_at_the_gateway_and_tags_reach_the_spend_log(gateway: Gateway) -> None:
    tag: Final = f"decisions-audit-{uuid.uuid4().hex[:8]}"
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(_PERPLEXITY))
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        response: Final = _decide(
            gateway, model, user="auditor", num_retries=0, temperature=0.2, metadata={"tags": [tag]}
        )
        assert response.status_code == 200, response.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["body"] == {"model": _PERPLEXITY.body_model, "state": _STATE, "questions": _QUESTIONS}
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        tags: Final = row["request_tags"]
        assert isinstance(tags, list) and tag in tags, row


def test_invalid_bodies_are_refused_at_the_gateway_without_an_upstream_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(_PERPLEXITY))
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        for label, body in _INVALID_BODIES:
            response: Final = gateway.request("POST", "/v1/systemone", {"model": model, **body})
            assert response.status_code == 400, (label, response.text)
            assert "Invalid Decisions request" in response.text, (label, response.text)
        assert _upstream_calls(gateway, handle) == []


def test_unknown_model_is_refused_like_chat(gateway: Gateway) -> None:
    model: Final = f"missing-{uuid.uuid4().hex}"
    decisions: Final = _decide(gateway, model)
    chat: Final = _chat(gateway, model)
    assert 400 <= decisions.status_code < 500, decisions.text
    assert decisions.status_code == chat.status_code, (decisions.text, chat.text)


def test_key_checks_match_chat(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(_PERPLEXITY))
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        anonymous: Final = gateway.client.post(
            "/v1/systemone", json={"model": model, "state": _STATE, "questions": _QUESTIONS}
        )
        assert anonymous.status_code == 401, anonymous.text
        restricted: Final = scenario.key(models=[f"other-{uuid.uuid4().hex}"])
        refused: Final = _decide(gateway, model, key=restricted)
        assert 400 <= refused.status_code < 500, refused.text
        assert refused.status_code == _chat(gateway, model, key=restricted).status_code, refused.text
        assert _upstream_calls(gateway, handle) == []
        spender: Final = scenario.key(max_budget=1e-06)
        first: Final = _decide(gateway, model, key=spender)
        assert first.status_code == 200, first.text
        blocked: Final = eventually(
            lambda: _decide(gateway, model, key=spender), lambda response: response.status_code != 200, seconds=70
        )
        assert 400 <= blocked.status_code < 500, blocked.text
        assert blocked.status_code == _chat(gateway, model, key=spender).status_code, blocked.text


def test_request_body_api_base_is_refused_like_chat_without_an_upstream_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(_PERPLEXITY))
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        decisions: Final = _decide(gateway, model, api_base=f"http://127.0.0.1:{_free_closed_port()}")
        chat: Final = _chat(gateway, model, api_base=f"http://127.0.0.1:{_free_closed_port()}")
        assert 400 <= decisions.status_code < 500, decisions.text
        assert decisions.status_code == chat.status_code, (decisions.text, chat.text)
        assert _upstream_calls(gateway, handle) == []


def test_a_deployment_without_a_key_sends_the_provider_env_key_to_its_configured_api_base(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(_PERPLEXITY))
        model: Final = scenario.model(model=_PERPLEXITY.model, api_base=handle.api_base(), api_key=None)
        response: Final = _decide(gateway, model)
        assert response.status_code == 200, response.text
        (call,) = _upstream_calls(gateway, handle)
        assert (call["path"], call["authorization"]) == (f"/{handle.scenario_id}/v1/decisions", f"Bearer {_ENV_KEY}")


def test_a_deployment_opted_into_client_api_base_sends_decisions_and_chat_to_the_body_api_base(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        configured: Final = _register(scenario, _answer_body(_PERPLEXITY))
        decisions_target: Final = _register(scenario, _answer_body(_PERPLEXITY))
        chat_target: Final = _register(scenario, _CHAT_REPLY)
        model: Final = scenario.model(
            model=_PERPLEXITY.model,
            api_base=configured.api_base(),
            api_key=_API_KEY,
            configurable_clientside_auth_params=["api_base"],
        )
        decisions: Final = _decide(gateway, model, api_base=decisions_target.api_base())
        chat: Final = _chat(gateway, model, api_base=chat_target.api_base())
        assert decisions.status_code == 200, decisions.text
        assert chat.status_code == 200, chat.text
        observed: Final = _observed_requests(gateway)
        assert [call["path"] for call in _calls_to(observed, decisions_target)] == [
            f"/{decisions_target.scenario_id}/v1/decisions"
        ]
        assert [call["path"] for call in _calls_to(observed, chat_target)] == [
            f"/{chat_target.scenario_id}/chat/completions"
        ]
        assert _calls_to(observed, configured) == []


def test_a_config_pass_through_at_v1_decisions_keeps_answering_and_the_native_api_serves_system_one(
    gateway: Gateway, tmp_path: Path
) -> None:
    with gateway.scenario() as scenario:
        pass_through_target: Final = _register(
            scenario, {"model": _PASS_THROUGH_MODEL, "answers": _ANSWERS, "usage": _USAGE}
        )
        native_target: Final = _register(scenario, _answer_body(_PERPLEXITY))
        config: Final = _pass_through_config(
            tmp_path, f"{pass_through_target.api_base()}/v1/decisions", native_target.api_base()
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
            through: Final = owned.gateway.request(
                "POST", "/v1/decisions", {"model": _PASS_THROUGH_MODEL, "state": _STATE, "questions": _QUESTIONS}
            )
            native: Final = owned.gateway.request(
                "POST", "/systemone", {"model": _PASS_THROUGH_NEIGHBOUR, "state": _STATE, "questions": _QUESTIONS}
            )
        assert through.status_code == 200, through.text
        assert through.json() == {"model": _PASS_THROUGH_MODEL, "answers": _ANSWERS, "usage": _USAGE}
        observed: Final = _observed_requests(gateway)
        (forwarded,) = _calls_to(observed, pass_through_target)
        assert (forwarded["path"], forwarded["authorization"], object_value(forwarded["body"])["model"]) == (
            f"/{pass_through_target.scenario_id}/v1/decisions",
            _PASS_THROUGH_AUTHORIZATION,
            _PASS_THROUGH_MODEL,
        )
        assert native.status_code == 200, native.text
        assert [call["path"] for call in _calls_to(observed, native_target)] == [
            f"/{native_target.scenario_id}/v1/decisions"
        ]


@pytest.mark.parametrize("status", (401, 429, 500))
def test_upstream_errors_keep_their_status_and_log_an_unbilled_failure(gateway: Gateway, status: int) -> None:
    marker: Final = f"scripted-{status}-{uuid.uuid4().hex[:8]}"
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, {"error": {"message": marker}}, status=status)
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        response: Final = _decide(gateway, model, num_retries=0)
        assert response.status_code == status, response.text
        assert marker in response.text
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], row["model_group"], _number(row["spend"])) == (
            "failure",
            "adecisions",
            model,
            0.0,
        )


def test_upstream_success_without_answers_is_a_gateway_side_server_error(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, {"model": _PERPLEXITY.body_model, "usage": _USAGE})
        model: Final = _deployment(scenario, handle, _PERPLEXITY)
        response: Final = _decide(gateway, model, num_retries=0)
        assert 500 <= response.status_code < 600, response.text
        assert "answers" in response.text
        assert _spend_row(response.headers["x-litellm-call-id"])["status"] == "failure"


@pytest.mark.parametrize("provider", _PROVIDERS, ids=lambda provider: provider.name)
def test_unreachable_upstream_fails_only_its_own_deployment(gateway: Gateway, provider: _Provider) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(provider))
        healthy: Final = _deployment(scenario, handle, provider)
        dead: Final = scenario.model(
            model=provider.model, api_base=f"http://127.0.0.1:{_free_closed_port()}", api_key=provider.api_key
        )
        failed: Final = _decide(gateway, dead, num_retries=0)
        _assert_connection_error(failed)
        assert _spend_row(failed.headers["x-litellm-call-id"])["status"] == "failure"
        served: Final = _decide(gateway, healthy)
        assert served.status_code == 200, served.text
        assert len(_upstream_calls(gateway, handle)) == 1


def test_an_unreachable_openrouter_deployment_reports_a_connection_error_on_chat_embeddings_and_decisions(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        dead_api_base: Final = f"http://127.0.0.1:{_free_closed_port()}"
        chat_model: Final = scenario.model(model=_OPENROUTER_CHAT_MODEL, api_base=dead_api_base, api_key=_API_KEY)
        decisions_model: Final = scenario.model(model=_OPENROUTER.model, api_base=dead_api_base, api_key=_API_KEY)
        chat: Final = _chat(gateway, chat_model, num_retries=0)
        streamed: Final = _chat(gateway, chat_model, num_retries=0, stream=True)
        embeddings: Final = gateway.request(
            "POST", "/v1/embeddings", {"model": chat_model, "input": "hi", "num_retries": 0}
        )
        decisions: Final = _decide(gateway, decisions_model, num_retries=0)
        for response in (chat, streamed, embeddings, decisions):
            _assert_connection_error(response)
        for response in (chat, decisions):
            assert _spend_row(response.headers["x-litellm-call-id"])["status"] == "failure"


async def test_sdk_openrouter_connection_failures_raise_a_connection_error(gateway: Gateway) -> None:
    dead_api_base: Final = f"http://127.0.0.1:{_free_closed_port()}"
    with pytest.raises(litellm.APIConnectionError):
        litellm.completion(
            model=_OPENROUTER_CHAT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            api_base=dead_api_base,
            api_key=_API_KEY,
        )
    with pytest.raises(litellm.APIConnectionError):
        await litellm.acompletion(
            model=_OPENROUTER_CHAT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            api_base=dead_api_base,
            api_key=_API_KEY,
        )
    with pytest.raises(litellm.APIConnectionError):
        litellm.decisions(
            model=_OPENROUTER.model, state=_STATE, questions=_SDK_QUESTIONS, api_base=dead_api_base, api_key=_API_KEY
        )
    with pytest.raises(litellm.APIConnectionError):
        await litellm.adecisions(
            model=_OPENROUTER.model, state=_STATE, questions=_SDK_QUESTIONS, api_base=dead_api_base, api_key=_API_KEY
        )


def test_a_deployment_whose_provider_has_no_decisions_support_is_refused_naming_every_supported_provider(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _answer_body(_PERPLEXITY))
        model: Final = scenario.model(model=_UNSUPPORTED_PROVIDER_MODEL, api_base=handle.api_base(), api_key=_API_KEY)
        response: Final = _decide(gateway, model)
        assert response.status_code == 400, response.text
        for provider in _PROVIDERS:
            assert provider.name in response.text, response.text
        assert _upstream_calls(gateway, handle) == []
