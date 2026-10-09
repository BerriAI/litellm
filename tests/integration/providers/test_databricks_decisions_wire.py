import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse
from pydantic import JsonValue

_ENDPOINT: Final = "databricks-openjev-qwen35-4b"
_API_KEY: Final = "synthetic-databricks-key"
_ENV_KEY: Final = "synthetic-databricks-env-key"
_ENV_TOKEN: Final = "synthetic-databricks-env-token"
_STATE: Final[dict[str, JsonValue]] = {"ticket": "The export job hangs at 99%", "component": "billing"}
_TIER_CRITERIA: Final[dict[str, JsonValue]] = {"SIMPLE": "a lookup", "MEDIUM": "some work", "COMPLEX": "deep work"}
_QUESTIONS: Final[dict[str, JsonValue]] = {"tier": {"type": "choice", "criteria": _TIER_CRITERIA}}
_OPENAI_INPUT: Final = "The export job hangs at 99% in billing"
_OPENAI_TIER_QUESTION: Final[dict[str, JsonValue]] = {
    "type": "choice",
    "name": "tier",
    "instructions": "Which tier?",
    "choices": [{"value": name, "description": description} for name, description in _TIER_CRITERIA.items()],
}
_ANSWERS: Final[dict[str, JsonValue]] = {
    "tier": {
        "type": "choice",
        "choice": "COMPLEX",
        "confidence": 0.91,
        "probabilities": {"SIMPLE": 0.03, "MEDIUM": 0.06, "COMPLEX": 0.91},
    }
}
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 367, "output_tokens": 3}
_ANSWER_BODY: Final[dict[str, JsonValue]] = {"model": _ENDPOINT, "answers": _ANSWERS, "usage": _USAGE}
_BARE_NAME_RULE: Final = (
    "must be the bare endpoint name (letters, digits, '-', '_' and '.', with no '/', '?', '#', spaces, or a "
    "leading '.'), e.g. databricks-openjev-qwen35-4b"
)
_MISSING_KEY: Final = "Missing API key for Decisions provider 'databricks'"
_NON_BARE_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("slash", "foo/bar"),
    ("space", "a b"),
    ("leading-dot", ".hidden"),
    ("question-mark", "a?b"),
    ("hash", "a#b"),
)
_DATABRICKS_ERRORS: Final[tuple[tuple[int, dict[str, JsonValue]], ...]] = (
    (400, {"error_code": "BAD_REQUEST", "message": "Invalid input: questions must not be empty"}),
    (404, {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": f"Endpoint with name '{_ENDPOINT}' does not exist."}),
)
_SPEND_QUERY: Final = (
    'SELECT spend, status, call_type, model_group, custom_llm_provider, api_base FROM "LiteLLM_SpendLogs" '
    "WHERE request_id = %s"
)
_OWNED_PROXY_BUDGET: Final = int(2 * graceful_stop_seconds() + 120)
_MEGABYTE_NAME: Final = "a" * 1_000_000 + "/"


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def _register(scenario: Scenario, body: dict[str, JsonValue], *, status: int = 200) -> ScenarioHandle:
    handle: Final = register_scenario(
        f"databricks-{uuid.uuid4().hex[:12]}", JsonResponse(content_type="application/json", body=body, status=status)
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _decide(gateway: Gateway, model: str, *, key: str | None = None) -> httpx.Response:
    return gateway.request("POST", "/v1/systemone", {"model": model, "state": _STATE, "questions": _QUESTIONS}, key=key)


def _observed(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        return tuple(map(object_value, upstream.get("/__observations").json()["requests"]))


def _calls_to(requests: Sequence[dict[str, JsonValue]], handle: ScenarioHandle) -> list[dict[str, JsonValue]]:
    return [request for request in requests if string_value(request["path"]).startswith(f"/{handle.scenario_id}/")]


def _upstream_calls(gateway: Gateway, handle: ScenarioHandle) -> list[dict[str, JsonValue]]:
    return _calls_to(_observed(gateway), handle)


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _wildcard_deployment(scenario: Scenario, **parameters: JsonValue) -> str:
    prefix: Final = f"databricks-{uuid.uuid4().hex[:8]}"
    created: Final = scenario.gateway.post(
        "/model/new",
        {"model_name": f"{prefix}/*", "litellm_params": {"model": "databricks/*", **parameters}, "model_info": {}},
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return prefix


def _router_config(classifier: Mapping[str, JsonValue], simple: str, complex_: str) -> dict[str, JsonValue]:
    return {
        "classifier_type": "oss_classifier",
        "opensource_classifier_config": dict(classifier),
        "tiers": {"SIMPLE": simple, "MEDIUM": simple, "COMPLEX": complex_, "REASONING": complex_},
    }


def _environment_classifier() -> dict[str, JsonValue]:
    return {"provider": "databricks", "model": _ENDPOINT, "timeout_ms": 20000, "circuit_breaker_enabled": False}


def _router(scenario: Scenario, classifier: Mapping[str, JsonValue]) -> str:
    simple: Final = scenario.model(model="openai/simple-tier")
    complex_: Final = scenario.model(model="openai/complex-tier")
    name: Final = f"router-{uuid.uuid4().hex[:10]}"
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_config": _router_config(classifier, simple, complex_),
            },
            "model_info": {},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _chat(gateway: Gateway, model: str, *, key: str | None = None) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"design a distributed cache {uuid.uuid4().hex}"}]},
        key=key,
    )


def _assert_classified_by_the_complex_tier(response: httpx.Response, router: str) -> None:
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"], response.text
    assert (
        response.headers["x-litellm-model-group"],
        response.headers["x-litellm-model-name"],
        response.headers["x-litellm-complexity-router-tier"],
        response.headers["x-litellm-complexity-router-cause"],
        response.headers["x-litellm-classifier-cost"],
    ) == (router, "openai/complex-tier", "COMPLEX", "jev_classifier", "0.0"), dict(response.headers)


def _assert_classifier_call(
    call: Mapping[str, JsonValue], *, path: str, authorization: str, prompt_marker: str
) -> None:
    assert call["path"] == path, call
    assert call["authorization"] == authorization, call
    body: Final = object_value(call["body"])
    assert body["model"] == _ENDPOINT, body
    assert prompt_marker in string_value(body["state"]), body
    assert sorted(object_value(object_value(body["questions"])["tier"])) == ["criteria", "instructions", "type"], body


def _assert_refused_before_any_upstream_call(gateway: Gateway, model: str, message: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _ANSWER_BODY)
        deployment: Final = scenario.model(model=model, api_base=handle.api_base(), api_key=_API_KEY)
        response: Final = _decide(gateway, deployment)
        assert response.status_code == 400, response.text
        assert message in response.text, response.text
        assert _upstream_calls(gateway, handle) == []
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], _number(row["spend"])) == ("failure", "adecisions", 0.0), row


@pytest.mark.parametrize("name", [name for _, name in _NON_BARE_NAMES], ids=[label for label, _ in _NON_BARE_NAMES])
def test_a_deployment_naming_a_non_bare_endpoint_is_refused_before_any_upstream_call(
    gateway: Gateway, name: str
) -> None:
    _assert_refused_before_any_upstream_call(
        gateway, f"databricks/{name}", f"Databricks serving endpoint name {name!r} {_BARE_NAME_RULE}"
    )


def test_a_deployment_naming_no_endpoint_is_refused_before_any_upstream_call(gateway: Gateway) -> None:
    _assert_refused_before_any_upstream_call(gateway, "databricks/", "A model name is required for the Decisions API")


def test_an_openai_format_request_at_v1_decisions_reaches_the_serving_endpoint_as_system_one(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _ANSWER_BODY)
        deployment: Final = scenario.model(
            model=f"databricks/{_ENDPOINT}", api_base=handle.api_base(), api_key=_API_KEY
        )
        response: Final = gateway.request(
            "POST",
            "/v1/decisions",
            {
                "model": deployment,
                "input": _OPENAI_INPUT,
                "questions": [_OPENAI_TIER_QUESTION],
                "safety_identifier": "end-user-1",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            "model": _ENDPOINT,
            "answers": [
                {
                    "type": "choice",
                    "name": "tier",
                    "choice": "COMPLEX",
                    "probabilities": [
                        {"value": "SIMPLE", "probability": 0.03},
                        {"value": "MEDIUM", "probability": 0.06},
                        {"value": "COMPLEX", "probability": 0.91},
                    ],
                    "confidence": 0.91,
                }
            ],
            "usage": {
                "input_tokens": 367,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 370,
            },
        }, response.text
        assert response.headers["x-litellm-model-name"] == f"databricks/{_ENDPOINT}", dict(response.headers)
        (call,) = _upstream_calls(gateway, handle)
        assert call["path"] == f"/{handle.scenario_id}/{_ENDPOINT}/invocations", call
        assert call["authorization"] == f"Bearer {_API_KEY}", call
        assert call["body"] == {
            "model": _ENDPOINT,
            "state": _OPENAI_INPUT,
            "questions": {"tier": {"type": "choice", "instructions": "Which tier?", "criteria": _TIER_CRITERIA}},
        }, call
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (
            row["status"],
            row["call_type"],
            row["custom_llm_provider"],
            row["model_group"],
            row["api_base"],
            _number(row["spend"]),
        ) == ("success", "adecisions", "databricks", deployment, f"{handle.api_base()}/{_ENDPOINT}/invocations", 0.0), (
            row
        )


def test_a_wildcard_deployment_sends_each_request_to_the_named_endpoint_and_bills_each_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _ANSWER_BODY)
        prefix: Final = _wildcard_deployment(scenario, api_base=handle.api_base(), api_key=_API_KEY)
        model: Final = f"{prefix}/{_ENDPOINT}"
        responses: Final = tuple(_decide(gateway, model) for _ in range(2))
        for response in responses:
            assert response.status_code == 200, response.text
            assert response.json() == _ANSWER_BODY, response.text
            assert response.headers["x-litellm-model-group"] == model, dict(response.headers)
            assert response.headers["x-litellm-model-name"] == f"databricks/{_ENDPOINT}", dict(response.headers)
        call_ids: Final = tuple(response.headers["x-litellm-call-id"] for response in responses)
        assert len(set(call_ids)) == 2, call_ids
        calls: Final = _upstream_calls(gateway, handle)
        assert len(calls) == 2, calls
        for call in calls:
            assert call["path"] == f"/{handle.scenario_id}/{_ENDPOINT}/invocations", call
            assert call["authorization"] == f"Bearer {_API_KEY}", call
            assert call["body"] == {"model": _ENDPOINT, "state": _STATE, "questions": _QUESTIONS}, call
        for call_id in call_ids:
            row: Final = _spend_row(call_id)
            assert (
                row["status"],
                row["call_type"],
                row["custom_llm_provider"],
                row["model_group"],
                row["api_base"],
                _number(row["spend"]),
            ) == ("success", "adecisions", "databricks", model, f"{handle.api_base()}/{_ENDPOINT}/invocations", 0.0), (
                row
            )


@pytest.mark.parametrize("model", (5, ["a"]), ids=("int", "list"))
def test_a_non_string_model_is_refused_at_the_gateway_without_an_upstream_call(
    gateway: Gateway, model: JsonValue
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _ANSWER_BODY)
        _wildcard_deployment(scenario, api_base=handle.api_base(), api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/systemone", {"model": model, "state": _STATE, "questions": _QUESTIONS}
        )
        assert 400 <= response.status_code < 500, response.text
        assert _upstream_calls(gateway, handle) == []


def test_a_five_kilobyte_endpoint_name_reaches_the_upstream_and_its_not_found_answer_is_kept(gateway: Gateway) -> None:
    name: Final = "n" * 5000
    with gateway.scenario() as scenario:
        handle: Final = _register(
            scenario,
            {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": f"Endpoint with name '{name}' does not exist."},
            status=404,
        )
        prefix: Final = _wildcard_deployment(scenario, api_base=handle.api_base(), api_key=_API_KEY)
        response: Final = _decide(gateway, f"{prefix}/{name}")
        assert response.status_code == 404, response.text[:500]
        assert "RESOURCE_DOES_NOT_EXIST" in response.text, response.text[:500]
        (call,) = _upstream_calls(gateway, handle)
        assert call["path"] == f"/{handle.scenario_id}/{name}/invocations", call["path"][:200]
        assert call["authorization"] == f"Bearer {_API_KEY}", call["authorization"]
        assert call["body"] == {"model": name, "state": _STATE, "questions": _QUESTIONS}
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], row["model_group"], _number(row["spend"])) == (
            "failure",
            "adecisions",
            f"{prefix}/{name}",
            0.0,
        ), row


@pytest.mark.parametrize("status,body", _DATABRICKS_ERRORS, ids=("400", "404"))
def test_a_databricks_error_keeps_its_status_and_message_and_bills_nothing(
    gateway: Gateway, status: int, body: dict[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, body, status=status)
        model: Final = scenario.model(model=f"databricks/{_ENDPOINT}", api_base=handle.api_base(), api_key=_API_KEY)
        response: Final = _decide(gateway, model)
        assert response.status_code == status, response.text
        assert string_value(body["message"]) in response.text, response.text
        (call,) = _upstream_calls(gateway, handle)
        assert call["path"] == f"/{handle.scenario_id}/{_ENDPOINT}/invocations", call
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], row["model_group"], _number(row["spend"])) == (
            "failure",
            "adecisions",
            model,
            0.0,
        ), row


def test_a_deployment_without_a_key_and_no_environment_key_is_refused_before_any_upstream_call(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _ANSWER_BODY)
        model: Final = scenario.model(model=f"databricks/{_ENDPOINT}", api_base=handle.api_base(), api_key=None)
        response: Final = _decide(gateway, model)
        assert response.status_code == 401, response.text
        assert _MISSING_KEY in response.text, response.text
        assert _upstream_calls(gateway, handle) == []


@pytest.mark.timeout(_OWNED_PROXY_BUDGET)
def test_the_environment_key_and_base_serve_a_deployment_a_classifier_and_a_team_member_that_leave_them_unset(
    gateway: Gateway, tmp_path: Path
) -> None:
    with gateway.scenario() as rig_scenario:
        handle: Final = _register(rig_scenario, _ANSWER_BODY)
        environment_base: Final = f"{handle.api_base()}/serving-endpoints"
        environment_path: Final = f"/{handle.scenario_id}/serving-endpoints/{_ENDPOINT}/invocations"
        environment: Final = {
            "DATABRICKS_API_KEY": _ENV_KEY,
            "DATABRICKS_TOKEN": _ENV_TOKEN,
            "DATABRICKS_API_BASE": environment_base,
        }
        with owned_proxy_process(gateway, tmp_path, environment) as owned, owned.gateway.scenario() as scenario:
            candidate: Final = owned.gateway
            with_base: Final = scenario.model(model=f"databricks/{_ENDPOINT}", api_base=handle.api_base(), api_key=None)
            bare: Final = scenario.model(model=f"databricks/{_ENDPOINT}", api_base=None, api_key=None)
            decided: Final = tuple(_decide(candidate, model) for model in (with_base, bare))
            for response in decided:
                assert response.status_code == 200, response.text
                assert response.json() == _ANSWER_BODY, response.text
            assert [(call["path"], call["authorization"]) for call in _upstream_calls(candidate, handle)] == [
                (f"/{handle.scenario_id}/{_ENDPOINT}/invocations", f"Bearer {_ENV_KEY}"),
                (environment_path, f"Bearer {_ENV_KEY}"),
            ]
            for response, model in zip(decided, (with_base, bare)):
                row: Final = _spend_row(response.headers["x-litellm-call-id"])
                assert (row["status"], row["model_group"], _number(row["spend"])) == ("success", model, 0.0), row
            assert (
                _spend_row(decided[1].headers["x-litellm-call-id"])["api_base"]
                == f"{environment_base}/{_ENDPOINT}/invocations"
            )

            router: Final = _router(scenario, _environment_classifier())
            routed: Final = _chat(candidate, router)
            _assert_classified_by_the_complex_tier(routed, router)
            (judge_call,) = _upstream_calls(candidate, handle)
            _assert_classifier_call(
                judge_call,
                path=environment_path,
                authorization=f"Bearer {_ENV_KEY}",
                prompt_marker="design a distributed cache",
            )

            team: Final = scenario.team(team_member_permissions=["/auto_router/manage"])
            member: Final = scenario.member(team)
            member_key: Final = scenario.key(user_id=member, team_id=team)
            simple: Final = scenario.model(model="openai/simple-tier")
            complex_: Final = scenario.model(model="openai/complex-tier")
            team_router: Final = f"team-router-{uuid.uuid4().hex[:8]}"
            created: Final = candidate.request(
                "POST",
                "/model/new",
                {
                    "model_name": team_router,
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "complexity_router_config": _router_config(_environment_classifier(), simple, complex_),
                    },
                    "model_info": {"team_id": team},
                },
                key=member_key,
            )
            assert created.status_code == 200, created.text
            team_router_id: Final = string_value(object_value(object_value(created.json())["model_info"])["id"])
            scenario.cleanups.callback(scenario.delete_model, team_router_id)
            stored: Final = candidate.request("GET", "/model/info", params={"litellm_model_id": team_router_id})
            assert stored.status_code == 200, stored.text
            (entry,) = stored.json()["data"]
            stored_classifier: Final = object_value(
                object_value(object_value(object_value(entry)["litellm_params"])["complexity_router_config"])[
                    "opensource_classifier_config"
                ]
            )
            assert (stored_classifier["provider"], stored_classifier["model"]) == ("databricks", _ENDPOINT), entry
            assert "api_key" not in stored_classifier and "api_base" not in stored_classifier, entry
            member_chat: Final = _chat(candidate, team_router, key=member_key)
            _assert_classified_by_the_complex_tier(member_chat, team_router)
            (member_judge_call,) = _upstream_calls(candidate, handle)
            _assert_classifier_call(
                member_judge_call,
                path=environment_path,
                authorization=f"Bearer {_ENV_KEY}",
                prompt_marker="design a distributed cache",
            )


@pytest.mark.timeout(_OWNED_PROXY_BUDGET)
def test_the_environment_token_serves_a_deployment_and_a_classifier_when_no_environment_key_is_set(
    gateway: Gateway, tmp_path: Path
) -> None:
    with gateway.scenario() as rig_scenario:
        handle: Final = _register(rig_scenario, _ANSWER_BODY)
        environment_path: Final = f"/{handle.scenario_id}/serving-endpoints/{_ENDPOINT}/invocations"
        environment: Final = {
            "DATABRICKS_TOKEN": _ENV_TOKEN,
            "DATABRICKS_API_BASE": f"{handle.api_base()}/serving-endpoints",
        }
        with (
            owned_proxy_process(gateway, tmp_path, environment, remove_environment=("DATABRICKS_API_KEY",)) as owned,
            owned.gateway.scenario() as scenario,
        ):
            candidate: Final = owned.gateway
            bare: Final = scenario.model(model=f"databricks/{_ENDPOINT}", api_base=None, api_key=None)
            decided: Final = _decide(candidate, bare)
            assert decided.status_code == 200, decided.text
            assert decided.json() == _ANSWER_BODY, decided.text
            (decision_call,) = _upstream_calls(candidate, handle)
            assert (decision_call["path"], decision_call["authorization"]) == (environment_path, f"Bearer {_ENV_TOKEN}")
            assert _spend_row(decided.headers["x-litellm-call-id"])["status"] == "success"
            router: Final = _router(scenario, _environment_classifier())
            routed: Final = _chat(candidate, router)
            _assert_classified_by_the_complex_tier(routed, router)
            (judge_call,) = _upstream_calls(candidate, handle)
            _assert_classifier_call(
                judge_call,
                path=environment_path,
                authorization=f"Bearer {_ENV_TOKEN}",
                prompt_marker="design a distributed cache",
            )


def test_a_megabyte_endpoint_name_is_refused_in_linear_time_while_liveliness_stays_fast(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _register(scenario, _ANSWER_BODY)
        prefix: Final = _wildcard_deployment(scenario, api_base=handle.api_base(), api_key=_API_KEY)
        model: Final = f"{prefix}/{_MEGABYTE_NAME}"
        anonymous: Final = gateway.client.post(
            "/v1/systemone", json={"model": model, "state": _STATE, "questions": _QUESTIONS}
        )
        assert anonymous.status_code == 401, anonymous.text[:300]
        liveliness: Final[list[tuple[int, float]]] = []
        stop: Final = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                started: Final = time.perf_counter()
                probe: Final = gateway.client.get("/health/liveliness")
                liveliness.append((probe.status_code, time.perf_counter() - started))

        poller: Final = threading.Thread(target=poll)
        poller.start()
        started: Final = time.perf_counter()
        response: Final = _decide(gateway, model)
        elapsed: Final = time.perf_counter() - started
        stop.set()
        poller.join()
        assert elapsed < 3, elapsed
        assert liveliness and all(status == 200 for status, _ in liveliness), liveliness
        assert max(seconds for _, seconds in liveliness) < 1, liveliness
        assert response.status_code == 400, response.text[:300]
        assert _BARE_NAME_RULE in response.text, response.text[-400:]
        assert _upstream_calls(gateway, handle) == []
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], _number(row["spend"])) == ("failure", "adecisions", 0.0), row
