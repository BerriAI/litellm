import asyncio
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_MODEL: Final = "ai_decide"
_ROUTE: Final = "/api/2.0/ai-functions/ai-decide"
_SECOND_WORKSPACE: Final = "/second-workspace"
_API_KEY: Final = "synthetic-databricks-key"
_MODEL_RULE: Final = "Databricks decisions model must be 'ai_decide' (the ai_decide AI Function)"
_REJECTED_AT_LOAD: Final = "The router would drop this deployment at load time, so the write is rejected instead."
_MODEL_REQUIRED: Final = (
    "opensource_classifier_config.model is required for provider 'databricks': set it to 'ai_decide'"
)
_BASE_NEEDS_KEY: Final = (
    "opensource_classifier_config.api_base requires opensource_classifier_config.api_key: DATABRICKS_API_KEY or "
    "DATABRICKS_TOKEN is only sent to DATABRICKS_API_BASE"
)
_BLANK_KEY: Final = (
    "opensource_classifier_config.api_key must be non-empty; omit it to use the provider environment key"
)
_CLASSIFIER_ANSWER: Final[dict[str, JsonValue]] = {
    "response": {
        "answers": {
            "tier": {
                "type": "choice",
                "choice": "COMPLEX",
                "confidence": 0.91,
                "probabilities": {"SIMPLE": 0.03, "MEDIUM": 0.06, "COMPLEX": 0.91},
            }
        }
    },
    "metadata": {"version": "1.0"},
}
_ROWS_QUERY: Final = (
    "SELECT request_id, status, metadata->>'internal_call_origin' AS origin "
    'FROM "LiteLLM_SpendLogs" WHERE model_group = %s'
)


def _classifier(request: Request) -> Reply:
    return Reply(body=json.dumps(_CLASSIFIER_ANSWER).encode())


def _tier(text: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        return chat_reply(f"{text}-{uuid.uuid4().hex[:8]}", "tier-model", text, stream=bool(body.get("stream")))

    return answering_model_discovery(respond)


def _classifier_config(api_base: str, **overrides: JsonValue) -> dict[str, JsonValue]:
    return {
        "provider": "databricks",
        "model": _MODEL,
        "api_base": api_base,
        "api_key": _API_KEY,
        "timeout_ms": 20000,
        "circuit_breaker_enabled": False,
        **overrides,
    }


def _without(config: Mapping[str, JsonValue], field: str) -> dict[str, JsonValue]:
    return {name: value for name, value in config.items() if name != field}


def _router_config(classifier: Mapping[str, JsonValue], simple: str, complex_: str) -> dict[str, JsonValue]:
    return {
        "classifier_type": "oss_classifier",
        "opensource_classifier_config": dict(classifier),
        "tiers": {"SIMPLE": simple, "MEDIUM": simple, "COMPLEX": complex_, "REASONING": complex_},
    }


def _validate(gateway: Gateway, config: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    response: Final = gateway.request(
        "POST", "/auto_router/validate_complexity_router_config", {"complexity_router_config": dict(config)}
    )
    assert response.status_code == 200, response.text
    return object_value(response.json())


def _create_router(scenario: Scenario, config: Mapping[str, JsonValue]) -> tuple[str, str]:
    name: Final = f"router-{uuid.uuid4().hex[:10]}"
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": "auto_router/complexity_router", "complexity_router_config": dict(config)},
            "model_info": {},
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return name, identity


def _stored_classifier(gateway: Gateway, identity: str) -> dict[str, JsonValue]:
    stored: Final = gateway.request("GET", "/model/info", params={"litellm_model_id": identity})
    assert stored.status_code == 200, stored.text
    (entry,) = stored.json()["data"]
    router_config: Final = object_value(object_value(object_value(entry)["litellm_params"])["complexity_router_config"])
    return object_value(router_config["opensource_classifier_config"])


def _judge_targets(judge: Wire) -> tuple[str, ...]:
    return tuple(request.target for request in judge.drain() if request.method == "POST")


def _chat_body(model: str) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": f"design a distributed cache {uuid.uuid4().hex}"}]}


async def _burst(base_url: str, key: str, model: str, count: int) -> tuple[httpx.Response, ...]:
    async with httpx.AsyncClient(
        base_url=base_url, timeout=60, trust_env=False, headers={"Authorization": f"Bearer {key}"}
    ) as client:
        return tuple(
            await asyncio.gather(*(client.post("/v1/chat/completions", json=_chat_body(model)) for _ in range(count)))
        )


def _request_rows(rows: Sequence[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(row for row in rows if row["origin"] != "autorouter_classifier")


def _rows(router: str, *, requests: int) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        eventually(
            lambda: read_rows(_ROWS_QUERY, (router,)),
            lambda found: len(_request_rows(found)) >= requests,
            seconds=70,
        )
    )


def test_validate_accepts_a_databricks_ai_decide_classifier_with_or_without_the_serving_endpoints_suffix(
    gateway: Gateway,
) -> None:
    for api_base in (
        "https://dbc-example.cloud.databricks.com",
        "https://dbc-example.cloud.databricks.com/serving-endpoints",
    ):
        config: Final = _router_config(_classifier_config(api_base), "simple", "complex")
        assert _validate(gateway, config) == {"valid": True, "error": None}, api_base


def test_validate_names_each_databricks_classifier_misconfiguration(gateway: Gateway) -> None:
    base: Final = _classifier_config("https://dbc-example.cloud.databricks.com/serving-endpoints")
    cases: Final[tuple[tuple[str, dict[str, JsonValue], str], ...]] = (
        ("model omitted", _without(base, "model"), f"at opensource_classifier_config: Value error, {_MODEL_REQUIRED}"),
        (
            "api_base without api_key",
            _without(base, "api_key"),
            f"at opensource_classifier_config: Value error, {_BASE_NEEDS_KEY}",
        ),
        (
            "serving endpoint name",
            {**base, "model": "databricks-openjev-qwen35-4b"},
            _MODEL_RULE,
        ),
        ("dash", {**base, "model": "ai-decide"}, _MODEL_RULE),
        ("empty name", {**base, "model": ""}, _MODEL_RULE),
        (
            "blank api_key",
            {**base, "api_key": ""},
            f"at opensource_classifier_config.api_key: Value error, {_BLANK_KEY}",
        ),
        ("int name", {**base, "model": 5}, "at opensource_classifier_config.model: Input should be a valid string"),
        (
            "list name",
            {**base, "model": ["a"]},
            "at opensource_classifier_config.model: Input should be a valid string",
        ),
        ("null name", {**base, "model": None}, "at opensource_classifier_config.model: Input should be a valid string"),
    )
    for label, classifier, expected in cases:
        verdict: Final = _validate(gateway, _router_config(classifier, "simple", "complex"))
        assert verdict["valid"] is False, (label, verdict)
        error: Final = string_value(verdict["error"])
        assert error.startswith("complexity_router_config is invalid "), (label, error)
        assert expected in error, (label, error)
        assert error.endswith(_REJECTED_AT_LOAD), (label, error)


def test_test_routing_classifies_through_databricks_ai_decide_without_calling_the_tier_it_picked(
    gateway: Gateway,
) -> None:
    prompt: Final = f"design a distributed cache {uuid.uuid4().hex}"
    with (
        wire_server(_classifier) as judge,
        wire_server(_tier("simple answer")) as simple,
        wire_server(_tier("complex answer")) as complex_tier,
        gateway.scenario() as scenario,
    ):
        simple_model: Final = scenario.model(model="openai/simple-tier", api_base=simple.url)
        complex_model: Final = scenario.model(model="openai/complex-tier", api_base=complex_tier.url)
        config: Final = _router_config(_classifier_config(judge.url), simple_model, complex_model)
        response: Final = gateway.request(
            "POST", "/auto_router/test_routing", {"prompt": prompt, "complexity_router_config": config}
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            "routed_model": complex_model,
            "routed_model_configured": True,
            "routing_decision": {
                "router_model_name": "auto_router_routing_test",
                "router_type": "complexity",
                "routed_model": complex_model,
                "cause": "jev_classifier",
                "tier": "COMPLEX",
                "signals": [
                    "databricks-classifier:COMPLEX",
                    "databricks-confidence=0.910000",
                    "tier-probability:SIMPLE=0.030000",
                    "tier-probability:MEDIUM=0.060000",
                    "tier-probability:COMPLEX=0.910000",
                ],
                "classifier_model": f"databricks/{_MODEL}",
                "classifier_probabilities": {"SIMPLE": 0.03, "MEDIUM": 0.06, "COMPLEX": 0.91},
                "classifier_confidence": 0.91,
                "conversation_continuing": False,
            },
        }, response.text
        (call,) = [request for request in judge.drain() if request.method == "POST"]
        assert call.target == _ROUTE, call.target
        assert call.headers.get("authorization") == f"Bearer {_API_KEY}", call.headers
        body: Final = json.loads(call.body)
        assert "model" not in body, body
        assert body["state"] == f"\nClassify this message:\n{prompt}", body
        assert _judge_targets(simple) == () and _judge_targets(complex_tier) == ()


def test_model_new_refuses_a_databricks_classifier_without_a_model_and_stores_nothing(
    gateway: Gateway,
) -> None:
    name: Final = f"router-{uuid.uuid4().hex[:10]}"
    config: Final = _router_config(
        _without(_classifier_config("https://dbc-example.cloud.databricks.com/serving-endpoints"), "model"),
        "simple",
        "complex",
    )
    response: Final = gateway.request(
        "POST",
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": "auto_router/complexity_router", "complexity_router_config": config},
            "model_info": {},
        },
    )
    assert response.status_code == 400, response.text
    error: Final = object_value(object_value(response.json())["error"])
    assert _MODEL_REQUIRED in string_value(error["message"]), error
    assert (error["type"], error["param"]) == ("validation_error", "litellm_params.model"), error
    listed: Final = gateway.get("/model/info")["data"]
    assert isinstance(listed, list)
    assert [entry for entry in map(object_value, listed) if entry["model_name"] == name] == []


def test_model_update_refuses_dropping_the_model_and_keeps_the_stored_classifier_serving(
    gateway: Gateway,
) -> None:
    with (
        wire_server(_classifier) as judge,
        wire_server(_tier("simple answer")) as simple,
        wire_server(_tier("complex answer")) as complex_tier,
        gateway.scenario() as scenario,
    ):
        simple_model: Final = scenario.model(model="openai/simple-tier", api_base=simple.url)
        complex_model: Final = scenario.model(model="openai/complex-tier", api_base=complex_tier.url)
        classifier: Final = _classifier_config(judge.url)
        name, identity = _create_router(scenario, _router_config(classifier, simple_model, complex_model))
        dropped: Final = _router_config(_without(classifier, "model"), simple_model, complex_model)
        response: Final = gateway.request(
            "PATCH",
            f"/model/{identity}/update",
            {"litellm_params": {"model": "auto_router/complexity_router", "complexity_router_config": dropped}},
        )
        assert response.status_code == 400, response.text
        assert _MODEL_REQUIRED in string_value(object_value(object_value(response.json())["error"])["message"]), (
            response.text
        )
        assert _stored_classifier(gateway, identity)["model"] == _MODEL
        chat: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(name))
        assert chat.status_code == 200, chat.text
        assert chat.json()["choices"][0]["message"]["content"] == "complex answer", chat.text
        assert chat.headers["x-litellm-complexity-router-cause"] == "jev_classifier", dict(chat.headers)
        assert _judge_targets(judge) == (_ROUTE,)


async def test_moving_the_workspace_under_a_burst_moves_the_classifier_and_logs_every_request_once(
    gateway: Gateway,
) -> None:
    base_url: Final = str(gateway.client.base_url)
    with (
        wire_server(_classifier) as judge,
        wire_server(_tier("simple answer")) as simple,
        wire_server(_tier("complex answer")) as complex_tier,
        gateway.scenario() as scenario,
    ):
        simple_model: Final = scenario.model(model="openai/simple-tier", api_base=simple.url)
        complex_model: Final = scenario.model(model="openai/complex-tier", api_base=complex_tier.url)
        classifier: Final = _classifier_config(judge.url)
        name, identity = _create_router(scenario, _router_config(classifier, simple_model, complex_model))
        burst: Final = asyncio.create_task(_burst(base_url, gateway.key, name, 24))
        moved: Final = _router_config(
            {**classifier, "api_base": f"{judge.url}{_SECOND_WORKSPACE}"}, simple_model, complex_model
        )
        patched: Final = gateway.request(
            "PATCH",
            f"/model/{identity}/update",
            {"litellm_params": {"model": "auto_router/complexity_router", "complexity_router_config": moved}},
        )
        assert patched.status_code == 200, patched.text
        during: Final = await burst
        assert [response.status_code for response in during] == [200] * 24, [response.text for response in during]
        assert set(_judge_targets(judge)) <= {_ROUTE, f"{_SECOND_WORKSPACE}{_ROUTE}"}
        assert _stored_classifier(gateway, identity)["api_base"] == f"{judge.url}{_SECOND_WORKSPACE}"

        polled: Final[list[str]] = []

        def classifier_target() -> tuple[str, ...]:
            chat: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(name))
            assert chat.status_code == 200, chat.text
            polled.append(string_value(chat.json()["id"]))
            return _judge_targets(judge)

        eventually(classifier_target, lambda targets: targets == (f"{_SECOND_WORKSPACE}{_ROUTE}",), seconds=30)
        after: Final = await _burst(base_url, gateway.key, name, 6)
        assert [response.status_code for response in after] == [200] * 6, [response.text for response in after]
        assert set(_judge_targets(judge)) == {f"{_SECOND_WORKSPACE}{_ROUTE}"}
        identities: Final = (*(string_value(response.json()["id"]) for response in (*during, *after)), *polled)
        assert len(set(identities)) == len(identities), identities
        rows: Final = _rows(name, requests=len(identities))
        request_rows: Final = _request_rows(rows)
        assert sorted(string_value(row["request_id"]) for row in request_rows) == sorted(identities), request_rows
        assert all(row["status"] == "success" for row in rows), rows
