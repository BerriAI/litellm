from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, list_value, object_value
from tests.integration._support.openai_wire import answering_model_discovery, chat_reply
from tests.integration._support.process import graceful_stop_seconds, owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

SUFFIX: Final = uuid.uuid4().hex[:10]
CONFIGURED_DEFAULT: Final = "gpt-6-luna"
EXPLICIT_WIRE_MODEL: Final = f"policy-enricher-{SUFFIX}"
EXPLICIT: Final = f"openai/{EXPLICIT_WIRE_MODEL}"
OUTAGE: Final = "outage"
ENRICH: Final = "/policy/templates/enrich"
ENRICH_STREAM: Final = "/policy/templates/enrich/stream"
SUGGEST: Final = "/policy/templates/suggest"
ENDPOINTS: Final = (ENRICH, ENRICH_STREAM)
ENDPOINT_IDS: Final = ("enrich", "stream")
TEMPLATES: Final = ("competitor-mention-detection", "aviation-operations-security")
COMPETITOR_TEMPERATURE: Final = 0.3
SUGGEST_TEMPERATURE: Final = 0.2
BRAND_PATTERN: Final = re.compile(r"brand[0-9a-f]{12}")
VARIATIONS_MARKER: Final = "common misspellings"
TEMPERATURE_REJECTION: Final = {
    "error": {
        "message": (
            "Unsupported value: 'temperature' does not support 0.3 with this model. "
            "Only the default (1) value is supported."
        ),
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
}


def _brand() -> str:
    return f"brand{uuid.uuid4().hex[:12]}"


def _rivals(brand: str) -> tuple[str, str, str]:
    return (f"{brand} Air", f"{brand} Jet", f"{brand} Wings")


def _variations(name: str) -> list[str]:
    return [name.replace(" ", ""), f"{name} Co"]


def _prompt(body: dict[str, JsonValue]) -> str:
    return json.dumps(body.get("messages"))


def _rejects_temperature(body: dict[str, JsonValue]) -> bool:
    temperature: Final = body.get("temperature")
    return temperature is not None and temperature != 1 and body.get("reasoning_effort") != "none"


def _error(status: int, payload: dict[str, JsonValue]) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def _unknown_model(model: JsonValue) -> Reply:
    return _error(
        404,
        {
            "error": {
                "message": f"The model `{model}` does not exist or you do not have access to it.",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            }
        },
    )


def _variation_text(prompt: str) -> str:
    names: Final = prompt.split("Names:\\n", 1)[1].split("\\n\\nReturn", 1)[0].split("\\n")
    return "\n".join(f"{name}: {', '.join(_variations(name))}" for name in names)


def _suggestion_reply(model: str) -> Reply:
    arguments: Final = {
        "selected_templates": [
            {"template_id": "competitor-mention-detection", "reason": "blocks rival brands"},
            {"template_id": f"not-a-template-{SUFFIX}", "reason": "unknown"},
        ],
        "explanation": "competitor mentions",
    }
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-suggest-{SUFFIX}",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_select",
                                    "type": "function",
                                    "function": {
                                        "name": "select_policy_templates",
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        ).encode()
    )


def _respond(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request.target
    body: Final = JSON_OBJECT.validate_json(request.body)
    model: Final = body.get("model")
    if model not in (CONFIGURED_DEFAULT, EXPLICIT_WIRE_MODEL):
        return _unknown_model(model)
    if model == CONFIGURED_DEFAULT and _rejects_temperature(body):
        return _error(400, TEMPERATURE_REJECTION)
    prompt: Final = _prompt(body)
    if "tools" in body:
        return _suggestion_reply(str(model))
    if model == CONFIGURED_DEFAULT and OUTAGE in prompt:
        return _error(500, {"error": {"message": "scripted outage", "type": "server_error", "code": None}})
    brand: Final = BRAND_PATTERN.search(prompt)
    assert brand is not None, prompt
    text: Final = _variation_text(prompt) if VARIATIONS_MARKER in prompt else "\n".join(_rivals(brand.group()))
    return chat_reply(f"chatcmpl-{SUFFIX}", str(model), text, stream=body.get("stream") is True)


class Rig:
    def __init__(self, gateway: Gateway, wire: Wire) -> None:
        self.gateway = gateway
        self.wire = wire
        self.seen: tuple[dict[str, JsonValue], ...] = ()

    def bodies_mentioning(self, token: str) -> tuple[dict[str, JsonValue], ...]:
        self.seen = (*self.seen, *(JSON_OBJECT.validate_json(request.body) for request in self.wire.drain()))
        return tuple(body for body in self.seen if token in _prompt(body))


def _config(directory: Path, upstream_url: str) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **base,
        "litellm_settings": {
            name: value for name, value in base["litellm_settings"].items() if name not in ("cache", "cache_params")
        },
        "router_settings": {},
        "model_list": [
            {
                "model_name": CONFIGURED_DEFAULT,
                "litellm_params": {
                    "model": f"openai/{CONFIGURED_DEFAULT}",
                    "api_base": f"{upstream_url}/v1",
                    "api_key": "synthetic-openai-key",
                },
            },
            {
                "model_name": EXPLICIT,
                "litellm_params": {
                    "model": EXPLICIT,
                    "api_base": f"{upstream_url}/v1",
                    "api_key": "synthetic-openai-key",
                },
            },
        ],
    }
    path: Final = directory / "policy_template_default_model.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("policy-template-default-model")
    with (
        wire_server(answering_model_discovery(_respond)) as wire,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"OPENAI_API_KEY": "synthetic-openai-key", "OPENAI_BASE_URL": f"{wire.url}/v1"},
            config=_config(directory, wire.url),
            remove_environment=("OPENAI_API_BASE",),
        ) as gateway,
    ):
        yield Rig(gateway, wire)


def _events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    )


def _enrich(rig: Rig, path: str, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    response: Final = rig.gateway.request("POST", path, body)
    assert response.status_code == 200, response.text
    if path == ENRICH:
        return JSON_OBJECT.validate_json(response.content)
    events: Final = _events(response.text)
    streamed: Final = [event["name"] for event in events if event.get("type") == "competitor"]
    done: Final = [event for event in events if event.get("type") == "done"]
    assert len(done) == 1, events
    assert streamed == done[0]["competitors"], events
    return done[0]


def _enrich_body(template: str, brand: str, model: str | None = None) -> dict[str, JsonValue]:
    return {"template_id": template, "parameters": {"brand_name": brand}, **({"model": model} if model else {})}


def _blocked_keywords(result: dict[str, JsonValue]) -> set[str]:
    definitions: Final = [object_value(definition) for definition in list_value(result["guardrailDefinitions"])]
    blocker: Final = next(
        object_value(definition["litellm_params"])
        for definition in definitions
        if definition.get("guardrail_name") in ("competitor-input-blocker", "competitor-name-input-blocker")
    )
    return {object_value(word)["keyword"] for word in list_value(blocker["blocked_words"])}


def _expected_variations(brand: str) -> dict[str, list[str]]:
    return {name: _variations(name) for name in _rivals(brand)}


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("path", ENDPOINTS, ids=ENDPOINT_IDS)
def test_enrich_without_a_model_discovers_competitors_through_the_configured_deployment_regression(
    rig: Rig, path: str, template: str
) -> None:
    brand: Final = _brand()
    result: Final = _enrich(rig, path, _enrich_body(template, brand))
    assert result["competitors"] == list(_rivals(brand)), result
    assert result["competitor_variations"] == _expected_variations(brand), result
    assert set(_rivals(brand)) <= _blocked_keywords(result), result
    sent: Final = rig.bodies_mentioning(brand)
    assert [body["model"] for body in sent] == [CONFIGURED_DEFAULT, CONFIGURED_DEFAULT], sent
    assert not any(_rejects_temperature(body) for body in sent), sent


@pytest.mark.parametrize("path", ENDPOINTS, ids=ENDPOINT_IDS)
def test_enrich_with_an_explicit_model_keeps_sending_the_competitor_temperature_guard(rig: Rig, path: str) -> None:
    brand: Final = _brand()
    result: Final = _enrich(rig, path, _enrich_body(TEMPLATES[0], brand, EXPLICIT))
    assert result["competitors"] == list(_rivals(brand)), result
    assert result["competitor_variations"] == _expected_variations(brand), result
    sent: Final = rig.bodies_mentioning(brand)
    assert [(body["model"], body.get("temperature")) for body in sent] == [
        (EXPLICIT_WIRE_MODEL, COMPETITOR_TEMPERATURE),
        (EXPLICIT_WIRE_MODEL, COMPETITOR_TEMPERATURE),
    ], sent
    assert [body.get("stream") is True for body in sent] == [path == ENRICH_STREAM, False], sent


def test_enrich_returns_no_competitors_when_the_default_model_call_fails_guard(rig: Rig) -> None:
    brand: Final = f"{_brand()} {OUTAGE}"
    response: Final = rig.gateway.request("POST", ENRICH, _enrich_body(TEMPLATES[0], brand))
    assert response.status_code == 200, response.text
    result: Final = JSON_OBJECT.validate_json(response.content)
    assert result["competitors"] == [], result
    assert result["competitor_variations"] == {}, result


def test_enrich_stream_reports_an_error_event_when_the_default_model_call_fails_guard(rig: Rig) -> None:
    brand: Final = f"{_brand()} {OUTAGE}"
    response: Final = rig.gateway.request("POST", ENRICH_STREAM, _enrich_body(TEMPLATES[0], brand))
    assert response.status_code == 200, response.text
    types: Final = [event.get("type") for event in _events(response.text)]
    assert types == ["error"], response.text


def _suggest(rig: Rig, model: str | None) -> tuple[dict[str, JsonValue], tuple[dict[str, JsonValue], ...]]:
    token: Final = f"suggest{uuid.uuid4().hex[:12]}"
    response: Final = rig.gateway.request(
        "POST",
        SUGGEST,
        {"description": f"stop answers that recommend rival brands {token}", **({"model": model} if model else {})},
    )
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content), rig.bodies_mentioning(token)


def _assert_forced_suggestion(result: dict[str, JsonValue], sent: tuple[dict[str, JsonValue], ...]) -> None:
    assert result == {
        "selected_templates": [{"template_id": "competitor-mention-detection", "reason": "blocks rival brands"}],
        "explanation": "competitor mentions",
    }, result
    assert len(sent) == 1, sent
    assert sent[0]["tool_choice"] == {"type": "function", "function": {"name": "select_policy_templates"}}, sent
    assert [object_value(object_value(tool)["function"])["name"] for tool in list_value(sent[0]["tools"])] == [
        "select_policy_templates"
    ], sent


def test_suggest_without_a_model_selects_templates_with_the_default_model_regression(rig: Rig) -> None:
    result, sent = _suggest(rig, None)
    _assert_forced_suggestion(result, sent)
    assert sent[0]["model"] == CONFIGURED_DEFAULT, sent
    assert not _rejects_temperature(sent[0]), sent


def test_suggest_with_an_explicit_model_keeps_sending_its_temperature_guard(rig: Rig) -> None:
    result, sent = _suggest(rig, EXPLICIT)
    _assert_forced_suggestion(result, sent)
    assert (sent[0]["model"], sent[0].get("temperature")) == (EXPLICIT_WIRE_MODEL, SUGGEST_TEMPERATURE), sent
