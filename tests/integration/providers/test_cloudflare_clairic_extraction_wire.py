import json
import math
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, WorkerHealthcheck, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

Route = Literal["/v1/extraction", "/extraction"]

_API_KEY: Final = "synthetic-cloudflare-key"
_CLAIRIC: Final = "cloudflare/@cf/cloudflare/clairic"
_CLAIRIC_PATH: Final = "/ai/run/@cf/cloudflare/clairic"
_INPUT_TOKENS: Final = 412
_EMAIL: Final = "billing@acmetools.example"
_SCENARIO: Final = re.compile(r"scenario-([a-z-]+)")
_JSON_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
_FIELDS: Final[dict[str, JsonValue]] = {
    "vendor": {"type": "text", "instructions": "Who sent the invoice?"},
    "total": {"type": "number", "unit": "USD"},
}
_SPEND_QUERY: Final = (
    'SELECT spend, status, call_type, model_group, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
    "WHERE request_id = %s"
)
_OWNED_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
_REQUEST_GUARDRAIL: Final = "extraction-mask-email-in"
_RESPONSE_GUARDRAIL: Final = "extraction-mask-email-out"
_OWNED_MODEL: Final = "invoice-reader"


def _state(marker: str, scenario: str = "plain") -> str:
    return f"Invoice from Acme Tools, total $174. scenario-{scenario} marker-{marker}"


def _vendor_span(state: str) -> list[JsonValue]:
    start: Final = state.index("Acme Tools")
    return [start, start + len("Acme Tools")]


def _email_span(state: str) -> list[JsonValue]:
    start: Final = state.index(_EMAIL)
    return [start, start + len(_EMAIL)]


def _answers(state: str) -> dict[str, JsonValue]:
    return {
        "vendor": {"type": "text", "value": "Acme Tools", "confidence": 0.97, "span": _vendor_span(state)},
        "total": {"type": "number", "value": 174, "confidence": 0.95},
    }


def _envelope(result: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps({"result": result, "success": True, "errors": [], "messages": []}).encode())


def _clairic_reply(request: Request) -> Reply:
    state: Final = JSON_OBJECT.validate_json(request.body)["state"]
    assert isinstance(state, str), request.body
    return _envelope(
        {
            "model": "clairic",
            "answers": _answers(state),
            "truncated": False,
            "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
        }
    )


def _expected_response(model: str, state: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "answers": _answers(state),
        "truncated": False,
        "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
    }


def _upstream_body(request: Request) -> dict[str, JsonValue]:
    assert (request.method, request.target) == ("POST", _CLAIRIC_PATH), (request.method, request.target)
    assert request.headers["authorization"] == f"Bearer {_API_KEY}", request.headers
    return JSON_OBJECT.validate_json(request.body)


def _body(model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "state": _state(marker), "fields": _FIELDS, **extra}


def _extract(
    gateway: Gateway,
    model: str,
    marker: str,
    *,
    route: Route = "/v1/extraction",
    key: str | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    return gateway.request("POST", route, _body(model, marker, **extra), key=key)


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def _expected_spend() -> float:
    cost_map: Final = JSON_OBJECT.validate_json(Path("model_prices_and_context_window.json").read_bytes())
    priced: Final = cost_map.get(_CLAIRIC)
    assert priced is not None, f"{_CLAIRIC} has no cost map entry"
    return _INPUT_TOKENS * _number(object_value(priced)["input_cost_per_token"])


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _assert_billed(gateway: Gateway, response: httpx.Response, model: str) -> None:
    expected: Final = _expected_spend()
    assert math.isclose(float(response.headers["x-litellm-response-cost"]), expected, rel_tol=1e-9), response.headers
    call_id: Final = response.headers["x-litellm-call-id"]
    row: Final = _spend_row(call_id)
    assert (row["status"], row["call_type"], row["model_group"], row["prompt_tokens"], row["completion_tokens"]) == (
        "success",
        "aextraction",
        model,
        _INPUT_TOKENS,
        0,
    ), row
    assert math.isclose(_number(row["spend"]), expected, rel_tol=1e-9), row
    listed: Final = gateway.request("GET", "/spend/logs", params={"request_id": call_id})
    assert listed.status_code == 200, listed.text
    (reported,) = _JSON_ROWS.validate_json(listed.content)
    assert (reported["request_id"], reported["model_group"]) == (call_id, model), reported
    assert math.isclose(_number(reported["spend"]), expected, rel_tol=1e-9), reported


@pytest.mark.parametrize("route", ("/v1/extraction", "/extraction"))
def test_extraction_reaches_clairic_on_both_routes_and_is_billed(gateway: Gateway, route: Route) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clairic_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLAIRIC, api_base=wire.url, api_key=_API_KEY)
        response: Final = _extract(gateway, model, marker, route=route, metadata={"tags": [f"audit-{marker}"]})
        assert response.status_code == 200, response.text
        assert response.json() == _expected_response(model, _state(marker))
        (request,) = wire.drain()
        assert _upstream_body(request) == {"model": "clairic", "state": _state(marker), "fields": _FIELDS}
        _assert_billed(gateway, response, model)


_INVALID_BODIES: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
    ("unknown_type", {"fields": {"vendor": {"type": "paragraph"}}}),
    ("list_without_items", {"fields": {"items": {"type": "list"}}}),
    ("remote_image", {"images": ["https://images.example/receipt.png"]}),
)


@pytest.mark.parametrize(("label", "change"), _INVALID_BODIES, ids=[label for label, _ in _INVALID_BODIES])
def test_invalid_extraction_bodies_are_refused_before_any_upstream_call(
    gateway: Gateway, label: str, change: dict[str, JsonValue]
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clairic_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLAIRIC, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/extraction", {**_body(model, marker), **change})
        assert response.status_code == 400, (label, response.text)
        assert "Invalid Extraction request" in response.text, (label, response.text)
        assert wire.drain() == (), label


@pytest.mark.parametrize(("param", "value"), (("temperature", 0.2), ("stream", True)), ids=("temperature", "stream"))
def test_unsupported_top_level_params_are_refused_unless_the_deployment_drops_params(
    gateway: Gateway, param: str, value: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clairic_reply) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model=_CLAIRIC, api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(model=_CLAIRIC, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        refused: Final = _extract(gateway, strict, marker, **{param: value})
        assert refused.status_code == 400, refused.text
        assert param in refused.text, refused.text
        assert wire.drain() == ()
        dropped: Final = _extract(gateway, dropping, marker, **{param: value})
        assert dropped.status_code == 200, dropped.text
        assert dropped.headers["content-type"].startswith("application/json"), dropped.headers
        assert dropped.json() == _expected_response(dropping, _state(marker))
        (request,) = wire.drain()
        assert _upstream_body(request) == {"model": "clairic", "state": _state(marker), "fields": _FIELDS}
        _assert_billed(gateway, dropped, dropping)


def test_key_checks_on_extraction_match_chat(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clairic_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLAIRIC, api_base=wire.url, api_key=_API_KEY)
        anonymous: Final = gateway.client.post("/v1/extraction", json=_body(model, marker))
        assert anonymous.status_code == 401, anonymous.text
        restricted: Final = scenario.key(models=[f"other-{uuid.uuid4().hex}"])
        refused: Final = _extract(gateway, model, marker, key=restricted)
        chat_refused: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}]},
            key=restricted,
        )
        assert 400 <= refused.status_code < 500, refused.text
        assert refused.status_code == chat_refused.status_code, (refused.text, chat_refused.text)
        assert wire.drain() == ()
        allowed: Final = scenario.key(models=[model], max_budget=1e-06)
        first: Final = _extract(gateway, model, marker, key=allowed)
        assert first.status_code == 200, first.text
        assert len(wire.drain()) == 1
        blocked, upstream_calls = eventually(
            lambda: (_extract(gateway, model, uuid.uuid4().hex, key=allowed), len(wire.drain())),
            lambda answer: answer[0].status_code != 200,
            seconds=70,
        )
        chat_blocked: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}]},
            key=allowed,
        )
        assert 400 <= blocked.status_code < 500, blocked.text
        assert blocked.status_code == chat_blocked.status_code, (blocked.text, chat_blocked.text)
        assert upstream_calls == 0
        assert wire.drain() == ()


def test_a_small_key_tpm_limit_serves_extraction_and_counts_its_input_tokens(gateway: Gateway) -> None:
    with wire_server(_clairic_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLAIRIC, api_base=wire.url, api_key=_API_KEY)
        key: Final = scenario.key(models=[model], tpm_limit=100)
        first: Final = _extract(gateway, model, uuid.uuid4().hex, key=key)
        assert first.status_code == 200, first.text
        assert len(wire.drain()) == 1
        limited, upstream_calls = eventually(
            lambda: (_extract(gateway, model, uuid.uuid4().hex, key=key), len(wire.drain())),
            lambda answer: answer[0].status_code != 200,
            seconds=70,
        )
        assert limited.status_code == 429, limited.text
        assert upstream_calls == 0


def test_cloudflare_errors_keep_their_status_and_log_an_unbilled_failure(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    refusal: Final = json.dumps(
        {
            "result": None,
            "success": False,
            "errors": [{"code": 5006, "message": f"min above max {marker}"}],
            "messages": [],
        }
    ).encode()
    with wire_server(lambda _: Reply(status=422, body=refusal)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLAIRIC, api_base=wire.url, api_key=_API_KEY)
        response: Final = _extract(gateway, model, marker)
        assert response.status_code == 422, response.text
        assert f"min above max {marker}" in response.text, response.text
        assert len(wire.drain()) >= 1
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], row["model_group"], _number(row["spend"])) == (
            "failure",
            "aextraction",
            model,
            0.0,
        ), row


def _owned_reply(request: Request) -> Reply:
    body: Final = JSON_OBJECT.validate_json(request.body)
    state: Final = body["state"]
    assert isinstance(state, str), request.body
    scenario: Final = _SCENARIO.search(state)
    assert scenario is not None, state
    if scenario.group(1) != "mask-out":
        return _clairic_reply(request)
    return _envelope(
        {
            "model": "clairic",
            "answers": {
                "vendor": {"type": "text", "value": "Acme Tools", "confidence": 0.97, "span": _vendor_span(state)},
                "contact_email": {"type": "text", "value": _EMAIL, "confidence": 0.9, "span": _email_span(state)},
            },
            "truncated": False,
            "state_text": state,
            "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
        }
    )


def _owned_config(upstream_url: str) -> dict[str, JsonValue]:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    email_filter: Final[list[JsonValue]] = [{"pattern_type": "prebuilt", "pattern_name": "email", "action": "MASK"}]
    return {
        **base,
        "guardrails": [
            {
                "guardrail_name": _REQUEST_GUARDRAIL,
                "litellm_params": {"guardrail": "litellm_content_filter", "mode": "pre_call", "patterns": email_filter},
            },
            {
                "guardrail_name": _RESPONSE_GUARDRAIL,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "post_call",
                    "patterns": email_filter,
                },
            },
        ],
        "model_list": [
            {
                "model_name": _OWNED_MODEL,
                "litellm_params": {"model": _CLAIRIC, "api_base": upstream_url, "api_key": _API_KEY},
            }
        ],
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: OwnedProxy
    upstream: Wire


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with wire_server(_owned_reply) as upstream:
        directory: Final = tmp_path_factory.mktemp("clairic-extraction-owned")
        config_path: Final = directory / "clairic-extraction.yaml"
        config_path.write_text(yaml.safe_dump(_owned_config(upstream.url)))
        with owned_proxy_process(
            session_gateway,
            directory,
            {},
            config=config_path,
            workers=2,
            extra_arguments=("--max_tokens", "256", "--temperature", "0.2"),
            healthcheck=WorkerHealthcheck.LENIENT,
        ) as proxy:
            yield Rig(proxy, upstream)


def _owned_extract(rig: Rig, state: str, **extra: JsonValue) -> httpx.Response:
    return rig.proxy.gateway.request(
        "POST", "/v1/extraction", {"model": _OWNED_MODEL, "state": state, "fields": _FIELDS, **extra}
    )


def _applied_guardrails(response: httpx.Response) -> frozenset[str]:
    return frozenset(name.strip() for name in response.headers.get("x-litellm-applied-guardrails", "").split(","))


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_cli_generation_defaults_never_reach_an_extraction_call(rig: Rig) -> None:
    state: Final = _state(uuid.uuid4().hex)
    response: Final = _owned_extract(rig, state)
    assert response.status_code == 200, response.text
    assert response.json() == _expected_response(_OWNED_MODEL, state)
    (request,) = rig.upstream.drain()
    assert _upstream_body(request) == {"model": "clairic", "state": state, "fields": _FIELDS}


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_a_request_guardrail_masks_the_state_before_clairic_and_drops_spans_into_it(rig: Rig) -> None:
    state: Final = f"{_state(uuid.uuid4().hex, 'mask-in')} Questions go to {_EMAIL}."
    response: Final = _owned_extract(rig, state, guardrails=[_REQUEST_GUARDRAIL])
    assert response.status_code == 200, response.text
    assert _REQUEST_GUARDRAIL in _applied_guardrails(response), response.headers
    (request,) = rig.upstream.drain()
    sent: Final = _upstream_body(request)
    sent_state: Final = sent["state"]
    assert isinstance(sent_state, str) and _EMAIL not in sent_state, sent
    assert sent_state.startswith(state.removesuffix(f"{_EMAIL}.")) and "REDACTED" in sent_state, sent
    assert sent == {"model": "clairic", "state": sent_state, "fields": _FIELDS}
    answers: Final = object_value(response.json()["answers"])
    assert object_value(answers["vendor"]) == {"type": "text", "value": "Acme Tools", "confidence": 0.97}, answers


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_a_response_guardrail_masks_text_answers_and_state_text(rig: Rig) -> None:
    state: Final = f"{_state(uuid.uuid4().hex, 'mask-out')} Questions go to {_EMAIL}."
    fields: Final[dict[str, JsonValue]] = {**_FIELDS, "contact_email": {"type": "text"}}
    response: Final = rig.proxy.gateway.request(
        "POST",
        "/v1/extraction",
        {
            "model": _OWNED_MODEL,
            "state": state,
            "fields": fields,
            "guardrails": [_RESPONSE_GUARDRAIL],
            "options": {"return_state_text": True},
        },
    )
    assert response.status_code == 200, response.text
    assert _RESPONSE_GUARDRAIL in _applied_guardrails(response), response.headers
    (request,) = rig.upstream.drain()
    assert _upstream_body(request)["state"] == state
    body: Final = response.json()
    state_text: Final = body["state_text"]
    assert isinstance(state_text, str) and _EMAIL not in state_text and "REDACTED" in state_text, body
    answers: Final = object_value(body["answers"])
    contact: Final = object_value(answers["contact_email"])
    assert "span" not in contact, answers
    masked: Final = contact["value"]
    assert isinstance(masked, str) and _EMAIL not in masked and "REDACTED" in masked, answers
    assert object_value(answers["vendor"]) == {"type": "text", "value": "Acme Tools", "confidence": 0.97}, answers
