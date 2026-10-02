import json
import os
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest
import yaml
from integration._support.bedrock_runtime_peer import answer, forwarded_effort, marker_of, respond, target_of
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

GPT: Final = "us.openai.gpt-5.6-sol"
TOKEN: Final = "synthetic-bedrock-bearer"
BAD_KEY: Final = "sk-synthetic-bad-key"
NATIVE_TARGET: Final = "/openai/v1/chat/completions"
CONVERSE_TARGET: Final = f"/model/{GPT}/converse"
LONG_VERSION_GPT: Final = "openai.gpt-" + "1" * 30000
PNG_DATA_URL: Final = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)
GPT_DEPLOYMENT: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"model": f"bedrock/{GPT}", "api_key": TOKEN, "aws_region_name": "us-east-1"}
)
_ALLOWLISTED_MODEL: Final = "bedrock-gpt-image-allowlist"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _prompt(marker: str) -> str:
    return f"synthetic sad request marker-{marker}"


def _messages(marker: str) -> list[dict[str, JsonValue]]:
    return [{"role": "user", "content": _prompt(marker)}]


def _image_messages(marker: str, url: str) -> list[dict[str, JsonValue]]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": _prompt(marker)}, {"type": "image_url", "image_url": {"url": url}}],
        }
    ]


def _deployment(scenario: Scenario, wire: Wire, **overrides: JsonValue) -> str:
    return scenario.model(**{**GPT_DEPLOYMENT, "aws_bedrock_runtime_endpoint": wire.url, **overrides})


def _chat(gateway: Gateway, model: str, marker: str, *, key: str | None = None, **params: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": _messages(marker), "cache": {"no-cache": True}, **params},
        key=key,
    )


def _payload(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def _content(response: httpx.Response) -> JsonValue:
    choices: Final = _payload(response)["choices"]
    assert isinstance(choices, list), response.text
    return object_value(object_value(choices[0])["message"])["content"]


def _error_message(response: httpx.Response) -> str:
    return string_value(object_value(_JSON_OBJECT.validate_json(response.content)["error"])["message"])


def _call_id(response: httpx.Response) -> str:
    return response.headers["x-litellm-call-id"]


def _body(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _routes(received: tuple[Request, ...]) -> list[tuple[str, str]]:
    return [(request.method, target_of(request)) for request in received]


def _only_request(wire: Wire, marker: str) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, _routes(received)
    assert marker_of(received[0]) == marker, received[0].body
    return received[0]


def _spend_rows(identity: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, model_group, status, cache_hit, spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
        (identity,),
    )


def _spend_row(identity: str) -> dict[str, JsonValue]:
    return eventually(lambda: _spend_rows(identity), lambda found: len(found) == 1, seconds=70)[0]


def _assert_row(identity: str, model: str, status: str) -> None:
    row: Final = _spend_row(identity)
    assert (row["model_group"], row["status"]) == (model, status), row


def _timed_liveliness(gateway: Gateway) -> tuple[int, float]:
    started: Final = time.monotonic()
    response: Final = gateway.request("GET", "/health/liveliness")
    return response.status_code, time.monotonic() - started


def _pooled_database_url(url: str) -> str:
    parts: Final = urlsplit(url)
    query: Final = "&".join(part for part in (parts.query, "connection_limit=5") if part)
    return urlunsplit(parts._replace(query=query))


def _allowlist_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    path: Final = tmp_path / "bedrock-gpt-image-allowlist.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": _ALLOWLISTED_MODEL,
                        "litellm_params": {**GPT_DEPLOYMENT, "aws_bedrock_runtime_endpoint": wire.url},
                    }
                ],
                "general_settings": {
                    **object_value(config["general_settings"]),
                    "user_url_allowed_hosts": ["127.0.0.1"],
                },
            }
        )
    )
    return path


def test_remote_image_url_on_the_shared_proxy_is_rejected_before_any_fetch(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _image_messages(marker, f"{wire.url}/image.png"), "cache": {"no-cache": True}},
        )
        assert response.status_code == 400, response.text
        message: Final = _error_message(response)
        assert "Unable to fetch image from URL" in message and "user_url_allowed_hosts" in message, response.text
        _assert_row(_call_id(response), model, "failure")
        assert _routes(wire.drain()) == []


@pytest.mark.timeout(180)
def test_allowlisted_remote_image_is_inlined_for_the_native_route(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    missing_marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire:
        path: Final = _allowlist_config(wire, tmp_path)
        overrides: Final = {"DATABASE_URL": _pooled_database_url(os.environ["DATABASE_URL"])}
        with owned_proxy_process(gateway, tmp_path, overrides, config=path) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": _ALLOWLISTED_MODEL,
                    "messages": _image_messages(marker, f"{wire.url}/image.png"),
                    "cache": {"no-cache": True},
                },
            )
            assert _content(response) == answer(marker), response.text
            received: Final = wire.drain()
            assert _routes(received) == [("GET", "/image.png"), ("POST", NATIVE_TARGET)], received
            assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
            assert _body(received[1]) == {
                "model": GPT,
                "messages": _image_messages(marker, PNG_DATA_URL),
                "stream": False,
            }, received[1].body
            _assert_row(f"chatcmpl-{marker}", _ALLOWLISTED_MODEL, "success")
            missing: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": _ALLOWLISTED_MODEL,
                    "messages": _image_messages(missing_marker, f"{wire.url}/missing.png"),
                    "cache": {"no-cache": True},
                },
            )
            assert missing.status_code == 400, missing.text
            assert "Unable to fetch image from URL. Status code: 404" in _error_message(missing), missing.text
            _assert_row(_call_id(missing), _ALLOWLISTED_MODEL, "failure")
            assert _routes(wire.drain()) == [("GET", "/missing.png")]


def test_response_cache_twin_serves_the_second_request_without_a_second_wire_call(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        body: Final[dict[str, JsonValue]] = {"model": model, "messages": _messages(marker)}
        first: Final = gateway.request("POST", "/v1/chat/completions", body)
        second: Final = gateway.request("POST", "/v1/chat/completions", body)
        identity: Final = string_value(_payload(first)["id"])
        assert _content(first) == answer(marker), first.text
        assert _payload(second)["id"] == identity, (first.text, second.text)
        assert _content(second) == answer(marker), second.text
        _only_request(wire, marker)
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, cache_hit, spend FROM "LiteLLM_SpendLogs" WHERE starts_with(request_id, %s)'
                " ORDER BY request_id",
                (identity,),
            ),
            lambda found: len(found) == 2,
            seconds=70,
        )
        assert [(row["request_id"] == identity, row["cache_hit"]) for row in rows] == [(True, "None"), (False, "True")]
        assert string_value(rows[1]["request_id"]).startswith(f"{identity}_cache_hit"), rows
        assert rows[1]["spend"] == 0.0, rows
        assert isinstance(rows[0]["spend"], float) and rows[0]["spend"] > 0.0, rows


def test_model_group_info_lists_the_native_supported_params(gateway: Gateway) -> None:
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        groups: Final = gateway.get("/model_group/info", {"model_group": model})["data"]
        assert isinstance(groups, list) and len(groups) == 1, groups
        group: Final = object_value(groups[0])
        assert group["model_group"] == model, group
        params: Final = group["supported_openai_params"]
        assert isinstance(params, list), group
        assert {"reasoning_effort", "logprobs", "top_logprobs"} <= set(params) and "n" not in params, params
        assert _routes(wire.drain()) == []


def test_thirty_thousand_digit_version_is_classified_quickly_and_served_by_converse(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario, ThreadPoolExecutor(max_workers=1) as pool:
        model: Final = _deployment(scenario, wire, model=f"bedrock/{LONG_VERSION_GPT}")
        liveliness: Final = pool.submit(_timed_liveliness, gateway)
        started: Final = time.monotonic()
        response: Final = _chat(gateway, model, marker)
        elapsed: Final = time.monotonic() - started
        health_status, health_elapsed = liveliness.result()
        assert _content(response) == answer(marker), response.text
        assert elapsed < 10, elapsed
        assert (health_status, health_elapsed < 2) == (200, True), (health_status, health_elapsed)
        request: Final = _only_request(wire, marker)
        assert (request.method, target_of(request)) == ("POST", f"/model/{LONG_VERSION_GPT}/converse"), request.target
        _assert_row(string_value(_payload(response)["id"]), model, "success")


def test_bad_key_on_the_long_version_model_is_refused_before_any_route(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    control_marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, model=f"bedrock/{LONG_VERSION_GPT}")
        started: Final = time.monotonic()
        refused: Final = _chat(gateway, model, marker, key=BAD_KEY)
        elapsed: Final = time.monotonic() - started
        assert refused.status_code == 401, refused.text
        assert elapsed < 2, elapsed
        assert "Authentication Error" in _error_message(refused), refused.text
        refused_rows: Final = eventually(
            lambda: read_rows(
                "SELECT request_id, status, spend, metadata->'error_information'->>'error_code' AS error_code"
                ' FROM "LiteLLM_SpendLogs" WHERE model_group=%s AND api_key=%s',
                (model, sha256(BAD_KEY.encode()).hexdigest()),
            ),
            lambda found: len(found) == 1,
            seconds=70,
        )
        assert (refused_rows[0]["status"], refused_rows[0]["spend"], refused_rows[0]["error_code"]) == (
            "failure",
            0.0,
            "401",
        ), refused_rows
        control: Final = _chat(gateway, model, control_marker)
        control_id: Final = string_value(_payload(control)["id"])
        _assert_row(control_id, model, "success")
        landed: Final = read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,))
        assert {row["request_id"] for row in landed} == {control_id, refused_rows[0]["request_id"]}, landed
        received: Final = wire.drain()
        assert [marker_of(request) for request in received] == [control_marker], _routes(received)


@pytest.mark.parametrize("effort", [pytest.param("", id="empty"), pytest.param("x" * 5120, id="five_kb")])
def test_invalid_reasoning_effort_reaches_the_peer_and_its_400_reaches_the_caller(
    gateway: Gateway, effort: str
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, reasoning_effort=effort)
        assert response.status_code == 400, response.text
        peer_error: Final = json.dumps({"message": f"Invalid reasoning effort: {json.dumps(effort)}"})
        assert f"BedrockException - {peer_error}" in _error_message(response), response.text
        request: Final = _only_request(wire, marker)
        assert forwarded_effort(request) == effort, request.body
        _assert_row(_call_id(response), model, "failure")


NON_STRING_EFFORTS: Final = (pytest.param(7, id="int"), pytest.param(["high"], id="list"))


@pytest.mark.parametrize("effort", NON_STRING_EFFORTS)
def test_non_string_reasoning_effort_is_refused_before_any_wire_request(gateway: Gateway, effort: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, reasoning_effort=effort)
        assert response.status_code == 400, response.text
        message: Final = _error_message(response)
        assert message.startswith("litellm.UnsupportedParamsError"), response.text
        assert "reasoning_effort as a string" in message and "drop_params" in message, response.text
        _assert_row(_call_id(response), model, "failure")
        assert _routes(wire.drain()) == []


@pytest.mark.parametrize("effort", NON_STRING_EFFORTS)
def test_drop_params_deployment_drops_a_non_string_reasoning_effort(gateway: Gateway, effort: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, drop_params=True)
        response: Final = _chat(gateway, model, marker, reasoning_effort=effort)
        assert _content(response) == answer(marker), response.text
        request: Final = _only_request(wire, marker)
        assert target_of(request) == NATIVE_TARGET, request.body
        assert "reasoning_effort" not in _body(request), request.body
        _assert_row(string_value(_payload(response)["id"]), model, "success")


def test_duplicated_reasoning_effort_key_lets_the_last_value_win(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        prefix: Final = json.dumps({"model": model, "messages": _messages(marker), "cache": {"no-cache": True}})[:-1]
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            content=f'{prefix}, "reasoning_effort": "low", "reasoning_effort": "high"}}'.encode(),
            headers={"Authorization": f"Bearer {gateway.key}", "content-type": "application/json"},
        )
        assert _content(response) == answer(marker), response.text
        request: Final = _only_request(wire, marker)
        assert forwarded_effort(request) == "high", request.body
        _assert_row(string_value(_payload(response)["id"]), model, "success")


def test_string_temperature_is_refused_before_any_wire_request(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, temperature="0.2")
        assert response.status_code == 400, response.text
        message: Final = _error_message(response)
        assert message.startswith("litellm.UnsupportedParamsError") and "['temperature']" in message, response.text
        _assert_row(_call_id(response), model, "failure")
        assert _routes(wire.drain()) == []


@pytest.mark.parametrize(
    ("scripted", "expected"),
    [pytest.param(401, 401, id="401"), pytest.param(429, 429, id="429"), pytest.param(500, 503, id="500")],
)
def test_peer_error_status_reaches_the_caller_and_unrelated_deployments_keep_serving(
    gateway: Gateway, scripted: int, expected: int
) -> None:
    marker: Final = uuid.uuid4().hex
    control_marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, num_retries=0)
        unrelated: Final = scenario.model()
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"status={scripted} marker-{marker}"}],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == expected, response.text
        assert f'BedrockException - {{"message": "scripted {scripted}"}}' in _error_message(response), response.text
        _only_request(wire, marker)
        _assert_row(_call_id(response), model, "failure")
        control: Final = _chat(gateway, unrelated, control_marker)
        assert control.status_code == 200, control.text
        _assert_row(string_value(_payload(control)["id"]), unrelated, "success")
        assert _routes(wire.drain()) == []


@pytest.mark.parametrize(
    "params", [pytest.param({"reasoning_effort": None}, id="null"), pytest.param({}, id="missing")]
)
def test_absent_reasoning_effort_is_forwarded_as_absent_on_every_repeat(
    gateway: Gateway, params: dict[str, JsonValue]
) -> None:
    markers: Final = tuple(uuid.uuid4().hex for _ in range(3))
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        responses: Final = tuple(_chat(gateway, model, marker, **params) for marker in markers)
        assert [_content(response) for response in responses] == [answer(marker) for marker in markers]
        ids: Final = tuple(string_value(_payload(response)["id"]) for response in responses)
        assert len(set(ids)) == 3, ids
        received: Final = wire.drain()
        assert [marker_of(request) for request in received] == list(markers), _routes(received)
        assert [forwarded_effort(request) for request in received] == [None, None, None], [_body(r) for r in received]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id IN (%s, %s, %s)', ids
            ),
            lambda found: len(found) == 3,
            seconds=70,
        )
        assert {(string_value(row["request_id"]), row["status"]) for row in rows} == {
            (identity, "success") for identity in ids
        }, rows
