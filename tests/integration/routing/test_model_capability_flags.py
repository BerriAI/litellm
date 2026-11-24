import asyncio
import json
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import unquote

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.anthropic_sse import error_body, message_json
from integration._support.client import Gateway, Scenario, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply, responses_reply
from integration._support.process import WorkerHealthcheck, graceful_stop_seconds, owned_proxy_process
from integration._support.provider import MODEL_DISCOVERY
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_MARKER: Final = re.compile(r"capflag-[0-9a-f]{32}")
_FLAG: Final = re.compile(r"^(supports_|bedrock_output_config_effort_ceiling$)")
_OPENAI_MODEL: Final = "openai/gpt-4o-mini"
_EMBEDDING_MODEL: Final = "openai/text-embedding-3-small"
_CLAUDE_MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"
_BEDROCK_ID: Final = "anthropic.claude-3-haiku-20240307-v1:0"
_PROVIDER_KEY: Final = "integration-provider-key"
_REVOKED_KEY: Final = "sk-revoked-capflag"
_AWS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
        "aws_secret_access_key": "scripted-secret",
        "aws_region_name": "us-east-1",
    }
)
_VISION: Final[Mapping[str, JsonValue]] = MappingProxyType({"supports_vision": True})
_EMBEDDING: Final = [0.25, 0.5]
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_FRESH: Final = httpx.Limits(max_keepalive_connections=0)


@dataclass(frozen=True, slots=True)
class _Vendor:
    wire: Wire
    outage: threading.Event

    def outbound(self) -> tuple[Request, ...]:
        return tuple(request for request in self.wire.drain() if (request.method, request.target) != MODEL_DISCOVERY)


@dataclass(frozen=True, slots=True)
class _Exchange:
    marker: str
    text: str
    request: Request


Send = Callable[[str, str], str]


def _marker() -> str:
    return f"capflag-{uuid.uuid4().hex}"


def _echo(marker: str) -> str:
    return f"echo {marker}"


def _body(request: Request) -> dict[str, JsonValue]:
    return _JSON.validate_json(request.body) if request.body else {}


def _flags(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, dict):
        own: Final = tuple(key for key in value if _FLAG.match(key))
        return own + tuple(chain.from_iterable(_flags(item) for item in value.values()))
    if isinstance(value, list):
        return tuple(chain.from_iterable(_flags(item) for item in value))
    return ()


def _openai_error(status: int, message: str, code: str, param: str | None) -> Reply:
    error: Final = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    return Reply(status=status, body=json.dumps({"error": error}).encode())


def _rejection(target: str, flag: str) -> Reply:
    if target.startswith("/model/"):
        return Reply(
            status=400,
            body=json.dumps(
                {"message": f"The model returned the following errors: {flag}: Extra inputs are not permitted"}
            ).encode(),
            headers=MappingProxyType({"x-amzn-ErrorType": "ValidationException"}),
        )
    if target == "/v1/messages":
        return Reply(status=400, body=error_body(400, f"{flag}: Extra inputs are not permitted"))
    return _openai_error(400, f"Unknown parameter: '{flag}'.", "unknown_parameter", flag)


def _converse_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _converse_stream_reply(text: str) -> Reply:
    events: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"delta": {"text": text}, "contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
    )
    return Reply(
        content_type="application/vnd.amazon.eventstream",
        chunks=tuple(
            aws_event_stream_frame(
                {":event-type": kind, ":content-type": "application/json", ":message-type": "event"},
                json.dumps(payload).encode(),
            )
            for kind, payload in events
        ),
    )


def _embedding_reply(model: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": _EMBEDDING}],
                "model": model,
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        ).encode()
    )


def _answer(target: str, body: Mapping[str, JsonValue], marker: str) -> Reply:
    text: Final = _echo(marker)
    model: Final = str(body.get("model", "scripted"))
    streamed: Final = body.get("stream") is True
    match target:
        case "/v1/chat/completions":
            return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", model, text, stream=streamed)
        case "/v1/responses":
            return responses_reply(f"resp_{uuid.uuid4().hex}", model, text, stream=streamed)
        case "/v1/embeddings":
            return _embedding_reply(model)
        case "/v1/messages":
            return Reply(body=message_json(f"msg_{uuid.uuid4().hex}", model, text))
        case _ if target.endswith("/converse-stream"):
            return _converse_stream_reply(text)
        case _ if target.endswith("/converse"):
            return _converse_reply(text)
        case _ if target.endswith("/invoke"):
            return Reply(body=message_json(f"msg_{uuid.uuid4().hex}", "claude-3-haiku", text))
        case _:
            return Reply(status=404, body=b'{"error": "unscripted target"}')


def _vendor(outage: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if outage.is_set():
            return Reply(drop_connection=True)
        if request.headers.get("authorization", "") == f"Bearer {_REVOKED_KEY}":
            return _openai_error(401, "Incorrect API key provided: sk-revo****flag.", "invalid_api_key", None)
        target: Final = unquote(request.target.split("?")[0])
        body: Final = _body(request)
        flags: Final = _flags(body)
        if flags:
            return _rejection(target, flags[0])
        found: Final = _MARKER.search(request.body.decode())
        return _answer(target, body, found.group(0) if found else "probe")

    return answering_model_discovery(respond)


@pytest.fixture
def vendor() -> Iterator[_Vendor]:
    outage: Final = threading.Event()
    with wire_server(_vendor(outage)) as wire:
        yield _Vendor(wire, outage)


def _proxy(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{_proxy(gateway)}/v1", api_key=gateway.key, max_retries=0)


def _chat(gateway: Gateway, extra_body: Mapping[str, JsonValue] | None = None) -> Send:
    def send(model: str, marker: str) -> str:
        with _openai_client(gateway) as client:
            completion: Final = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": marker}], extra_body=dict(extra_body or {})
            )
        return completion.choices[0].message.content or ""

    return send


def _chat_stream(gateway: Gateway) -> Send:
    async def streamed(model: str, marker: str) -> str:
        async with openai.AsyncOpenAI(base_url=f"{_proxy(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
            stream: Final = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": marker}], stream=True
            )
            return "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])

    def send(model: str, marker: str) -> str:
        return asyncio.run(streamed(model, marker))

    return send


def _responses(gateway: Gateway) -> Send:
    def send(model: str, marker: str) -> str:
        with _openai_client(gateway) as client:
            return client.responses.create(model=model, input=marker).output_text

    return send


def _embeddings(gateway: Gateway) -> Send:
    def send(model: str, marker: str) -> str:
        with _openai_client(gateway) as client:
            return json.dumps(client.embeddings.create(model=model, input=marker).data[0].embedding)

    return send


def _messages(gateway: Gateway) -> Send:
    def send(model: str, marker: str) -> str:
        with anthropic.Anthropic(base_url=_proxy(gateway), api_key=gateway.key, max_retries=0) as client:
            message: Final = client.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": marker}]
            )
        return "".join(block.text for block in message.content if block.type == "text")

    return send


def _family(name: str, wire: Wire) -> dict[str, JsonValue]:
    match name:
        case "openai":
            return {"model": _OPENAI_MODEL, "api_base": f"{wire.url}/v1"}
        case "anthropic":
            return {"model": _CLAUDE_MODEL, "api_base": wire.url, "api_key": "sk-ant-scripted"}
        case "converse":
            return {"model": f"bedrock/converse/{_BEDROCK_ID}", "api_base": wire.url, "api_key": None, **_AWS}
        case "invoke":
            return {
                "model": f"bedrock/invoke/{_BEDROCK_ID}",
                "api_base": wire.url,
                "aws_bedrock_runtime_endpoint": wire.url,
                "api_key": None,
                **_AWS,
            }
        case _:
            raise AssertionError(f"unknown family {name}")


def _deployment(
    scenario: Scenario,
    params: Mapping[str, JsonValue],
    model_info: Mapping[str, JsonValue] | None = None,
    name: str | None = None,
) -> str:
    model_name: Final = name or f"capability-flag-{uuid.uuid4().hex}"
    body: Final[dict[str, JsonValue]] = {
        "model_name": model_name,
        "litellm_params": {"api_key": _PROVIDER_KEY, **params},
        "model_info": dict(model_info or {}),
    }
    created: Final = scenario.gateway.post("/model/new", body)
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return model_name


def _idle(gateway: Gateway) -> dict[str, JsonValue]:
    return {"model": _OPENAI_MODEL, "api_base": f"{gateway.upstream_url}/v1"}


def _pair(scenario: Scenario, params: Mapping[str, JsonValue], flag: Mapping[str, JsonValue]) -> tuple[str, str]:
    return _deployment(scenario, {**params, **flag}), _deployment(scenario, params)


def _exchange(vendor: _Vendor, send: Send, model: str) -> _Exchange:
    marker: Final = _marker()
    text: Final = send(model, marker)
    (request,) = vendor.outbound()
    return _Exchange(marker, text, request)


def _shape(request: Request) -> tuple[str, JsonValue]:
    return unquote(request.target), json.loads(_MARKER.sub("capflag-marker", request.body.decode()))


def _probe_shape(request: Request) -> tuple[str, dict[str, JsonValue]]:
    return unquote(request.target), {key: value for key, value in _body(request).items() if key != "messages"}


def _assert_sends_the_control_request(
    vendor: _Vendor, send: Send, flagged: str, control: str, expected: Callable[[str], str] = _echo
) -> None:
    sent: Final = _exchange(vendor, send, flagged)
    baseline: Final = _exchange(vendor, send, control)
    assert sent.text == expected(sent.marker), sent.text
    assert baseline.text == expected(baseline.marker), baseline.text
    assert _flags(_body(sent.request)) == (), sent.request.body
    assert _shape(sent.request) == _shape(baseline.request)


def _entry(gateway: Gateway, path: str, name: str, params: Mapping[str, str] | None = None) -> dict[str, JsonValue]:
    entries: Final = list_value(gateway.get(path, params)["data"])
    (found,) = tuple(object_value(entry) for entry in entries if object_value(entry)["model_name"] == name)
    return found


def _model_id(gateway: Gateway, name: str) -> str:
    return string_value(object_value(_entry(gateway, "/model/info", name)["model_info"])["id"])


def _health(gateway: Gateway, name: str) -> tuple[int, JsonValue, JsonValue]:
    response: Final = gateway.request("GET", "/health", params={"model": name})
    health: Final = _JSON.validate_json(response.content)
    return response.status_code, health["healthy_count"], health["unhealthy_count"]


def _fresh_request(gateway: Gateway, method: str, path: str, body: Mapping[str, JsonValue] | None) -> httpx.Response:
    with httpx.Client(base_url=_proxy(gateway), timeout=60, trust_env=False, limits=_FRESH) as client:
        return client.request(method, path, json=body, headers={"Authorization": f"Bearer {gateway.key}"})


def _fresh_chat(gateway: Gateway, model: str, marker: str) -> str:
    response: Final = _fresh_request(
        gateway, "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
    )
    assert response.status_code == 200, response.text
    choice: Final = object_value(list_value(_JSON.validate_json(response.content)["choices"])[0])
    return string_value(object_value(choice["message"])["content"])


@pytest.mark.parametrize(
    ("family", "flag"),
    [
        ("openai", {"supports_vision": True}),
        ("openai", {"supports_speed": True}),
        ("openai", {"bedrock_output_config_effort_ceiling": "high"}),
        ("anthropic", {"supports_vision": True}),
        ("converse", {"supports_max_reasoning_effort": False}),
        ("invoke", {"supports_max_reasoning_effort": False}),
    ],
    ids=["openai-static", "openai-catalog_only", "openai-ceiling", "anthropic", "bedrock_converse", "bedrock_invoke"],
)
def test_chat_completions_send_a_flagged_deployment_the_control_request(
    gateway: Gateway, vendor: _Vendor, family: str, flag: Mapping[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        flagged, control = _pair(scenario, _family(family, vendor.wire), flag)
        _assert_sends_the_control_request(vendor, _chat(gateway), flagged, control)


@pytest.mark.parametrize("family", ["openai", "converse"])
def test_streamed_chat_completions_send_a_flagged_deployment_the_control_request(
    gateway: Gateway, vendor: _Vendor, family: str
) -> None:
    flag: Final = {"supports_max_reasoning_effort": False} if family == "converse" else _VISION
    with gateway.scenario() as scenario:
        flagged, control = _pair(scenario, _family(family, vendor.wire), flag)
        _assert_sends_the_control_request(vendor, _chat_stream(gateway), flagged, control)


@pytest.mark.parametrize("family", ["openai", "anthropic", "converse"])
def test_messages_send_a_flagged_deployment_the_control_request(gateway: Gateway, vendor: _Vendor, family: str) -> None:
    with gateway.scenario() as scenario:
        flagged, control = _pair(scenario, _family(family, vendor.wire), _VISION)
        _assert_sends_the_control_request(vendor, _messages(gateway), flagged, control)


def test_responses_send_a_flagged_deployment_the_control_request(gateway: Gateway, vendor: _Vendor) -> None:
    with gateway.scenario() as scenario:
        flagged, control = _pair(scenario, _family("openai", vendor.wire), _VISION)
        _assert_sends_the_control_request(vendor, _responses(gateway), flagged, control)


def test_embeddings_send_a_flagged_deployment_the_control_request(gateway: Gateway, vendor: _Vendor) -> None:
    params: Final = {"model": _EMBEDDING_MODEL, "api_base": f"{vendor.wire.url}/v1"}
    with gateway.scenario() as scenario:
        flagged, control = _pair(scenario, params, {"supports_embedding_image_input": True})
        _assert_sends_the_control_request(
            vendor, _embeddings(gateway), flagged, control, expected=lambda _marker: json.dumps(_EMBEDDING)
        )


def _wildcard(scenario: Scenario, wire: Wire, flag: Mapping[str, JsonValue]) -> str:
    prefix: Final = f"capflag{uuid.uuid4().hex}"
    _deployment(scenario, {"model": "openai/*", "api_base": f"{wire.url}/v1", **flag}, name=f"{prefix}/*")
    return f"{prefix}/gpt-4o-mini"


def test_a_flagged_wildcard_deployment_sends_the_control_request(gateway: Gateway, vendor: _Vendor) -> None:
    with gateway.scenario() as scenario:
        flagged: Final = _wildcard(scenario, vendor.wire, _VISION)
        control: Final = _wildcard(scenario, vendor.wire, {})
        _assert_sends_the_control_request(vendor, _chat(gateway), flagged, control)


def test_health_probes_a_flagged_deployment_with_the_control_probe(gateway: Gateway, vendor: _Vendor) -> None:
    with gateway.scenario() as scenario:
        flagged, control = _pair(scenario, _family("openai", vendor.wire), _VISION)
        flagged_health: Final = _health(gateway, flagged)
        (flagged_probe,) = vendor.outbound()
        control_health: Final = _health(gateway, control)
        (control_probe,) = vendor.outbound()
    assert flagged_health == control_health == (200, 1, 0)
    assert _flags(_body(flagged_probe)) == (), flagged_probe.body
    assert _probe_shape(flagged_probe) == _probe_shape(control_probe)


def test_test_connection_probes_flagged_params_with_the_unflagged_probe(gateway: Gateway, vendor: _Vendor) -> None:
    params: Final[dict[str, JsonValue]] = {
        "model": _OPENAI_MODEL,
        "api_base": f"{vendor.wire.url}/v1",
        "api_key": _PROVIDER_KEY,
    }
    flagged: Final = gateway.post("/health/test_connection", {"mode": "chat", "litellm_params": {**params, **_VISION}})
    (flagged_probe,) = vendor.outbound()
    unflagged: Final = gateway.post("/health/test_connection", {"mode": "chat", "litellm_params": params})
    (unflagged_probe,) = vendor.outbound()
    assert (flagged["status"], unflagged["status"]) == ("success", "success"), (flagged, unflagged)
    assert _flags(_body(flagged_probe)) == (), flagged_probe.body
    assert _probe_shape(flagged_probe) == _probe_shape(unflagged_probe)


@pytest.mark.parametrize(
    "value",
    [7, ["vision"], "", "v" * 5120, None, False],
    ids=["int", "list", "empty_string", "five_kb", "null", "false"],
)
def test_any_flag_value_sends_the_control_request(gateway: Gateway, vendor: _Vendor, value: JsonValue) -> None:
    with gateway.scenario() as scenario:
        flagged, control = _pair(scenario, _family("openai", vendor.wire), {"supports_vision": value})
        _assert_sends_the_control_request(vendor, _chat(gateway), flagged, control)


def test_a_flag_in_litellm_params_and_model_info_keeps_the_model_info_value(gateway: Gateway, vendor: _Vendor) -> None:
    with gateway.scenario() as scenario:
        flagged: Final = _deployment(
            scenario, {**_family("openai", vendor.wire), **_VISION}, {"supports_vision": False}
        )
        control: Final = _deployment(scenario, _family("openai", vendor.wire))
        _assert_sends_the_control_request(vendor, _chat(gateway), flagged, control)
        model_info: Final = object_value(_entry(gateway, "/model/info", flagged)["model_info"])
    assert model_info["supports_vision"] is False, model_info


@pytest.mark.parametrize("key", ["supports_litellm_qa_unknown", "supports_system_prompt"])
def test_an_undeclared_or_provider_supports_key_still_reaches_the_vendor(
    gateway: Gateway, vendor: _Vendor, key: str
) -> None:
    with gateway.scenario() as scenario:
        model: Final = _deployment(scenario, {**_family("openai", vendor.wire), key: True})
        with pytest.raises(openai.BadRequestError) as raised:
            _chat(gateway)(model, _marker())
        (request,) = vendor.outbound()
    assert f"Unknown parameter: '{key}'." in str(raised.value), raised.value
    assert _flags(_body(request)) == (key,), request.body


def test_a_flagged_deployment_with_a_revoked_key_returns_the_vendor_401(gateway: Gateway, vendor: _Vendor) -> None:
    with gateway.scenario() as scenario:
        flagged: Final = _deployment(scenario, {**_family("openai", vendor.wire), "api_key": _REVOKED_KEY, **_VISION})
        control: Final = _deployment(scenario, _family("openai", vendor.wire))
        with pytest.raises(openai.AuthenticationError) as raised:
            _chat(gateway)(flagged, _marker())
        assert len(vendor.outbound()) == 1
        marker: Final = _marker()
        answered: Final = _chat(gateway)(control, marker)
    assert "Incorrect API key provided" in str(raised.value), raised.value
    assert answered == _echo(marker)


def test_a_fallback_to_a_flagged_deployment_sends_the_control_request(gateway: Gateway, vendor: _Vendor) -> None:
    with gateway.scenario() as scenario:
        broken: Final = _deployment(scenario, {**_family("openai", vendor.wire), "api_key": _REVOKED_KEY})
        flagged, control = _pair(scenario, _family("openai", vendor.wire), _VISION)
        marker: Final = _marker()
        answered: Final = _chat(gateway, {"fallbacks": [flagged]})(broken, marker)
        rejected, fallback = vendor.outbound()
        baseline: Final = _exchange(vendor, _chat(gateway), control)
    assert answered == _echo(marker)
    assert rejected.headers["authorization"] == f"Bearer {_REVOKED_KEY}"
    assert _flags(_body(fallback)) == (), fallback.body
    assert _shape(fallback) == _shape(baseline.request)


@pytest.mark.parametrize(
    ("path", "by_name"), [("/model/info", False), ("/v2/model/info", True)], ids=["model_info", "v2_model_info"]
)
def test_model_info_lists_the_served_params_without_the_flag(gateway: Gateway, path: str, by_name: bool) -> None:
    with gateway.scenario() as scenario:
        flagged: Final = _deployment(scenario, {**_idle(gateway), **_VISION})
        entry: Final = _entry(gateway, path, flagged, {"search": flagged} if by_name else None)
    litellm_params: Final = object_value(entry["litellm_params"])
    assert litellm_params["model"] == _OPENAI_MODEL, litellm_params
    assert "supports_vision" not in litellm_params, litellm_params


def test_model_group_info_reports_a_flagged_group_like_a_plain_one(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        flagged: Final = _deployment(scenario, {**_idle(gateway), **_VISION})
        plain: Final = _deployment(scenario, _idle(gateway))
        groups: Final = tuple(
            object_value(list_value(gateway.get("/model_group/info", {"model_group": name})["data"])[0])
            for name in (flagged, plain)
        )
    flagged_group, plain_group = (
        {key: value for key, value in group.items() if key != "model_group"} for group in groups
    )
    assert flagged_group == plain_group


def test_supports_system_message_false_still_folds_the_system_prompt(gateway: Gateway, vendor: _Vendor) -> None:
    instruction: Final = f"answer briefly {uuid.uuid4().hex}"
    marker: Final = _marker()
    with gateway.scenario() as scenario:
        model: Final = _deployment(scenario, {**_family("openai", vendor.wire), "supports_system_message": False})
        with _openai_client(gateway) as client:
            completion: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": instruction}, {"role": "user", "content": marker}],
            )
        (request,) = vendor.outbound()
    messages: Final = list_value(_body(request)["messages"])
    assert completion.choices[0].message.content == _echo(marker)
    assert [object_value(message)["role"] for message in messages] == ["user"], messages
    assert instruction in json.dumps(messages) and marker in json.dumps(messages), messages
    assert _flags(_body(request)) == (), request.body


def test_three_identical_requests_to_a_flagged_deployment_each_log_once(gateway: Gateway, vendor: _Vendor) -> None:
    marker: Final = _marker()
    with gateway.scenario() as scenario:
        flagged: Final = _deployment(scenario, {**_family("openai", vendor.wire), **_VISION})
        with _openai_client(gateway) as client:
            identities: Final = tuple(
                client.chat.completions.create(
                    model=flagged,
                    messages=[{"role": "user", "content": marker}],
                    extra_body={"cache": {"no-cache": True}},
                ).id
                for _ in range(3)
            )
        received: Final = vendor.outbound()
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id IN (%s, %s, %s)', identities
            ),
            lambda found: sorted(str(row["request_id"]) for row in found) == sorted(identities),
            seconds=70,
        )
    assert len(set(identities)) == 3, identities
    assert len(rows) == 3, rows
    assert len(received) == 3 and len({request.body for request in received}) == 1, received
    assert _flags(_body(received[0])) == (), received[0].body


def _rpm_on_fresh_connections(gateway: Gateway, name: str) -> tuple[JsonValue, ...]:
    def read() -> JsonValue:
        response: Final = _fresh_request(gateway, "GET", "/model/info", None)
        entries: Final = list_value(_JSON.validate_json(response.content)["data"])
        (entry,) = tuple(object_value(item) for item in entries if object_value(item)["model_name"] == name)
        return object_value(entry["litellm_params"]).get("rpm")

    return tuple(read() for _ in range(8))


def test_a_flag_patched_onto_a_db_deployment_never_reaches_the_vendor(gateway: Gateway, vendor: _Vendor) -> None:
    with gateway.scenario() as scenario:
        patched: Final = _deployment(scenario, _family("openai", vendor.wire))
        control: Final = _deployment(scenario, _family("openai", vendor.wire))
        response: Final = gateway.request(
            "PATCH",
            f"/model/{_model_id(gateway, patched)}/update",
            {"litellm_params": {"supports_vision": True, "rpm": 7}},
        )
        assert response.status_code == 200, response.text
        eventually(lambda: _rpm_on_fresh_connections(gateway, patched), lambda seen: seen == (7,) * 8, seconds=60)
        for _ in range(3):
            _assert_sends_the_control_request(vendor, _chat(gateway), patched, control)


def test_updates_during_a_burst_never_send_the_flag(gateway: Gateway, vendor: _Vendor) -> None:
    markers: Final = tuple(_marker() for _ in range(20))
    with gateway.scenario() as scenario:
        flagged: Final = _deployment(scenario, {**_family("openai", vendor.wire), **_VISION})
        model_id: Final = _model_id(gateway, flagged)
        send: Final = _chat(gateway)
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers: Final = tuple(pool.submit(send, flagged, marker) for marker in markers)
            updates: Final = tuple(
                gateway.request(
                    "PATCH", f"/model/{model_id}/update", {"litellm_params": {"rpm": rpm, "supports_vision": True}}
                ).status_code
                for rpm in (11, 12, 13)
            )
            texts: Final = tuple(answer.result() for answer in answers)
        received: Final = vendor.outbound()
    assert updates == (200, 200, 200), updates
    assert texts == tuple(_echo(marker) for marker in markers), texts
    assert len(received) == 20, received
    assert [_flags(_body(request)) for request in received] == [()] * 20, received


def _config_with_flagged_model(directory: Path, name: str, wire: Wire) -> Path:
    base: Final = _JSON.validate_python(yaml.safe_load(_PROXY_CONFIG.read_text()))
    flagged: Final[dict[str, JsonValue]] = {
        "model_name": name,
        "litellm_params": {"model": _OPENAI_MODEL, "api_key": _PROVIDER_KEY, "api_base": f"{wire.url}/v1", **_VISION},
    }
    path: Final = directory / "capability_flags_config.yaml"
    path.write_text(yaml.safe_dump({**base, "model_list": [*list_value(base["model_list"]), flagged]}))
    return path


@pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120))
def test_each_worker_and_a_replacement_send_config_and_db_flagged_deployments_without_the_flag(
    gateway: Gateway, vendor: _Vendor, tmp_path: Path
) -> None:
    configured: Final = f"capflag-config-{uuid.uuid4().hex}"
    config: Final = _config_with_flagged_model(tmp_path, configured, vendor.wire)
    with owned_proxy_process(
        gateway, tmp_path, {}, config=config, workers=2, healthcheck=WorkerHealthcheck.LENIENT
    ) as owned:
        with owned.gateway.scenario() as scenario:
            stored: Final = _deployment(scenario, {**_family("openai", vendor.wire), **_VISION})
            stored_id: Final = _model_id(owned.gateway, stored)
            eventually(lambda: owned.serves_only_on_every_worker(stored, stored_id), bool, seconds=120)
            first, second = sorted(eventually(owned.worker_pids, lambda pids: len(pids) == 2))
            with owned.suspended(second):
                on_first: Final = tuple((model, _marker()) for model in (configured, stored))
                first_texts: Final = tuple(_fresh_chat(owned.gateway, model, marker) for model, marker in on_first)
            with owned.suspended(first):
                on_second: Final = tuple((model, _marker()) for model in (configured, stored))
                second_texts: Final = tuple(_fresh_chat(owned.gateway, model, marker) for model, marker in on_second)
            owned.kill_worker(first)
            eventually(
                lambda: owned.serves_only_on_every_worker(stored, stored_id),
                bool,
                seconds=graceful_stop_seconds() + 60,
            )
            replacement: Final = next(iter(owned.worker_pids() - {second}))
            with owned.suspended(second):
                on_replacement: Final = tuple((model, _marker()) for model in (configured, stored))
                replacement_texts: Final = tuple(
                    _fresh_chat(owned.gateway, model, marker) for model, marker in on_replacement
                )
            assert replacement not in (first, second)
        received: Final = vendor.outbound()
    sent: Final = (*on_first, *on_second, *on_replacement)
    assert (*first_texts, *second_texts, *replacement_texts) == tuple(_echo(marker) for _, marker in sent)
    assert len(received) == 6, received
    assert [_flags(_body(request)) for request in received] == [()] * 6, received


def _burst(send: Send, model: str, size: int) -> tuple[str, ...]:
    def attempt(marker: str) -> str:
        try:
            return "ok" if send(model, marker) == _echo(marker) else "wrong text"
        except openai.APIError as error:
            return type(error).__name__

    with ThreadPoolExecutor(max_workers=size) as pool:
        return tuple(pool.map(attempt, tuple(_marker() for _ in range(size))))


def test_a_vendor_outage_between_bursts_reports_unhealthy_then_healthy_and_never_sends_the_flag(
    gateway: Gateway, vendor: _Vendor
) -> None:
    with gateway.scenario() as scenario:
        flagged: Final = _deployment(scenario, {**_family("openai", vendor.wire), **_VISION})
        send: Final = _chat(gateway)
        before: Final = _burst(send, flagged, 10)
        healthy_before: Final = _health(gateway, flagged)
        vendor.outage.set()
        during: Final = _burst(send, flagged, 10)
        down: Final = _health(gateway, flagged)
        vendor.outage.clear()
        up: Final = _health(gateway, flagged)
        after: Final = _burst(send, flagged, 10)
        received: Final = vendor.outbound()
    assert before == after == ("ok",) * 10, (before, after)
    assert "ok" not in during, during
    assert (healthy_before, down, up) == ((200, 1, 0), (503, 0, 1), (200, 1, 0))
    assert [_flags(_body(request)) for request in received] == [()] * len(received), received
