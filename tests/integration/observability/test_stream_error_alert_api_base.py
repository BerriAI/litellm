import asyncio
import hashlib
import json
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import openai
import pytest
import yaml
from anthropic.types import RawContentBlockDeltaEvent, TextDelta
from integration._support.anthropic_sse import DROP_PAUSE, dropping_reply, stream_reply
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_stream import created, delta, failed, frame
from integration._support.responses_ws_peer import PROVIDER_MODEL, Script, events, responses_peer, session
from integration._support.tls import write_self_signed_cert
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import ResponseFailedEvent, ResponseTextDeltaEvent
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120), pytest.mark.shares_worker_state]

AZURE_KEY: Final = "synthetic-azure-key"
API_VERSION: Final = "2025-04-01-preview"
AZURE_CHAT_BACKEND: Final = "azure/gpt-4o-mini"
PARTIAL_TEXT: Final = "partial answer"
ALERT_SECONDS: Final = 40
NO_RETRY_NO_CACHE: Final[Mapping[str, JsonValue]] = MappingProxyType({"num_retries": 0, "cache": {"no-cache": True}})
EMPTY_MODEL_LIST: Final = Reply(body=b'{"object": "list", "data": []}')
VERTEX_PROJECT: Final = "alert-vertex-project"
VERTEX_LOCATION: Final = "us-east5"
VERTEX_PARTNER_BACKEND: Final = "claude-opus-5-5@default"
VERTEX_PARTNER_URL: Final = (
    f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/{VERTEX_PROJECT}"
    f"/locations/{VERTEX_LOCATION}/publishers/anthropic/models/{VERTEX_PARTNER_BACKEND}"
)
VERTEX_OVERLOADED: Final = Reply(
    status=500, body=b'{"error": {"code": 500, "message": "vertex partner overloaded", "status": "INTERNAL"}}'
)

API_BASES: Final[Mapping[str, Callable[[str, str], str]]] = MappingProxyType(
    {
        "plain": lambda url, _: f"{url}/gateway",
        "code": lambda url, secret: f"{url}/gateway?code={secret}",
        "subscription-key": lambda url, secret: f"{url}/gateway?subscription-key={secret}",
        "userinfo": lambda url, secret: url.replace("://", f"://alice:{secret}@", 1) + "/gateway",
    }
)


@dataclass(frozen=True, slots=True)
class AlertingProxy:
    gateway: Gateway
    alerts: Wire
    vertex: Wire
    cert: tuple[Path, Path]


@dataclass(frozen=True, slots=True)
class Caller:
    key: str
    key_alias: str
    team_alias: str


@dataclass(frozen=True, slots=True)
class Outcome:
    text: str
    failure: str


def accept_alert(request: Request) -> Reply:
    return Reply(body=b"ok", content_type="text/plain")


def overloaded_vertex(request: Request) -> Reply:
    return VERTEX_OVERLOADED


def alerting_config(directory: Path) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {
            **object_value(base["general_settings"]),
            "alerting": ["slack"],
            "alert_types": ["llm_exceptions"],
        },
    }
    path: Final = directory / "alerting.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def alerting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AlertingProxy]:
    directory: Final = tmp_path_factory.mktemp("stream-error-alerts")
    cert: Final = write_self_signed_cert(directory)
    with (
        gateway_from_environment() as shared,
        wire_server(accept_alert) as alerts,
        wire_server(overloaded_vertex) as vertex,
    ):
        with owned_proxy(
            shared,
            directory,
            {
                "SLACK_WEBHOOK_URL": alerts.url,
                "DEFAULT_FLUSH_INTERVAL_SECONDS": "1",
                "SSL_CERT_FILE": str(cert[0]),
                "VERTEXAI_API_BASE": vertex.url,
            },
            config=alerting_config(directory),
        ) as gateway:
            yield AlertingProxy(gateway, alerts, vertex, cert)


def alert_naming(alerts: Wire, marker: str) -> str:
    def matching() -> tuple[str, ...]:
        texts: Final = (string_value(JSON_OBJECT.validate_json(request.body)["text"]) for request in alerts.drain())
        return tuple(text for text in texts if marker in text)

    return eventually(matching, lambda found: len(found) > 0, seconds=ALERT_SECONDS)[0]


def api_base_line(alert: str) -> str | None:
    return next((line for line in alert.splitlines() if line.startswith("API Base:")), None)


def assert_names_deployment(alert: str, api_base: str, model_group: str, backend: str, caller: Caller) -> None:
    assert api_base_line(alert) == f"API Base: `{api_base}`", alert
    assert f"model_group: `{model_group}`" in alert, alert
    assert f"deployment: `{backend}`" in alert, alert
    assert f"Key Name: `{caller.key_alias}`\nTeam: `{caller.team_alias}`" in alert, alert


def keyed_caller(scenario: Scenario) -> Caller:
    team_alias: Final = f"alert-team-{uuid.uuid4().hex[:12]}"
    key_alias: Final = f"alert-key-{uuid.uuid4().hex[:12]}"
    team: Final = scenario.team(team_alias=team_alias)
    return Caller(scenario.key(key_alias=key_alias, team_id=team), key_alias, team_alias)


def azure_chat(scenario: Scenario, api_base: str, *, bridged: bool = False) -> str:
    return scenario.model(
        model=AZURE_CHAT_BACKEND,
        api_base=api_base,
        api_key=AZURE_KEY,
        api_version=API_VERSION,
        num_retries=0,
        use_chat_completions_api=bridged,
    )


def provider_message(marker: str) -> str:
    return f"upstream overloaded {marker}"


def chat_chunk(marker: str, content: Mapping[str, JsonValue]) -> bytes:
    body: Final = {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": f"gpt-4o-mini-{marker}",
        "choices": [{"index": 0, "delta": dict(content), "finish_reason": None}],
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def first_text(marker: str) -> bytes:
    return chat_chunk(marker, {"role": "assistant", "content": PARTIAL_TEXT})


def cut_after_text(marker: str) -> Reply:
    return dropping_reply((first_text(marker), chat_chunk(marker, {"content": " never sent"})), abort_after=1)


def error_frame_after_text(marker: str) -> Reply:
    error: Final = {"error": {"message": provider_message(marker), "type": "server_error", "code": "server_error"}}
    return stream_reply((first_text(marker), b"data: " + json.dumps(error).encode() + b"\n\n"), pause=DROP_PAUSE)


def rejected_before_stream(marker: str) -> Reply:
    return Reply(
        status=500, body=json.dumps({"error": {"message": provider_message(marker), "type": "server_error"}}).encode()
    )


def chat_upstream(reply: Callable[[str], Reply], marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return EMPTY_MODEL_LIST
        return reply(marker)

    return respond


def items_or_error(open_stream: Callable[[], Iterable[object]], error_type: type[Exception]) -> Iterator[object]:
    try:
        yield from open_stream()
    except error_type as error:
        yield error


def chunk_text(item: object) -> str:
    if not isinstance(item, ChatCompletionChunk):
        return ""
    return "".join(choice.delta.content or "" for choice in item.choices)


def message_text(item: object) -> str:
    if isinstance(item, RawContentBlockDeltaEvent) and isinstance(item.delta, TextDelta):
        return item.delta.text
    return ""


def responses_text(item: object) -> str:
    return item.delta if isinstance(item, ResponseTextDeltaEvent) else ""


def failure_of(item: object) -> str:
    if isinstance(item, Exception):
        return str(item)
    if isinstance(item, ResponseFailedEvent) and item.response.error is not None:
        return item.response.error.message
    return ""


def outcome(items: tuple[object, ...], text_of: Callable[[object], str]) -> Outcome:
    return Outcome("".join(text_of(item) for item in items), "".join(failure_of(item) for item in items))


def openai_client(gateway: Gateway, key: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0, timeout=60)


def chat_outcome(gateway: Gateway, key: str, model: str) -> Outcome:
    with openai_client(gateway, key) as client:
        items: Final = tuple(
            items_or_error(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "stream an answer"}],
                    stream=True,
                    extra_body=dict(NO_RETRY_NO_CACHE),
                ),
                openai.APIError,
            )
        )
        return outcome(items, chunk_text)


def messages_outcome(gateway: Gateway, key: str, model: str) -> Outcome:
    with anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=key, max_retries=0, timeout=60) as client:
        items: Final = tuple(
            items_or_error(
                lambda: client.messages.create(
                    model=model,
                    max_tokens=64,
                    messages=[{"role": "user", "content": "stream an answer"}],
                    stream=True,
                    extra_body=dict(NO_RETRY_NO_CACHE),
                ),
                anthropic.APIError,
            )
        )
        return outcome(items, message_text)


def responses_outcome(gateway: Gateway, key: str, model: str) -> Outcome:
    with openai_client(gateway, key) as client:
        items: Final = tuple(
            items_or_error(
                lambda: client.responses.create(
                    model=model, input="stream an answer", stream=True, extra_body=dict(NO_RETRY_NO_CACHE)
                ),
                openai.APIError,
            )
        )
        return outcome(items, responses_text)


ENTRYPOINTS: Final[Mapping[str, Callable[[Gateway, str, str], Outcome]]] = MappingProxyType(
    {"chat": chat_outcome, "messages": messages_outcome, "responses-bridge": responses_outcome}
)
FAILURES: Final[Mapping[str, Callable[[str], Reply]]] = MappingProxyType(
    {"cut": cut_after_text, "error-frame": error_frame_after_text}
)


@pytest.mark.parametrize(
    ("entrypoint", "failure"),
    [("chat", "cut"), ("chat", "error-frame"), ("messages", "error-frame"), ("responses-bridge", "error-frame")],
)
def test_mid_stream_failure_alert_names_the_deployment(alerting: AlertingProxy, entrypoint: str, failure: str) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(chat_upstream(FAILURES[failure], marker)) as wire, alerting.gateway.scenario() as scenario:
        model: Final = azure_chat(scenario, wire.url, bridged=entrypoint == "responses-bridge")
        caller: Final = keyed_caller(scenario)
        seen: Final = ENTRYPOINTS[entrypoint](alerting.gateway, caller.key, model)
        assert seen.text == PARTIAL_TEXT and seen.failure, seen
        assert failure == "cut" or provider_message(marker) in seen.failure, seen
        assert_names_deployment(alert_naming(alerting.alerts, marker), wire.url, model, AZURE_CHAT_BACKEND, caller)


def test_native_responses_failed_event_alert_names_the_deployment(alerting: AlertingProxy) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"resp_{marker}"
    stream: Final = (
        frame(created(identity)),
        frame(delta(identity, PARTIAL_TEXT)),
        frame(failed(identity, "server_error", provider_message(marker))),
    )

    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return EMPTY_MODEL_LIST
        assert "/responses" in request.target, request.target
        return stream_reply(stream)

    with wire_server(respond) as wire, alerting.gateway.scenario() as scenario:
        model: Final = scenario.model(model=AZURE_CHAT_BACKEND, api_base=wire.url, api_key=AZURE_KEY, num_retries=0)
        caller: Final = keyed_caller(scenario)
        seen: Final = responses_outcome(alerting.gateway, caller.key, model)
        assert seen.text == PARTIAL_TEXT and provider_message(marker) in seen.failure, seen
        assert_names_deployment(alert_naming(alerting.alerts, marker), wire.url, model, AZURE_CHAT_BACKEND, caller)


def failing_turn(marker: str) -> Script:
    def script(
        response_id: str,
    ) -> tuple[dict[str, JsonValue], ...]:  # mutable-ok: responses_peer's Script returns dicts
        opened, text_delta, completed_event = events(response_id, PARTIAL_TEXT)
        response: Final = object_value(completed_event["response"])
        return (
            opened,
            text_delta,
            {
                "type": "response.failed",
                "response": {
                    **response,
                    "status": "failed",
                    "output": [],
                    "error": {"code": "server_error", "message": provider_message(marker)},
                },
            },
        )

    return script


def test_responses_websocket_turn_failure_alert_names_the_deployment(alerting: AlertingProxy) -> None:
    marker: Final = uuid.uuid4().hex
    backend: Final = f"openai/{PROVIDER_MODEL}"
    with responses_peer(alerting.cert, failing_turn(marker)) as peer, alerting.gateway.scenario() as scenario:
        model: Final = scenario.model(model=backend, api_base=peer.url, num_retries=0)
        caller: Final = keyed_caller(scenario)
        (received,) = asyncio.run(session(str(alerting.gateway.client.base_url), caller.key, model, (marker,)))
        assert [event["type"] for event in received] == [
            "response.created",
            "response.output_text.delta",
            "response.failed",
        ], received
        assert_names_deployment(alert_naming(alerting.alerts, marker), peer.url, model, backend, caller)
        (row,) = eventually(
            lambda: read_rows(
                'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (hashlib.sha256(caller.key.encode()).hexdigest(),),
            ),
            lambda rows: len(rows) >= 1,
            seconds=70,
        )
        assert row["status"] == "failure", row
        error_information: Final = object_value(object_value(row["metadata"])["error_information"])
        assert provider_message(marker) in string_value(error_information["error_message"]), error_information


@pytest.mark.parametrize(
    ("phase", "placement"),
    [
        ("pre-stream", "plain"),
        ("pre-stream", "code"),
        ("pre-stream", "subscription-key"),
        ("pre-stream", "userinfo"),
        ("mid-stream", "code"),
    ],
)
def test_alert_api_base_line_keeps_credentials_out(alerting: AlertingProxy, phase: str, placement: str) -> None:
    marker: Final = uuid.uuid4().hex
    secret: Final = f"credential{uuid.uuid4().hex}"
    pre_stream: Final = phase == "pre-stream"
    reply: Final = rejected_before_stream if pre_stream else error_frame_after_text
    with wire_server(chat_upstream(reply, marker)) as wire, alerting.gateway.scenario() as scenario:
        model: Final = azure_chat(scenario, API_BASES[placement](wire.url, secret))
        caller: Final = keyed_caller(scenario)
        seen: Final = chat_outcome(alerting.gateway, caller.key, model)
        assert seen.failure and seen.text == ("" if pre_stream else PARTIAL_TEXT), seen
        alert: Final = alert_naming(alerting.alerts, model if pre_stream else marker)
        assert secret not in alert, alert
        assert_names_deployment(alert, f"{wire.url}/gateway", model, AZURE_CHAT_BACKEND, caller)


def test_alert_api_base_line_keeps_a_malformed_password_out(alerting: AlertingProxy) -> None:
    password_start: Final = f"credential{uuid.uuid4().hex[:8]}"
    password_end: Final = uuid.uuid4().hex
    with alerting.gateway.scenario() as scenario:
        model: Final = azure_chat(scenario, f"http://alice:{password_start}?{password_end}@127.0.0.1:9/gateway")
        caller: Final = keyed_caller(scenario)
        seen: Final = chat_outcome(alerting.gateway, caller.key, model)
        assert seen.failure and not seen.text, seen
        line: Final = api_base_line(alert_naming(alerting.alerts, model)) or ""
        assert not any(part in line for part in ("alice", password_start, password_end)), line


def test_alert_api_base_line_names_a_vertex_partner_model_version(alerting: AlertingProxy) -> None:
    with alerting.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"vertex_ai/{VERTEX_PARTNER_BACKEND}",
            api_base=None,
            api_key=None,
            vertex_project=VERTEX_PROJECT,
            vertex_location=VERTEX_LOCATION,
            vertex_credentials=service_account_json(VERTEX_PROJECT, alerting.gateway.upstream_url.rstrip("/")),
            num_retries=0,
        )
        caller: Final = keyed_caller(scenario)
        seen: Final = chat_outcome(alerting.gateway, caller.key, model)
        assert seen.failure and not seen.text, seen
        assert any(
            f"/publishers/anthropic/models/{VERTEX_PARTNER_BACKEND}:" in request.target
            for request in alerting.vertex.drain()
        )
        line: Final = api_base_line(alert_naming(alerting.alerts, model)) or ""
        assert line.startswith(f"API Base: `{VERTEX_PARTNER_URL}:"), line


def test_fallback_mid_stream_failure_alert_names_the_fallback_deployment(alerting: AlertingProxy) -> None:
    marker: Final = uuid.uuid4().hex
    primary_reply: Final = Reply(status=500, body=b'{"error": {"message": "primary is down", "type": "server_error"}}')
    with (
        wire_server(chat_upstream(lambda _: primary_reply, marker)) as primary_wire,
        wire_server(chat_upstream(cut_after_text, marker)) as backup_wire,
        alerting.gateway.scenario() as scenario,
    ):
        primary: Final = azure_chat(scenario, primary_wire.url)
        backup: Final = azure_chat(scenario, backup_wire.url)
        caller: Final = keyed_caller(scenario)
        with openai_client(alerting.gateway, caller.key) as client:
            items: Final = tuple(
                items_or_error(
                    lambda: client.chat.completions.create(
                        model=primary,
                        messages=[{"role": "user", "content": "stream an answer"}],
                        stream=True,
                        extra_body={**NO_RETRY_NO_CACHE, "fallbacks": [backup]},
                    ),
                    openai.APIError,
                )
            )
        seen: Final = outcome(items, chunk_text)
        assert seen.text == PARTIAL_TEXT and seen.failure, seen
        assert [request.method for request in primary_wire.drain()].count("POST") == 1
        assert [request.method for request in backup_wire.drain()].count("POST") == 1
        alert: Final = alert_naming(alerting.alerts, marker)
        assert_names_deployment(alert, backup_wire.url, backup, AZURE_CHAT_BACKEND, caller)
        assert primary_wire.url not in alert, alert
