import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlparse

import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.openai_wire import answering_model_discovery, chat_reply, openai_error, responses_reply
from integration._support.otlp_sink import Span, SpanSinks, recorded_spans
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

MARKER: Final = re.compile(rb"lfenv-(?:fail-)?[0-9a-f]{32}")
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
SERVER: Final = 2
CALL_ID: Final = "litellm.call_id"
OPERATION: Final = "gen_ai.operation.name"
ENVIRONMENT: Final = "langfuse.environment"
RELEASE: Final = "langfuse.release"
VERSION: Final = "langfuse.version"
PROXY_ENVIRONMENT: Final = "itest-proxy"
Endpoint = Literal["chat", "responses", "messages"]


def _marker(kind: str = "") -> str:
    return f"lfenv-{kind}{uuid.uuid4().hex}"


def _upstream(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(status=404, body=b'{"error":"no marker"}')
    marker: Final = found.group(0).decode()
    if marker.startswith("lfenv-fail-"):
        return openai_error(500)
    stream: Final = object_value(JSON.validate_json(request.body)).get("stream") is True
    identity: Final = uuid.uuid4().hex
    if request.target.endswith("/responses"):
        return responses_reply(f"resp_{identity}", "gpt-4o-mini", marker, stream=stream)
    return chat_reply(f"chatcmpl-{identity}", "gpt-4o-mini", marker, stream=stream)


def _body(model: str, endpoint: Endpoint, marker: str, stream: bool) -> tuple[str, dict[str, JsonValue]]:
    if endpoint == "responses":
        return "/v1/responses", {"model": model, "input": marker, "stream": stream}
    messages: Final[list[JsonValue]] = [{"role": "user", "content": marker}]
    if endpoint == "messages":
        return "/v1/messages", {"model": model, "max_tokens": 16, "messages": messages, "stream": stream}
    return "/v1/chat/completions", {"model": model, "messages": messages, "stream": stream}


@dataclass(frozen=True, slots=True)
class Sent:
    marker: str
    trace_id: str
    call_id: str
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    scenario: Scenario
    model: str
    sinks: SpanSinks

    def langfuse(self, environment: str | None = None) -> dict[str, str]:
        credentials: Final = {
            "langfuse_public_key": "pk-lf-tenant",
            "langfuse_secret_key": "sk-lf-tenant",
            "langfuse_host": self.sinks.tenant,
        }
        return credentials if environment is None else {**credentials, "langfuse_environment": environment}

    def send(
        self,
        key: str | None = None,
        endpoint: Endpoint = "chat",
        stream: bool = False,
        *,
        marker: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Sent:
        tagged: Final = marker or _marker()
        path, body = _body(self.model, endpoint, tagged, stream)
        trace_id: Final = uuid.uuid4().hex
        call_id: Final = str(uuid.uuid4())
        sent_headers: Final = {
            "Authorization": f"Bearer {self.proxy.key if key is None else key}",
            "traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01",
            "x-litellm-call-id": call_id,
            **(headers or {}),
        }
        payload: Final = body if metadata is None else {**body, "metadata": dict(metadata)}
        with self.proxy.client.stream("POST", path, json=payload, headers=sent_headers) as response:
            text: Final = response.read().decode()
            return Sent(tagged, trace_id, call_id, response.status_code, text)


def _served(sent: Sent) -> Sent:
    assert sent.status == 200, sent.text
    assert sent.marker in sent.text, sent.text
    return sent


def _of_request(url: str, since: int, sent: Sent) -> tuple[Span, ...]:
    spans: Final = recorded_spans(url, since)[1]
    traces: Final = {sent.trace_id} | {
        span["trace_id"] for span in spans if span["attributes"].get(CALL_ID) == sent.call_id
    }
    return tuple(span for span in spans if span["trace_id"] in traces)


def _root(spans: tuple[Span, ...]) -> Span | None:
    return next((span for span in spans if span["kind"] == SERVER and span["name"].startswith("POST")), None)


def _model(spans: tuple[Span, ...]) -> Span | None:
    return next((span for span in spans if OPERATION in span["attributes"]), None)


def _trace(url: str, since: int, sent: Sent) -> tuple[Span, ...]:
    return eventually(
        lambda: _of_request(url, since, sent),
        lambda found: _root(found) is not None and _model(found) is not None,
        seconds=40,
    )


def _environments(spans: tuple[Span, ...]) -> set[JsonValue]:
    return {span["resource"].get(ENVIRONMENT) for span in spans}


def _cursor(url: str) -> int:
    return recorded_spans(url)[0]


@pytest.mark.parametrize(
    ("endpoint", "stream"),
    [("chat", False), ("chat", True), ("responses", False), ("messages", True)],
    ids=["chat-unary", "chat-stream", "responses-unary", "messages-stream"],
)
def test_every_span_of_a_request_names_the_proxy_environment(rig: Rig, endpoint: Endpoint, stream: bool) -> None:
    since: Final = _cursor(rig.sinks.operator)
    sent: Final = _served(rig.send(endpoint=endpoint, stream=stream))
    assert _environments(_trace(rig.sinks.operator, since, sent)) == {PROXY_ENVIRONMENT}


def test_a_failed_call_names_the_proxy_environment(rig: Rig) -> None:
    since: Final = _cursor(rig.sinks.operator)
    sent: Final = rig.send(marker=_marker("fail-"))
    assert sent.status >= 500, sent.text
    assert _environments(_trace(rig.sinks.operator, since, sent)) == {PROXY_ENVIRONMENT}


@pytest.mark.parametrize("own", [True, False], ids=["own-environment", "no-environment"])
def test_a_key_destination_names_its_own_environment_else_the_proxys(rig: Rig, own: bool) -> None:
    environment: Final = f"key-{uuid.uuid4().hex[:8]}" if own else None
    logging: Final[list[JsonValue]] = [
        {"callback_name": "langfuse_otel", "callback_type": "success", "callback_vars": rig.langfuse(environment)}
    ]
    key: Final = rig.scenario.key(metadata={"logging": logging})
    since: Final = _cursor(rig.sinks.tenant)
    sent: Final = _served(rig.send(key))
    assert _environments(_trace(rig.sinks.tenant, since, sent)) == {environment or PROXY_ENVIRONMENT}


def test_a_team_destination_names_its_own_environment(rig: Rig) -> None:
    team: Final = rig.scenario.team()
    environment: Final = f"team-{uuid.uuid4().hex[:8]}"
    attached: Final = rig.proxy.request(
        "POST",
        f"/team/{team}/callback",
        {"callback_name": "langfuse_otel", "callback_vars": rig.langfuse(environment)},
    )
    assert attached.status_code == 200, attached.text
    since: Final = _cursor(rig.sinks.tenant)
    sent: Final = _served(rig.send(rig.scenario.key(team_id=team)))
    assert _environments(_trace(rig.sinks.tenant, since, sent)) == {environment}


@pytest.mark.parametrize(
    ("metadata", "headers", "expected"),
    [
        ({"trace_release": "r-body", "trace_version": "v-body"}, {}, ("r-body", "v-body")),
        (
            {"trace_release": "r-body", "trace_version": "v-body"},
            {"langfuse_trace_release": "r-header", "langfuse_trace_version": "v-header"},
            ("r-header", "v-header"),
        ),
        ({"version": "v-plain"}, {}, (None, "v-plain")),
        ({}, {}, (None, None)),
    ],
    ids=["metadata", "headers-win", "plain-version", "neither"],
)
def test_release_and_version_land_on_the_root_and_model_spans(
    rig: Rig, metadata: Mapping[str, JsonValue], headers: Mapping[str, str], expected: tuple[str | None, str | None]
) -> None:
    since: Final = _cursor(rig.sinks.operator)
    sent: Final = _served(rig.send(metadata=metadata, headers=headers))
    spans: Final = _trace(rig.sinks.operator, since, sent)
    root, model = _root(spans), _model(spans)
    assert root is not None and model is not None
    for attributes in (root["attributes"], model["attributes"]):
        assert (attributes.get(RELEASE), attributes.get(VERSION)) == expected, sorted(attributes)


def test_no_environment_set_names_none(bare_rig: Rig) -> None:
    since: Final = _cursor(bare_rig.sinks.operator)
    sent: Final = _served(bare_rig.send())
    assert _environments(_trace(bare_rig.sinks.operator, since, sent)) == {None}


def _write_config(directory: Path, sinks: SpanSinks) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"] = {
        **config.get("litellm_settings", {}),
        "callbacks": ["langfuse_otel"],
        "provider_url_destination_allowed_hosts": [urlparse(sinks.tenant).netloc],
    }
    config["general_settings"] = {**config.get("general_settings", {}), "user_api_key_cache_ttl": 2}
    path: Final = directory / "langfuse-environment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _started(provider: Wire, sinks: SpanSinks, directory: Path, overrides: Mapping[str, str]) -> Iterator[Rig]:
    environment: Final = {
        "LITELLM_OTEL_V2": "1",
        "OTEL_BSP_SCHEDULE_DELAY": "300",
        "LANGFUSE_HOST": sinks.operator,
        "LANGFUSE_PUBLIC_KEY": "pk-lf-operator",
        "LANGFUSE_SECRET_KEY": "sk-lf-operator",
        **overrides,
    }
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            environment,
            config=_write_config(directory, sinks),
            remove_environment=tuple(
                name
                for name in ("LANGFUSE_TRACING_ENVIRONMENT", "OTEL_ENVIRONMENT_NAME", "LANGFUSE_RELEASE")
                if name not in overrides
            ),
            workers=2,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        yield Rig(owned.gateway, scenario, scenario.model(api_base=provider.url + "/v1"), sinks)


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(answering_model_discovery(_upstream)) as wire:
        yield wire


@pytest.fixture(scope="module")
def rig(provider: Wire, audit_sinks: SpanSinks, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _started(
        provider,
        audit_sinks,
        tmp_path_factory.mktemp("langfuse-environment"),
        {"LANGFUSE_TRACING_ENVIRONMENT": PROXY_ENVIRONMENT},
    ) as started:
        yield started


@pytest.fixture(scope="module")
def bare_rig(provider: Wire, audit_sinks: SpanSinks, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _started(provider, audit_sinks, tmp_path_factory.mktemp("langfuse-environment-unset"), {}) as started:
        yield started
