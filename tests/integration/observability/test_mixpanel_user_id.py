"""Mixpanel joins agent traces to user events through ``user.id`` and drops a span without one, so every span a request
produces once auth has named its caller reaches the project with that id, whichever callback setup the operator runs,
and nothing else gets the id. Database and cache spans that close during auth, before the id is known, carry none."""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from _openinference_support import _spans as _wire_span_attributes
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, held
from integration._support.mcp import INITIALIZE, EntryPoint, McpCaller, echo_tool, register_mcp, scripted_peer
from integration._support.otlp_sink import (
    AuditConfigWriter,
    ConnectSink,
    Span,
    SpanSinks,
    owned_connect_sink,
    recorded_requests,
    recorded_spans,
    spans_for_trace,
)
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

OPERATION: Final = "gen_ai.operation.name"
USER_ID: Final = "user.id"
TEAM_TOKEN: Final = "team-project-token"
KEY_TOKEN: Final = "key-project-token"
OPERATOR_TOKEN: Final = "operator-project-token"
OPERATOR_HOST: Final = "ingestion-us.mixpanel.com"
PROVIDER_FAILS: Final = "provider-fails"
CUSTOM_AUTH: Final = """
from fastapi import Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


async def authenticate(request: Request, api_key: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key=api_key, user_role=LitellmUserRoles.PROXY_ADMIN)
"""
CHAT_COMPLETION: Final = {
    "id": "chatcmpl-mixpanel",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
}
CHAT_CHUNKS: Final = (
    b'data: {"id":"chatcmpl-mixpanel","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-mixpanel","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
    b"data: [DONE]\n\n",
)
RESPONSE: Final = {
    "id": "resp_mixpanel",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "type": "message",
            "id": "msg_mixpanel",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
}


def _provider_reply(request: Request) -> Reply:
    if PROVIDER_FAILS.encode() in request.body:
        return Reply(status=400, body=json.dumps({"error": {"message": "rejected by the provider"}}).encode())
    if request.target.endswith("/responses"):
        return Reply(body=json.dumps(RESPONSE).encode())
    if json.loads(request.body).get("stream"):
        return Reply(content_type="text/event-stream", chunks=CHAT_CHUNKS)
    return Reply(body=json.dumps(CHAT_COMPLETION).encode())


@dataclass(frozen=True, slots=True)
class Call:
    path: str
    body: Callable[[str], dict[str, JsonValue]]


def _chat_call(content: str) -> dict[str, JsonValue]:
    return {"messages": [{"role": "user", "content": content}]}


CALLS: Final[Mapping[str, Call]] = {
    "chat": Call("/v1/chat/completions", _chat_call),
    "chat-stream": Call("/v1/chat/completions", lambda content: {**_chat_call(content), "stream": True}),
    "responses": Call("/v1/responses", lambda content: {"input": content}),
    "messages": Call("/v1/messages", lambda content: {**_chat_call(content), "max_tokens": 20}),
}
IDENTIFICATIONS: Final[Mapping[str, Callable[[str], dict[str, JsonValue]]]] = {
    "user": lambda caller: {"user": caller},
    "metadata-user-dot-id": lambda caller: {"metadata": {"user.id": caller}},
    "metadata-user-id": lambda caller: {"metadata": {"user_id": caller}},
    "metadata-trace-user-id": lambda caller: {"metadata": {"trace_user_id": caller}},
    "litellm-metadata-user-dot-id": lambda caller: {"litellm_metadata": {"user.id": caller}},
    "metadata-json-user-dot-id": lambda caller: {"metadata": json.dumps({"user.id": caller})},
}


def _authorization(token: str) -> str:
    return f"Basic {base64.b64encode(f'{token}:'.encode()).decode()}"


def _team_or_key(scenario: Scenario, level: str, **access: JsonValue) -> tuple[str, str, str]:
    """A key with a Mixpanel project of its own, or under a team with one, plus the host and token that project uses."""
    token, region, host = (
        (TEAM_TOKEN, "eu", "ingestion-eu.mixpanel.com")
        if level == "team"
        else (KEY_TOKEN, "in", "ingestion-in.mixpanel.com")
    )
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "mixpanel",
            "callback_type": "success",
            "callback_vars": {"mixpanel_project_token": token, "mixpanel_region": region},
        }
    ]
    alias: Final = f"mixpanel-{uuid.uuid4().hex}"
    key: Final = (
        scenario.key(key_alias=alias, team_id=scenario.team(metadata={"logging": logging}, **access), **access)
        if level == "team"
        else scenario.key(key_alias=alias, metadata={"logging": logging}, **access)
    )
    return key, host, token


def _call_on_a_new_connection(gateway: Gateway, key: str, body: Mapping[str, JsonValue]) -> str:
    """A worker takes each connection, and only a new connection can land on one that has not served this key yet."""
    trace_id: Final = uuid.uuid4().hex
    with httpx.Client(base_url=gateway.client.base_url, trust_env=False, timeout=60) as client:
        response: Final = client.post(
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {key}", "traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
    assert response.status_code == 200, response.text
    return trace_id


def _traced_call(gateway: Gateway, key: str, body: Mapping[str, JsonValue], path: str = "/v1/chat/completions") -> str:
    trace_id: Final = uuid.uuid4().hex
    response: Final = gateway.request(
        "POST", path, body, key=key, headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"}
    )
    assert response.status_code == 200, response.text
    return trace_id


def _request_finished(spans: tuple[Span, ...]) -> bool:
    return any(span["kind"] == 2 for span in spans) and any(OPERATION in span["attributes"] for span in spans)


def _tool_call_finished(spans: tuple[Span, ...]) -> bool:
    return any(span["kind"] == 2 for span in spans) and any(_is_tool_call(span) for span in spans)


def _is_tool_call(span: Span) -> bool:
    return str(span["name"]).startswith("tools/call ")


def _has_model_call(spans: tuple[Span, ...]) -> bool:
    return any(OPERATION in span["attributes"] for span in spans)


def _trace(url: str, trace_id: str) -> tuple[Span, ...]:
    return spans_for_trace(recorded_spans(url)[1], trace_id)


def _settled_trace(
    url: str, trace_id: str, finished: Callable[[tuple[Span, ...]], bool] = _request_finished
) -> tuple[Span, ...]:
    held(lambda: _trace(url, trace_id), finished, holding=3, seconds=40)
    return _trace(url, trace_id)


def _without_the_id(spans: tuple[Span, ...], caller: str) -> list[tuple[str, JsonValue]]:
    return [
        (str(span["name"]), span["attributes"].get(USER_ID))
        for span in spans
        if span["attributes"].get(USER_ID) != caller
    ]


def _outside_auth(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    auth_spans: Final = {span["span_id"] for span in spans if str(span["name"]).startswith("auth ")}
    return tuple(span for span in spans if span["parent_span_id"] not in auth_spans)


def _inside_the_mcp_session(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    """Auth on ``/mcp`` runs on a task inside the transport, so the transport's ``POST /mcp`` span and the server
    lookups that come before auth never see the id. The model trace in the same list has no such span to leave out."""
    return tuple(
        span
        for span in _outside_auth(spans)
        if str(span["name"]) != "POST /mcp" and not str(span["name"]).startswith("postgres.select LiteLLM_")
    )


def _ingest_accepted(_request: Request) -> Reply:
    return Reply(body=b"", content_type="application/x-protobuf")


def _reached(sink: ConnectSink, spans: tuple[Span, ...], host: str) -> tuple[Span, ...]:
    exports: Final = tuple(
        cast("list[str]", export.get("span_ids", []))
        for export in recorded_requests(sink.control_url)
        if str(export.get("host", "")).partition(":")[0] == host
    )
    exported: Final = frozenset(chain.from_iterable(exports))
    return tuple(span for span in spans if span["span_id"] in exported)


def _exports_of(sink: ConnectSink, spans: tuple[Span, ...]) -> tuple[Mapping[str, JsonValue], ...]:
    span_ids: Final = {span["span_id"] for span in spans}
    return tuple(
        request
        for request in recorded_requests(sink.control_url)
        if span_ids & set(cast("list[str]", request.get("span_ids", [])))
    )


def _sent_to(exports: tuple[Mapping[str, JsonValue], ...]) -> set[tuple[str, str]]:
    return {
        (
            str(export["host"]).partition(":")[0],
            next(
                str(value)
                for name, value in cast("dict[str, JsonValue]", export["headers"]).items()
                if name.lower() == "authorization"
            ),
        )
        for export in exports
    }


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_provider_reply) as wire:
        yield wire


@pytest.fixture(scope="module")
def mixpanel(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ConnectSink]:
    with owned_connect_sink(tmp_path_factory.mktemp("mixpanel-ingest")) as sink:
        yield sink


def _with_general_settings(config: Path, settings: Mapping[str, JsonValue]) -> Path:
    configured: Final = yaml.safe_load(config.read_text())
    config.write_text(
        yaml.safe_dump({**configured, "general_settings": {**configured["general_settings"], **settings}})
    )
    return config


@contextmanager
def _booted(
    directory: Path,
    config: Path,
    mixpanel: ConnectSink,
    operator_token: str | None,
    collector: str | None = None,
    environment: Mapping[str, str] = MappingProxyType({}),
) -> Iterator[OwnedProxy]:
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {
                "LITELLM_OTEL_V2": "1",
                "OTEL_BSP_SCHEDULE_DELAY": "300",
                "HTTPS_PROXY": mixpanel.proxy_url,
                "NO_PROXY": "127.0.0.1,localhost",
                "REQUESTS_CA_BUNDLE": mixpanel.ca_pem,
                **({"MIXPANEL_PROJECT_TOKEN": operator_token} if operator_token else {}),
                **({"OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": collector} if collector else {}),
                **environment,
            },
            config=config,
            remove_environment=(
                *(() if operator_token else ("MIXPANEL_PROJECT_TOKEN",)),
                *(() if "MIXPANEL_REGION" in environment else ("MIXPANEL_REGION",)),
                *(() if "MIXPANEL_ENDPOINT" in environment else ("MIXPANEL_ENDPOINT",)),
                *(() if collector else ("OTEL_EXPORTER", "OTEL_ENDPOINT")),
            ),
            workers=2,
        ) as owned,
    ):
        yield owned


@pytest.fixture(
    scope="module",
    params=[(["mixpanel"], None), ([], None), (["mixpanel"], OPERATOR_TOKEN)],
    ids=["mixpanel-listed-no-token", "not-listed", "mixpanel-listed-with-token"],
)
def tenant_setup(
    request: pytest.FixtureRequest,
    mixpanel: ConnectSink,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("mixpanel-tenant")
    callbacks, operator_token = request.param
    with _booted(
        directory,
        otel_audit_config(directory, {"callbacks": cast("list[JsonValue]", callbacks)}),
        mixpanel,
        operator_token,
    ) as owned:
        yield owned


@pytest.fixture(scope="module")
def operator_setup(
    mixpanel: ConnectSink, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("mixpanel-operator")
    with _booted(
        directory, otel_audit_config(directory, {"callbacks": ["mixpanel"]}), mixpanel, OPERATOR_TOKEN
    ) as owned:
        yield owned


@pytest.fixture(scope="module")
def eu_operator_setup(
    mixpanel: ConnectSink, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("mixpanel-eu-operator")
    with _booted(
        directory,
        _with_general_settings(
            otel_audit_config(directory, {"callbacks": ["mixpanel"]}), {"allow_client_side_credentials": True}
        ),
        mixpanel,
        OPERATOR_TOKEN,
        environment={"MIXPANEL_REGION": "eu"},
    ) as owned:
        yield owned


@pytest.fixture(scope="module")
def collector_beside_operator_setup(
    mixpanel: ConnectSink,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("mixpanel-beside-collector")
    with _booted(
        directory,
        otel_audit_config(directory, {"callbacks": ["mixpanel"]}),
        mixpanel,
        OPERATOR_TOKEN,
        collector=audit_sinks.operator,
    ) as owned:
        yield owned


@pytest.fixture(scope="module")
def custom_auth_setup(
    mixpanel: ConnectSink, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("mixpanel-custom-auth")
    (directory / "mixpanel_custom_auth.py").write_text(CUSTOM_AUTH)
    with _booted(
        directory,
        _with_general_settings(
            otel_audit_config(directory, {"callbacks": ["mixpanel"]}),
            {"custom_auth": "mixpanel_custom_auth.authenticate"},
        ),
        mixpanel,
        OPERATOR_TOKEN,
    ) as owned:
        yield owned


@pytest.fixture(scope="module", params=[{}, {"otel_tenant_destination_mode": "additive"}], ids=["default", "additive"])
def collector_setup(
    request: pytest.FixtureRequest,
    mixpanel: ConnectSink,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("mixpanel-collector")
    with _booted(
        directory,
        otel_audit_config(directory, {"callbacks": ["otel"], **request.param}),
        mixpanel,
        None,
        collector=audit_sinks.operator,
    ) as owned:
        yield owned


@pytest.fixture(scope="module")
def additive_setup(
    mixpanel: ConnectSink, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("mixpanel-additive")
    with _booted(
        directory,
        otel_audit_config(directory, {"callbacks": ["mixpanel"], "otel_tenant_destination_mode": "additive"}),
        mixpanel,
        OPERATOR_TOKEN,
    ) as owned:
        yield owned


@pytest.fixture(
    scope="module",
    params=[("mixpanel", "otel"), ("otel", "mixpanel"), ("mixpanel", "langfuse_otel"), ("langfuse_otel", "mixpanel")],
    ids="+".join,
)
def combined_setup(
    request: pytest.FixtureRequest,
    mixpanel: ConnectSink,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[OwnedProxy, tuple[str, ...]]]:
    callbacks: Final[tuple[str, ...]] = request.param
    directory: Final = tmp_path_factory.mktemp("mixpanel-combined")
    with _booted(
        directory,
        otel_audit_config(directory, {"callbacks": cast("list[JsonValue]", list(callbacks))}),
        mixpanel,
        OPERATOR_TOKEN,
        collector=audit_sinks.operator if "otel" in callbacks else None,
        environment={
            "LANGFUSE_HOST": audit_sinks.operator,
            "LANGFUSE_PUBLIC_KEY": "pk-lf-operator",
            "LANGFUSE_SECRET_KEY": "sk-lf-operator",
        }
        if "langfuse_otel" in callbacks
        else {},
    ) as owned:
        yield owned, callbacks


@pytest.mark.parametrize("identified", ["user", "metadata-user-dot-id"])
@pytest.mark.parametrize("level", ["team", "key"])
def test_a_team_or_key_project_gets_the_callers_id_on_every_span(
    level: str, identified: str, tenant_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = tenant_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key, host, token = _team_or_key(scenario, level)
        trace_id: Final = _traced_call(
            proxy, key, {"model": model, **_chat_call("hi"), **IDENTIFICATIONS[identified](caller)}
        )
        spans: Final = _settled_trace(mixpanel.control_url, trace_id)

    at_project: Final = _reached(mixpanel, spans, host)
    assert _without_the_id(at_project, caller) == []
    names: Final = [str(span["name"]) for span in at_project]
    assert any(name.startswith("auth ") for name in names), names
    assert (host, _authorization(token)) in _sent_to(_exports_of(mixpanel, spans))
    assert OPERATION not in json.dumps([span["attributes"] for span in _reached(mixpanel, spans, OPERATOR_HOST)]), [
        span["name"] for span in spans
    ]


@pytest.mark.parametrize("call", sorted(CALLS))
def test_every_endpoint_carries_the_id_to_a_team_project(
    call: str, tenant_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = tenant_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key, host, _ = _team_or_key(scenario, "team")
        trace_id: Final = _traced_call(
            proxy, key, {"model": model, **CALLS[call].body("hi"), "metadata": {"user.id": caller}}, CALLS[call].path
        )
        spans: Final = _settled_trace(mixpanel.control_url, trace_id)

    at_project: Final = _reached(mixpanel, spans, host)
    assert at_project, [span["name"] for span in spans]
    assert _without_the_id(at_project, caller) == []


@pytest.mark.parametrize("level", ["team", "key"])
def test_every_request_on_a_long_lived_key_names_its_own_caller(
    level: str, tenant_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = tenant_setup.gateway
    callers: Final = tuple(f"caller-{uuid.uuid4().hex}" for _ in range(12))
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key, host, _ = _team_or_key(scenario, level)
        traces: Final = tuple(
            (caller, _call_on_a_new_connection(proxy, key, {"model": model, **_chat_call("hi"), "user": caller}))
            for caller in callers
        )
        settled: Final = tuple((caller, _settled_trace(mixpanel.control_url, trace_id)) for caller, trace_id in traces)

    at_project: Final = tuple((caller, _reached(mixpanel, spans, host)) for caller, spans in settled)
    assert [caller for caller, spans in at_project if not spans] == []
    assert [(caller, _without_the_id(spans, caller)) for caller, spans in at_project] == [
        (caller, []) for caller in callers
    ]


def test_a_project_token_listed_without_one_stays_quiet_on_plain_traffic(
    tenant_setup: OwnedProxy, provider: Wire
) -> None:
    proxy: Final = tenant_setup.gateway
    logged_before: Final = len(tenant_setup.log.read_text())
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        _traced_call(
            proxy, scenario.key(key_alias=f"mixpanel-plain-{uuid.uuid4().hex}"), {"model": model, **_chat_call("hi")}
        )
        held(
            lambda: tenant_setup.log.read_text()[logged_before:],
            lambda logged: "Error initializing custom logger" not in logged,
            holding=5,
            seconds=40,
        )


def test_a_renamed_service_keeps_the_model_call_of_an_eu_proxy_in_its_region(
    eu_operator_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = eu_operator_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = scenario.key(
            key_alias=f"mixpanel-renamed-{uuid.uuid4().hex}", metadata={"otel_service_name": "renamed-service"}
        )
        _traced_call(
            proxy,
            key,
            {"model": model, **_chat_call("hi"), "metadata": {USER_ID: caller, "langsmith_project": "any-project"}},
        )
        model_spans: Final = eventually(
            lambda: tuple(
                span
                for span in recorded_spans(mixpanel.control_url)[1]
                if OPERATION in span["attributes"] and span["attributes"].get(USER_ID) == caller
            ),
            bool,
            seconds=40,
        )

    assert _reached(mixpanel, model_spans, "ingestion-eu.mixpanel.com") == model_spans
    assert _reached(mixpanel, model_spans, OPERATOR_HOST) == ()


@pytest.mark.parametrize("identified", sorted(IDENTIFICATIONS))
def test_the_operators_project_gets_the_callers_id_on_every_span_after_auth(
    identified: str, operator_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = operator_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = scenario.key(key_alias=f"mixpanel-operator-{uuid.uuid4().hex}")
        trace_id: Final = _traced_call(
            proxy, key, {"model": model, **_chat_call("hi"), **IDENTIFICATIONS[identified](caller)}
        )
        spans: Final = _settled_trace(mixpanel.control_url, trace_id)

    assert any(str(span["name"]).startswith("auth ") for span in spans), [span["name"] for span in spans]
    assert _without_the_id(_outside_auth(spans), caller) == []
    assert _sent_to(_exports_of(mixpanel, spans)) == {(OPERATOR_HOST, _authorization(OPERATOR_TOKEN))}


@pytest.mark.parametrize(
    "bad_id",
    [
        {"metadata": {"user.id": "   "}},
        {"metadata": {"user.id": ""}},
        {"metadata": {"user.id": True}},
        {"metadata": {"user_id": False}},
        {"metadata": {"user.id": None}},
        {"metadata": {"user.id": ["list"]}},
        {"metadata": {"user.id": "x" * 2000}},
    ],
    ids=["whitespace", "empty", "true", "false", "null", "list", "oversized"],
)
def test_an_unusable_id_gives_way_to_the_callers_real_one(
    bad_id: dict[str, JsonValue], operator_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = operator_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = scenario.key(key_alias=f"mixpanel-bad-id-{uuid.uuid4().hex}")
        trace_id: Final = _traced_call(proxy, key, {"model": model, **_chat_call("hi"), "user": caller, **bad_id})
        spans: Final = _settled_trace(mixpanel.control_url, trace_id)

    model_call: Final = [span for span in spans if OPERATION in span["attributes"]]
    assert [span["attributes"].get(USER_ID) for span in model_call] == [caller]


def test_the_baggage_header_names_the_caller_only_once_auth_has_passed(
    operator_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = operator_setup.gateway
    named: Final = f"named-{uuid.uuid4().hex}"
    rejected_trace: Final = uuid.uuid4().hex
    accepted_trace: Final = uuid.uuid4().hex
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = scenario.key(key_alias=f"mixpanel-baggage-{uuid.uuid4().hex}")
        rejected: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, **_chat_call("hi")},
            key="sk-not-a-key",
            headers={
                "baggage": f"{USER_ID}={named}",
                "traceparent": f"00-{rejected_trace}-{uuid.uuid4().hex[:16]}-01",
            },
        )
        assert rejected.status_code == 401, rejected.text
        accepted: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, **_chat_call("hi")},
            key=key,
            headers={
                "baggage": f"{USER_ID}={named}",
                "traceparent": f"00-{accepted_trace}-{uuid.uuid4().hex[:16]}-01",
            },
        )
        assert accepted.status_code == 200, accepted.text
        accepted_spans: Final = _settled_trace(mixpanel.control_url, accepted_trace)
        rejected_spans: Final = _settled_trace(
            mixpanel.control_url, rejected_trace, lambda spans: any(span["kind"] == 2 for span in spans)
        )

    assert named not in json.dumps([span["attributes"] for span in rejected_spans])
    assert _without_the_id(_outside_auth(accepted_spans), named) == []


def test_a_collector_never_gets_the_id_only_a_team_project_asked_for(
    collector_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink, audit_sinks: SpanSinks
) -> None:
    proxy: Final = collector_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key, _, _ = _team_or_key(scenario, "team")
        trace_id: Final = _traced_call(
            proxy, key, {"model": model, **_chat_call("hi"), "metadata": {"user.id": caller}}
        )
        at_mixpanel: Final = _settled_trace(mixpanel.control_url, trace_id)
        at_collector: Final = _settled_trace(audit_sinks.operator, trace_id)

    assert _without_the_id(at_mixpanel, caller) == []
    assert caller not in json.dumps([span["attributes"] for span in at_collector])


@pytest.mark.parametrize(
    "token",
    ["", "   ", None, "None", " None ", "***REDACTED***", " ***REDACTED*** "],
    ids=["empty", "whitespace", "null", "none-string", "padded-none-string", "masked", "padded-masked"],
)
def test_a_team_or_key_callback_refuses_a_token_it_could_never_send(
    token: JsonValue, operator_setup: OwnedProxy, mixpanel: ConnectSink
) -> None:
    proxy: Final = operator_setup.gateway
    callback_vars: Final[dict[str, JsonValue]] = {"mixpanel_project_token": token}
    logging: Final[list[JsonValue]] = [
        {"callback_name": "mixpanel", "callback_type": "success", "callback_vars": callback_vars}
    ]
    with proxy.scenario() as scenario:
        team_id: Final = scenario.team()
        team: Final = proxy.request(
            "POST",
            f"/team/{team_id}/callback",
            {"callback_name": "mixpanel", "callback_type": "success", "callback_vars": callback_vars},
        )
        new_team: Final = proxy.request(
            "POST", "/team/new", {"team_alias": f"mixpanel-{uuid.uuid4().hex}", "metadata": {"logging": logging}}
        )
        updated_team: Final = proxy.request(
            "POST", "/team/update", {"team_id": team_id, "metadata": {"logging": logging}}
        )
        generated: Final = proxy.request(
            "POST", "/key/generate", {"key_alias": f"mixpanel-{uuid.uuid4().hex}", "metadata": {"logging": logging}}
        )
        created: Final = scenario.key(key_alias=f"mixpanel-{uuid.uuid4().hex}")
        updated: Final = proxy.request("POST", "/key/update", {"key": created, "metadata": {"logging": logging}})

    for refused in (team, new_team, updated_team, generated, updated):
        assert refused.status_code in (400, 422), refused.text
        assert "mixpanel_project_token" in refused.text, refused.text


def test_each_project_of_an_additive_proxy_gets_the_callers_id_on_every_span(
    additive_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = additive_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key, host, token = _team_or_key(scenario, "team")
        trace_id: Final = _traced_call(proxy, key, {"model": model, **_chat_call("hi"), "user": caller})
        spans: Final = _settled_trace(
            mixpanel.control_url,
            trace_id,
            lambda found: len([span for span in found if OPERATION in span["attributes"]]) == 2,
        )

    assert _without_the_id(_outside_auth(spans), caller) == []
    assert [str(span["name"]) for span in _reached(mixpanel, spans, host) if OPERATION in span["attributes"]]
    assert [str(span["name"]) for span in _reached(mixpanel, spans, OPERATOR_HOST) if OPERATION in span["attributes"]]
    assert _sent_to(_exports_of(mixpanel, spans)) == {
        (host, _authorization(token)),
        (OPERATOR_HOST, _authorization(OPERATOR_TOKEN)),
    }


def test_mixpanel_listed_with_another_callback_gets_one_labeled_copy_of_the_model_call(
    combined_setup: tuple[OwnedProxy, tuple[str, ...]], provider: Wire, mixpanel: ConnectSink, audit_sinks: SpanSinks
) -> None:
    owned, callbacks = combined_setup
    proxy: Final = owned.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = scenario.key(key_alias=f"mixpanel-combined-{uuid.uuid4().hex}")
        trace_id: Final = _traced_call(proxy, key, {"model": model, **_chat_call("hi"), "user": caller})
        at_mixpanel: Final = _settled_trace(mixpanel.control_url, trace_id, _has_model_call)
        at_other: Final = _settled_trace(audit_sinks.operator, trace_id, _has_model_call)

    assert len([span for span in at_mixpanel if OPERATION in span["attributes"]]) == 1, at_mixpanel
    assert len([span for span in at_other if OPERATION in span["attributes"]]) == 1, at_other
    assert _without_the_id(_outside_auth(at_mixpanel), caller) == []
    assert any(span["kind"] == 2 for span in at_mixpanel) == (callbacks[0] == "mixpanel"), [
        span["name"] for span in at_mixpanel
    ]
    assert _sent_to(_exports_of(mixpanel, at_mixpanel)) == {(OPERATOR_HOST, _authorization(OPERATOR_TOKEN))}


def test_the_operators_project_names_one_caller_when_a_request_also_sends_a_trace_user_id(
    combined_setup: tuple[OwnedProxy, tuple[str, ...]], provider: Wire, mixpanel: ConnectSink, audit_sinks: SpanSinks
) -> None:
    owned, callbacks = combined_setup
    proxy: Final = owned.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    traced_as: Final = f"traced-as-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = scenario.key(key_alias=f"mixpanel-two-ids-{uuid.uuid4().hex}")
        trace_id: Final = _traced_call(
            proxy,
            key,
            {"model": model, **_chat_call("hi"), "metadata": {"user.id": caller, "trace_user_id": traced_as}},
        )
        at_mixpanel: Final = _settled_trace(mixpanel.control_url, trace_id, _has_model_call)
        at_other: Final = _settled_trace(audit_sinks.operator, trace_id, _has_model_call)

    assert _without_the_id(_outside_auth(at_mixpanel), caller) == []
    assert {span["attributes"].get(USER_ID) for span in at_other if OPERATION in span["attributes"]} == (
        {traced_as} if "langfuse_otel" in callbacks else {None}
    )


def test_a_team_project_names_one_caller_when_a_request_also_sends_a_trace_user_id(
    combined_setup: tuple[OwnedProxy, tuple[str, ...]], provider: Wire, mixpanel: ConnectSink
) -> None:
    owned, _ = combined_setup
    proxy: Final = owned.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key, host, _token = _team_or_key(scenario, "team")
        trace_id: Final = _traced_call(
            proxy,
            key,
            {
                "model": model,
                **_chat_call("hi"),
                "metadata": {"user.id": caller, "trace_user_id": f"traced-as-{uuid.uuid4().hex}"},
            },
        )
        spans: Final = _settled_trace(mixpanel.control_url, trace_id, _has_model_call)

    at_project: Final = _reached(mixpanel, spans, host)
    assert [span for span in at_project if OPERATION in span["attributes"]], at_project
    assert _without_the_id(at_project, caller) == []


@pytest.mark.parametrize("entry", ["mcp", "rest"])
def test_a_team_project_gets_the_callers_id_on_an_mcp_tool_call(
    entry: EntryPoint, tenant_setup: OwnedProxy, mixpanel: ConnectSink
) -> None:
    proxy: Final = tenant_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    trace_id: Final = uuid.uuid4().hex
    alias: Final = f"mixpanelmcp{uuid.uuid4().hex[:8]}"
    with scripted_peer(echo_tool("echo")) as peer, proxy.scenario() as scenario:
        server_id: Final = register_mcp(scenario, peer, alias)
        key, host, _ = _team_or_key(scenario, "team", object_permission={"mcp_servers": [server_id]})
        outcome: Final = McpCaller(
            proxy,
            key,
            entry,
            alias,
            headers={
                "baggage": f"{USER_ID}={caller}",
                "traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01",
            },
        ).call(f"{alias}-echo", {"note": "hi"}, server_id=server_id)
        assert outcome.error is None, outcome.raw
        spans: Final = _settled_trace(mixpanel.control_url, trace_id, _tool_call_finished)

    at_project: Final = _reached(mixpanel, spans, host)
    assert [span for span in at_project if _is_tool_call(span)], [span["name"] for span in spans]
    assert _without_the_id(at_project, caller) == []


@pytest.mark.parametrize("later_message_names", [True, False], ids=["its-own-caller", "nobody"])
def test_a_later_message_of_an_mcp_session_carries_its_own_caller_and_never_the_first(
    later_message_names: bool, tenant_setup: OwnedProxy, mixpanel: ConnectSink
) -> None:
    proxy: Final = tenant_setup.gateway
    first_caller: Final = f"caller-{uuid.uuid4().hex}"
    later_caller: Final = f"caller-{uuid.uuid4().hex}"
    trace_id: Final = uuid.uuid4().hex
    alias: Final = f"mixpanelmcp{uuid.uuid4().hex[:8]}"
    with scripted_peer(echo_tool("echo")) as peer, proxy.scenario() as scenario:
        server_id: Final = register_mcp(scenario, peer, alias)
        key, host, _ = _team_or_key(scenario, "team", object_permission={"mcp_servers": [server_id]})
        opened: Final = McpCaller(proxy, key, "mcp", alias, headers={"baggage": f"{USER_ID}={first_caller}"}).rpc(
            "initialize", INITIALIZE
        )
        assert opened.status_code == 200, opened.text
        session: Final = {"Mcp-Session-Id": opened.headers["mcp-session-id"]}
        initialized: Final = proxy.client.post(
            "/mcp",
            headers=McpCaller(proxy, key, "mcp", alias, headers=session)._headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert initialized.status_code == 202, initialized.text
        outcome: Final = McpCaller(
            proxy,
            key,
            "mcp",
            alias,
            headers={
                **session,
                **({"baggage": f"{USER_ID}={later_caller}"} if later_message_names else {}),
                "traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01",
            },
        ).call(f"{alias}-echo", {"note": "hi"}, server_id=server_id)
        assert outcome.error is None, outcome.raw
        spans: Final = _settled_trace(mixpanel.control_url, trace_id, _tool_call_finished)

    at_project: Final = _reached(mixpanel, spans, host)
    assert [span for span in at_project if _is_tool_call(span)], [span["name"] for span in spans]
    assert first_caller not in json.dumps([span["attributes"] for span in spans])
    assert {span["attributes"].get(USER_ID) for span in at_project} == (
        {later_caller} if later_message_names else {None}
    )


@pytest.mark.parametrize("entry", ["mcp", "rest"])
def test_a_collector_beside_the_operators_project_gets_no_id_on_a_model_or_tool_call(
    entry: EntryPoint,
    collector_beside_operator_setup: OwnedProxy,
    provider: Wire,
    mixpanel: ConnectSink,
    audit_sinks: SpanSinks,
) -> None:
    proxy: Final = collector_beside_operator_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    tool_trace: Final = uuid.uuid4().hex
    alias: Final = f"mixpanelmcp{uuid.uuid4().hex[:8]}"
    with scripted_peer(echo_tool("echo")) as peer, proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        server_id: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(
            key_alias=f"mixpanel-collector-{uuid.uuid4().hex}", object_permission={"mcp_servers": [server_id]}
        )
        model_trace: Final = _traced_call(proxy, key, {"model": model, **_chat_call("hi"), "user": caller})
        outcome: Final = McpCaller(
            proxy,
            key,
            entry,
            alias,
            headers={
                "x-litellm-customer-id": caller,
                "traceparent": f"00-{tool_trace}-{uuid.uuid4().hex[:16]}-01",
            },
        ).call(f"{alias}-echo", {"note": "hi"}, server_id=server_id)
        assert outcome.error is None, outcome.raw
        at_mixpanel: Final = (
            *_settled_trace(mixpanel.control_url, model_trace),
            *_settled_trace(mixpanel.control_url, tool_trace, _tool_call_finished),
        )
        at_collector: Final = (
            *_settled_trace(audit_sinks.operator, model_trace),
            *_settled_trace(audit_sinks.operator, tool_trace, _tool_call_finished),
        )

    assert [span for span in at_mixpanel if _is_tool_call(span)], [span["name"] for span in at_mixpanel]
    labeled: Final = _outside_auth(at_mixpanel) if entry == "rest" else _inside_the_mcp_session(at_mixpanel)
    assert _without_the_id(labeled, caller) == []
    assert [span["name"] for span in at_collector if USER_ID in span["attributes"]] == []


def test_a_custom_auth_proxy_names_the_caller_from_the_request(
    custom_auth_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = custom_auth_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        trace_id: Final = _traced_call(proxy, "sk-any-caller", {"model": model, **_chat_call("hi"), "user": caller})
        spans: Final = _settled_trace(mixpanel.control_url, trace_id)

    assert _without_the_id(_outside_auth(spans), caller) == []


def test_a_failed_model_call_reaches_a_team_project_with_the_callers_id(
    tenant_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = tenant_setup.gateway
    caller: Final = f"caller-{uuid.uuid4().hex}"
    trace_id: Final = uuid.uuid4().hex
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key, host, _ = _team_or_key(scenario, "team")
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, **_chat_call(PROVIDER_FAILS), "user": caller},
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        assert response.status_code == 400, response.text
        spans: Final = _settled_trace(mixpanel.control_url, trace_id)

    at_project: Final = _reached(mixpanel, spans, host)
    assert at_project, [span["name"] for span in spans]
    assert _without_the_id(at_project, caller) == []


def test_concurrent_callers_never_get_each_others_id(
    operator_setup: OwnedProxy, provider: Wire, mixpanel: ConnectSink
) -> None:
    proxy: Final = operator_setup.gateway
    callers: Final = tuple(
        (f"caller-{uuid.uuid4().hex}", sorted(CALLS)[index % len(CALLS)], uuid.uuid4().hex) for index in range(32)
    )
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = scenario.key(key_alias=f"mixpanel-burst-{uuid.uuid4().hex}")
        team_key, _, _ = _team_or_key(scenario, "team")
        with ThreadPoolExecutor(max_workers=len(callers)) as executor:
            statuses: Final = tuple(
                executor.map(
                    lambda indexed: (
                        proxy.request(
                            "POST",
                            CALLS[indexed[1][1]].path,
                            {"model": model, **CALLS[indexed[1][1]].body("hi"), "metadata": {"user.id": indexed[1][0]}},
                            key=team_key if indexed[0] % 2 else key,
                            headers={"traceparent": f"00-{indexed[1][2]}-{uuid.uuid4().hex[:16]}-01"},
                        ).status_code
                    ),
                    enumerate(callers),
                )
            )
        assert statuses == (200,) * len(callers), statuses
        traces: Final = tuple(
            (caller, _settled_trace(mixpanel.control_url, trace_id)) for caller, _, trace_id in callers
        )

    assert [(caller, _without_the_id(_outside_auth(spans), caller)) for caller, spans in traces] == [
        (caller, []) for caller, _ in traces
    ]


def test_a_project_outage_costs_spans_and_never_a_request(
    mixpanel: ConnectSink, provider: Wire, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    with wire_server(_ingest_accepted) as first_listener:
        port: Final = urlsplit(first_listener.url).port
    caller: Final = f"caller-{uuid.uuid4().hex}"
    with _booted(
        tmp_path,
        otel_audit_config(tmp_path, {"callbacks": ["mixpanel"]}),
        mixpanel,
        OPERATOR_TOKEN,
        environment={"MIXPANEL_ENDPOINT": f"http://127.0.0.1:{port}/v1/ai/otel/traces"},
    ) as owned:
        proxy: Final = owned.gateway
        with proxy.scenario() as scenario:
            model: Final = scenario.model(api_base=provider.url + "/v1")
            key: Final = scenario.key(key_alias=f"mixpanel-outage-{uuid.uuid4().hex}")
            during: Final = tuple(
                proxy.request("POST", "/v1/chat/completions", {"model": model, **_chat_call("hi")}, key=key).status_code
                for _ in range(10)
            )
            assert during == (200,) * len(during), during
            with wire_server(_ingest_accepted, port=port) as recovered:
                assert (
                    proxy.request(
                        "POST", "/v1/chat/completions", {"model": model, **_chat_call("hi"), "user": caller}, key=key
                    ).status_code
                    == 200
                )
                landed: Final = eventually(
                    lambda: tuple(_wire_span_attributes(recovered.drain())),
                    lambda spans: any(span.get(USER_ID) == caller and OPERATION in span for span in spans),
                    seconds=40,
                )
    assert any(span.get(USER_ID) == caller for span in landed), landed
