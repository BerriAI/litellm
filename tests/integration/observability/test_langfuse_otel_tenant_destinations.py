"""A key's failure-only Langfuse entry and a team in ``litellm_settings.default_team_settings`` reach their own
Langfuse project under OTel V2 when the operator has no Langfuse keys, the way they do with V2 off. A failed request
reaches a failure-only project and a successful one does not, whatever else failed on the way."""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Final, Literal

import pytest
import yaml
from integration._support.anthropic_sse import stream_reply
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.otlp_sink import (
    AuditConfigWriter,
    Span,
    SpanSinks,
    langfuse_callback_variables,
    recorded_requests,
    recorded_spans,
    spans_for_trace,
)
from integration._support.ports import allocate_port
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

OPERATION: Final = "gen_ai.operation.name"
ERROR_TYPE: Final = "error.type"
TAGS: Final = ["agent:orders", "run:r1"]
FAIL: Final = "please-fail"
BREAK_STREAM: Final = "please-break-stream"
CONFIG_TEAM: Final = f"lf-config-team-{uuid.uuid4().hex[:8]}"
CONFIG_TEAM_SECRET: Final = "sk-lf-config-team"
CONFIG_TEAM_SECRET_ENV: Final = "LF_CONFIG_TEAM_SECRET_KEY"
OPERATOR_KEYS: Final = ("pk-lf-operator", "sk-lf-operator")
LANGFUSE_KEYS: Final = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
COLLECTOR_ENVIRONMENT: Final = (
    "OTEL_EXPORTER",
    "OTEL_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
)


def _basic(public: str, secret: str) -> str:
    return "Basic " + base64.b64encode(f"{public}:{secret}".encode()).decode()


def _chunk(content: Mapping[str, JsonValue]) -> bytes:
    body: Final = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": dict(content), "finish_reason": None}],
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _chat(request: Request) -> Reply:
    marker: Final = json.loads(request.body)["messages"][-1]["content"]
    if BREAK_STREAM in marker:
        error: Final = {"error": {"message": "upstream broke mid-stream", "type": "server_error"}}
        return stream_reply(
            (_chunk({"role": "assistant", "content": "partial"}), b"data: " + json.dumps(error).encode() + b"\n\n"),
            pause=0.2,
        )
    if FAIL in marker:
        return Reply(
            status=500, body=json.dumps({"error": {"message": "upstream boom", "type": "server_error"}}).encode()
        )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            }
        ).encode()
    )


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_chat) as wire:
        yield wire


@contextmanager
def _owned_gateway(
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
    settings: Mapping[str, JsonValue],
    environment: Mapping[str, str],
    removed: tuple[str, ...],
) -> Iterator[Gateway]:
    """A one-worker proxy whose config gives ``CONFIG_TEAM`` its own Langfuse project in ``default_team_settings``,
    with the secret key read from the env."""
    directory: Final = tmp_path_factory.mktemp("langfuse-tenant-destinations")
    variables: Final = {
        **langfuse_callback_variables(audit_sinks),
        "langfuse_secret_key": f"os.environ/{CONFIG_TEAM_SECRET_ENV}",
    }
    config: Final = otel_audit_config(
        directory,
        {
            "default_team_settings": [
                {"team_id": CONFIG_TEAM, "success_callback": ["langfuse_otel"], "callback_vars": variables}
            ],
            **settings,
        },
    )
    loaded: Final = yaml.safe_load(config.read_text())
    loaded.pop("callback_settings", None)
    config.write_text(yaml.safe_dump(loaded))
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {"OTEL_BSP_SCHEDULE_DELAY": "300", CONFIG_TEAM_SECRET_ENV: CONFIG_TEAM_SECRET, **environment},
            config=config,
            remove_environment=removed,
            workers=1,
        ) as owned,
    ):
        yield owned.gateway


def _v2_with_collector(audit_sinks: SpanSinks) -> dict[str, str]:
    return {"LITELLM_OTEL_V2": "1", "OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": audit_sinks.operator}


@pytest.fixture(scope="module", params=[["langfuse_otel"], []], ids=["listed", "not-listed"])
def proxy(
    request: pytest.FixtureRequest,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Gateway]:
    """V2 on, no operator Langfuse keys, and the operator's collector on the env."""
    with _owned_gateway(
        audit_sinks,
        otel_audit_config,
        tmp_path_factory,
        {"callbacks": request.param},
        _v2_with_collector(audit_sinks),
        LANGFUSE_KEYS,
    ) as gateway:
        yield gateway


@pytest.fixture(scope="module")
def redis_down_proxy(
    audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    """The same proxy with a response cache whose Redis is down, so a successful request carries failed Redis spans.
    Deleting a key fails on the same Redis, so tests here create what they need without the scenario's cleanup."""
    with _owned_gateway(
        audit_sinks,
        otel_audit_config,
        tmp_path_factory,
        {
            "callbacks": ["langfuse_otel"],
            "cache": True,
            "cache_params": {"type": "redis", "host": "127.0.0.1", "port": allocate_port()},
        },
        _v2_with_collector(audit_sinks),
        LANGFUSE_KEYS,
    ) as gateway:
        yield gateway


@pytest.fixture(scope="module")
def operator_langfuse_proxy(
    audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    """``langfuse_otel`` listed with the operator's own Langfuse keys, whose project is the third sink."""
    with _owned_gateway(
        audit_sinks,
        otel_audit_config,
        tmp_path_factory,
        {"callbacks": ["langfuse_otel"]},
        {
            "LITELLM_OTEL_V2": "1",
            "LANGFUSE_PUBLIC_KEY": OPERATOR_KEYS[0],
            "LANGFUSE_SECRET_KEY": OPERATOR_KEYS[1],
            "LANGFUSE_HOST": audit_sinks.arize,
        },
        COLLECTOR_ENVIRONMENT,
    ) as gateway:
        yield gateway


@pytest.fixture(scope="module")
def v2_off_proxy(
    audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    with _owned_gateway(
        audit_sinks,
        otel_audit_config,
        tmp_path_factory,
        {"callbacks": []},
        {},
        ("LITELLM_OTEL_V2", *LANGFUSE_KEYS, *COLLECTOR_ENVIRONMENT),
    ) as gateway:
        yield gateway


def _failure_only_logging(audit_sinks: SpanSinks) -> list[JsonValue]:
    return [
        {
            "callback_name": "langfuse_otel",
            "callback_type": "failure",
            "callback_vars": langfuse_callback_variables(audit_sinks),
        }
    ]


def _send(gateway: Gateway, model: str, key: str, outcome: Literal["success", "failure", "broken-stream"]) -> str:
    trace_id: Final = uuid.uuid4().hex
    marker: Final = {"success": "", "failure": f" {FAIL}", "broken-stream": f" {BREAK_STREAM}"}[outcome]
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"lf-tenant-destinations-{trace_id}{marker}"}],
            "metadata": {"tags": TAGS, "session_id": f"sess-{trace_id}", "trace_user_id": "user-42"},
            **({"stream": True} if outcome == "broken-stream" else {}),
        },
        key=key,
        headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
    )
    assert response.status_code == (500 if outcome == "failure" else 200), response.text
    return trace_id


def _trace(url: str, since: int, trace_id: str) -> tuple[Span, ...]:
    return spans_for_trace(recorded_spans(url, since)[1], trace_id)


def _model_spans(url: str, since: int, trace_id: str) -> tuple[Span, ...]:
    return tuple(span for span in _trace(url, since, trace_id) if OPERATION in span["attributes"])


def _labels(spans: tuple[Span, ...]) -> list[dict[str, JsonValue]]:
    return [{name: span["attributes"].get(name) for name in ("user.id", "langfuse.trace.tags")} for span in spans]


def _legacy_tags(span: Span) -> JsonValue:
    """The legacy logger writes the tag list as one JSON string."""
    tags: Final = span["attributes"].get("langfuse.trace.tags")
    return json.loads(tags) if isinstance(tags, str) else tags


def _authorizations(url: str, since: int) -> set[str]:
    return {
        str({key.lower(): value for key, value in object_value(posted["headers"]).items()}.get("authorization"))
        for posted in recorded_requests(url)[since:]
    }


def test_a_failure_only_key_gets_its_failed_request_and_not_its_successful_one(
    proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1", num_retries=0)
        key: Final = scenario.key(metadata={"logging": _failure_only_logging(audit_sinks)})
        since: Final = recorded_spans(audit_sinks.tenant)[0]
        posts_since: Final = len(recorded_requests(audit_sinks.tenant))
        succeeded: Final = _send(proxy, model, key, "success")
        failed: Final = _send(proxy, model, key, "failure")

        delivered: Final = eventually(
            lambda: _model_spans(audit_sinks.tenant, since, failed), lambda spans: bool(spans), seconds=40
        )

    assert _trace(audit_sinks.tenant, since, succeeded) == ()
    assert _labels(delivered) == [{"user.id": "user-42", "langfuse.trace.tags": TAGS}]
    assert _authorizations(audit_sinks.tenant, posts_since) == {_basic("pk-lf-audit", "sk-lf-audit")}


def test_a_failed_redis_span_does_not_send_a_successful_request_to_a_failure_only_key(
    redis_down_proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    gateway: Final = redis_down_proxy
    created_model: Final = gateway.post(
        "/model/new",
        {
            "model_name": f"integration-{uuid.uuid4().hex}",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": provider.url + "/v1",
                "num_retries": 0,
            },
        },
    )
    model: Final = string_value(created_model["model_name"])
    key: Final = string_value(
        gateway.post("/key/generate", {"metadata": {"logging": _failure_only_logging(audit_sinks)}})["key"]
    )
    tenant_since: Final = recorded_spans(audit_sinks.tenant)[0]
    operator_since: Final = recorded_spans(audit_sinks.operator)[0]
    succeeded: Final = _send(gateway, model, key, "success")
    failed: Final = _send(gateway, model, key, "failure")

    delivered: Final = eventually(
        lambda: _model_spans(audit_sinks.tenant, tenant_since, failed), lambda spans: bool(spans), seconds=40
    )
    operator_success: Final = eventually(
        lambda: _trace(audit_sinks.operator, operator_since, succeeded),
        lambda spans: any(OPERATION in span["attributes"] for span in spans),
        seconds=40,
    )

    assert any(ERROR_TYPE in span["attributes"] for span in operator_success), [
        (span["name"], sorted(span["attributes"])) for span in operator_success
    ]
    assert _trace(audit_sinks.tenant, tenant_since, succeeded) == ()
    assert _labels(delivered) == [{"user.id": "user-42", "langfuse.trace.tags": TAGS}]


def test_a_stream_that_breaks_after_the_200_reaches_a_failure_only_key(
    proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1", num_retries=0)
        key: Final = scenario.key(metadata={"logging": _failure_only_logging(audit_sinks)})
        since: Final = recorded_spans(audit_sinks.tenant)[0]
        broken: Final = _send(proxy, model, key, "broken-stream")

        delivered: Final = eventually(
            lambda: _model_spans(audit_sinks.tenant, since, broken), lambda spans: bool(spans), seconds=40
        )

    assert [ERROR_TYPE in span["attributes"] for span in delivered] == [True]
    assert _labels(delivered) == [{"user.id": "user-42", "langfuse.trace.tags": TAGS}]


def test_a_default_team_settings_team_gets_its_trace_with_its_env_secret(
    proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1", num_retries=0)
        key: Final = scenario.key(team_id=scenario.team(team_id=CONFIG_TEAM))
        since: Final = recorded_spans(audit_sinks.tenant)[0]
        posts_since: Final = len(recorded_requests(audit_sinks.tenant))
        trace_id: Final = _send(proxy, model, key, "success")

        delivered: Final = eventually(
            lambda: _model_spans(audit_sinks.tenant, since, trace_id), lambda spans: bool(spans), seconds=40
        )

    assert _labels(delivered) == [{"user.id": "user-42", "langfuse.trace.tags": TAGS}]
    assert _authorizations(audit_sinks.tenant, posts_since) == {_basic("pk-lf-audit", CONFIG_TEAM_SECRET)}


def test_the_operators_langfuse_project_keeps_every_request_of_a_failure_only_key(
    operator_langfuse_proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    gateway: Final = operator_langfuse_proxy
    with gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1", num_retries=0)
        key: Final = scenario.key(metadata={"logging": _failure_only_logging(audit_sinks)})
        tenant_since: Final = recorded_spans(audit_sinks.tenant)[0]
        operator_since: Final = recorded_spans(audit_sinks.arize)[0]
        operator_posts_since: Final = len(recorded_requests(audit_sinks.arize))
        succeeded: Final = _send(gateway, model, key, "success")
        failed: Final = _send(gateway, model, key, "failure")

        delivered: Final = eventually(
            lambda: _model_spans(audit_sinks.tenant, tenant_since, failed), lambda spans: bool(spans), seconds=40
        )
        operator: Final = eventually(
            lambda: {
                trace_id: _model_spans(audit_sinks.arize, operator_since, trace_id) for trace_id in (succeeded, failed)
            },
            lambda traces: all(traces.values()),
            seconds=40,
        )

    assert {trace_id: len(spans) for trace_id, spans in operator.items()} == {succeeded: 1, failed: 1}
    assert _authorizations(audit_sinks.arize, operator_posts_since) == {_basic(*OPERATOR_KEYS)}
    assert _trace(audit_sinks.tenant, tenant_since, succeeded) == ()
    assert _labels(delivered) == [{"user.id": "user-42", "langfuse.trace.tags": TAGS}]


def test_with_otel_v2_off_both_reach_langfuse_through_the_legacy_logger(
    v2_off_proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    gateway: Final = v2_off_proxy
    with gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1", num_retries=0)
        failure_only: Final = scenario.key(metadata={"logging": _failure_only_logging(audit_sinks)})
        config_team: Final = scenario.key(team_id=scenario.team(team_id=CONFIG_TEAM))
        since: Final = recorded_spans(audit_sinks.tenant)[0]
        succeeded: Final = _send(gateway, model, failure_only, "success")
        failed: Final = _send(gateway, model, failure_only, "failure")
        team_trace: Final = _send(gateway, model, config_team, "success")

        delivered: Final = eventually(
            lambda: {trace_id: _trace(audit_sinks.tenant, since, trace_id) for trace_id in (failed, team_trace)},
            lambda traces: all(
                any(span["attributes"].get("user.id") == "user-42" for span in spans) for spans in traces.values()
            ),
            seconds=40,
        )

    tagged: Final = {
        trace_id: any(_legacy_tags(span) == TAGS for span in spans) for trace_id, spans in delivered.items()
    }
    assert tagged == {failed: True, team_trace: True}, {
        trace_id: [span["attributes"].get("langfuse.trace.tags") for span in spans]
        for trace_id, spans in delivered.items()
    }
    assert _trace(audit_sinks.tenant, since, succeeded) == ()
