import asyncio
import base64
import json
import signal
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from _s3_v2_support import _chat_stream_frames, _responses_stream_frames
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows, scratch_database
from integration._support.process import group_members, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from pydantic import BaseModel, JsonValue, TypeAdapter

PUBLIC_KEY: Final = "pk-lf-integration"
SECRET_KEY: Final = "sk-lf-integration"
PROJECTS_PATH: Final = "/api/public/projects"
TRACES_PATH: Final = "/api/public/otel/v1/traces"
PROMPTS_PATH: Final = "/api/public/v2/prompts/"
STOCK_CONFIG: Final = Path("tests/integration/proxy_config.yaml")
CONFIG_SECTIONS: Final = ("litellm_settings", "environment_variables")
LANGFUSE_ENVIRONMENT: Final = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
INHERITED_ENVIRONMENT: Final = (*LANGFUSE_ENVIRONMENT, "DATABASE_URL_READ_REPLICA")
_PROXY_CONFIG: Final = TypeAdapter(dict[str, object])
_SETTINGS: Final = TypeAdapter(dict[str, object])


class _ProviderBody(BaseModel):
    messages: list[object]


def _completion(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + text,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


def _projects() -> Reply:
    return Reply(body=json.dumps({"data": [{"id": "integration-project", "name": "integration"}]}).encode())


def _text_prompt(name: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "type": "text",
                "name": name,
                "version": 1,
                "prompt": "Say {{word}}",
                "config": {},
                "labels": ["production"],
                "tags": [],
            }
        ).encode()
    )


def _langfuse_config(tmp_path: Path, general_settings: Mapping[str, JsonValue] | None = None) -> Path:
    config: Final = _PROXY_CONFIG.validate_python(yaml.safe_load(STOCK_CONFIG.read_text()))
    settings: Final = {
        **_SETTINGS.validate_python(config["litellm_settings"]),
        "success_callback": ["langfuse"],
        "failure_callback": ["langfuse"],
    }
    general: Final = {
        **_SETTINGS.validate_python(config["general_settings"]),
        **(general_settings or {}),
    }
    name: Final = "langfuse.yaml" if general_settings is None else "langfuse-merged.yaml"
    path: Final = tmp_path / name
    path.write_text(yaml.safe_dump({**config, "litellm_settings": settings, "general_settings": general}))
    return path


def _langfuse_environment(langfuse: Wire) -> dict[str, str]:
    return {
        "LANGFUSE_HOST": langfuse.url,
        "LANGFUSE_PUBLIC_KEY": PUBLIC_KEY,
        "LANGFUSE_SECRET_KEY": SECRET_KEY,
        "LANGFUSE_FLUSH_INTERVAL": "1",
    }


def _config_rows(database_url: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT param_name, param_value FROM "LiteLLM_Config" WHERE param_name IN (%s, %s) ORDER BY param_name',
        CONFIG_SECTIONS,
        database_url=database_url,
    )


def _attribute(entries: Sequence[KeyValue], key: str) -> str | list[str] | None:
    for entry in entries:
        if entry.key != key:
            continue
        if entry.value.HasField("array_value"):
            return [item.string_value for item in entry.value.array_value.values]
        return entry.value.string_value
    return None


def _spans(batches: Sequence[Request]) -> tuple[Span, ...]:
    return tuple(
        span
        for batch in batches
        if batch.target == TRACES_PATH and batch.headers.get("content-type") == "application/x-protobuf"
        for resource_spans in ExportTraceServiceRequest.FromString(batch.body).resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    )


def test_langfuse_callback_delivers_the_generation_over_otlp_v4_with_the_caller_trace_fields(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "langfuse" + uuid.uuid4().hex
    trace_id: Final = uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker

    def upstream(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {provider_secret}"
        return _completion(marker + "-answer")

    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        return Reply(body=b"", content_type="application/x-protobuf")

    with (
        wire_server(upstream) as provider,
        wire_server(langfuse) as destination,
        owned_proxy(
            gateway, tmp_path, _langfuse_environment(destination), config=_langfuse_config(tmp_path)
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker + "-question"}],
                "metadata": {
                    "trace_id": trace_id,
                    "trace_name": marker + "-trace",
                    "generation_name": marker,
                    "trace_user_id": marker + "-user",
                    "session_id": marker + "-session",
                    "tags": [marker],
                },
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones

        def exported() -> tuple[Span, ...]:
            received.extend(destination.drain())
            return tuple(span for span in _spans(received) if span.name == marker)

        spans: Final = eventually(exported, lambda values: len(values) == 1, seconds=20)
        span: Final = spans[0]
        posts: Final = tuple(request for request in received if request.method == "POST")
        assert {request.target for request in posts} == {TRACES_PATH}, [request.target for request in received]
        basic: Final = "Basic " + base64.b64encode(f"{PUBLIC_KEY}:{SECRET_KEY}".encode()).decode()
        for request in posts:
            assert request.headers["authorization"] == basic
            assert request.headers["content-type"] == "application/x-protobuf"
            assert request.headers["x-langfuse-ingestion-version"] == "4"
            assert provider_secret.encode() not in request.body
            assert candidate.key.encode() not in request.body

        assert span.trace_id.hex() == trace_id
        assert span.parent_span_id == b""
        attributes: Final = span.attributes
        assert _attribute(attributes, "langfuse.observation.type") == "generation"
        assert _attribute(attributes, "langfuse.trace.name") == marker + "-trace"
        assert _attribute(attributes, "user.id") == marker + "-user"
        assert _attribute(attributes, "session.id") == marker + "-session"
        assert marker in (_attribute(attributes, "langfuse.trace.tags") or ())
        assert _attribute(attributes, "langfuse.observation.model.name") == "openai/gpt-4o-mini"
        assert json.loads(str(_attribute(attributes, "langfuse.observation.usage_details"))) == {
            "input": 11,
            "output": 4,
            "total": 15,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        assert marker + "-question" in str(_attribute(attributes, "langfuse.observation.input"))
        assert marker + "-answer" in str(_attribute(attributes, "langfuse.observation.output"))
        assert (
            _attribute(attributes, "langfuse.observation.metadata.litellm_call_id")
            == response.headers["x-litellm-call-id"]
        )


def test_langfuse_callback_stored_in_the_db_through_config_update_delivers_the_generation_over_otlp_v4(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "langfusedb" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    public_key: Final = "pk-lf-db-" + marker
    secret_key: Final = "sk-lf-db-" + marker
    stock_settings: Final = _SETTINGS.validate_python(
        _PROXY_CONFIG.validate_python(yaml.safe_load(STOCK_CONFIG.read_text()))["litellm_settings"]
    )
    assert "langfuse" not in json.dumps(
        [stock_settings.get(key) for key in ("callbacks", "success_callback", "failure_callback")]
    )

    def upstream(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {provider_secret}"
        return _completion(marker + "-answer")

    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        return Reply(body=b"", content_type="application/x-protobuf")

    with (
        scratch_database() as scratch_url,
        wire_server(upstream) as provider,
        wire_server(langfuse) as destination,
        owned_proxy(
            gateway,
            tmp_path,
            {"DATABASE_URL": scratch_url, "LANGFUSE_FLUSH_INTERVAL": "1"},
            remove_environment=INHERITED_ENVIRONMENT,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        candidate.post(
            "/config/update",
            {
                "litellm_settings": {"success_callback": ["langfuse"]},
                "environment_variables": {
                    "LANGFUSE_HOST": destination.url,
                    "LANGFUSE_PUBLIC_KEY": public_key,
                    "LANGFUSE_SECRET_KEY": secret_key,
                },
            },
        )
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        body: Final = candidate.post(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker + "-question"}],
                "metadata": {"generation_name": marker},
                "cache": {"no-cache": True},
            },
        )
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones

        def exported() -> tuple[Span, ...]:
            received.extend(destination.drain())
            return tuple(span for span in _spans(received) if span.name == marker)

        spans: Final = eventually(exported, lambda values: len(values) == 1, seconds=20)
        posts: Final = tuple(request for request in received if request.method == "POST")
        assert {request.target for request in posts} == {TRACES_PATH}, [request.target for request in received]
        basic: Final = "Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        for request in posts:
            assert request.headers["authorization"] == basic
            assert request.headers["content-type"] == "application/x-protobuf"
            assert request.headers["x-langfuse-ingestion-version"] == "4"
            assert provider_secret.encode() not in request.body
            assert candidate.key.encode() not in request.body

        attributes: Final = spans[0].attributes
        assert _attribute(attributes, "langfuse.observation.type") == "generation"
        assert _attribute(attributes, "langfuse.observation.metadata.response_id") == string_value(body["id"])
        assert marker + "-question" in str(_attribute(attributes, "langfuse.observation.input"))
        assert marker + "-answer" in str(_attribute(attributes, "langfuse.observation.output"))

        stored: Final = {string_value(row["param_name"]): row["param_value"] for row in _config_rows(scratch_url)}
        callbacks: Final = TypeAdapter(list[str]).validate_python(
            object_value(stored["litellm_settings"]).get("success_callback") or []
        )
        assert "langfuse" in callbacks, stored
        assert set(object_value(stored["environment_variables"])) >= set(LANGFUSE_ENVIRONMENT), stored
        assert secret_key not in json.dumps(stored["environment_variables"]), stored


def test_prompt_fetch_encodes_the_name_retries_a_5xx_once_and_keeps_langfuse_headers_off_the_client(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "prompt" + uuid.uuid4().hex
    leak: Final = "leak-" + marker
    flaky_prompt: Final = f"{marker}/what?"
    encoded_flaky_prompt: Final = f"{marker}%2Fwhat%3F"
    missing_prompt: Final = marker + "-missing"

    seen_prompt_gets: Final[list[str]] = []  # mutable-ok: the double counts attempts across requests

    def upstream(request: Request) -> Reply:
        return _completion(marker + "-answer")

    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        if request.method == "POST":
            return Reply(body=b"", content_type="application/x-protobuf")
        assert request.target.startswith(PROMPTS_PATH), request.target
        assert request.headers["authorization"].startswith("Basic ")
        if request.target.startswith(PROMPTS_PATH + encoded_flaky_prompt):
            prior: Final = sum(1 for seen in seen_prompt_gets if seen.startswith(PROMPTS_PATH + encoded_flaky_prompt))
            seen_prompt_gets.append(request.target)
            if prior == 0:
                return Reply(status=503, body=b'{"message":"try later"}', headers={"retry-after": "30"})
            return _text_prompt(flaky_prompt)
        seen_prompt_gets.append(request.target)
        return Reply(
            status=404,
            body=b'{"message":"Prompt not found","error":"LangfuseNotFoundError"}',
            headers={"set-cookie": f"session={leak}; Path=/", "x-upstream-internal": leak, "server": leak},
        )

    with (
        wire_server(upstream) as provider,
        wire_server(langfuse) as destination,
        owned_proxy(
            gateway, tmp_path, _langfuse_environment(destination), config=_langfuse_config(tmp_path)
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        flaky: Final = scenario.model(
            model="langfuse/gpt-4o-mini", prompt_id=flaky_prompt, api_base=provider.url + "/v1", api_key="synthetic"
        )
        missing: Final = scenario.model(
            model="langfuse/gpt-4o-mini", prompt_id=missing_prompt, api_base=provider.url + "/v1", api_key="synthetic"
        )
        started: Final = time.monotonic()
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": flaky, "messages": [{"role": "user", "content": marker}], "prompt_variables": {"word": marker}},
        )
        elapsed: Final = time.monotonic() - started
        assert response.status_code == 200, response.text
        assert elapsed < 5, f"a retried cold prompt miss took {elapsed:.1f}s"
        attempts: Final = tuple(
            target for target in seen_prompt_gets if target.startswith(PROMPTS_PATH + encoded_flaky_prompt)
        )
        assert len(attempts) == 2, seen_prompt_gets
        assert all(target.split("?", 1)[0] == PROMPTS_PATH + encoded_flaky_prompt for target in attempts), attempts
        sent: Final = _ProviderBody.model_validate_json(provider.drain()[-1].body).messages
        assert any("Say " + marker in json.dumps(message) for message in sent), sent

        failure: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": missing, "messages": [{"role": "user", "content": marker}], "prompt_variables": {"word": marker}},
        )
        assert failure.status_code == 404, failure.text
        assert "Prompt not found" in failure.text
        assert leak not in failure.text
        assert leak not in json.dumps(dict(failure.headers))
        assert "set-cookie" not in failure.headers and "x-upstream-internal" not in failure.headers
        assert sum(1 for target in seen_prompt_gets if target.startswith(PROMPTS_PATH + missing_prompt)) == 1


def _responses_result(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": "msg_" + identity,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "integration answer", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
            }
        ).encode()
    )


def _trace_body(kind: str, model: str, marker: str, metadata: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    metadata_field: Final[dict[str, JsonValue]] = {} if metadata is None else {"metadata": dict(metadata)}
    if kind == "responses":
        return {"model": model, "input": marker + "-question", **metadata_field}
    if kind == "messages":
        return {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": marker + "-question"}],
            **metadata_field,
        }
    return {
        "model": model,
        "messages": [{"role": "user", "content": marker + "-question"}],
        "cache": {"no-cache": True},
        **metadata_field,
    }


def _w3c_headers(header_trace: str, baggage_session: str | None) -> dict[str, str]:
    baggage_field: Final = {} if baggage_session is None else {"baggage": f"session.id={baggage_session}"}
    return {"traceparent": f"00-{header_trace}-00f067aa0ba902b7-01", **baggage_field}


def _await_span(received: list[Request], destination: Wire, call_id: str) -> Span:
    def exported() -> tuple[Span, ...]:
        received.extend(destination.drain())
        return tuple(
            span
            for span in _spans(received)
            if _attribute(span.attributes, "langfuse.observation.metadata.litellm_call_id") == call_id
        )

    spans: Final = eventually(exported, lambda values: len(values) == 1, seconds=20)
    return spans[0]


@pytest.mark.parametrize(
    ("endpoint", "kind", "metadata_mode", "expected_trace", "expected_session", "expected_target"),
    (
        pytest.param(
            "/v1/chat/completions", "chat", "both", "caller", "caller", "/v1/chat/completions", id="chat_caller_ids"
        ),
        pytest.param(
            "/v1/responses", "responses", "both", "caller", "caller", "/v1/responses", id="responses_caller_ids"
        ),
        pytest.param(
            "/v1/messages", "messages", "both", "caller", "caller", "/v1/responses", id="messages_caller_ids"
        ),
        pytest.param(
            "/v1/chat/completions", "chat", "none", "header", "baggage", "/v1/chat/completions", id="chat_header_ids"
        ),
        pytest.param(
            "/v1/chat/completions",
            "chat",
            "trace",
            "caller",
            "baggage",
            "/v1/chat/completions",
            id="chat_caller_trace_header_session",
        ),
        pytest.param(
            "/v1/chat/completions",
            "chat",
            "empty_session",
            "header",
            "baggage",
            "/v1/chat/completions",
            id="chat_empty_session_header_ids",
        ),
    ),
)
def test_langfuse_trace_and_session_prefer_caller_metadata_over_w3c_headers(
    gateway: Gateway,
    tmp_path: Path,
    endpoint: str,
    kind: str,
    metadata_mode: str,
    expected_trace: str,
    expected_session: str,
    expected_target: str,
) -> None:
    marker: Final = "w3c" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    caller_trace: Final = uuid.uuid4().hex
    caller_session: Final = f"my-session-id-{marker}"
    metadata: Final = {
        "both": {"trace_id": caller_trace, "session_id": caller_session},
        "trace": {"trace_id": caller_trace},
        "empty_session": {"session_id": ""},
        "none": None,
    }[metadata_mode]
    expected_trace_value: Final = {"caller": caller_trace, "header": header_trace}[expected_trace]
    expected_session_value: Final = {"caller": caller_session, "baggage": baggage_session}[expected_session]
    upstream_targets: Final[list[str]] = []  # mutable-ok: records which upstream endpoint each call hit

    def upstream(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {provider_secret}"
        upstream_targets.append(request.target)
        if request.target == "/v1/responses":
            return _responses_result("resp-" + marker)
        assert request.target == "/v1/chat/completions", request.target
        return _completion(marker + "-answer")

    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        return Reply(body=b"", content_type="application/x-protobuf")

    with (
        wire_server(upstream) as provider,
        wire_server(langfuse) as destination,
        owned_proxy(
            gateway, tmp_path, _langfuse_environment(destination), config=_langfuse_config(tmp_path)
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, metadata),
            headers=_w3c_headers(header_trace, baggage_session),
        )
        assert response.status_code == 200, response.text
        assert upstream_targets == [expected_target], upstream_targets
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        span: Final = _await_span(received, destination, response.headers["x-litellm-call-id"])
        assert span.trace_id.hex() == expected_trace_value
        assert _attribute(span.attributes, "session.id") == expected_session_value


def test_missing_session_id_reject_accepts_caller_metadata_and_baggage_fallback(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "reject" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    upstream_targets: Final[list[str]] = []  # mutable-ok: records which upstream endpoint each call hit

    def upstream(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {provider_secret}"
        upstream_targets.append(request.target)
        if request.target == "/v1/responses":
            return _responses_result("resp-" + marker)
        assert request.target == "/v1/chat/completions", request.target
        return _completion(marker + "-answer")

    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        return Reply(body=b"", content_type="application/x-protobuf")

    with (
        wire_server(upstream) as provider,
        wire_server(langfuse) as destination,
        owned_proxy(
            gateway,
            tmp_path,
            _langfuse_environment(destination),
            config=_langfuse_config(tmp_path, {"missing_session_id": "reject"}),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones

        caller_session: Final = f"my-session-id-{marker}-r1"
        header_trace: Final = uuid.uuid4().hex
        first: Final = candidate.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": marker + "-r1", "metadata": {"session_id": caller_session}},
            headers=_w3c_headers(header_trace, None),
        )
        assert first.status_code == 200, first.text
        first_span: Final = _await_span(received, destination, first.headers["x-litellm-call-id"])
        assert _attribute(first_span.attributes, "session.id") == caller_session

        baggage_session: Final = "baggage-" + marker + "-r2"
        second: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker + "-r2"}],
                "metadata": {"session_id": ""},
            },
            headers=_w3c_headers(uuid.uuid4().hex, baggage_session),
        )
        assert second.status_code == 200, second.text
        second_span: Final = _await_span(received, destination, second.headers["x-litellm-call-id"])
        assert _attribute(second_span.attributes, "session.id") == baggage_session

        third: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker + "-r3"}]},
            headers=_w3c_headers(uuid.uuid4().hex, None),
        )
        assert third.status_code == 400, third.text
        assert upstream_targets == ["/v1/responses", "/v1/chat/completions"], upstream_targets


@pytest.mark.parametrize(
    ("endpoint", "kind", "expected_target"),
    (
        pytest.param("/v1/responses", "responses", "/v1/responses", id="responses_caller_trace"),
        pytest.param("/v1/messages", "messages", "/v1/responses", id="messages_caller_trace"),
    ),
)
def test_missing_session_id_generate_derives_session_from_caller_trace(
    gateway: Gateway, tmp_path: Path, endpoint: str, kind: str, expected_target: str
) -> None:
    marker: Final = "gen" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    caller_trace: Final = uuid.uuid4().hex
    upstream_targets: Final[list[str]] = []  # mutable-ok: records which upstream endpoint each call hit

    def upstream(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {provider_secret}"
        upstream_targets.append(request.target)
        if request.target == "/v1/responses":
            return _responses_result("resp-" + marker)
        assert request.target == "/v1/chat/completions", request.target
        return _completion(marker + "-answer")

    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        return Reply(body=b"", content_type="application/x-protobuf")

    with (
        wire_server(upstream) as provider,
        wire_server(langfuse) as destination,
        owned_proxy(
            gateway,
            tmp_path,
            _langfuse_environment(destination),
            config=_langfuse_config(tmp_path, {"missing_session_id": "generate"}),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, {"trace_id": caller_trace}),
            headers=_w3c_headers(uuid.uuid4().hex, None),
        )
        assert response.status_code == 200, response.text
        assert upstream_targets == [expected_target], upstream_targets
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        span: Final = _await_span(received, destination, response.headers["x-litellm-call-id"])
        assert _attribute(span.attributes, "session.id") == caller_trace


_AUDIT_ENDPOINTS: Final = (
    pytest.param("/v1/chat/completions", "chat", "/v1/chat/completions", id="chat"),
    pytest.param("/v1/responses", "responses", "/v1/responses", id="responses"),
    pytest.param("/v1/messages", "messages", "/v1/responses", id="messages"),
)


def _audit_upstream(provider_secret: str, marker: str, targets: list[str]):
    def upstream(request: Request) -> Reply:
        if not request.body:
            return Reply(status=404)
        assert request.headers["authorization"] == f"Bearer {provider_secret}"
        targets.append(request.target)
        body: Final = TypeAdapter(dict[str, JsonValue]).validate_json(request.body)
        index: Final = len(targets)
        if request.target == "/v1/responses":
            if body.get("stream"):
                return Reply(
                    content_type="text/event-stream", chunks=_responses_stream_frames(f"resp-{marker}-{index}")
                )
            return _responses_result(f"resp-{marker}-{index}")
        assert request.target == "/v1/chat/completions", request.target
        if body.get("stream"):
            return Reply(
                content_type="text/event-stream", chunks=_chat_stream_frames(f"chatcmpl-{marker}-{index}")
            )
        return _completion(f"{marker}-{index}-answer")

    return upstream


def _audit_sink():
    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        return Reply(body=b"", content_type="application/x-protobuf")

    return langfuse


@dataclass(frozen=True, slots=True)
class _AuditRig:
    candidate: Gateway
    scenario: Scenario
    destination: Wire


@pytest.fixture(scope="module")
def audit_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_AuditRig]:
    directory: Final = tmp_path_factory.mktemp("audit")
    with (
        gateway_from_environment() as gateway,
        wire_server(_audit_sink()) as destination,
        owned_proxy(
            gateway, directory, _langfuse_environment(destination), config=_langfuse_config(directory)
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield _AuditRig(candidate, scenario, destination)


@pytest.fixture(scope="module")
def audit_generate_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_AuditRig]:
    directory: Final = tmp_path_factory.mktemp("audit-generate")
    with (
        gateway_from_environment() as gateway,
        wire_server(_audit_sink()) as destination,
        owned_proxy(
            gateway,
            directory,
            _langfuse_environment(destination),
            config=_langfuse_config(directory, {"missing_session_id": "generate"}),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield _AuditRig(candidate, scenario, destination)


@pytest.fixture(scope="module")
def audit_reject_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_AuditRig]:
    directory: Final = tmp_path_factory.mktemp("audit-reject")
    with (
        gateway_from_environment() as gateway,
        wire_server(_audit_sink()) as destination,
        owned_proxy(
            gateway,
            directory,
            _langfuse_environment(destination),
            config=_langfuse_config(directory, {"missing_session_id": "reject"}),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield _AuditRig(candidate, scenario, destination)


@pytest.fixture(scope="module")
def audit_omit_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_AuditRig]:
    directory: Final = tmp_path_factory.mktemp("audit-omit")
    with (
        gateway_from_environment() as gateway,
        wire_server(_audit_sink()) as destination,
        owned_proxy(
            gateway,
            directory,
            _langfuse_environment(destination),
            config=_langfuse_config(directory, {"missing_session_id": "omit"}),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield _AuditRig(candidate, scenario, destination)


@pytest.fixture(scope="module")
def audit_otel_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_AuditRig]:
    directory: Final = tmp_path_factory.mktemp("audit-otel")

    def otlp(request: Request) -> Reply:
        return Reply()

    with (
        gateway_from_environment() as gateway,
        wire_server(otlp) as destination,
        owned_proxy(
            gateway,
            directory,
            {"LITELLM_OTEL_V2": "1", "OTEL_BSP_SCHEDULE_DELAY": "300"},
            config=_otel_config(directory, destination.url),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield _AuditRig(candidate, scenario, destination)


def _await_spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT session_id, status, metadata, cache_hit FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s',
            (call_id,),
        ),
        lambda values: len(values) == 1,
        seconds=240,
    )
    return rows[0]


def _assert_call(
    response: httpx.Response,
    received: list[Request],
    destination: Wire,
    targets: list[str],
    expected_target: str,
    expected_trace: str | None,
    expected_session: str | None,
) -> None:
    call_id: Final = response.headers["x-litellm-call-id"]
    assert targets == [expected_target], targets
    _assert_span_spend(response, received, destination, expected_trace, expected_session)


def _assert_span_spend(
    response: httpx.Response,
    received: list[Request],
    destination: Wire,
    expected_trace: str | None,
    expected_session: str | None,
) -> None:
    call_id: Final = response.headers["x-litellm-call-id"]
    span: Final = _await_span(received, destination, call_id)
    if expected_trace is not None:
        assert span.trace_id.hex() == expected_trace, f"call {call_id}: trace id"
    assert _attribute(span.attributes, "session.id") == expected_session, f"call {call_id}: session.id"
    row: Final = _await_spend_row(call_id)
    assert row["session_id"] == expected_session, f"call {call_id}: spend session {row}"


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
@pytest.mark.parametrize("metadata_mode", ("both", "none", "trace", "session"))
def test_audit_caller_metadata_wins_over_w3c_per_field(
    audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str, metadata_mode: str
) -> None:
    marker: Final = "audit" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    caller_trace: Final = uuid.uuid4().hex
    caller_session: Final = f"my-session-id-{marker}"
    metadata: Final = {
        "both": {"trace_id": caller_trace, "session_id": caller_session},
        "trace": {"trace_id": caller_trace},
        "session": {"session_id": caller_session},
        "none": None,
    }[metadata_mode]
    expected_trace: Final = caller_trace if metadata_mode in ("both", "trace") else header_trace
    expected_session: Final = caller_session if metadata_mode in ("both", "session") else baggage_session
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_rig.candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, metadata),
            headers=_w3c_headers(header_trace, baggage_session),
        )
        assert response.status_code == 200, response.text
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        _assert_call(
            response, received, audit_rig.destination, targets, expected_target, expected_trace, expected_session
        )


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
def test_audit_caller_metadata_wins_on_streamed_calls(
    audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str
) -> None:
    marker: Final = "auditstream" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    caller_trace: Final = uuid.uuid4().hex
    caller_session: Final = f"my-session-id-{marker}"
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_rig.candidate.request(
            "POST",
            endpoint,
            {
                **_trace_body(kind, model, marker, {"trace_id": caller_trace, "session_id": caller_session}),
                "stream": True,
            },
            headers=_w3c_headers(header_trace, baggage_session),
        )
        assert response.status_code == 200, response.text
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        _assert_call(response, received, audit_rig.destination, targets, expected_target, caller_trace, caller_session)


def test_audit_caller_metadata_wins_through_official_sdk_clients(audit_rig: _AuditRig) -> None:
    marker: Final = "auditsdk" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        candidate: Final = audit_rig.candidate
        destination: Final = audit_rig.destination
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        base_url: Final = str(candidate.client.base_url).rstrip("/")
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones

        def drive(client_kind: str) -> tuple[str, str, httpx.Headers]:
            caller_trace: Final = uuid.uuid4().hex
            caller_session: Final = f"{client_kind}-session-{marker}"
            body_metadata: Final = {
                "trace_id": caller_trace,
                "session_id": caller_session,
                "generation_name": marker + "-" + client_kind,
            }
            headers: Final = _w3c_headers(uuid.uuid4().hex, "baggage-" + marker + "-" + client_kind)
            if client_kind == "chat_openai_sync":
                return caller_trace, caller_session, (
                    openai.OpenAI(base_url=f"{base_url}/v1", api_key=candidate.key)
                    .chat.completions.with_raw_response.create(
                        model=model,
                        messages=[{"role": "user", "content": marker + "-chat"}],
                        extra_body={"metadata": body_metadata, "cache": {"no-cache": True}},
                        extra_headers=headers,
                    )
                    .headers
                )
            if client_kind == "responses_openai_async":

                async def responses_call() -> httpx.Headers:
                    answer: Final = await openai.AsyncOpenAI(
                        base_url=f"{base_url}/v1", api_key=candidate.key
                    ).responses.with_raw_response.create(
                        model=model,
                        input=marker + "-responses",
                        extra_body={"metadata": body_metadata, "cache": {"no-cache": True}},
                        extra_headers=headers,
                    )
                    return answer.headers

                return caller_trace, caller_session, asyncio.run(responses_call())
            if client_kind == "messages_anthropic_sync":
                return caller_trace, caller_session, (
                    anthropic.Anthropic(base_url=base_url, api_key=candidate.key)
                    .messages.with_raw_response.create(
                        model=model,
                        max_tokens=16,
                        messages=[{"role": "user", "content": marker + "-messages"}],
                        extra_body={"metadata": body_metadata},
                        extra_headers=headers,
                    )
                    .headers
                )

            async def messages_call() -> httpx.Headers:
                answer: Final = await anthropic.AsyncAnthropic(
                    base_url=base_url, api_key=candidate.key
                ).messages.with_raw_response.create(
                    model=model,
                    max_tokens=16,
                    messages=[{"role": "user", "content": marker + "-messages-async"}],
                    extra_body={"metadata": body_metadata},
                    extra_headers=headers,
                )
                return answer.headers

            return caller_trace, caller_session, asyncio.run(messages_call())

        expected: Final = {
            call_headers["x-litellm-call-id"]: (caller_trace, caller_session, client_kind)
            for client_kind, (caller_trace, caller_session, call_headers) in (
                (kind, drive(kind))
                for kind in (
                    "chat_openai_sync",
                    "responses_openai_async",
                    "messages_anthropic_sync",
                    "messages_anthropic_async",
                )
            )
        }

        def spans_named() -> tuple[Span, ...]:
            received.extend(destination.drain())
            return tuple(
                span
                for span in _spans(received)
                if _attribute(span.attributes, "langfuse.observation.metadata.litellm_call_id") in expected
            )

        spans: Final = eventually(spans_named, lambda values: len(values) == len(expected), seconds=60)
        for span in spans:
            call_id: Final = str(
                _attribute(span.attributes, "langfuse.observation.metadata.litellm_call_id")
            )
            caller_trace, caller_session, client_kind = expected[call_id]
            assert span.trace_id.hex() == caller_trace, f"{client_kind}: trace id"
            assert _attribute(span.attributes, "session.id") == caller_session, f"{client_kind}: session.id"
            row: Final = _await_spend_row(call_id)
            assert row["session_id"] == caller_session, f"{client_kind} {call_id}: spend session {row}"


def _otel_config(tmp_path: Path, sink_url: str) -> Path:
    config: Final = _PROXY_CONFIG.validate_python(yaml.safe_load(STOCK_CONFIG.read_text()))
    settings: Final = {**_SETTINGS.validate_python(config["litellm_settings"]), "callbacks": ["otel"]}
    path: Final = tmp_path / "otel.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "litellm_settings": settings,
                "callback_settings": {
                    "otel": {"exporter": "http/json", "endpoint": sink_url, "mapper_names": ["genai"]}
                },
            }
        )
    )
    return path


_OTEL_SPAN: Final = TypeAdapter(dict[str, JsonValue])


def _otel_spans(batches: Sequence[Request]) -> tuple[dict[str, JsonValue], ...]:
    spans: list[dict[str, JsonValue]] = []  # mutable-ok: flattens nested OTLP batches into a tuple
    for batch in batches:
        if not batch.target.endswith("/v1/traces"):
            continue
        payload: Final = TypeAdapter(JsonValue).validate_json(batch.body)
        envelopes: Final = payload if isinstance(payload, list) else [payload]
        for envelope in envelopes:
            for resource in TypeAdapter(list[JsonValue]).validate_python(
                object_value(envelope)["resourceSpans"]
            ):
                for scope in object_value(resource)["scopeSpans"]:
                    spans.extend(TypeAdapter(list[JsonValue]).validate_python(object_value(scope)["spans"]))
    return tuple(_OTEL_SPAN.validate_python(span) for span in spans)


def _otel_attribute(span: Mapping[str, JsonValue], key: str) -> str | None:
    for attribute in TypeAdapter(list[JsonValue]).validate_python(span.get("attributes") or []):
        entry: Final = object_value(attribute)
        if entry["key"] == key:
            value: Final = object_value(entry["value"])
            raw: Final = value.get("stringValue")
            return str(raw) if raw is not None else None
    return None


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS[:2])
def test_audit_otel_span_carries_caller_ids(
    audit_otel_rig: _AuditRig, endpoint: str, kind: str, expected_target: str
) -> None:
    marker: Final = "auditotel" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    caller_trace: Final = uuid.uuid4().hex
    caller_session: Final = f"my-session-id-{marker}"
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        destination: Final = audit_otel_rig.destination
        model: Final = audit_otel_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_otel_rig.candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, {"trace_id": caller_trace, "session_id": caller_session}),
            headers=_w3c_headers(header_trace, baggage_session),
        )
        assert response.status_code == 200, response.text
        assert targets == [expected_target], targets
        response_id: Final = string_value(object_value(response.json())["id"])
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones

        def exported() -> tuple[dict[str, JsonValue], ...]:
            received.extend(destination.drain())
            return tuple(
                span
                for span in _otel_spans(received)
                if _otel_attribute(span, "gen_ai.response.id") == response_id
            )

        spans: Final = eventually(exported, lambda values: len(values) == 1, seconds=60)
        span: Final = spans[0]
        print(
            f"H7 record: otel span trace={span['traceId']} header={header_trace} "
            f"session.id={_otel_attribute(span, 'session.id')} "
            f"gen_ai.conversation.id={_otel_attribute(span, 'gen_ai.conversation.id')}"
        )
        assert span["traceId"] == header_trace, (
            f"otel span for {response_id}: trace id must be the ambient W3C header trace"
        )
        assert _otel_attribute(span, "gen_ai.conversation.id") == caller_session, (
            f"otel span conversation id for {response_id}"
        )


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
@pytest.mark.parametrize(
    ("case", "metadata_mode", "headers_mode", "expected_session"),
    (
        pytest.param("g1", "trace", "traceparent", "caller_trace", id="g1_caller_trace_and_header"),
        pytest.param("g3", "none", "traceparent", "header_trace", id="g3_header_only"),
        pytest.param("g5", "session", "traceparent", "caller_session", id="g5_caller_session"),
    ),
)
def test_audit_generate_policy_derives_session_from_caller_trace(
    audit_generate_rig: _AuditRig,
    endpoint: str,
    kind: str,
    expected_target: str,
    case: str,
    metadata_mode: str,
    headers_mode: str,
    expected_session: str,
) -> None:
    marker: Final = "auditgen" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    caller_trace: Final = uuid.uuid4().hex
    caller_session: Final = f"my-session-id-{marker}"
    metadata: Final = {"trace": {"trace_id": caller_trace}, "session": {"session_id": caller_session}, "none": None}[
        metadata_mode
    ]
    expected: Final = {"caller_trace": caller_trace, "header_trace": header_trace, "caller_session": caller_session}[
        expected_session
    ]
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_generate_rig.scenario.model(
            api_base=provider.url + "/v1", api_key=provider_secret
        )
        response: Final = audit_generate_rig.candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, metadata),
            headers={"traceparent": f"00-{header_trace}-00f067aa0ba902b7-01"},
        )
        assert response.status_code == 200, response.text
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        call_id: Final = response.headers["x-litellm-call-id"]
        assert targets == [expected_target], targets
        span: Final = _await_span(received, audit_generate_rig.destination, call_id)
        assert _attribute(span.attributes, "session.id") == expected, f"call {call_id}: session.id"
        row: Final = _await_spend_row(call_id)
        assert row["session_id"] == expected, f"call {call_id}: spend session {row}"



@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
def test_audit_generate_policy_is_stable_across_repeated_caller_trace(
    audit_generate_rig: _AuditRig, endpoint: str, kind: str, expected_target: str
) -> None:
    marker: Final = "auditgen2" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    caller_trace: Final = uuid.uuid4().hex
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_generate_rig.scenario.model(
            api_base=provider.url + "/v1", api_key=provider_secret
        )
        responses: Final = tuple(
            audit_generate_rig.candidate.request(
                "POST",
                endpoint,
                _trace_body(kind, model, f"{marker}-{attempt}", {"trace_id": caller_trace}),
            )
            for attempt in ("first", "second")
        )
        sessions: Final[list[str | None]] = []  # mutable-ok: collects the two observed sessions in order
        for attempt, response in zip(("first", "second"), responses):
            assert response.status_code == 200, f"{attempt}: {response.text}"
            row: Final = _await_spend_row(response.headers["x-litellm-call-id"])
            sessions.append(string_value(row["session_id"]))
        assert sessions == [caller_trace, caller_trace], (
            f"generate must derive both sessions from the caller trace {caller_trace}"
        )


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
def test_audit_generate_policy_fresh_session_without_any_ids(
    audit_generate_rig: _AuditRig, endpoint: str, kind: str, expected_target: str
) -> None:
    marker: Final = "auditgen4" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_generate_rig.scenario.model(
            api_base=provider.url + "/v1", api_key=provider_secret
        )
        sessions: Final[list[str | None]] = []  # mutable-ok: collects the two observed sessions in order
        for attempt in ("first", "second"):
            response: Final = audit_generate_rig.candidate.request(
                "POST", endpoint, _trace_body(kind, model, f"{marker}-{attempt}", None)
            )
            assert response.status_code == 200, response.text
            session: Final = _await_spend_row(response.headers["x-litellm-call-id"])["session_id"]
            assert session, f"{attempt}: generated session id must be non-empty"
            sessions.append(string_value(session))
        assert sessions[0] != sessions[1], f"two id-less calls must not share a session: {sessions}"
        assert targets and set(targets) == {expected_target}, targets


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
@pytest.mark.parametrize(
    ("case", "metadata", "baggage", "expected_status", "expected_session"),
    (
        pytest.param(
            "r1", "caller_session", None, 200, "caller_session", id="r1_caller_session_no_baggage"
        ),
        pytest.param("r2", "empty_session", "baggage", 200, "baggage", id="r2_empty_session_baggage"),
        pytest.param("r3", "none", None, 400, None, id="r3_nothing_rejected"),
    ),
)
def test_audit_reject_policy(
    audit_reject_rig: _AuditRig,
    endpoint: str,
    kind: str,
    expected_target: str,
    case: str,
    metadata: str,
    baggage: str,
    expected_status: int,
    expected_session: str | None,
) -> None:
    marker: Final = "auditreject" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    baggage_session: Final = "baggage-" + marker
    caller_session: Final = f"my-session-id-{marker}"
    body_metadata: Final = {
        "caller_session": {"session_id": caller_session},
        "empty_session": {"session_id": ""},
        "none": None,
    }[metadata]
    expected: Final = {"caller_session": caller_session, "baggage": baggage_session}[expected_session] if expected_session else None
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_reject_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_reject_rig.candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, body_metadata),
            headers=_w3c_headers(uuid.uuid4().hex, baggage_session if baggage else None),
        )
        assert response.status_code == expected_status, response.text
        if expected_status != 200:
            assert targets == [], targets
            rejected_rows: Final = eventually(
                lambda: read_rows(
                    'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s',
                    (response.headers["x-litellm-call-id"],),
                ),
                lambda values: len(values) == 1,
                seconds=240,
            )
            assert rejected_rows[0]["status"] == "failure", (
                f"rejected call {response.headers['x-litellm-call-id']} must write exactly one failure spend row: {rejected_rows}"
            )
            return
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        _assert_call(response, received, audit_reject_rig.destination, targets, expected_target, None, expected)


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
def test_audit_omit_policy_records_no_session(audit_omit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str) -> None:
    marker: Final = "auditomit" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_omit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_omit_rig.candidate.request(
            "POST", endpoint, _trace_body(kind, model, marker, None)
        )
        assert response.status_code == 200, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        assert targets == [expected_target], targets
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        span: Final = _await_span(received, audit_omit_rig.destination, call_id)
        row: Final = _await_spend_row(call_id)
        span_session: Final = _attribute(span.attributes, "session.id")
        assert row["session_id"] == span_session, (
            f"call {call_id}: spend session {row['session_id']!r} must match span session {span_session!r}"
        )


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
@pytest.mark.parametrize("bad_value", (123, ["x"]), ids=["int", "list"])
def test_audit_non_string_caller_ids_are_ignored_consistently(
    audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str, bad_value: JsonValue
) -> None:
    marker: Final = "auditbad" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        candidate: Final = audit_rig.candidate
        destination: Final = audit_rig.destination
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        outcomes: Final[list[tuple[str | None, str | None, str | None]]] = (
            []
        )  # mutable-ok: collects (span trace, span session, spend session) per leg
        for attempt, headers in (
            ("with_headers", _w3c_headers(uuid.uuid4().hex, "baggage-" + marker)),
            ("no_headers", {}),
        ):
            response: Final = candidate.request(
                "POST",
                endpoint,
                _trace_body(
                    kind,
                    model,
                    f"{marker}-{attempt}",
                    {"trace_id": bad_value, "session_id": bad_value},
                ),
                headers=headers,
            )
            assert response.status_code == 200, f"{attempt}: {response.text}"
            call_id: Final = response.headers["x-litellm-call-id"]
            span: Final = _await_span(received, destination, call_id)
            row: Final = _await_spend_row(call_id)
            outcomes.append(
                (
                    span.trace_id.hex(),
                    str(_attribute(span.attributes, "session.id")),
                    str(row["session_id"]),
                )
            )
        assert outcomes[0] == outcomes[1], (
            f"W3C headers must not change the outcome for caller {bad_value!r}: {outcomes}"
        )


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
@pytest.mark.parametrize(
    "metadata",
    ({"trace_id": "", "session_id": ""}, {"trace_id": None, "session_id": None}),
    ids=["empty", "null"],
)
def test_audit_empty_and_null_caller_ids_fall_back_to_w3c(
    audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str, metadata: dict[str, object]
) -> None:
    marker: Final = "auditempty" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_rig.candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, dict(metadata)),
            headers=_w3c_headers(header_trace, baggage_session),
        )
        assert response.status_code == 200, response.text
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        _assert_call(response, received, audit_rig.destination, targets, expected_target, header_trace, baggage_session)


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS[:2])
def test_audit_five_kilobyte_caller_ids_win_verbatim(
    audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str
) -> None:
    marker: Final = "auditbig" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    caller_trace: Final = "T" * 5120
    caller_session: Final = "S" * 5120
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_rig.candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, {"trace_id": caller_trace, "session_id": caller_session}),
            headers=_w3c_headers(header_trace, baggage_session),
        )
        assert response.status_code == 200, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        assert targets == [expected_target], targets
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        span: Final = _await_span(received, audit_rig.destination, call_id)
        assert span.trace_id.hex() != header_trace, f"call {call_id}: caller trace must beat the W3C header"
        assert _attribute(span.attributes, "session.id") == caller_session, f"call {call_id}: session.id"
        assert _await_spend_row(call_id)["session_id"] == caller_session, f"call {call_id}: spend session"


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS)
def test_audit_identical_requests_twice_log_per_call(
    audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str
) -> None:
    marker: Final = "auditdup" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    caller_trace: Final = uuid.uuid4().hex
    caller_session: Final = f"my-session-id-{marker}"
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        for attempt in ("first", "second"):
            response: Final = audit_rig.candidate.request(
                "POST",
                endpoint,
                _trace_body(
                    kind, model, marker, {"trace_id": caller_trace, "session_id": caller_session}
                ),
                headers=_w3c_headers(header_trace, baggage_session),
            )
            assert response.status_code == 200, f"{attempt}: {response.text}"
            call_id: Final = response.headers["x-litellm-call-id"]
            span: Final = _await_span(received, audit_rig.destination, call_id)
            assert span.trace_id.hex() == caller_trace, f"{attempt} {call_id}: trace id"
            assert _attribute(span.attributes, "session.id") == caller_session, f"{attempt} {call_id}: session.id"
            assert _await_spend_row(call_id)["session_id"] == caller_session, f"{attempt} {call_id}: spend"
        assert targets and set(targets) == {expected_target}, targets


def test_audit_malformed_w3c_headers_are_ignored(audit_rig: _AuditRig) -> None:
    marker: Final = "auditmal" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            _trace_body("chat", model, marker, None),
            headers={"traceparent": "00-zz-00f067aa0ba902b7-01", "baggage": "not-a-session-key"},
        )
        assert response.status_code == 200, response.text
        assert targets == ["/v1/chat/completions"], targets
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        span: Final = _await_span(received, audit_rig.destination, response.headers["x-litellm-call-id"])
        assert len(span.trace_id.hex()) == 32 and "zz" not in span.trace_id.hex(), span.trace_id.hex()
        row: Final = _await_spend_row(response.headers["x-litellm-call-id"])
        span_session: Final = _attribute(span.attributes, "session.id")
        print(f"S7 record: span session.id={span_session!r} spend session={row['session_id']!r}")
        assert span_session in (None, row["session_id"]), (
            f"span session {span_session!r} diverges from spend session {row['session_id']!r}"
        )
        assert row["session_id"], row


def test_audit_unauthenticated_call_leaves_no_spend_row(audit_rig: _AuditRig) -> None:
    marker: Final = "auditunauth" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        candidate: Final = audit_rig.candidate
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        denied: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            _trace_body(
                "chat",
                model,
                marker,
                {"trace_id": uuid.uuid4().hex, "session_id": f"my-session-id-{marker}"},
            ),
            headers=_w3c_headers(uuid.uuid4().hex, "baggage-" + marker),
            key="sk-wrong-key",
        )
        assert denied.status_code == 401, denied.text
        assert targets == [], targets
        control: Final = candidate.request(
            "POST", "/v1/chat/completions", _trace_body("chat", model, marker + "-control", None)
        )
        assert control.status_code == 200, control.text
        _await_spend_row(control.headers["x-litellm-call-id"])
        assert (
            read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s',
                (denied.headers.get("x-litellm-call-id") or "",),
            )
            == []
        ), "an unauthenticated call must not write a spend row"


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS[:2])
@pytest.mark.parametrize("upstream_status", (500, 401), ids=["upstream_500", "upstream_401"])
def test_audit_upstream_error_still_logs_caller_session(
    audit_rig: _AuditRig,
    endpoint: str,
    kind: str,
    expected_target: str,
    upstream_status: int,
) -> None:
    marker: Final = "auditerr" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    caller_trace: Final = uuid.uuid4().hex
    caller_session: Final = f"my-session-id-{marker}"
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    def upstream(request: Request) -> Reply:
        if request.body:
            targets.append(request.target)
        return Reply(status=upstream_status, body=b'{"error": {"message": "scripted upstream failure"}}')

    with wire_server(upstream) as provider:
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        response: Final = audit_rig.candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, {"trace_id": caller_trace, "session_id": caller_session}),
            headers=_w3c_headers(uuid.uuid4().hex, "baggage-" + marker),
        )
        assert response.status_code == upstream_status, response.text
        assert targets and set(targets) == {expected_target}, targets
        call_id: Final = response.headers["x-litellm-call-id"]
        row: Final = _await_spend_row(call_id)
        assert row["session_id"] == caller_session, f"call {call_id}: failure spend session {row}"


def test_audit_sink_rejection_does_not_break_the_caller(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "auditreject" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit
    attempts: Final[list[int]] = []  # mutable-ok: sink status sequence counter

    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        if request.target == TRACES_PATH:
            attempts.append(1)
            if len(attempts) == 1:
                return Reply(status=403)
            if len(attempts) == 2:
                return Reply(status=404)
        return Reply(body=b"", content_type="application/x-protobuf")

    with (
        wire_server(_audit_upstream(provider_secret, marker, targets)) as provider,
        wire_server(langfuse) as destination,
        owned_proxy(
            gateway, tmp_path, _langfuse_environment(destination), config=_langfuse_config(tmp_path)
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        for attempt in ("first", "second", "third"):
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                _trace_body(
                    "chat",
                    model,
                    f"{marker}-{attempt}",
                    {"trace_id": uuid.uuid4().hex, "session_id": f"my-session-id-{marker}-{attempt}"},
                ),
            )
            assert response.status_code == 200, f"{attempt}: {response.text}"
            _await_spend_row(response.headers["x-litellm-call-id"])
        assert targets == ["/v1/chat/completions"] * 3, targets


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS[:2])
def test_audit_string_metadata_body_does_not_crash(audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str) -> None:
    marker: Final = "auditstr" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        candidate: Final = audit_rig.candidate
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        body: Final = {**_trace_body(kind, model, marker, None), "metadata": "x"}
        response: Final = candidate.request(
            "POST", endpoint, body, headers=_w3c_headers(uuid.uuid4().hex, "baggage-" + marker)
        )
        assert response.status_code < 500, f"metadata string must not crash the proxy: {response.status_code} {response.text}"
        follow_up: Final = candidate.request(
            "POST", "/v1/chat/completions", _trace_body("chat", model, marker + "-follow", None)
        )
        assert follow_up.status_code == 200, follow_up.text


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS[:2])
def test_audit_key_metadata_session_still_honoured(audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str) -> None:
    marker: Final = "auditkey" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    key_session: Final = f"key-session-{marker}"
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        token: Final = audit_rig.scenario.key(metadata={"session_id": key_session})
        response: Final = audit_rig.candidate.request(
            "POST",
            endpoint,
            _trace_body(kind, model, marker, None),
            headers=_w3c_headers(uuid.uuid4().hex, "baggage-" + marker),
            key=token,
        )
        assert response.status_code == 200, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        assert targets == [expected_target], targets
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        span: Final = _await_span(received, audit_rig.destination, call_id)
        row: Final = _await_spend_row(call_id)
        assert _attribute(span.attributes, "session.id") == row["session_id"], (
            f"call {call_id}: spend session {row['session_id']!r} must match span session"
        )


@pytest.mark.parametrize(("endpoint", "kind", "expected_target"), _AUDIT_ENDPOINTS[:2])
@pytest.mark.parametrize("metadata_mode", ("both", "none"), ids=["caller_ids", "no_metadata"])
def test_audit_cache_hit_call_keeps_winning_ids(
    audit_rig: _AuditRig, endpoint: str, kind: str, expected_target: str, metadata_mode: str
) -> None:
    marker: Final = "auditcache" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    caller_trace: Final = uuid.uuid4().hex
    caller_session: Final = f"my-session-id-{marker}"
    metadata: Final = (
        {"trace_id": caller_trace, "session_id": caller_session} if metadata_mode == "both" else None
    )
    expected_trace: Final = caller_trace if metadata_mode == "both" else header_trace
    expected_session: Final = caller_session if metadata_mode == "both" else baggage_session
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        candidate: Final = audit_rig.candidate
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        body: Final = {key: value for key, value in _trace_body(kind, model, marker, metadata).items() if key != "cache"}
        headers: Final = _w3c_headers(header_trace, baggage_session)
        first: Final = candidate.request("POST", endpoint, body, headers=headers)
        assert first.status_code == 200, first.text
        second: Final = candidate.request("POST", endpoint, body, headers=headers)
        assert second.status_code == 200, second.text
        assert second.headers.get("x-litellm-cache-key"), (
            f"second identical call must be a cache hit: {dict(second.headers)}"
        )
        call_id: Final = second.headers["x-litellm-call-id"]
        assert targets == [expected_target], targets
        span: Final = _await_span(received, audit_rig.destination, call_id)
        if metadata_mode == "both":
            assert span.trace_id.hex() == expected_trace, f"cache hit {call_id}: trace id"
            assert _attribute(span.attributes, "session.id") == expected_session, (
                f"cache hit {call_id}: session.id"
            )
        else:
            span_session: Final = _attribute(span.attributes, "session.id")
            print(
                f"E2 record: cache-hit span trace={span.trace_id.hex()} session={span_session!r} "
                f"header trace={header_trace} baggage session={baggage_session!r}"
            )
            assert len(span.trace_id.hex()) == 32, span.trace_id.hex()
        assert _await_spend_row(call_id)["session_id"] == expected_session, f"cache hit {call_id}: spend"


def test_audit_concurrent_requests_each_keep_their_caller_ids(audit_rig: _AuditRig) -> None:
    marker: Final = "auditconc" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with wire_server(_audit_upstream(provider_secret, marker, targets)) as provider:
        candidate: Final = audit_rig.candidate
        model: Final = audit_rig.scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        jobs: Final = tuple(
            (index, endpoint, kind, uuid.uuid4().hex, f"my-session-id-{marker}-{index}")
            for index, (endpoint, kind, _) in tuple(
                enumerate(tuple(row.values for row in _AUDIT_ENDPOINTS) * 4)
            )[:10]
        )

        def fire(job: tuple[int, str, str, str, str]) -> tuple[str, str, httpx.Response]:
            index, endpoint, kind, caller_trace, caller_session = job
            response: Final = candidate.request(
                "POST",
                endpoint,
                _trace_body(
                    kind, model, f"{marker}-{index}", {"trace_id": caller_trace, "session_id": caller_session}
                ),
                headers=_w3c_headers(header_trace, baggage_session),
            )
            return caller_trace, caller_session, response

        with ThreadPoolExecutor(max_workers=10) as pool:
            answered: Final = tuple(pool.map(fire, jobs))
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        assert len(targets) == 10, targets
        for caller_trace, caller_session, response in answered:
            assert response.status_code == 200, response.text
            _assert_span_spend(response, received, audit_rig.destination, caller_trace, caller_session)


def test_audit_sink_outage_mid_burst_loses_no_spend_rows(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "auditoutage" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit
    outage: Final = threading.Event()

    def langfuse(request: Request) -> Reply:
        if request.method == "GET" and request.target.startswith(PROJECTS_PATH):
            return _projects()
        if outage.is_set():
            return Reply(status=503)
        return Reply(body=b"", content_type="application/x-protobuf")

    with (
        wire_server(_audit_upstream(provider_secret, marker, targets)) as provider,
        wire_server(langfuse) as destination,
        owned_proxy(
            gateway, tmp_path, _langfuse_environment(destination), config=_langfuse_config(tmp_path)
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        jobs: Final = tuple(
            (index, endpoint, kind, uuid.uuid4().hex, f"my-session-id-{marker}-{index}")
            for index, (endpoint, kind, _) in enumerate(tuple(row.values for row in _AUDIT_ENDPOINTS) * 10)
        )

        def fire(job: tuple[int, str, str, str, str]) -> tuple[str, str, httpx.Response]:
            index, endpoint, kind, caller_trace, caller_session = job
            stream: Final = index % 3 == 0
            response: Final = candidate.request(
                "POST",
                endpoint,
                {
                    **_trace_body(
                        kind,
                        model,
                        f"{marker}-{index}",
                        {"trace_id": caller_trace, "session_id": caller_session},
                    ),
                    "stream": stream,
                },
                headers=_w3c_headers(header_trace, baggage_session),
            )
            return caller_trace, caller_session, response

        outage.set()
        with ThreadPoolExecutor(max_workers=30) as pool:
            first_wave: Final = tuple(pool.map(fire, jobs[:10]))
        outage.clear()
        with ThreadPoolExecutor(max_workers=30) as pool:
            answered: Final = first_wave + tuple(pool.map(fire, jobs[10:]))
        for _, _, response in answered:
            assert response.status_code == 200, response.text
        expected_by_call: Final = {
            response.headers["x-litellm-call-id"]: caller_session
            for caller_session, response in ((session, res) for _, session, res in answered)
        }
        call_ids: Final = sorted(expected_by_call)
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        delivered: Final = eventually(
            lambda: (
                received.extend(destination.drain()) or tuple(
                    span
                    for span in _spans(received)
                    if _attribute(span.attributes, "langfuse.observation.metadata.litellm_call_id")
                    in expected_by_call
                )
            ),
            lambda spans: len(spans) >= len(answered),
            seconds=45,
            return_last_on_timeout=True,
        )
        delivered_calls: Final = frozenset(
            str(_attribute(span.attributes, "langfuse.observation.metadata.litellm_call_id"))
            for span in delivered
        )
        lost: Final = frozenset(call_ids) - delivered_calls
        print(f"sink outage lost {len(lost)} of {len(answered)} spans")
        for span in delivered:
            span_call: Final = str(
                _attribute(span.attributes, "langfuse.observation.metadata.litellm_call_id")
            )
            assert _attribute(span.attributes, "session.id") == expected_by_call[span_call], (
                f"call {span_call}: session.id"
            )
        spend_rows: Final = eventually(
            lambda: read_rows(
                'SELECT session_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = ANY(%s)',
                (call_ids,),
            ),
            lambda values: len(values) == len(answered),
            seconds=150,
            return_last_on_timeout=True,
        )
        print(f"C1 record: {len(spend_rows)} of {len(answered)} spend rows written")
        for row in spend_rows:
            assert row["session_id"] in frozenset(expected_by_call.values()), row


def test_audit_surviving_worker_keeps_serving_after_kill(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "auditworker" + uuid.uuid4().hex
    provider_secret: Final = "synthetic-provider-secret-" + marker
    header_trace: Final = uuid.uuid4().hex
    baggage_session: Final = "baggage-" + marker
    targets: Final[list[str]] = []  # mutable-ok: the upstream records every hit

    with (
        wire_server(_audit_upstream(provider_secret, marker, targets)) as provider,
        wire_server(_audit_sink()) as destination,
        owned_proxy_process(
            gateway,
            tmp_path,
            _langfuse_environment(destination),
            config=_langfuse_config(tmp_path),
            workers=2,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        candidate: Final = owned.gateway
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=provider_secret)
        workers: Final = tuple(
            member for member in group_members(owned.process.pid) if member.pid != owned.process.pid
        )
        assert workers, "expected worker processes under the owned proxy"
        workers[0].send_signal(signal.SIGKILL)
        psutil.wait_procs([workers[0]], timeout=5)

        def fire(index: int) -> tuple[str, httpx.Response]:
            caller_trace: Final = uuid.uuid4().hex
            caller_session: Final = f"my-session-id-{marker}-{index}"
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                _trace_body(
                    "chat",
                    model,
                    f"{marker}-{index}",
                    {"trace_id": caller_trace, "session_id": caller_session},
                ),
                headers=_w3c_headers(header_trace, baggage_session),
            )
            return caller_session, response

        with ThreadPoolExecutor(max_workers=10) as pool:
            answered: Final = tuple(pool.map(fire, range(10)))
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls keep earlier ones
        for caller_session, response in answered:
            assert response.status_code == 200, response.text
            call_id: Final = response.headers["x-litellm-call-id"]
            span: Final = _await_span(received, destination, call_id)
            assert _attribute(span.attributes, "session.id") == caller_session, f"{call_id}: session.id"
        expected_by_call: Final = {
            response.headers["x-litellm-call-id"]: caller_session
            for caller_session, response in answered
        }
        spend_rows: Final = eventually(
            lambda: read_rows(
                'SELECT session_id, litellm_call_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = ANY(%s)',
                (list(expected_by_call),),
            ),
            lambda values: len(values) == len(answered),
            seconds=150,
            return_last_on_timeout=True,
        )
        print(f"C2 record: {len(spend_rows)} of {len(answered)} spend rows written")
        assert spend_rows, "no spend rows survived the worker kill"
        for row in spend_rows:
            assert row["session_id"] == expected_by_call[string_value(row["litellm_call_id"])], row
