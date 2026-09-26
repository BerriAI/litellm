import base64
import json
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
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


def _langfuse_config(tmp_path: Path) -> Path:
    config: Final = _PROXY_CONFIG.validate_python(yaml.safe_load(STOCK_CONFIG.read_text()))
    settings: Final = {
        **_SETTINGS.validate_python(config["litellm_settings"]),
        "success_callback": ["langfuse"],
        "failure_callback": ["langfuse"],
    }
    path: Final = tmp_path / "langfuse.yaml"
    path.write_text(yaml.safe_dump({**config, "litellm_settings": settings}))
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
    assert "langfuse" not in json.dumps([stock_settings.get(key) for key in ("callbacks", "success_callback")])

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
