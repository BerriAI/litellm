import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from opentelemetry.proto.common.v1.common_pb2 import AnyValue


def _attribute_value(value: AnyValue) -> object:
    kind: Final = value.WhichOneof("value")
    return getattr(value, kind) if kind is not None else None


def _spans(body: bytes) -> tuple[tuple[str, dict[str, object]], ...]:
    export: Final = trace_service_pb2.ExportTraceServiceRequest()
    export.ParseFromString(body)
    return tuple(
        (
            span.trace_id.hex(),
            {attribute.key: _attribute_value(attribute.value) for attribute in span.attributes},
        )
        for resource in export.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )


def _upstream_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": marker,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"reply {marker}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ).encode()
    )


def _sink(_request: Request) -> Reply:
    return Reply(body=b"", content_type="application/x-protobuf")


def _drained_spans(sink: Wire, batches: list[bytes]) -> tuple[tuple[str, dict[str, object]], ...]:
    batches.extend(request.body for request in sink.drain())
    return tuple(span for body in batches for span in _spans(body))


def _generation_span_attributes(sink: Wire, batches: list[bytes], marker: str) -> tuple[dict[str, object], ...]:
    return tuple(
        attributes
        for _trace_id, attributes in _drained_spans(sink, batches)
        if attributes.get("llm.response.id") == marker or attributes.get("gen_ai.response.id") == marker
    )


def _span_attributes_containing_marker(sink: Wire, batches: list[bytes], marker: str) -> tuple[dict[str, object], ...]:
    return tuple(
        attributes
        for _trace_id, attributes in _drained_spans(sink, batches)
        if any(isinstance(value, str) and marker in value for value in attributes.values())
    )


def _trace_user_span_attributes(sink: Wire, batches: list[bytes], marker: str) -> tuple[dict[str, object], ...]:
    spans: Final = _drained_spans(sink, batches)
    generation_trace: Final = next(
        (
            trace_id
            for trace_id, attributes in spans
            if attributes.get("llm.response.id") == marker or attributes.get("gen_ai.response.id") == marker
        ),
        None,
    )
    if generation_trace is None:
        return ()
    return tuple(
        attributes for trace_id, attributes in spans if trace_id == generation_trace and "user.id" in attributes
    )


@contextmanager
def _langfuse_proxy(
    gateway: Gateway, directory: Path, collector_url: str, overrides: Mapping[str, str] | None = None
) -> Iterator[Gateway]:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["langfuse_otel"]})
    path: Final = directory / "langfuse_otel.yaml"
    path.write_text(yaml.safe_dump(config))
    with owned_proxy(
        gateway,
        directory,
        {
            "LANGFUSE_PUBLIC_KEY": "pk-integration",
            "LANGFUSE_SECRET_KEY": "sk-integration",
            "LANGFUSE_HOST": collector_url,
            "OTEL_BSP_SCHEDULE_DELAY": "100",
            **(overrides or {}),
        },
        config=path,
    ) as candidate:
        yield candidate


@pytest.mark.covers("other.observability.langfuse_otel.header_end_user_in_user_id_over_internal_user")
def test_langfuse_otel_header_end_user_lands_in_user_id_not_session_id_for_a_key_owned_by_an_internal_user(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key=key,
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, attributes


@pytest.mark.covers("other.observability.langfuse_otel.header_end_user_in_user_id_service_key")
def test_langfuse_otel_header_end_user_lands_in_user_id_for_a_service_account_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key()
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key=key,
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, attributes


@pytest.mark.covers("other.observability.langfuse_otel.body_user_never_a_session")
def test_langfuse_otel_body_user_is_never_a_session(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "user": f"end-user-{marker}",
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, attributes


@pytest.mark.covers("other.observability.langfuse_otel.caller_trace_user_id_wins")
def test_langfuse_otel_caller_trace_user_id_wins_over_the_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "user": f"end-user-{marker}",
                "metadata": {"trace_user_id": f"caller-{marker}"},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"caller-{marker}",
            "session.id": None,
        }, attributes


@pytest.mark.covers("other.observability.langfuse_otel.caller_session_id_stays_session")
def test_langfuse_otel_caller_session_id_stays_the_session_beside_the_end_user(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "user": f"end-user-{marker}",
                "metadata": {"session_id": f"sess-{marker}"},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": f"sess-{marker}",
        }, attributes


@pytest.mark.covers("other.observability.langfuse_otel.v2_header_end_user_in_user_id")
def test_langfuse_otel_v2_header_end_user_lands_in_user_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url, {"LITELLM_OTEL_V2": "1"}) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key=key,
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        user_spans: Final = eventually(
            lambda: _trace_user_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )
        assert {key: user_spans[0].get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, user_spans[0]


@pytest.mark.covers("other.observability.langfuse_otel.messages_caller_trace_user_id_under_litellm_metadata")
def test_langfuse_otel_messages_caller_trace_user_id_under_litellm_metadata_wins_over_the_end_user(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "max_tokens": 5,
                "litellm_metadata": {"trace_user_id": f"caller-{marker}"},
                "cache": {"no-cache": True},
            },
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _span_attributes_containing_marker(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"caller-{marker}",
            "session.id": None,
        }, attributes
