import json
import threading
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from _s3_v2_support import SURFACES, call_surface, mixed_burst, surface_reply
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue

REDACTED: Final = "redacted-by-litellm"
SURFACE_CALL_TYPES: Final = {
    "chat": "acompletion",
    "chat_stream": "acompletion",
    "messages": "anthropic_messages",
    "messages_stream": "anthropic_messages",
    "responses": "aresponses",
    "responses_stream": "aresponses",
}
ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"


@dataclass(frozen=True, slots=True)
class SpanRecord:
    trace_id: str
    parent_span_id: str
    attributes: Mapping[str, object]

    def tags(self) -> list[object]:
        """The trace tags Langfuse ingestion reads from the raw attribute."""
        raw: Final = self.attributes.get("langfuse.trace.tags")
        if isinstance(raw, tuple):
            return list(raw)
        if not isinstance(raw, str) or not raw:
            return []
        if raw.strip().startswith("["):
            return json.loads(raw)
        return [tag.strip() for tag in raw.split(",")] if "," in raw else [raw]


def _attribute_value(value: AnyValue) -> object:
    kind: Final = value.WhichOneof("value")
    if kind == "array_value":
        return tuple(_attribute_value(item) for item in value.array_value.values)
    return getattr(value, kind) if kind is not None else None


def _span_records(body: bytes) -> tuple[SpanRecord, ...]:
    export: Final = ExportTraceServiceRequest.FromString(body)
    return tuple(
        SpanRecord(
            trace_id=span.trace_id.hex(),
            parent_span_id=span.parent_span_id.hex(),
            attributes={pair.key: _attribute_value(pair.value) for pair in span.attributes},
        )
        for resource in export.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )


class LangfuseSink:
    """The Langfuse OTLP endpoint: keeps every accepted generation and root span, or answers 503 during a scripted
    outage."""

    def __init__(self) -> None:
        self.lock: Final = threading.Lock()
        self.outage: Final = threading.Event()
        self.accepted: Final[list[SpanRecord]] = []  # mutable-ok: spans accumulate as the exporter delivers them
        self.roots: Final[list[SpanRecord]] = []  # mutable-ok: root spans accumulate as the exporter delivers them
        self.refusals: Final[list[int]] = []  # mutable-ok: one entry per batch refused during the outage

    def respond(self, request: Request) -> Reply:
        if not request.target.endswith("/v1/traces"):
            return Reply(body=b"")
        assert request.headers.get("x-langfuse-ingestion-version") == "4", request.headers
        if self.outage.is_set():
            with self.lock:
                self.refusals.append(1)
            return Reply(status=503, body=b'{"error": "scripted outage"}')
        records: Final = _span_records(request.body)
        with self.lock:
            self.accepted.extend(
                record for record in records if record.attributes.get("langfuse.observation.type") == "generation"
            )
            self.roots.extend(record for record in records if not record.parent_span_id)
        return Reply(body=b"")

    def matching(self, marker: str) -> tuple[SpanRecord, ...]:
        with self.lock:
            snapshot: Final = tuple(self.accepted)
        return tuple(record for record in snapshot if marker in str(tuple(record.attributes.values())))

    def one(self, marker: str) -> SpanRecord:
        return eventually(lambda: self.matching(marker), lambda found: len(found) == 1, seconds=70)[0]

    def root_of(self, span: SpanRecord) -> SpanRecord:
        def roots() -> tuple[SpanRecord, ...]:
            with self.lock:
                return tuple(root for root in self.roots if root.trace_id == span.trace_id)

        return eventually(roots, lambda found: len(found) == 1, seconds=70)[0]


def _chat_completion(marker: str, text: str = "langfuse-otel scripted reply") -> bytes:
    return json.dumps(
        {
            "id": marker,
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        }
    ).encode()


def _provider(request: Request) -> Reply:
    """Scripted provider: the three surfaces echo the caller marker, a `fail-` marker answers 500, embeddings succeed."""
    if request.method != "POST" or not request.body:
        return Reply(body=b'{"object": "list", "data": []}')
    body: Final = json.loads(request.body)
    if request.target.endswith("/embeddings"):
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                }
            ).encode()
        )
    messages: Final = body.get("messages")
    first: Final = messages[0]["content"] if isinstance(messages, list) and messages else None
    if isinstance(first, str) and first.startswith("fail-"):
        return Reply(status=500, body=b'{"error": {"message": "scripted provider failure", "type": "server_error"}}')
    return surface_reply(request)


def _langfuse_config(directory: Path, name: str, settings: Mapping[str, object]) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **base,
        "litellm_settings": {**base["litellm_settings"], "callbacks": ["langfuse_otel"], **settings},
    }
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    sink: LangfuseSink
    openai_model: str
    anthropic_model: str
    embedding_model: str
    key: str


def _rig(
    tmp_path_factory: pytest.TempPathFactory, name: str, environment: Mapping[str, str], settings: Mapping[str, object]
) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp(name)
    sink: Final = LangfuseSink()
    with (
        wire_server(sink.respond) as destination,
        wire_server(_provider) as provider,
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway,
            directory,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                "LANGFUSE_SECRET_KEY": "sk-lf-test",
                "LANGFUSE_HOST": destination.url,
                **environment,
            },
            config=_langfuse_config(directory, name, settings),
            workers=2,
        ) as proxy,
        proxy.scenario() as scenario,
    ):
        openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
        anthropic_model: Final = scenario.model(
            model=ANTHROPIC_MODEL, api_base=provider.url, api_key="synthetic-provider-key"
        )
        embedding_model: Final = scenario.model(
            model="openai/text-embedding-3-small",
            api_base=provider.url + "/v1",
            api_key="synthetic-provider-key",
            model_info={"mode": "embedding"},
        )
        key: Final = scenario.key(models=[openai_model, anthropic_model, embedding_model])
        yield Rig(proxy, sink, openai_model, anthropic_model, embedding_model, key)


@pytest.fixture(scope="module")
def v1(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _rig(tmp_path_factory, "langfuse-v1", {}, {"langfuse_default_tags": ["cache_hit"]})


V2_ENVIRONMENT: Final = {"LITELLM_OTEL_V2": "1", "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only"}


@pytest.fixture(scope="module")
def v2(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _rig(tmp_path_factory, "langfuse-v2", V2_ENVIRONMENT, {})


@pytest.fixture(scope="module")
def v2_llm_only(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _rig(
        tmp_path_factory, "langfuse-v2-llm-only", {**V2_ENVIRONMENT, "LITELLM_OTEL_LANGFUSE_SPAN_SCOPE": "llm_only"}, {}
    )


def _chat(
    rig: Rig, content: str, metadata: Mapping[str, object] | None = None, headers: Mapping[str, str] | None = None
) -> str:
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": rig.openai_model,
            "messages": [{"role": "user", "content": content}],
            **({"metadata": dict(metadata)} if metadata is not None else {}),
        },
        key=rig.key,
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.text


def _assert_derived(span: SpanRecord, call_type: str) -> None:
    attributes: Final = span.attributes
    assert attributes.get("langfuse.trace.name") == f"litellm-{call_type}", attributes
    assert attributes.get("langfuse.trace.input") == attributes.get("langfuse.observation.input"), attributes
    assert attributes.get("langfuse.trace.output") == attributes.get("langfuse.observation.output"), attributes
    tags: Final = span.tags()
    assert any(str(tag).startswith("User-Agent: ") for tag in tags), tags
    assert len(tags) == len(set(map(str, tags))), tags


@pytest.mark.parametrize("surface", SURFACES)
def test_every_surface_derives_trace_name_io_and_tags(v1: Rig, surface: str) -> None:
    marker: Final = f"lit8281-{surface}-{uuid.uuid4().hex}"
    call_surface(v1.proxy, surface, v1.openai_model, v1.anthropic_model, v1.key, marker)
    span: Final = v1.sink.one(marker)
    _assert_derived(span, SURFACE_CALL_TYPES[surface])
    assert marker in str(span.attributes["langfuse.trace.input"]), span.attributes
    assert span.attributes.get("langfuse.trace.output") is not None, span.attributes
    assert any(str(tag).startswith("cache_hit:") for tag in span.tags()), span.tags()


def test_inbound_traceparent_keeps_the_caller_trace_and_derives_trace_fields(v1: Rig) -> None:
    marker: Final = "lit8281-parent-" + uuid.uuid4().hex
    trace_id: Final = uuid.uuid4().hex
    parent_id: Final = uuid.uuid4().hex[:16]
    _chat(v1, marker, headers={"traceparent": f"00-{trace_id}-{parent_id}-01"})
    span: Final = v1.sink.one(marker)
    assert (span.trace_id, span.parent_span_id) == (trace_id, parent_id), span
    _assert_derived(span, "acompletion")


def test_caller_trace_name_and_tags_win_over_derived_values(v1: Rig) -> None:
    marker: Final = "lit8281-caller-" + uuid.uuid4().hex
    _chat(v1, marker, metadata={"trace_name": "caller-trace", "tags": ["caller-tag", "second-tag"]})
    span: Final = v1.sink.one(marker)
    assert span.attributes["langfuse.trace.name"] == "caller-trace", span.attributes
    assert span.tags()[:2] == ["caller-tag", "second-tag"], span.tags()
    assert span.attributes.get("langfuse.trace.input") == span.attributes.get("langfuse.observation.input")


@pytest.mark.parametrize(
    ("metadata", "headers", "expected"),
    [
        pytest.param({"tags": "c1, c2"}, None, ["c1", "c2"], id="body_comma_string"),
        pytest.param({"tags": '["j1", "j2"]'}, None, ["j1", "j2"], id="body_json_array_string"),
        pytest.param(None, {"langfuse_tags": "h1,h2"}, ["h1", "h2"], id="header_comma_string"),
    ],
)
def test_string_caller_tags_keep_the_tags_langfuse_split_them_into(
    v1: Rig, metadata: Mapping[str, object] | None, headers: Mapping[str, str] | None, expected: list[str]
) -> None:
    marker: Final = "lit8281-strtags-" + uuid.uuid4().hex
    _chat(v1, marker, metadata=metadata, headers=headers)
    span: Final = v1.sink.one(marker)
    assert span.tags()[: len(expected)] == expected, span.attributes


def test_trace_name_header_wins_over_the_derived_name(v1: Rig) -> None:
    marker: Final = "lit8281-header-" + uuid.uuid4().hex
    _chat(v1, marker, headers={"langfuse_trace_name": "header-trace"})
    span: Final = v1.sink.one(marker)
    assert span.attributes["langfuse.trace.name"] == "header-trace", span.attributes


def test_joined_trace_keeps_caller_tags_and_gets_no_derived_trace_fields(v1: Rig) -> None:
    marker: Final = "lit8281-join-" + uuid.uuid4().hex
    existing: Final = uuid.uuid4().hex
    _chat(v1, marker, metadata={"existing_trace_id": existing, "tags": ["join-tag"]})
    span: Final = v1.sink.one(marker)
    attributes: Final = span.attributes
    assert attributes.get("langfuse.trace.existing_id") == existing, attributes
    assert span.tags() == ["join-tag"], attributes
    assert "langfuse.trace.name" not in attributes, attributes
    assert "langfuse.trace.input" not in attributes, attributes
    assert "langfuse.trace.output" not in attributes, attributes
    assert marker in str(attributes.get("langfuse.observation.input")), attributes


def test_joined_trace_without_caller_tags_gets_no_trace_tags(v1: Rig) -> None:
    marker: Final = "lit8281-join-bare-" + uuid.uuid4().hex
    _chat(v1, marker, metadata={"existing_trace_id": uuid.uuid4().hex})
    attributes: Final = v1.sink.one(marker).attributes
    assert not {
        key for key in attributes if key.startswith("langfuse.trace.") and key != "langfuse.trace.existing_id"
    }, attributes


def test_mask_flags_redact_the_derived_trace_io(v1: Rig) -> None:
    marker: Final = "lit8281-mask-" + uuid.uuid4().hex
    _chat(v1, marker, metadata={"mask_input": True, "mask_output": True})
    attributes: Final = v1.sink.one(marker).attributes
    assert attributes.get("langfuse.trace.input") == REDACTED, attributes
    assert attributes.get("langfuse.trace.output") == REDACTED, attributes
    assert attributes.get("langfuse.trace.name") == "litellm-acompletion", attributes


@pytest.mark.parametrize(
    "tags",
    [
        pytest.param(5, id="int"),
        pytest.param(["ok-tag", 7, None], id="mixed_list"),
        pytest.param("", id="empty_string"),
        pytest.param("x" * 5000, id="five_kb_string"),
    ],
)
def test_malformed_caller_tags_still_log_the_generation_with_derived_fields(v1: Rig, tags: object) -> None:
    marker: Final = "lit8281-badtags-" + uuid.uuid4().hex
    _chat(v1, marker, metadata={"tags": tags})
    span: Final = v1.sink.one(marker)
    assert span.attributes.get("langfuse.trace.name") == "litellm-acompletion", span.attributes
    assert span.attributes.get("langfuse.trace.input") == span.attributes.get("langfuse.observation.input")
    assert all(isinstance(tag, str) for tag in span.tags()), span.tags()


def test_provider_failure_span_derives_name_and_input_without_output(v1: Rig) -> None:
    marker: Final = "fail-lit8281-" + uuid.uuid4().hex
    response: Final = v1.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": v1.openai_model, "messages": [{"role": "user", "content": marker}]},
        key=v1.key,
    )
    assert response.status_code >= 500, response.text
    span: Final = v1.sink.one(marker)
    assert span.attributes.get("langfuse.trace.name") == "litellm-acompletion", span.attributes
    assert span.attributes.get("langfuse.trace.input") == span.attributes.get("langfuse.observation.input")
    assert "langfuse.trace.output" not in span.attributes, span.attributes


def test_embedding_call_derives_the_trace_name(v1: Rig) -> None:
    marker: Final = "lit8281-embed-" + uuid.uuid4().hex
    response: Final = v1.proxy.request(
        "POST", "/v1/embeddings", {"model": v1.embedding_model, "input": marker, "user": marker}, key=v1.key
    )
    assert response.status_code == 200, response.text
    span: Final = v1.sink.one(marker)
    assert span.attributes.get("langfuse.trace.name") == "litellm-aembedding", span.attributes
    assert span.attributes.get("langfuse.trace.output") == span.attributes.get("langfuse.observation.output")


def test_identical_requests_each_log_exactly_one_span_with_derived_fields(v1: Rig) -> None:
    marker: Final = "lit8281-twice-" + uuid.uuid4().hex
    _chat(v1, marker)
    _chat(v1, marker)
    spans: Final = eventually(lambda: v1.sink.matching(marker), lambda found: len(found) == 2, seconds=70)
    for span in spans:
        _assert_derived(span, "acompletion")


def test_otel_v2_derives_the_trace_name_and_leaves_trace_io_to_the_root_observation(v2: Rig) -> None:
    derived: Final = "lit8281-v2-" + uuid.uuid4().hex
    caller: Final = "lit8281-v2-caller-" + uuid.uuid4().hex
    _chat(v2, derived)
    _chat(v2, caller, metadata={"trace_name": "caller-trace", "tags": ["caller-tag"]})
    derived_span: Final = v2.sink.one(derived)
    assert derived_span.attributes.get("langfuse.trace.name") == "litellm-acompletion", derived_span.attributes
    assert not {"langfuse.trace.input", "langfuse.trace.output"} & set(derived_span.attributes), derived_span.attributes
    root: Final = v2.sink.root_of(derived_span)
    assert root.attributes.get("langfuse.trace.name") == "litellm-acompletion", root.attributes
    assert derived in str(root.attributes.get("langfuse.observation.input")), root.attributes
    assert root.attributes.get("langfuse.observation.output") is not None, root.attributes
    caller_span: Final = v2.sink.one(caller)
    assert caller_span.attributes.get("langfuse.trace.name") == "caller-trace", caller_span.attributes
    assert caller_span.tags() == ["caller-tag"], caller_span.attributes


def test_otel_v2_llm_only_root_generation_is_named_after_the_call_type(v2_llm_only: Rig) -> None:
    marker: Final = "lit8281-v2-llm-only-" + uuid.uuid4().hex
    _chat(v2_llm_only, marker)
    span: Final = v2_llm_only.sink.one(marker)
    assert span.parent_span_id == "", span
    assert span.attributes.get("langfuse.trace.name") == "litellm-acompletion", span.attributes
    assert marker in str(span.attributes.get("langfuse.observation.input")), span.attributes


def test_destination_outage_mid_burst_lands_every_span_once_with_derived_fields(v1: Rig) -> None:
    marker: Final = "lit8281-chaos-" + uuid.uuid4().hex[:12]
    v1.sink.outage.set()
    try:
        answered: Final = mixed_burst(v1.proxy, v1.openai_model, v1.anthropic_model, v1.key, marker, per_surface=4)
        eventually(lambda: len(v1.sink.refusals), lambda refused: refused >= 1, seconds=70)
    finally:
        v1.sink.outage.clear()
    assert len(answered) == 4 * len(SURFACES)
    expected: Final = {f"{marker}-{surface}-{index}": surface for surface in SURFACES for index in range(4)}

    def landed() -> dict[str, tuple[SpanRecord, ...]]:
        return {identity: v1.sink.matching(identity) for identity in expected}

    spans: Final = eventually(landed, lambda found: all(len(records) >= 1 for records in found.values()), seconds=120)
    assert {identity: len(records) for identity, records in spans.items()} == dict.fromkeys(expected, 1)
    for identity, records in spans.items():
        _assert_derived(records[0], SURFACE_CALL_TYPES[expected[identity]])
