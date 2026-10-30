import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from _langfuse_otel_metadata_support import (
    LangfuseRig,
    chat_identity_fields,
    decoded_response_id,
    exported_llm_span,
    exported_spans,
    identity_attributes,
    langfuse_rig,
    scenario_identity,
    span_records,
    watched_spans,
)
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, wire_server


def _scalar_requester_metadata_request(route: str, model: str, marker: str) -> dict[str, object]:
    caller: Final = {"litellm_metadata": {"requester_metadata": 1}}
    if route == "/v1/messages":
        return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}], **caller}
    return {"model": model, "input": marker, **caller}


@pytest.mark.parametrize(
    ("callback", "route"),
    [
        ("otel", "/v1/messages"),
        ("otel", "/v1/responses"),
        ("langfuse_otel", "/v1/messages"),
        ("langfuse_otel", "/v1/responses"),
    ],
)
def test_otel_v2_scalar_requester_metadata_keeps_the_llm_span(
    callback: str, route: str, gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-v2-scalar-" + uuid.uuid4().hex
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {"callbacks": [callback]},
            marker,
            overrides={
                **({"OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": "__collector__"} if callback == "otel" else {}),
            },
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        model: Final = (
            scenario.model(model="anthropic/claude-sonnet-4-6", api_base=rig.provider.url)
            if route == "/v1/messages"
            else identity["model"]
        )
        response: Final = rig.candidate.request(
            "POST", route, _scalar_requester_metadata_request(route, str(model), marker), key=key
        )
        assert response.status_code == 200, response.text

        attrs: Final = exported_llm_span(rig.collector, response.json()["id"])
        assert attrs.get("gen_ai.usage.input_tokens") == 3, attrs
        if callback == "langfuse_otel":
            observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
            assert "requester_metadata" not in observation, observation
            assert observation["user_api_key_alias"] == identity["key_alias"], observation


@pytest.mark.parametrize(
    "callbacks",
    [["langfuse_otel", "otel"], ["langfuse_otel"]],
    ids=["langfuse-listed-before-otel", "langfuse-with-otel-env"],
)
def test_otel_v2_collector_beside_langfuse_gets_no_langfuse_request_metadata(
    callbacks: list[str], gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-v2-generic-" + uuid.uuid4().hex
    probe: Final = "lf-probe-" + uuid.uuid4().hex
    with (
        wire_server(lambda _request: Reply()) as generic,
        langfuse_rig(
            gateway,
            tmp_path,
            {"callbacks": callbacks},
            marker,
            overrides={"OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": generic.url},
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "metadata": {"caller_note": probe},
                "cache": {"no-cache": True},
            },
            key=key,
            headers={"x-lit8283-probe": probe},
        )
        assert response.status_code == 200, response.text

        langfuse: Final = exported_llm_span(rig.collector, response.json()["id"])
        observation: Final = json.loads(str(langfuse["langfuse.observation.metadata"]))
        assert observation["user_api_key_alias"] == identity["key_alias"], observation
        assert langfuse["langfuse.trace.metadata.user_api_key_alias"] == identity["key_alias"], langfuse
        assert probe not in json.dumps(langfuse, default=str), langfuse

        collector: Final = exported_llm_span(generic, response.json()["id"])
        assert "langfuse.observation.metadata" not in collector, sorted(collector)
        assert not [name for name in collector if name.startswith("langfuse.trace.metadata.user_api_key_")], sorted(
            collector
        )
        assert probe not in json.dumps(collector, default=str), collector
        assert collector["langfuse.trace.metadata.team_id"] == identity["team_id"], collector
        assert collector.get("gen_ai.usage.input_tokens") == 3, collector


def _limit_sweep(
    rig: LangfuseRig,
    key: str,
    marker: str,
    bodies: Mapping[int, Mapping[str, object]],
) -> dict[int, tuple[dict[str, object], int]]:
    """The exported span and dropped-attribute count per size; each body's prompt carries ``<marker>:<size>:``."""
    drained: list[
        tuple[dict[str, object], int]
    ] = []  # mutable-ok: drain() consumes the wire queue, so observed spans must accumulate across eventually polls
    for body in bodies.values():
        response = rig.candidate.request("POST", "/v1/chat/completions", dict(body), key=key)
        assert response.status_code == 200, response.text

    def by_size() -> dict[int, tuple[dict[str, object], int] | None]:
        drained.extend(record for batch in rig.collector.drain() for record in span_records(batch.body))
        return {
            size: next(
                (
                    (attrs, dropped)
                    for attrs, dropped in drained
                    if f"{marker}:{size}:" in str(attrs.get("langfuse.observation.input", ""))
                ),
                None,
            )
            for size in bodies
        }

    observed: Final = eventually(by_size, lambda spans: None not in spans.values(), seconds=30)
    return {size: record for size, record in observed.items() if record is not None}


def test_otel_v2_langfuse_identity_sheds_before_any_message_attribute(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-v2-limit-" + uuid.uuid4().hex
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {},
            marker,
            overrides={
                "MAPPER_NAMES": "genai,openinference",
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only",
            },
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)

        def body(message_count: int) -> dict[str, object]:
            return {
                "model": identity["model"],
                "messages": [
                    {"role": ("user", "assistant")[n % 2], "content": f"{marker} turn {n}"}
                    for n in range(message_count - 1)
                ]
                + [{"role": "user", "content": f"{marker}:{message_count}: turn {message_count - 1}"}],
                "user": identity["end_user"],
                "cache": {"no-cache": True},
            }

        spans: Final = _limit_sweep(
            rig, key, marker, {message_count: body(message_count) for message_count in (1, *range(20, 61))}
        )

        reference, _ = spans[1]
        identity_count: Final = len(identity_attributes(reference))
        assert identity_count == len(chat_identity_fields(identity)) + 1, sorted(reference)
        shed_by_size: Final[dict[int, tuple[int, int]]] = {}  # mutable-ok: filled once per swept span below
        for message_count, (attrs, _) in spans.items():
            kept = {n for n in range(message_count) if f"llm.input_messages.{n}.message.content" in attrs}
            shed_messages = message_count - len(kept)
            shed_identity = identity_count - len(identity_attributes(attrs))
            shed_by_size[message_count] = (shed_identity, shed_messages)
            if shed_messages:
                assert shed_identity == identity_count, (message_count, shed_identity, shed_messages)
        assert [
            size for size, (shed_identity, shed_messages) in shed_by_size.items() if shed_identity and not shed_messages
        ], shed_by_size
        assert [size for size, (_, shed_messages) in shed_by_size.items() if shed_messages], shed_by_size


_APP_SPAN_HOOK: Final = """
from opentelemetry import trace

from litellm.integrations.custom_logger import CustomLogger


class AppSpanHook(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        with trace.get_tracer("lit8283-app").start_as_current_span("app.audit") as span:
            span.set_attributes(
                {
                    "app.response_id": response_obj.id,
                    "langfuse.observation.metadata": '{"source": "app"}',
                    "langfuse.trace.metadata.user_api_key_alias": "app-set-alias",
                }
            )


instance = AppSpanHook()
"""


def test_otel_v2_span_another_tracer_wrote_keeps_its_langfuse_metadata_keys_at_the_collector(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-v2-app-" + uuid.uuid4().hex
    (tmp_path / "lit8283_app_span.py").write_text(_APP_SPAN_HOOK)
    with (
        wire_server(lambda _request: Reply()) as generic,
        langfuse_rig(
            gateway,
            tmp_path,
            {"callbacks": ["langfuse_otel", "otel", "lit8283_app_span.instance"]},
            marker,
            overrides={"OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": generic.url},
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text

        response_id: Final = response.json()["id"]

        def app(span: Mapping[str, object]) -> bool:
            return span.get("app.response_id") == response_id

        def llm(span: Mapping[str, object]) -> bool:
            return decoded_response_id(span.get("gen_ai.response.id")) == decoded_response_id(response_id)

        spans: Final = eventually(
            watched_spans(generic), lambda seen: any(map(app, seen)) and any(map(llm, seen)), seconds=25
        )
        app_span: Final = next(span for span in spans if app(span))
        assert app_span["langfuse.observation.metadata"] == '{"source": "app"}', app_span
        assert app_span["langfuse.trace.metadata.user_api_key_alias"] == "app-set-alias", app_span
        collector_llm: Final = next(span for span in spans if llm(span))
        assert not identity_attributes(collector_llm), sorted(collector_llm)


def _chat_span_drops(body: bytes) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (span.dropped_attributes_count, span.dropped_events_count, span.dropped_links_count)
        for span in exported_spans(body)
        if span.name.startswith("chat ")
    )


def test_otel_v2_collector_keeps_the_dropped_counts_of_a_span_without_langfuse_request_metadata(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-v2-drops-" + uuid.uuid4().hex
    with (
        wire_server(lambda _request: Reply()) as generic,
        langfuse_rig(
            gateway,
            tmp_path,
            {"callbacks": ["langfuse_otel", "otel"]},
            marker,
            overrides={
                "OTEL_EXPORTER": "otlp_http",
                "OTEL_ENDPOINT": generic.url,
                "OTEL_SPAN_EVENT_COUNT_LIMIT": "0",
            },
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": f"{marker} scripted -fail- upstream"}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code >= 500, response.text

        drops: Final[dict[str, list[tuple[int, int, int]]]] = {"langfuse": [], "generic": []}  # mutable-ok: drained

        def observed() -> dict[str, list[tuple[int, int, int]]]:
            for name, wire in (("langfuse", rig.collector), ("generic", generic)):
                drops[name].extend(drop for batch in wire.drain() for drop in _chat_span_drops(batch.body))
            return drops

        settled: Final = eventually(
            observed, lambda seen: bool(seen["langfuse"]) and len(seen["generic"]) == len(seen["langfuse"]), seconds=25
        )
        assert any(events for _, events, _ in settled["langfuse"]), settled
        assert sorted(settled["generic"]) == sorted(settled["langfuse"]), settled


def test_otel_v2_langfuse_mapper_sends_no_langfuse_request_metadata_to_a_plain_collector(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-v2-mapper-" + uuid.uuid4().hex
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {"callbacks": ["otel"]},
            marker,
            overrides={
                "MAPPER_NAMES": "genai,langfuse",
                "OTEL_EXPORTER": "otlp_http",
                "OTEL_ENDPOINT": "__collector__",
            },
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "user": identity["end_user"],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text

        collector: Final = exported_llm_span(rig.collector, response.json()["id"])
        assert collector["langfuse.trace.metadata.team_id"] == identity["team_id"], collector
        assert not identity_attributes(collector), sorted(collector)
