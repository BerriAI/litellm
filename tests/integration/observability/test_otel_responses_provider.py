import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.openai_wire import answering_model_discovery, responses_reply
from integration._support.otlp_sink import Span, owned_sinks, recorded_spans, spans_for_trace
from integration._support.process import owned_proxy_process
from integration._support.prometheus_series import scrape
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

INPUT_PRICE: Final = 0.001
OUTPUT_PRICE: Final = 0.002
INPUT_TOKENS: Final = 5
OUTPUT_TOKENS: Final = 3
UPSTREAM_MODEL: Final = "provider-label-model"
REFUSED_DELETE: Final = "refuseddelete"
MARKER: Final = re.compile(rb"providerlabel(?:refuseddelete)?[0-9a-f]{32}")
REFUSE: Final = "refuse-this-request"
REFUSAL: Final = "upstream refused this request"
PRIMARY_SPAN: Final = "litellm_request"
RAW_SPAN: Final = "raw_gen_ai_request"
OTLP_STATUS_ERROR: Final = 2
STREAM_EVENTS: Final = (
    "response.created",
    "response.output_item.added",
    "response.output_text.delta",
    "response.output_item.done",
    "response.completed",
)
DEPLOYMENT_METRICS: Final = (
    "litellm_deployment_state",
    "litellm_deployment_tpm_limit",
    "litellm_deployment_rpm_limit",
)


@dataclass(frozen=True, slots=True)
class Deployment:
    group: str
    model_id: str
    provider: str
    params: Mapping[str, JsonValue]
    info: Mapping[str, JsonValue]
    base_path: str

    def entry(self, upstream: Wire) -> dict[str, JsonValue]:
        return {
            "model_name": self.group,
            "litellm_params": {
                **self.params,
                "api_key": "integration-provider-key",
                "api_base": upstream.url + self.base_path,
                "input_cost_per_token": INPUT_PRICE,
                "output_cost_per_token": OUTPUT_PRICE,
            },
            "model_info": {"id": self.model_id, **self.info},
        }


def _deployment(
    name: str,
    provider: str,
    params: Mapping[str, JsonValue],
    info: Mapping[str, JsonValue] = MappingProxyType({}),
    base_path: str = "/v1",
) -> Deployment:
    suffix: Final = uuid.uuid4().hex[:12]
    return Deployment(f"{name}-{suffix}", f"{name}-id-{suffix}", provider, params, info, base_path)


PREFIXED: Final = _deployment("responses-openai-prefix", "openai", {"model": f"openai/{UPSTREAM_MODEL}"})
DECLARED: Final = _deployment(
    "responses-declared-openai", "openai", {"model": UPSTREAM_MODEL, "custom_llm_provider": "openai"}
)
AZURE: Final = _deployment(
    "responses-azure-prefix",
    "azure",
    {"model": f"azure/{UPSTREAM_MODEL}", "api_version": "2024-10-21"},
    base_path="",
)
METERED: Final = _deployment(
    "responses-metered", "openai", {"model": f"openai/{UPSTREAM_MODEL}"}, info={"tpm": 100000, "rpm": 1000}
)
DEPLOYMENTS: Final = (PREFIXED, DECLARED, AZURE, METERED)


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    sink: str
    upstream: Wire


@dataclass(frozen=True, slots=True)
class Traced:
    response: httpx.Response
    trace_id: str


def _marker() -> str:
    return "providerlabel" + uuid.uuid4().hex


def _marker_refused_on_delete() -> str:
    return f"providerlabel{REFUSED_DELETE}{uuid.uuid4().hex}"


def _answer(marker: str) -> str:
    return "answer to " + marker


def _upstream_id(marker: str) -> str:
    return "resp_" + marker


def _json_reply(body: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _refusal() -> Reply:
    return Reply(
        status=400,
        body=json.dumps({"error": {"message": REFUSAL, "type": "invalid_request_error", "code": "refused"}}).encode(),
    )


def _response_reply(marker: str, *, stream: bool = False) -> Reply:
    return responses_reply(_upstream_id(marker), UPSTREAM_MODEL, _answer(marker), stream=stream)


def _delete_reply(marker: str) -> Reply:
    if REFUSED_DELETE in marker:
        return _refusal()
    return _json_reply({"id": _upstream_id(marker), "object": "response", "deleted": True})


def _generation_reply(request: Request, marker: str) -> Reply:
    if REFUSE.encode() in request.body:
        return _refusal()
    return _response_reply(marker, stream=JSON_OBJECT.validate_json(request.body).get("stream") is True)


def _respond(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    found: Final = MARKER.search(request.body + path.encode())
    assert found is not None, request
    marker: Final = found.group().decode()
    if request.method == "DELETE":
        return _delete_reply(marker)
    if path.endswith("/input_items"):
        return _json_reply({"object": "list", "data": [{"id": "msg_input_" + marker}]})
    if request.method == "GET" or path.endswith("/cancel"):
        return _response_reply(marker)
    return _generation_reply(request, marker)


def _config(upstream: Wire, telemetry: Mapping[str, JsonValue], callbacks: tuple[str, ...]) -> dict[str, JsonValue]:
    base: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    return {
        **base,
        **telemetry,
        "model_list": [
            *list_value(base.get("model_list") or []),
            *(deployment.entry(upstream) for deployment in DEPLOYMENTS),
        ],
        "litellm_settings": {
            **object_value(base["litellm_settings"]),
            "cache": False,
            **({"callbacks": list(callbacks)} if callbacks else {}),
        },
        "router_settings": {**object_value(base.get("router_settings") or {}), "num_retries": 0},
    }


def _write_config(directory: Path, config: Mapping[str, JsonValue]) -> Path:
    path: Final = directory / f"responses-provider-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(dict(config)))
    return path


@pytest.fixture(scope="module")
def upstream() -> Iterator[Wire]:
    with wire_server(answering_model_discovery(_respond)) as wire:
        yield wire


@pytest.fixture(scope="module")
def rig(upstream: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("responses-provider-otel")
    samples: Final = directory / "prometheus-multiproc"
    samples.mkdir()
    with owned_sinks(directory / "sinks") as sinks:
        config: Final = _config(
            upstream,
            {
                "callback_settings": {
                    "otel": {"exporter": "http/json", "endpoint": sinks.operator, "use_simple_processor": True}
                }
            },
            ("otel", "prometheus"),
        )
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway,
                directory,
                {
                    "OTEL_EXPORTER": "http/json",
                    "OTEL_ENDPOINT": sinks.operator,
                    "PROMETHEUS_MULTIPROC_DIR": str(samples),
                },
                config=_write_config(directory, config),
                remove_environment=("LITELLM_OTEL_V2",),
                workers=2,
                client_timeout=60,
            ) as owned,
        ):
            yield Rig(owned.gateway, sinks.operator, upstream)


@pytest.fixture(scope="module")
def plain(upstream: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("responses-provider-plain")
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {},
            config=_write_config(directory, _config(upstream, {}, ())),
            remove_environment=("LITELLM_OTEL_V2", "OTEL_EXPORTER", "OTEL_ENDPOINT"),
            workers=2,
            client_timeout=60,
        ) as owned,
    ):
        yield owned.gateway


def _send(gateway: Gateway, method: str, path: str, body: Mapping[str, JsonValue] | None = None) -> Traced:
    trace_id: Final = uuid.uuid4().hex
    response: Final = gateway.request(
        method, path, body, headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"}
    )
    return Traced(response, trace_id)


def _create(rig: Rig, deployment: Deployment, prompt: str, *, stream: bool = False) -> Traced:
    body: Final[dict[str, JsonValue]] = {"model": deployment.group, "input": prompt}
    return _send(rig.proxy, "POST", "/v1/responses", {**body, "stream": True} if stream else body)


def _encoded_id(rig: Rig, marker: str) -> str:
    created: Final = _create(rig, PREFIXED, marker)
    assert created.response.status_code == 200, created.response.text
    rig.upstream.drain()
    encoded: Final = string_value(object_value(created.response.json())["id"])
    assert encoded != _upstream_id(marker), created.response.text
    return encoded


def _forwarded(upstream: Wire, marker: str) -> tuple[Request, ...]:
    return tuple(request for request in upstream.drain() if marker.encode() in request.body or marker in request.target)


def _calls(upstream: Wire, marker: str) -> tuple[tuple[str, str], ...]:
    return tuple((request.method, urlsplit(request.target).path) for request in _forwarded(upstream, marker))


def _trace(rig: Rig, trace_id: str, *names: str) -> tuple[Span, ...]:
    return eventually(
        lambda: spans_for_trace(recorded_spans(rig.sink)[1], trace_id),
        lambda spans: all(any(span["name"] == name for span in spans) for name in names),
        seconds=30,
    )


def _only(spans: tuple[Span, ...], name: str) -> Span:
    found: Final = tuple(span for span in spans if span["name"] == name)
    assert len(found) == 1, spans
    return found[0]


def _keys_under(span: Span, prefix: str) -> tuple[str, ...]:
    return tuple(key for key in span["attributes"] if key.startswith(prefix))


def _none_provider_keys(spans: tuple[Span, ...]) -> tuple[str, ...]:
    return tuple(chain.from_iterable(_keys_under(span, "llm.None.") for span in spans))


def _none_valued_attributes(span: Span) -> tuple[str, ...]:
    return tuple(key for key, value in span["attributes"].items() if value is None or value == "None")


def _assert_labeled_with(spans: tuple[Span, ...], provider: str) -> None:
    primary: Final = _only(spans, PRIMARY_SPAN)
    raw: Final = _only(spans, RAW_SPAN)
    assert primary["attributes"].get("gen_ai.system") == provider, primary
    assert _keys_under(raw, f"llm.{provider}."), raw
    assert _none_provider_keys(spans) == (), spans


def _assert_error_labeled_with(spans: tuple[Span, ...], provider: str) -> Span:
    primary: Final = _only(spans, PRIMARY_SPAN)
    assert primary.get("status_code") == OTLP_STATUS_ERROR, primary
    assert primary["attributes"].get("gen_ai.system") == provider, primary
    return primary


def _output_text(body: Mapping[str, JsonValue]) -> str:
    message: Final = object_value(list_value(body["output"])[0])
    return string_value(object_value(list_value(message["content"])[0])["text"])


def _sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _spend_row(gateway: Gateway, call_id: str) -> Mapping[str, JsonValue]:
    rows: Final = eventually(
        lambda: gateway.rows(
            'SELECT status, spend, custom_llm_provider FROM "LiteLLM_SpendLogs" '
            "WHERE request_id=%s OR litellm_call_id=%s",
            (call_id, call_id),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


@pytest.mark.parametrize(
    "deployment",
    (PREFIXED, DECLARED, AZURE),
    ids=("openai-model-prefix", "declared-openai-with-bare-model", "azure-model-prefix"),
)
def test_a_responses_call_labels_its_spans_with_the_deployment_provider(rig: Rig, deployment: Deployment) -> None:
    marker: Final = _marker()

    created: Final = _create(rig, deployment, marker)

    assert created.response.status_code == 200, created.response.text
    assert _output_text(object_value(created.response.json())) == _answer(marker), created.response.text
    assert len(_forwarded(rig.upstream, marker)) == 1
    _assert_labeled_with(_trace(rig, created.trace_id, PRIMARY_SPAN, RAW_SPAN), deployment.provider)


def test_a_streamed_responses_call_delivers_the_full_stream_and_labels_its_spans(rig: Rig) -> None:
    marker: Final = _marker()

    created: Final = _create(rig, PREFIXED, marker, stream=True)

    assert created.response.status_code == 200, created.response.text
    events: Final = _sse_events(created.response.text)
    kinds: Final = tuple(string_value(event["type"]) for event in events)
    assert tuple(kind for kind in kinds if kind in STREAM_EVENTS) == STREAM_EVENTS, kinds
    deltas: Final = tuple(
        string_value(event["delta"]) for event in events if event["type"] == "response.output_text.delta"
    )
    assert "".join(deltas) == _answer(marker), events
    completed: Final = next(event for event in events if event["type"] == "response.completed")
    assert _output_text(object_value(completed["response"])) == _answer(marker), events
    _assert_labeled_with(_trace(rig, created.trace_id, PRIMARY_SPAN, RAW_SPAN), PREFIXED.provider)


@pytest.mark.parametrize(
    ("method", "suffix"),
    (("GET", ""), ("POST", "/cancel")),
    ids=("retrieve", "cancel"),
)
def test_a_follow_up_on_an_encoded_response_id_reports_the_provider(rig: Rig, method: str, suffix: str) -> None:
    marker: Final = _marker()
    encoded: Final = _encoded_id(rig, marker)

    follow_up: Final = _send(rig.proxy, method, f"/v1/responses/{encoded}{suffix}")

    assert follow_up.response.status_code == 200, follow_up.response.text
    assert _calls(rig.upstream, marker) == ((method, f"/v1/responses/{_upstream_id(marker)}{suffix}"),)
    spans: Final = _trace(rig, follow_up.trace_id, PRIMARY_SPAN)
    assert _only(spans, PRIMARY_SPAN)["attributes"].get("gen_ai.system") == PREFIXED.provider, spans


def test_listing_input_items_of_an_encoded_response_id_labels_the_raw_span(rig: Rig) -> None:
    marker: Final = _marker()
    encoded: Final = _encoded_id(rig, marker)

    listed: Final = _send(rig.proxy, "GET", f"/v1/responses/{encoded}/input_items")

    assert listed.response.status_code == 200, listed.response.text
    assert marker in listed.response.text
    assert _calls(rig.upstream, marker) == (("GET", f"/v1/responses/{_upstream_id(marker)}/input_items"),)
    spans: Final = _trace(rig, listed.trace_id, PRIMARY_SPAN, RAW_SPAN)
    assert _keys_under(_only(spans, RAW_SPAN), f"llm.{PREFIXED.provider}."), spans
    assert _none_provider_keys(spans) == (), spans


def test_compacting_labels_its_spans_with_the_provider_inferred_from_the_model(rig: Rig) -> None:
    marker: Final = _marker()

    compacted: Final = _send(rig.proxy, "POST", "/v1/responses/compact", {"model": PREFIXED.group, "input": marker})

    assert compacted.response.status_code == 200, compacted.response.text
    assert _calls(rig.upstream, marker) == (("POST", "/v1/responses/compact"),)
    _assert_labeled_with(_trace(rig, compacted.trace_id, PRIMARY_SPAN, RAW_SPAN), PREFIXED.provider)


@pytest.mark.parametrize("stream", (False, True), ids=("plain", "streaming"))
def test_an_upstream_400_reaches_the_client_and_labels_the_error_span(rig: Rig, stream: bool) -> None:
    marker: Final = _marker()

    refused: Final = _create(rig, PREFIXED, f"{REFUSE} {marker}", stream=stream)

    assert refused.response.status_code == 400, refused.response.text
    assert REFUSAL in refused.response.text
    primary: Final = _assert_error_labeled_with(_trace(rig, refused.trace_id, PRIMARY_SPAN), PREFIXED.provider)
    assert _none_valued_attributes(primary) == (), primary


@pytest.mark.parametrize("metric", DEPLOYMENT_METRICS)
def test_deployment_metrics_of_a_responses_call_carry_the_provider(rig: Rig, metric: str) -> None:
    created: Final = _create(rig, METERED, _marker())
    assert created.response.status_code == 200, created.response.text

    samples: Final = eventually(
        lambda: tuple(
            sample
            for sample in scrape(rig.proxy)
            if sample.name == metric and sample.labels.get("model_id") == METERED.model_id
        ),
        bool,
        seconds=30,
    )

    assert {sample.labels["api_provider"] for sample in samples} == {METERED.provider}, samples


def test_an_upstream_400_on_delete_reaches_the_client_and_labels_the_error_span(rig: Rig) -> None:
    marker: Final = _marker_refused_on_delete()
    encoded: Final = _encoded_id(rig, marker)

    deleted: Final = _send(rig.proxy, "DELETE", f"/v1/responses/{encoded}")

    assert deleted.response.status_code == 400, deleted.response.text
    assert REFUSAL in deleted.response.text
    assert _calls(rig.upstream, marker) == (("DELETE", f"/v1/responses/{_upstream_id(marker)}"),)
    _assert_error_labeled_with(_trace(rig, deleted.trace_id, PRIMARY_SPAN), PREFIXED.provider)


def test_without_telemetry_a_responses_call_returns_the_upstream_answer_and_is_billed(
    plain: Gateway, upstream: Wire
) -> None:
    marker: Final = _marker()

    response: Final = plain.request("POST", "/v1/responses", {"model": PREFIXED.group, "input": marker})

    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    usage: Final = object_value(body["usage"])
    assert (body["status"], _output_text(body), usage["input_tokens"], usage["output_tokens"]) == (
        "completed",
        _answer(marker),
        INPUT_TOKENS,
        OUTPUT_TOKENS,
    ), response.text
    forwarded: Final = _forwarded(upstream, marker)
    assert len(forwarded) == 1, forwarded
    assert b"custom_llm_provider" not in forwarded[0].body, forwarded[0].body
    row: Final = _spend_row(plain, response.headers["x-litellm-call-id"])
    assert (row["status"], row["custom_llm_provider"]) == ("success", PREFIXED.provider), row
    assert _number(row["spend"]) == pytest.approx(INPUT_TOKENS * INPUT_PRICE + OUTPUT_TOKENS * OUTPUT_PRICE), row


def test_without_telemetry_an_upstream_400_reaches_the_client_and_is_logged_unbilled(
    plain: Gateway, upstream: Wire
) -> None:
    marker: Final = _marker()

    response: Final = plain.request("POST", "/v1/responses", {"model": PREFIXED.group, "input": f"{REFUSE} {marker}"})

    assert response.status_code == 400, response.text
    assert REFUSAL in response.text
    forwarded: Final = _forwarded(upstream, marker)
    assert len(forwarded) == 1, forwarded
    assert b"custom_llm_provider" not in forwarded[0].body, forwarded[0].body
    row: Final = _spend_row(plain, response.headers["x-litellm-call-id"])
    assert (row["status"], row["custom_llm_provider"]) == ("failure", PREFIXED.provider), row
    assert _number(row["spend"]) == 0, row
