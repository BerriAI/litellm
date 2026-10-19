"""A team's Arize sampling rate decides once per trace, so pods serving calls of one trace agree.

Two proxies share one database, as two pods behind a load balancer do. Each trace sends one call
to each pod under the caller's ``traceparent``; the team's Arize space must receive both calls of a
trace or neither.
"""

import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import pytest
import yaml
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    until_asserted,
)
from integration._support.database import scratch_database
from integration._support.otlp_sink import Span, SpanSinks, recorded_requests, recorded_spans
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

MODEL: Final = "team-sampling"
SERVER: Final = 2
CALL_ID: Final = "litellm.call_id"
OPERATION: Final = "gen_ai.operation.name"
SPACE: Final = "space-sampled"
CREDENTIALS: Final = {"arize_space_key": SPACE, "arize_api_key": "arize-sampled"}
FAILING: Final = re.compile(rb"sampling-fail")
TRACES_PER_CASE: Final = 16


def _upstream(request: Request) -> Reply:
    if FAILING.search(request.body) is not None:
        return Reply(status=500, body=b'{"error":{"message":"scripted upstream failure","type":"server_error"}}')
    return Reply(
        body=(
            b'{"id":"chatcmpl-sampling","object":"chat.completion","created":1,"model":"gpt-4o-mini",'
            b'"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}'
        )
    )


@dataclass(frozen=True, slots=True)
class Sent:
    trace_id: str
    parent_span_id: str
    call_id: str


@dataclass(frozen=True, slots=True)
class Pods:
    first: Gateway
    second: Gateway
    scenario: Scenario
    sinks: SpanSinks

    def team_key(self, variables: Mapping[str, str]) -> str:
        team: Final = self.scenario.team()
        attached: Final = self.first.request(
            "POST", f"/team/{team}/callback", {"callback_name": "arize", "callback_vars": dict(variables)}
        )
        assert attached.status_code == 200, attached.text
        return self.scenario.key(team_id=team)

    def send(self, pod: Gateway, key: str, trace_id: str, content: str = "hello") -> Sent:
        parent: Final = uuid.uuid4().hex[:16]
        call_id: Final = str(uuid.uuid4())
        body: Final[dict[str, JsonValue]] = {"model": MODEL, "messages": [{"role": "user", "content": content}]}
        response: Final = pod.client.post(
            "/v1/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {key}",
                "traceparent": f"00-{trace_id}-{parent}-01",
                "x-litellm-call-id": call_id,
            },
        )
        assert response.status_code == (500 if FAILING.search(content.encode()) else 200), response.text
        return Sent(trace_id, parent, call_id)


def _space_span_ids(url: str) -> frozenset[str]:
    """Span ids the Arize sink received in export requests sent with the team's space id."""
    return frozenset(
        span_id
        for request in recorded_requests(url)
        if {name.lower(): value for name, value in object_value(request.get("headers") or {}).items()}.get(
            "arize-space-id"
        )
        == SPACE
        for span_id in request.get("span_ids") or ()
        if isinstance(span_id, str)
    )


def _roots(spans: tuple[Span, ...], sent: tuple[Sent, ...]) -> frozenset[str]:
    """Call ids of ``sent`` whose server span is among ``spans``."""
    by_parent: Final = {(call.trace_id, call.parent_span_id): call.call_id for call in sent}
    return frozenset(
        by_parent[(span["trace_id"], span["parent_span_id"])]
        for span in spans
        if span["kind"] == SERVER and (span["trace_id"], span["parent_span_id"]) in by_parent
    )


def _team_roots(pods: Pods, since: int, sent: tuple[Sent, ...]) -> frozenset[str]:
    owned: Final = _space_span_ids(pods.sinks.arize)
    spans: Final = recorded_spans(pods.sinks.arize, since)[1]
    return _roots(tuple(span for span in spans if span["span_id"] in owned), sent)


def _logged(pods: Pods, since: int, sent: tuple[Sent, ...]) -> frozenset[str]:
    """Call ids of ``sent`` whose model span reached the operator's unsampled copy."""
    spans: Final = recorded_spans(pods.sinks.operator, since)[1]
    return frozenset(
        str(span["attributes"][CALL_ID])
        for span in spans
        if OPERATION in span["attributes"] and CALL_ID in span["attributes"]
    ) & {call.call_id for call in sent}


def _trace_ids(case: str) -> tuple[str, ...]:
    """Fixed RFC 4122 shaped trace ids, like the uuid4 hex a caller mints for ``traceparent``."""
    return tuple(
        uuid.uuid5(uuid.NAMESPACE_URL, f"team-sampling/{case}/{index}").hex for index in range(TRACES_PER_CASE)
    )


@pytest.mark.parametrize(
    ("variables", "content"),
    [({"arize_success_sampling_rate": "0.5"}, "hello"), ({"arize_error_sampling_rate": "0.5"}, "sampling-fail")],
    ids=["success-rate", "error-rate"],
)
def test_each_trace_reaches_the_team_space_whole_or_not_at_all(
    pods: Pods, variables: Mapping[str, str], content: str
) -> None:
    sampled: Final = pods.team_key({**CREDENTIALS, **variables})
    barrier: Final = pods.team_key(CREDENTIALS)
    operator_since: Final = recorded_spans(pods.sinks.operator)[0]
    arize_since: Final = recorded_spans(pods.sinks.arize)[0]
    case: Final = "-".join(sorted(variables))
    calls: Final = tuple(
        pods.send(pod, sampled, trace_id, content) for trace_id in _trace_ids(case) for pod in (pods.first, pods.second)
    )
    eventually(
        lambda: _logged(pods, operator_since, calls),
        lambda logged: len(logged) == len(calls),
        seconds=60,
    )
    # The unsampled team shares the sampled team's exporter, which sends spans in the order they were handed to it,
    # so its calls arriving means every decided call of the sampled team has been sent too.
    fences: Final = tuple(pods.send(pod, barrier, uuid.uuid4().hex) for pod in (pods.first, pods.second))

    def whole_or_nothing() -> dict[str, tuple[bool, bool]]:
        delivered: Final = _team_roots(pods, arize_since, calls + fences)
        assert {fence.call_id for fence in fences} <= delivered, "the unsampled team's calls have not arrived yet"
        by_trace: Final = {
            trace_id: (calls[2 * index].call_id in delivered, calls[2 * index + 1].call_id in delivered)
            for index, trace_id in enumerate(_trace_ids(case))
        }
        split: Final = {trace_id: kept for trace_id, kept in by_trace.items() if kept[0] != kept[1]}
        assert split == {}, f"pods disagreed on {len(split)} of {len(by_trace)} traces (first pod, second pod): {split}"
        return by_trace

    verdicts: Final = until_asserted(whole_or_nothing, seconds=30)
    kept: Final = sum(1 for first, _ in verdicts.values() if first)
    assert 0 < kept < TRACES_PER_CASE, f"a rate of 0.5 kept {kept} of {TRACES_PER_CASE} traces"


def _write_config(directory: Path, sinks: SpanSinks, provider: Wire) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        *config.get("model_list", []),
        {
            "model_name": MODEL,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": provider.url + "/v1",
            },
        },
    ]
    config["litellm_settings"] = {
        **config.get("litellm_settings", {}),
        "callbacks": ["langfuse_otel"],
        "otel_tenant_destination_mode": "additive",
        "provider_url_destination_allowed_hosts": [urlparse(sinks.operator).netloc, urlparse(sinks.arize).netloc],
    }
    config["router_settings"] = {**config.get("router_settings", {}), "num_retries": 0}
    config["general_settings"] = {**config.get("general_settings", {}), "user_api_key_cache_ttl": 2}
    path: Final = directory / "team-sampling.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _two_pods(provider: Wire, sinks: SpanSinks, directory: Path) -> Iterator[Pods]:
    environment: Final = {
        "LITELLM_OTEL_V2": "1",
        "OTEL_BSP_SCHEDULE_DELAY": "300",
        "MAPPER_NAMES": "genai,openinference",
        "LANGFUSE_HOST": sinks.operator,
        "LANGFUSE_PUBLIC_KEY": "pk-lf-operator",
        "LANGFUSE_SECRET_KEY": "sk-lf-operator",
        "ARIZE_SPACE_KEY": "space-operator",
        "ARIZE_API_KEY": "arize-operator",
        "ARIZE_HTTP_ENDPOINT": sinks.arize + "/v1/traces",
    }
    removed: Final = ("LITELLM_OTEL_TENANT_DESTINATION_MODE", "OTEL_TRACES_SAMPLER", "OTEL_TRACES_SAMPLER_ARG")
    config: Final = _write_config(directory, sinks, provider)
    for pod in ("first", "second"):
        (directory / pod).mkdir()
    with gateway_from_environment() as shared, scratch_database("team_sampling") as database_url:
        overrides: Final = {**environment, "DATABASE_URL": database_url}
        with (
            owned_proxy_process(
                shared, directory / "first", overrides, config=config, remove_environment=removed
            ) as first,
            owned_proxy_process(
                shared, directory / "second", overrides, config=config, remove_environment=removed
            ) as second,
            replace(first.gateway, database_url=database_url).scenario() as scenario,
        ):
            yield Pods(first.gateway, second.gateway, scenario, sinks)


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_upstream) as wire:
        yield wire


@pytest.fixture(scope="module")
def pods(provider: Wire, audit_sinks: SpanSinks, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pods]:
    with _two_pods(provider, audit_sinks, tmp_path_factory.mktemp("team-sampling")) as started:
        yield started
