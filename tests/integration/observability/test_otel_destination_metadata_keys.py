"""A team's or key's own OTel v2 destination can list extra request metadata keys in ``baggage_metadata_keys``,
exported as ``litellm.metadata.*`` on that destination's spans only, on top of the operator's global allowlist."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.otlp_sink import (
    ConnectSink,
    Span,
    SpanSinks,
    audit_configuration_writer,
    owned_connect_sink,
    owned_sinks,
    recorded_spans,
    spans_for_trace,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

OPERATION: Final = "gen_ai.operation.name"
SERVER_KIND: Final = 2
METADATA_PREFIX: Final = "litellm.metadata."
TEAM_KEYS: Final = "requester_metadata.tenant_id, requester_metadata.order.id"

Backend = Literal["newrelic", "arize"]


def _chat(_request: Request) -> Reply:
    return Reply(
        body=(
            b'{"id":"chatcmpl-meta","object":"chat.completion","created":1,"model":"gpt-4o-mini",'
            b'"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}'
        )
    )


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    sinks: SpanSinks
    newrelic: ConnectSink
    upstream: Wire

    def sink(self, backend: Backend) -> str:
        return self.newrelic.control_url if backend == "newrelic" else self.sinks.arize

    def destination(self, backend: Backend, extra: Mapping[str, str]) -> dict[str, JsonValue]:
        tag: Final = uuid.uuid4().hex
        credentials: Final[Mapping[str, str]] = (
            {"newrelic_api_key": f"nr-{tag}", "newrelic_region": "us"}
            if backend == "newrelic"
            else {"arize_space_key": f"space-{tag}", "arize_api_key": f"arize-{tag}"}
        )
        return {
            "callback_name": backend,
            "callback_type": "success_and_failure",
            "callback_vars": {**credentials, **extra},
        }

    def team_key(self, scenario: Scenario, backend: Backend, extra: Mapping[str, str]) -> str:
        team: Final = scenario.team()
        attached: Final = self.proxy.request("POST", f"/team/{team}/callback", self.destination(backend, extra))
        assert attached.status_code == 200, attached.text
        return scenario.key(team_id=team)

    def send(self, key: str, model: str, metadata: Mapping[str, JsonValue]) -> str:
        trace_id: Final = uuid.uuid4().hex
        response: Final = self.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "hi"}], "metadata": dict(metadata)},
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        assert response.status_code == 200, response.text
        return trace_id

    def request_spans(self, url: str, since: int, trace_id: str) -> tuple[Span, ...]:
        return _server_and_model_spans(
            eventually(
                lambda: spans_for_trace(recorded_spans(url, since)[1], trace_id),
                lambda found: len(_server_and_model_spans(found)) >= 2,
                seconds=40,
            )
        )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("destination-metadata-keys")
    with (
        wire_server(_chat) as upstream,
        owned_sinks(directory / "sinks") as sinks,
        owned_connect_sink(directory / "newrelic") as newrelic,
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {
                "LITELLM_OTEL_V2": "1",
                "OTEL_BSP_SCHEDULE_DELAY": "300",
                "LITELLM_OTEL_BAGGAGE_METADATA_KEYS": "requester_metadata.correlation_id",
                "ARIZE_HTTP_ENDPOINT": sinks.arize + "/v1/traces",
                "HTTPS_PROXY": newrelic.proxy_url,
                "NO_PROXY": "127.0.0.1,localhost",
                "REQUESTS_CA_BUNDLE": newrelic.ca_pem,
            },
            config=audit_configuration_writer(sinks)(directory, {}),
            remove_environment=("OTEL_ENDPOINT", "OTEL_EXPORTER", "OTEL_EXPORTER_OTLP_ENDPOINT"),
            workers=2,
        ) as owned,
    ):
        yield Rig(owned.gateway, sinks, newrelic, upstream)


def _server_and_model_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    return tuple(span for span in spans if span["kind"] == SERVER_KIND or OPERATION in span["attributes"])


def _exported_metadata(spans: tuple[Span, ...]) -> list[dict[str, JsonValue]]:
    return [
        {name: value for name, value in span["attributes"].items() if name.startswith(METADATA_PREFIX)}
        for span in spans
    ]


def _request_metadata(marker: str) -> dict[str, JsonValue]:
    return {
        "correlation_id": f"corr-{marker}",
        "tenant_id": f"tenant-{marker}",
        "order": {"id": f"order-{marker}"},
        "unlisted": f"unlisted-{marker}",
    }


@pytest.mark.parametrize("level", ["team", "key"])
@pytest.mark.parametrize("backend", ["newrelic", "arize"])
def test_destination_keys_reach_that_destination_and_not_the_operator(
    rig: Rig, backend: Backend, level: Literal["team", "key"]
) -> None:
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.upstream.url + "/v1")
        keys: Final = {"baggage_metadata_keys": TEAM_KEYS}
        key: Final = (
            rig.team_key(scenario, backend, keys)
            if level == "team"
            else scenario.key(metadata={"logging": [rig.destination(backend, keys)]})
        )
        tenant_since: Final = recorded_spans(rig.sink(backend))[0]
        operator_since: Final = recorded_spans(rig.sinks.operator)[0]
        marker: Final = uuid.uuid4().hex
        trace_id: Final = rig.send(key, model, _request_metadata(marker))
        tenant: Final = rig.request_spans(rig.sink(backend), tenant_since, trace_id)
        operator: Final = rig.request_spans(rig.sinks.operator, operator_since, trace_id)

    assert sorted(span["kind"] == SERVER_KIND for span in tenant) == [False, True], [span["name"] for span in tenant]
    assert _exported_metadata(tenant) == [
        {
            "litellm.metadata.correlation_id": f"corr-{marker}",
            "litellm.metadata.tenant_id": f"tenant-{marker}",
            "litellm.metadata.order.id": f"order-{marker}",
        }
    ] * len(tenant), [(span["name"], sorted(span["attributes"])) for span in tenant]
    assert _exported_metadata(operator) == [{"litellm.metadata.correlation_id": f"corr-{marker}"}] * len(operator), [
        (span["name"], sorted(span["attributes"])) for span in operator
    ]


@pytest.mark.parametrize("backend", ["newrelic", "arize"])
def test_a_destination_without_keys_and_a_request_body_cannot_add_them(rig: Rig, backend: Backend) -> None:
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.upstream.url + "/v1")
        listing: Final = rig.team_key(scenario, backend, {"baggage_metadata_keys": TEAM_KEYS})
        sibling: Final = rig.team_key(scenario, backend, {})
        since: Final = recorded_spans(rig.sink(backend))[0]
        marker: Final = uuid.uuid4().hex
        listed_trace: Final = rig.send(listing, model, _request_metadata(f"listed-{marker}"))
        sibling_trace: Final = rig.send(
            sibling,
            model,
            {
                **_request_metadata(marker),
                "baggage_metadata_keys": TEAM_KEYS,
                "callback_vars": {"baggage_metadata_keys": TEAM_KEYS},
            },
        )
        listed: Final = rig.request_spans(rig.sink(backend), since, listed_trace)
        unlisted: Final = rig.request_spans(rig.sink(backend), since, sibling_trace)

    assert {"litellm.metadata.tenant_id": f"tenant-listed-{marker}"}.items() <= _exported_metadata(listed)[0].items()
    assert _exported_metadata(unlisted) == [{"litellm.metadata.correlation_id": f"corr-{marker}"}] * len(unlisted), [
        (span["name"], sorted(span["attributes"])) for span in unlisted
    ]


def _attach(rig: Rig, team: str, keys: str, *, key: str | None = None) -> httpx.Response:
    return rig.proxy.request(
        "POST",
        f"/team/{team}/callback",
        rig.destination("newrelic", {"baggage_metadata_keys": keys}),
        key=key,
    )


def test_a_team_admin_can_set_keys_and_a_plain_member_cannot(rig: Rig) -> None:
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team()
        admin_key: Final = scenario.key(user_id=scenario.member(team, role="admin"), team_id=team)
        member_key: Final = scenario.key(user_id=scenario.member(team), team_id=team)

        refused: Final = _attach(rig, team, "requester_metadata.tenant_id", key=member_key)
        accepted: Final = _attach(rig, team, "requester_metadata.tenant_id", key=admin_key)

    assert refused.status_code == 403, refused.text
    assert accepted.status_code == 200, accepted.text


@pytest.mark.parametrize(
    ("keys", "reason"),
    [
        ("requester_metadata.user_api_key_user_id", "proxy identity attribute"),
        ("tenant_id", "requester_metadata.<key>"),
        ("requester_metadata.headers.x-internal-token", "would export request headers"),
        (",".join(f"requester_metadata.k{index}" for index in range(21)), "at most 20 keys"),
    ],
)
def test_unusable_keys_are_refused_on_team_and_key_callbacks(rig: Rig, keys: str, reason: str) -> None:
    with rig.proxy.scenario() as scenario:
        team_response: Final = _attach(rig, scenario.team(), keys)
        key_response: Final = rig.proxy.request(
            "POST",
            "/key/generate",
            {"metadata": {"logging": [rig.destination("newrelic", {"baggage_metadata_keys": keys})]}},
        )

    assert [(response.status_code, reason in response.text) for response in (team_response, key_response)] == [
        (422, True),
        (400, True),
    ], [team_response.text, key_response.text]


def test_unset_keys_leave_the_destination_on_the_global_allowlist(rig: Rig) -> None:
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.upstream.url + "/v1")
        key: Final = rig.team_key(scenario, "newrelic", MappingProxyType({}))
        since: Final = recorded_spans(rig.newrelic.control_url)[0]
        marker: Final = uuid.uuid4().hex
        spans: Final = rig.request_spans(
            rig.newrelic.control_url, since, rig.send(key, model, _request_metadata(marker))
        )

    assert _exported_metadata(spans) == [{"litellm.metadata.correlation_id": f"corr-{marker}"}] * len(spans)
