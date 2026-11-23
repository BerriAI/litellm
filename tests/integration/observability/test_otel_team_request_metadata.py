"""A team's or key's own New Relic destination gets the operator-allowlisted request metadata as
``litellm.metadata.*``, whether or not ``otel`` is listed in ``litellm_settings.callbacks`` (LIT-9783)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.otlp_sink import (
    ConnectSink,
    Span,
    owned_connect_sink,
    recorded_requests,
    recorded_spans,
    spans_for_trace,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

OPERATION: Final = "gen_ai.operation.name"
SERVER_KIND: Final = 2
PROMOTED: Final = ("litellm.metadata.correlation_id", "litellm.metadata.correlation.id")
ALLOWLIST: Final = (
    "requester_metadata.correlation_id",
    "requester_metadata.correlation.id",
    "requester_metadata.headers.authorization",
    "requester_metadata.headers.x-api-key",
    "requester_metadata.headers.cookie",
    "requester_metadata.headers.proxy-authorization",
    "user_api_key_user_id",
    "user_api_key_alias",
    "requester_ip_address",
)


def _chat(_request: Request) -> Reply:
    return Reply(
        body=(
            b'{"id":"chatcmpl-meta","object":"chat.completion","created":1,"model":"gpt-4o-mini",'
            b'"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}'
        )
    )


def _config(directory: Path, callbacks: tuple[str, ...]) -> Path:
    loaded: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **loaded,
        "litellm_settings": {**loaded.get("litellm_settings", {}), "callbacks": list(callbacks)},
        "general_settings": {**loaded.get("general_settings", {}), "user_api_key_cache_ttl": 2},
    }
    path: Final = directory / "team-request-metadata.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_chat) as wire:
        yield wire


@pytest.fixture(scope="module")
def newrelic(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ConnectSink]:
    with owned_connect_sink(tmp_path_factory.mktemp("newrelic-request-metadata")) as sink:
        yield sink


@pytest.fixture(scope="module", params=[(), ("otel",)], ids=["team-only", "with-proxy-otel"])
def proxy(
    request: pytest.FixtureRequest, newrelic: ConnectSink, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("team-request-metadata")
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {
                "LITELLM_OTEL_V2": "1",
                "OTEL_BSP_SCHEDULE_DELAY": "300",
                "LITELLM_OTEL_BAGGAGE_METADATA_KEYS": ",".join(ALLOWLIST),
                "HTTPS_PROXY": newrelic.proxy_url,
                "NO_PROXY": "127.0.0.1,localhost",
                "REQUESTS_CA_BUNDLE": newrelic.ca_pem,
            },
            config=_config(directory, request.param),
            remove_environment=("OTEL_ENDPOINT", "OTEL_EXPORTER", "OTEL_EXPORTER_OTLP_ENDPOINT"),
            workers=2,
        ) as owned,
    ):
        yield owned.gateway


def _server_and_model_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    return tuple(span for span in spans if span["kind"] == SERVER_KIND or OPERATION in span["attributes"])


def _team_callback_key(proxy: Gateway, scenario: Scenario, destination: Mapping[str, JsonValue]) -> str:
    team: Final = scenario.team()
    attached: Final = proxy.request("POST", f"/team/{team}/callback", dict(destination))
    assert attached.status_code == 200, attached.text
    return scenario.key(team_id=team)


@pytest.mark.parametrize("level", ["team", "key"])
def test_tenant_newrelic_spans_carry_allowlisted_request_metadata(
    level: Literal["team", "key"], proxy: Gateway, provider: Wire, newrelic: ConnectSink
) -> None:
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        destination: Final[Mapping[str, JsonValue]] = MappingProxyType(
            {
                "callback_name": "newrelic",
                "callback_type": "success_and_failure",
                "callback_vars": {"newrelic_api_key": f"nr-{level}-{uuid.uuid4().hex}", "newrelic_region": "us"},
            }
        )
        key: Final = (
            _team_callback_key(proxy, scenario, destination)
            if level == "team"
            else scenario.key(metadata={"logging": [dict(destination)]})
        )
        since: Final = recorded_spans(newrelic.control_url)[0]
        trace_id: Final = uuid.uuid4().hex
        correlation: Final = f"corr-{trace_id}"
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"correlation_id": correlation, "correlation.id": f"{correlation}-dot"},
            },
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        assert response.status_code == 200, response.text

        spans: Final = _server_and_model_spans(
            eventually(
                lambda: spans_for_trace(recorded_spans(newrelic.control_url, since)[1], trace_id),
                lambda found: len(_server_and_model_spans(found)) >= 2,
                seconds=40,
            )
        )

    assert sorted(span["kind"] == SERVER_KIND for span in spans) == [False, True], [span["name"] for span in spans]
    assert [{name: span["attributes"].get(name) for name in PROMOTED} for span in spans] == [
        {PROMOTED[0]: correlation, PROMOTED[1]: f"{correlation}-dot"}
    ] * len(spans), [(span["name"], sorted(span["attributes"])) for span in spans]


def _export_credentials(newrelic: ConnectSink, span_ids: frozenset[str]) -> frozenset[str]:
    return frozenset(
        str(object_value(request.get("headers") or {}).get("api-key"))
        for request in recorded_requests(newrelic.control_url)
        if span_ids & {span_id for span_id in (request.get("span_ids") or ()) if isinstance(span_id, str)}
    )


def test_tenant_newrelic_spans_never_carry_proxy_or_caller_credentials(
    proxy: Gateway, provider: Wire, newrelic: ConnectSink
) -> None:
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        team_newrelic_key: Final = f"nr-team-{uuid.uuid4().hex}"
        key: Final = _team_callback_key(
            proxy,
            scenario,
            {
                "callback_name": "newrelic",
                "callback_type": "success_and_failure",
                "callback_vars": {"newrelic_api_key": team_newrelic_key, "newrelic_region": "us"},
            },
        )
        planted_newrelic_key: Final = f"nr-planted-{uuid.uuid4().hex}"
        credential_headers: Final = MappingProxyType(
            {
                "cookie": f"session=cookie-{uuid.uuid4().hex}",
                "x-api-key": f"xapikey-{uuid.uuid4().hex}",
                "proxy-authorization": f"Basic proxyauth-{uuid.uuid4().hex}",
            }
        )
        since: Final = recorded_spans(newrelic.control_url)[0]
        trace_id: Final = uuid.uuid4().hex
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {
                    "correlation_id": f"corr-{trace_id}",
                    "callback_vars": {"newrelic_api_key": planted_newrelic_key},
                },
            },
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01", **credential_headers},
        )
        assert response.status_code == 200, response.text
        traced: Final = eventually(
            lambda: spans_for_trace(recorded_spans(newrelic.control_url, since)[1], trace_id),
            lambda found: len(_server_and_model_spans(found)) >= 2,
            seconds=40,
        )

    exported: Final = json.dumps([[span["attributes"], span["resource"]] for span in traced])
    secrets: Final = {
        "master key": proxy.key,
        "virtual key": key,
        "team New Relic key": team_newrelic_key,
        **credential_headers,
    }
    assert [name for name, secret in secrets.items() if secret in exported] == [], [
        sorted(span["attributes"]) for span in traced
    ]
    assert _export_credentials(newrelic, frozenset(span["span_id"] for span in traced)) == {team_newrelic_key}
