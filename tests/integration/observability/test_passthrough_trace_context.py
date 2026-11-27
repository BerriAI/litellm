import json
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_TRACEPARENT: Final = "00-e27af848748b606082a7eb70a4fdb1f4-5829f799163236b5-01"
_HEADERS: Final = MappingProxyType(
    {
        "traceparent": _TRACEPARENT,
        "tracestate": "vendor=original,another=kept",
        "baggage": "aiobs_project_name=xxx",
    }
)
_MANAGEMENT_ROUTE: Final = "/config/pass_through_endpoint"


def _echo_trace_context(request: Request) -> Reply:
    return Reply(body=json.dumps({header: request.headers.get(header) for header in _HEADERS}).encode())


@pytest.fixture(scope="module", params=("false", "true"), ids=("legacy", "v2"))
def trace_proxy(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("passthrough-trace-context")
    with gateway_from_environment() as base, wire_server(_echo_trace_context) as upstream:
        shared: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        path: Final = directory / "trace-context.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **shared,
                    "litellm_settings": {**shared["litellm_settings"], "callbacks": ["otel"]},
                    "general_settings": {
                        **shared["general_settings"],
                        "pass_through_endpoints": [
                            {
                                "path": f"/trace-{auth}-{forward}",
                                "target": upstream.url + "/trace",
                                "auth": auth,
                                "forward_headers": forward,
                                "include_subpath": True,
                                "preserve_trace_context": True,
                            }
                            for auth, forward in ((True, True), (True, False), (False, True), (False, False))
                        ]
                        + [
                            {
                                "path": "/default-trace",
                                "target": upstream.url + "/trace",
                                "auth": True,
                                "forward_headers": True,
                            }
                        ],
                    },
                }
            )
        )
        with owned_proxy(
            base, directory, {"LITELLM_OTEL_V2": request.param, "OTEL_EXPORTER": "console"}, config=path
        ) as proxy:
            yield Gateway(proxy.client, proxy.key, upstream.url)


@pytest.mark.parametrize(
    ("auth", "forward", "subpath"),
    (
        (True, True, ""),
        (True, False, "/nested"),
        (False, True, ""),
        (False, False, ""),
    ),
)
def test_configured_relay_preserves_caller_trace_context(
    trace_proxy: Gateway, auth: bool, forward: bool, subpath: str
) -> None:
    path: Final = f"/trace-{auth}-{forward}{subpath}"
    response: Final = (
        trace_proxy.request("POST", path, {}, headers=_HEADERS)
        if auth
        else trace_proxy.client.post(path, json={}, headers=_HEADERS)
    )
    assert response.status_code == 200, response.text
    assert response.json() == {**_HEADERS, "baggage": _HEADERS["baggage"] if forward else None}


def test_configured_relay_does_not_invent_an_absent_trace_context(trace_proxy: Gateway) -> None:
    response: Final = trace_proxy.request("POST", "/trace-True-True", {})
    assert response.status_code == 200, response.text
    assert response.json() == dict.fromkeys(_HEADERS)


def test_default_relay_still_propagates_litellm_span(trace_proxy: Gateway) -> None:
    response: Final = trace_proxy.request("POST", "/default-trace", {}, headers=_HEADERS)
    assert response.status_code == 200, response.text
    outgoing: Final = response.json()
    version, trace_id, parent_id, flags = outgoing["traceparent"].split("-")
    assert (version, trace_id, flags) == ("00", _TRACEPARENT.split("-")[1], "01")
    assert parent_id != _TRACEPARENT.split("-")[2]
    assert outgoing["baggage"] == _HEADERS["baggage"]


def test_management_relay_persists_and_updates_trace_preservation(trace_proxy: Gateway) -> None:
    route: Final = "/api-trace"
    payload: Final = {"path": route, "target": trace_proxy.upstream_url + "/trace", "preserve_trace_context": True}
    created: Final = trace_proxy.request("POST", _MANAGEMENT_ROUTE, payload)
    assert created.status_code == 200, created.text
    endpoint: Final = created.json()["endpoints"][0]
    stored: Final = trace_proxy.request("GET", _MANAGEMENT_ROUTE, params={"endpoint_id": endpoint["id"]})
    assert stored.status_code == 200, stored.text
    assert stored.json()["endpoints"][0].get("preserve_trace_context") is True
    preserved: Final = trace_proxy.request("POST", route, {}, headers=_HEADERS)
    assert preserved.status_code == 200, preserved.text
    assert preserved.json() == {**_HEADERS, "baggage": None}

    updated: Final = trace_proxy.request(
        "POST", f"{_MANAGEMENT_ROUTE}/{endpoint['id']}", {**payload, "preserve_trace_context": False}
    )
    assert updated.status_code == 200, updated.text
    propagated: Final = trace_proxy.request("POST", route, {}, headers=_HEADERS)
    assert propagated.status_code == 200, propagated.text
    assert propagated.json()["traceparent"] not in (None, _TRACEPARENT)
