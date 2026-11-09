"""Built-in loggers that override both hooks deliver one record per async request outside the async flag list:
Datadog one log entry and LangSmith one run. OpenTelemetry keeps its own emit-once guard, so it never exports two
spans for a request either way; its speech, OCR and search spans carry no `litellm.call_id`, so only "never two"
is checked for it.

A legacy sync callback is configured, because that is what makes the sync handler run for an async call."""

import gzip
import json
import uuid
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment, held
from integration._support.hook_counts import IN_THE_ASYNC_FLAG_LIST, SURFACE_CALL_TYPES, provider, send
from integration._support.hook_counts import write_hook_count_config
from integration._support.otlp_sink import decode_spans
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark = pytest.mark.timeout(240)

_PROBES: Final = "integration._support.hook_count_probe"
_FLUSH_SECONDS: Final = 1
_OUTSIDE_THE_ASYNC_FLAG_LIST: Final = sorted(SURFACE_CALL_TYPES.keys() - IN_THE_ASYNC_FLAG_LIST)


@dataclass(frozen=True)
class _Vendors:
    proxy: Gateway
    sink: Wire


@pytest.fixture(scope="module")
def vendors(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Vendors]:
    tmp_path: Final = tmp_path_factory.mktemp("custom-logger-hook-count-vendors")
    with (
        wire_server(provider) as upstream,
        wire_server(lambda request: Reply(status=202)) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = write_hook_count_config(
            tmp_path / "hook-count-vendors.yaml",
            upstream.url,
            {
                "callbacks": ["datadog", "langsmith", "otel"],
                "success_callback": [f"{_PROBES}.sync_success_function"],
                "failure_callback": [f"{_PROBES}.sync_failure_function"],
            },
        )
        overrides: Final = {
            "HOOK_COUNT_PROBE_URL": sink.url + "/probe",
            "STORE_MODEL_IN_DB": "False",
            "DEFAULT_FLUSH_INTERVAL_SECONDS": str(_FLUSH_SECONDS),
            "DD_API_KEY": "synthetic-datadog-key",
            "DD_SITE": "us5.datadoghq.com",
            "DD_BASE_URL": sink.url + "/datadog",
            "LANGSMITH_API_KEY": "synthetic-langsmith-key",
            "LANGSMITH_PROJECT": "hook-counts",
            "LANGSMITH_BASE_URL": sink.url + "/langsmith",
            "OTEL_EXPORTER": "otlp_http",
            "OTEL_ENDPOINT": sink.url + "/otel/v1/traces",
        }
        with owned_proxy(rig, tmp_path, overrides, config=config, workers=2) as proxy:
            yield _Vendors(proxy, sink)


def _unzipped(body: bytes) -> bytes:
    if body[:2] == b"\x1f\x8b":
        return gzip.decompress(body)
    try:
        return zlib.decompress(body, zlib.MAX_WBITS | 16)
    except zlib.error:
        return body


def _entries(body: bytes) -> tuple[object, ...]:
    decoded: Final = json.loads(_unzipped(body))
    return tuple(decoded) if isinstance(decoded, list) else (decoded,)


def _records(request: Request) -> tuple[tuple[str, str], ...]:
    """Every record one delivery carries, as (destination, serialized record). A LangSmith batch carries its new
    runs under `post`, and an OpenTelemetry export carries every span of the request's trace."""
    if request.target.startswith("/datadog"):
        return tuple(("datadog", json.dumps(entry)) for entry in _entries(request.body))
    if request.target.startswith("/langsmith") and request.method == "POST":
        runs: Final = tuple(
            run
            for entry in _entries(request.body)
            for run in (entry.get("post", []) if isinstance(entry, dict) and "post" in entry else [entry])
        )
        return tuple(("langsmith", json.dumps(run)) for run in runs)
    if request.target.startswith("/otel"):
        spans: Final = decode_spans(request.body, request.headers.get("content-type", ""))
        return tuple(("otel", json.dumps(span)) for span in spans if "litellm.call_id" in span["attributes"])
    return ()


@dataclass
class _Deliveries:
    sink: Wire
    seen: tuple[tuple[str, str], ...] = ()

    def per_destination(self, call_id: str) -> dict[str, int]:
        self.seen = (*self.seen, *(record for request in self.sink.drain() for record in _records(request)))
        destinations: Final = ("datadog", "langsmith", "otel")
        return {
            destination: sum(1 for name, record in self.seen if name == destination and call_id in record)
            for destination in destinations
        }


@pytest.mark.parametrize("surface", _OUTSIDE_THE_ASYNC_FLAG_LIST)
def test_built_in_loggers_deliver_one_record_per_async_request(vendors: _Vendors, surface: str) -> None:
    deliveries: Final = _Deliveries(vendors.sink)
    call_id: Final = str(uuid.uuid4())
    response: Final = send(vendors.proxy, surface, "success", call_id)
    assert response.status_code == 200, response.text
    held(
        lambda: deliveries.per_destination(call_id),
        lambda counts: counts["datadog"] == 1 and counts["langsmith"] == 1 and counts["otel"] <= 1,
        holding=4 * _FLUSH_SECONDS,
        seconds=60,
    )
