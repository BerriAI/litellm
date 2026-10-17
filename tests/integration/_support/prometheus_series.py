"""Rig and readers for the Prometheus series-cap cells: a capped proxy, keys that fill the cap, and the
scrape, the multiprocess sample files, and the spend log read back per request."""

from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.responses_vendor import ResponsesVendor, same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from prometheus_client.mmap_dict import MmapedDict
from prometheus_client.parser import text_string_to_metric_families
from pydantic import JsonValue

REQUESTS: Final = "litellm_requests_metric_total"
PROXY_REQUESTS: Final = "litellm_proxy_total_requests_metric_total"
PROXY_FAILURES: Final = "litellm_proxy_failed_requests_metric_total"
CACHE_HITS: Final = "litellm_cache_hits_metric_total"
REMAINING_REQUESTS: Final = "litellm_remaining_api_key_requests_for_model"
SUCCESSFUL_FALLBACKS: Final = "litellm_deployment_successful_fallbacks_total"
FAILED_FALLBACKS: Final = "litellm_deployment_failed_fallbacks_total"
OVERFLOW: Final = "other"
USER_AGENT: Final = "litellm-series-cap-audit/1"
AGENT_HEADERS: Final[Mapping[str, str]] = MappingProxyType({"User-Agent": USER_AGENT})
PROVIDER_OUTAGE: Final = "synthetic provider outage"
_SERIES_ATTRIBUTES: Final = frozenset({"le", "pid"})


@dataclass(frozen=True, slots=True)
class Call:
    """One request the cells can follow end to end: the call id the proxy keeps as the spend log's request id
    and the marker the scripted provider echoes in its answer."""

    call_id: str
    marker: str

    @classmethod
    def new(cls) -> Call:
        identity: Final = uuid.uuid4()
        return cls(str(identity), identity.hex)

    @property
    def text(self) -> str:
        return f"say marker-{self.marker}"

    @property
    def answer(self) -> str:
        return f"answer marker-{self.marker}"

    @property
    def message(self) -> dict[str, str]:
        return {"role": "user", "content": self.text}

    @property
    def headers(self) -> dict[str, str]:
        return {"x-litellm-call-id": self.call_id}


@dataclass(frozen=True, slots=True)
class Key:
    token: str
    alias: str


@dataclass(frozen=True, slots=True)
class Provider:
    vendor: ResponsesVendor
    outage: threading.Event
    failing_models: frozenset[str]

    def respond(self, request: Request) -> Reply:
        if self.outage.is_set() or self._failing(request):
            return Reply(status=500, body=json.dumps({"error": {"message": PROVIDER_OUTAGE}}).encode())
        return self.vendor.respond(request)

    def _failing(self, request: Request) -> bool:
        if request.method != "POST" or not self.failing_models:
            return False
        body: Final = json.loads(request.body)
        return isinstance(body, dict) and body.get("model") in self.failing_models


@dataclass(frozen=True, slots=True)
class Sample:
    family: str
    kind: str
    name: str
    labels: Mapping[str, str]
    value: float

    def identity(self) -> tuple[tuple[str, str], ...]:
        """The label set that makes this a series of its own: the histogram bucket and the multiprocess pid are
        attributes of one series, not separate ones."""
        return tuple(sorted((name, value) for name, value in self.labels.items() if name not in _SERIES_ATTRIBUTES))

    def is_overflow(self) -> bool:
        identity: Final = self.identity()
        return bool(identity) and all(value == OVERFLOW for _, value in identity)


@dataclass(frozen=True, slots=True)
class CapRig:
    proxy: OwnedProxy
    scenario: Scenario
    model: str
    provider: Wire
    outage: threading.Event
    warm: tuple[Key, ...]
    prom_dir: Path

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway

    @property
    def base_url(self) -> str:
        return str(self.gateway.client.base_url).rstrip("/")

    @property
    def openai_base(self) -> str:
        return self.base_url + "/v1"

    @property
    def warm_aliases(self) -> frozenset[str]:
        return frozenset(key.alias for key in self.warm)

    def key(self, cell: str) -> Key:
        alias: Final = f"{cell}-{uuid.uuid4().hex[:12]}"
        return Key(self.scenario.key(key_alias=alias), alias)

    def chat(self, key: Key, call: Call) -> httpx.Response:
        return chat_once(self.base_url, key, self.model, call)


def chat_once(base_url: str, key: Key, model: str, call: Call) -> httpx.Response:
    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as client:
        return client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [call.message]},
            headers={**AGENT_HEADERS, **call.headers, "Authorization": f"Bearer {key.token}"},
        )


def series_cap_config(
    directory: Path,
    settings: Mapping[str, JsonValue],
    *,
    model_list: Sequence[Mapping[str, JsonValue]] = (),
    router_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    assert isinstance(config, dict)
    config["litellm_settings"] = {**config["litellm_settings"], "callbacks": ["prometheus"], **settings}
    config["router_settings"] = {**config["router_settings"], "num_retries": 0, **(router_settings or {})}
    if model_list:
        config["model_list"] = [dict(entry) for entry in model_list]
    path: Final = directory / "series-cap.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def series_cap_rig(
    directory: Path,
    settings: Mapping[str, JsonValue],
    *,
    workers: int,
    warm_keys: int,
    multiproc_dir: Path | None = None,
    failing_models: frozenset[str] = frozenset(),
    deployments: Callable[[str], Sequence[Mapping[str, JsonValue]]] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[CapRig]:
    outage: Final = threading.Event()
    double: Final = Provider(ResponsesVendor(), outage, failing_models)
    prom_dir: Final = multiproc_dir if multiproc_dir is not None else directory / "prom"
    prom_dir.mkdir(exist_ok=True)
    shared_samples: Final = multiproc_dir is not None or workers > 1
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        provider: Final = stack.enter_context(wire_server(double.respond))
        config: Final = series_cap_config(
            directory,
            settings,
            model_list=deployments(provider.url) if deployments is not None else (),
            router_settings=router_settings,
        )
        owned: Final = stack.enter_context(
            owned_proxy_process(
                gateway,
                directory,
                {"PROMETHEUS_MULTIPROC_DIR": str(prom_dir)} if shared_samples else {},
                config=config,
                remove_environment=() if shared_samples else ("PROMETHEUS_MULTIPROC_DIR",),
                workers=workers,
            )
        )
        scenario: Final = stack.enter_context(owned.gateway.scenario())
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
        rig: Final = CapRig(owned, scenario, model, provider, outage, (), prom_dir)
        warm: Final = tuple(rig.key("warm") for _ in range(warm_keys))
        for key in warm:
            _warm_up(rig, key)
        eventually(
            lambda: alias_values(scrape(owned.gateway), REQUESTS),
            lambda seen: all(key.alias in seen for key in warm),
            seconds=60,
        )
        yield CapRig(owned, scenario, model, provider, outage, warm, prom_dir)


def _warm_up(rig: CapRig, key: Key) -> None:
    response: Final = rig.chat(key, Call.new())
    assert response.status_code == 200, response.text


def _samples(text: str) -> Iterator[Sample]:
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            yield Sample(
                family.name, family.type, sample.name, MappingProxyType(dict(sample.labels)), float(sample.value)
            )


def scrape(gateway: Gateway) -> tuple[Sample, ...]:
    response: Final = gateway.client.request(
        "GET", "/metrics", headers={"Authorization": f"Bearer {gateway.key}"}, follow_redirects=True
    )
    assert response.status_code == 200, f"GET /metrics: {response.status_code} {response.text[:300]}"
    return tuple(_samples(response.text))


def alias_values(samples: Sequence[Sample], name: str) -> frozenset[str]:
    """The key aliases holding a series of their own on the metric; the shared overflow series is not one."""
    return frozenset(
        sample.labels["api_key_alias"]
        for sample in samples
        if sample.name == name and sample.labels.get("api_key_alias") not in (None, OVERFLOW)
    )


def alias_total(samples: Sequence[Sample], name: str, alias: str) -> float:
    return sum(
        sample.value for sample in samples if sample.name == name and sample.labels.get("api_key_alias") == alias
    )


def overflow_total(samples: Sequence[Sample], name: str) -> float:
    return sum(sample.value for sample in samples if sample.name == name and sample.is_overflow())


def label_values(samples: Sequence[Sample]) -> frozenset[str]:
    return frozenset(chain.from_iterable(sample.labels.values() for sample in samples))


def gauge_samples(samples: Sequence[Sample]) -> tuple[Sample, ...]:
    return tuple(sample for sample in samples if sample.kind == "gauge")


def series_per_family(samples: Sequence[Sample]) -> Mapping[str, int]:
    """How many series of their own each metric family holds, the shared `other` series left out."""
    owned: Final = frozenset(
        (sample.family, sample.identity()) for sample in samples if sample.identity() and not sample.is_overflow()
    )
    return MappingProxyType(dict(Counter(family for family, _ in owned)))


def families_over(samples: Sequence[Sample], cap: int) -> tuple[tuple[str, int], ...]:
    """Every metric family holding more series of its own than the cap allows."""
    return tuple(sorted((family, count) for family, count in series_per_family(samples).items() if count > cap))


@dataclass(frozen=True, slots=True)
class SpendRow:
    request_id: str
    status: str


def spend_rows(alias: str) -> tuple[SpendRow, ...]:
    """Every spend log row the key wrote: a success row carries the response id the caller received, a failure
    row the call id the caller sent."""
    rows: Final = read_rows(
        "SELECT request_id, status FROM \"LiteLLM_SpendLogs\" WHERE metadata->>'user_api_key_alias' = %s",
        (alias,),
    )
    return tuple(SpendRow(str(row["request_id"]), str(row["status"])) for row in rows)


def expect_spend_rows(
    alias: str, response_ids: Sequence[str], call_ids: Sequence[str] = (), earlier: Sequence[SpendRow] = ()
) -> None:
    """One new row per request on top of the rows the key already had: successes found by the response id the
    caller got, failures by their call id."""
    expected: Final = len(earlier) + len(response_ids) + len(call_ids)
    rows: Final = eventually(lambda: spend_rows(alias), lambda found: len(found) >= expected, seconds=70)
    fresh: Final = tuple(row for row in rows if row not in earlier)
    assert len(rows) == expected and len(fresh) == len(response_ids) + len(call_ids), (rows, earlier)
    for response_id in response_ids:
        assert any(row.status == "success" and same_response(row.request_id, response_id) for row in fresh), (
            response_id,
            fresh,
        )
    for call_id in call_ids:
        assert any(row.status == "failure" and row.request_id == call_id for row in fresh), (call_id, fresh)


def sse_data(text: str) -> tuple[dict[str, JsonValue], ...]:
    """The JSON payload of every `data:` frame in a server-sent event stream, the `[DONE]` sentinel left out."""
    payloads: Final = tuple(
        line.removeprefix("data:").strip() for line in text.splitlines() if line.startswith("data:")
    )
    return tuple(object_value(json.loads(payload)) for payload in payloads if payload and payload != "[DONE]")


def received_markers(provider: Wire) -> tuple[str, ...]:
    return tuple(chain.from_iterable(_markers_in(request.body) for request in provider.drain()))


def _markers_in(body: bytes) -> tuple[str, ...]:
    return tuple(part[:32].decode() for part in body.split(b"marker-")[1:])


@dataclass(frozen=True, slots=True)
class WorkerSamples:
    pid: int
    aliases: frozenset[str]
    overflow: float


def worker_samples(prom_dir: Path, name: str) -> tuple[WorkerSamples, ...]:
    return tuple(_worker_samples(path, name) for path in sorted(prom_dir.glob("counter_*.db")))


def _worker_samples(path: Path, name: str) -> WorkerSamples:
    pid: Final = int(path.stem.rsplit("_", 1)[1])
    rows: Final = tuple(_counter_rows(path, name))
    return WorkerSamples(
        pid,
        frozenset(labels["api_key_alias"] for labels, _ in rows if labels.get("api_key_alias") not in (None, OVERFLOW)),
        sum(value for labels, value in rows if all(label == OVERFLOW for label in labels.values())),
    )


def _counter_rows(path: Path, name: str) -> Iterator[tuple[Mapping[str, str], float]]:
    for key, value, *_ in MmapedDict.read_all_values_from_file(str(path)):
        _, sample_name, labels, _ = json.loads(key)
        if sample_name == name:
            yield labels, float(value)
