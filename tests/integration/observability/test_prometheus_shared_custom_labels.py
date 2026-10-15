from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
import yaml
from integration._support import claude_code as cc
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.prometheus_series import Sample, scrape
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_GPT: Final = "prometheus-shared-label-gpt"
_CLAUDE: Final = "prometheus-shared-label-claude"
_DOWN: Final = "prometheus-shared-label-down"
_DOWN_UPSTREAM: Final = "gpt-4o-down"
_SHARED_LABELS: Final = ("project_alias", "team", "requested_model", "metadata.department")
_RPM: Final = 600
_TPM: Final = 1_000_000
_ALLOWED: Final = "litellm_project_model_rate_limit_allowed_metric"
_USED: Final = "litellm_project_model_rate_limit_used_metric"
_TOTAL_REQUESTS: Final = "litellm_proxy_total_requests_metric_total"
_FAILED_REQUESTS: Final = "litellm_proxy_failed_requests_metric_total"
_LATENCY_COUNT: Final = "litellm_request_total_latency_metric_count"
_CALLBACK_FAILURES: Final = "litellm_callback_logging_failures_metric_total"


@dataclass(frozen=True, slots=True)
class _Upstream:
    vendor: rv.ResponsesVendor
    outage: threading.Event

    def respond(self, request: Request) -> Reply:
        if request.method == "POST" and (
            self.outage.is_set() or json.loads(request.body).get("model") == _DOWN_UPSTREAM
        ):
            return rv.error(503, "synthetic upstream outage", "server_error")
        return self.vendor.respond(request)


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    upstream: Wire
    outage: threading.Event


@dataclass(frozen=True, slots=True)
class _Call:
    route: str
    model: str
    stream: bool

    @property
    def upstream(self) -> tuple[str, str, bool]:
        if self.model == _CLAUDE:
            return ("/v1/messages", cc.OPUS, self.stream)
        return (
            "/v1/chat/completions" if self.route == "/v1/chat/completions" else "/v1/responses",
            "gpt-4o-mini",
            self.stream,
        )


@dataclass(frozen=True, slots=True)
class _ProjectKey:
    key: str
    team: str
    project: str
    alias: str
    model: str

    @property
    def hashed(self) -> str:
        return sha256(self.key.encode()).hexdigest()


_CALLS: Final = (
    _Call("/v1/chat/completions", _GPT, stream=False),
    _Call("/v1/chat/completions", _GPT, stream=True),
    _Call("/v1/messages", _GPT, stream=False),
    _Call("/v1/messages", _GPT, stream=True),
    _Call("/v1/messages", _CLAUDE, stream=False),
    _Call("/v1/messages", _CLAUDE, stream=True),
    _Call("/v1/responses", _GPT, stream=False),
    _Call("/v1/responses", _GPT, stream=True),
)


def _config(directory: Path, upstream: Wire, custom_labels: Sequence[str], **settings: JsonValue) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        {
            "model_name": _GPT,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-fixture",
                "api_base": f"{upstream.url}/v1",
            },
        },
        {
            "model_name": _CLAUDE,
            "litellm_params": {
                "model": f"anthropic/{cc.OPUS}",
                "api_key": cc.ANTHROPIC_API_KEY,
                "api_base": upstream.url,
            },
        },
        {
            "model_name": _DOWN,
            "litellm_params": {
                "model": f"openai/{_DOWN_UPSTREAM}",
                "api_key": "sk-fixture",
                "api_base": f"{upstream.url}/v1",
            },
        },
    ]
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration["litellm_settings"]["custom_prometheus_metadata_labels"] = list(custom_labels)
    configuration["litellm_settings"].update(settings)
    configuration.setdefault("router_settings", {}).update({"num_retries": 0, "disable_cooldowns": True})
    path: Final = directory / "prometheus-shared-custom-labels.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@contextmanager
def _owned_rig(directory: Path, custom_labels: Sequence[str], **settings: JsonValue) -> Iterator[_Rig]:
    samples: Final = directory / "prometheus-multiproc"
    samples.mkdir()
    outage: Final = threading.Event()
    with (
        wire_server(_Upstream(rv.ResponsesVendor(), outage).respond) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"PROMETHEUS_MULTIPROC_DIR": str(samples)},
            config=_config(directory, upstream, custom_labels, **settings),
            workers=2,
        ) as owned,
    ):
        yield _Rig(owned, upstream, outage)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    with _owned_rig(tmp_path_factory.mktemp("prometheus-shared-custom-labels"), _SHARED_LABELS) as owned:
        yield owned


def _project_key(scenario: Scenario, model: str) -> _ProjectKey:
    team: Final = scenario.team()
    alias: Final = f"integration-project-{uuid.uuid4().hex}"
    project: Final = scenario.project(
        team, project_alias=alias, model_rpm_limit={model: _RPM}, model_tpm_limit={model: _TPM}
    )
    return _ProjectKey(scenario.key(team_id=team, project_id=project), team, project, alias, model)


def _body(call: _Call, **extra: JsonValue) -> dict[str, JsonValue]:
    text: Final = f"say marker-{uuid.uuid4().hex}"
    if call.route == "/v1/responses":
        return {"model": call.model, "input": text, "stream": call.stream, **extra}
    return {
        "model": call.model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": text}],
        "stream": call.stream,
        **extra,
    }


def _send(gateway: Gateway, call: _Call, key: str, **extra: JsonValue) -> int:
    with gateway.client.stream(
        "POST", call.route, json=_body(call, **extra), headers={"Authorization": f"Bearer {key}"}
    ) as response:
        body: Final = response.read()
    if response.status_code == 200 and call.stream:
        assert b"data:" in body, body[:300]
    return response.status_code


def _forwarded(request: Request) -> tuple[str, str, bool]:
    body: Final = json.loads(request.body)
    return (urlsplit(request.target).path, body["model"], bool(body.get("stream")))


def _series(samples: Sequence[Sample], name: str, **labels: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in samples
        if sample.name == name and all(sample.labels.get(label) == value for label, value in labels.items())
    )


def _total(samples: Sequence[Sample], name: str, **labels: str) -> float:
    return sum(sample.value for sample in _series(samples, name, **labels))


def _project_gauges(samples: Sequence[Sample], project: str) -> tuple[Sample, ...]:
    return _series(samples, _ALLOWED, project_id=project) + _series(samples, _USED, project_id=project)


def _shared_labels(sample: Sample) -> tuple[str | None, ...]:
    return tuple(
        sample.labels.get(label) for label in ("project_alias", "team", "requested_model", "metadata_department")
    )


@pytest.mark.parametrize(
    "call", _CALLS, ids=lambda call: f"{call.route}-{call.model}-{'stream' if call.stream else 'json'}"
)
def test_a_project_key_call_on_every_endpoint_lands_in_the_request_latency_and_project_metrics(
    rig: _Rig, call: _Call
) -> None:
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        owner: Final = _project_key(scenario, call.model)
        assert _send(rig.gateway, call, owner.key) == 200
        forwarded: Final = rig.upstream.drain()
        assert [_forwarded(request) for request in forwarded] == [call.upstream], forwarded
        samples: Final = eventually(
            lambda: scrape(rig.gateway),
            lambda found: (
                _total(found, _LATENCY_COUNT, hashed_api_key=owner.hashed) == 1
                and _total(found, _TOTAL_REQUESTS, hashed_api_key=owner.hashed) == 1
                and len(_project_gauges(found, owner.project)) == 4
            ),
            seconds=30,
        )
        requests: Final = _series(samples, _TOTAL_REQUESTS, hashed_api_key=owner.hashed)
        assert [(sample.labels["route"], sample.labels["status_code"], sample.value) for sample in requests] == [
            (call.route, "200", 1.0)
        ], requests
        assert [_shared_labels(sample) for sample in requests] == [(owner.alias, owner.team, call.model, "None")]
        gauges: Final = _project_gauges(samples, owner.project)
        assert {(sample.name, sample.labels["rate_limit_type"], int(sample.value)) for sample in gauges} >= {
            (_ALLOWED, "requests", _RPM),
            (_USED, "requests", 1),
            (_ALLOWED, "tokens", _TPM),
        }, gauges
        assert {_shared_labels(sample) for sample in gauges} == {(owner.alias, owner.team, call.model, "None")}, gauges


def test_a_failed_project_key_call_counts_as_one_failed_request_with_each_shared_label_once(rig: _Rig) -> None:
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        owner: Final = _project_key(scenario, _DOWN)
        call: Final = _Call("/v1/chat/completions", _DOWN, stream=False)
        status: Final = _send(rig.gateway, call, owner.key)
        assert status >= 500, status
        forwarded: Final = rig.upstream.drain()
        assert len(forwarded) == 1, forwarded
        samples: Final = eventually(
            lambda: scrape(rig.gateway),
            lambda found: (
                _total(found, _FAILED_REQUESTS, hashed_api_key=owner.hashed) == 1
                and _total(found, _TOTAL_REQUESTS, hashed_api_key=owner.hashed) == 1
            ),
            seconds=30,
        )
        failed: Final = _series(samples, _FAILED_REQUESTS, hashed_api_key=owner.hashed)
        assert [_shared_labels(sample) for sample in failed] == [("None", owner.team, _DOWN, "None")], failed
        requests: Final = _series(samples, _TOTAL_REQUESTS, hashed_api_key=owner.hashed)
        assert [(sample.labels["status_code"], sample.value) for sample in requests] == [(str(status), 1.0)], requests


def test_request_metadata_fills_the_shared_labels_on_the_request_and_project_metrics(rig: _Rig) -> None:
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        owner: Final = _project_key(scenario, _GPT)
        call: Final = _Call("/v1/chat/completions", _GPT, stream=False)
        metadata: Final[dict[str, JsonValue]] = {
            "project_alias": "caller-alias",
            "team": "caller-team",
            "department": "eng",
        }
        assert _send(rig.gateway, call, owner.key, metadata=metadata) == 200
        forwarded: Final = rig.upstream.drain()
        assert len(forwarded) == 1, forwarded
        samples: Final = eventually(
            lambda: scrape(rig.gateway),
            lambda found: (
                _total(found, _TOTAL_REQUESTS, hashed_api_key=owner.hashed) == 1
                and len(_project_gauges(found, owner.project)) == 4
            ),
            seconds=30,
        )
        expected: Final = ("caller-alias", "caller-team", _GPT, "eng")
        requests: Final = _series(samples, _TOTAL_REQUESTS, hashed_api_key=owner.hashed)
        assert [_shared_labels(sample) for sample in requests] == [expected], requests
        gauges: Final = _project_gauges(samples, owner.project)
        assert {_shared_labels(sample) for sample in gauges} == {expected}, gauges


def test_a_key_outside_any_team_or_project_still_counts_its_request(rig: _Rig) -> None:
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        hashed: Final = sha256(key.encode()).hexdigest()
        assert _send(rig.gateway, _Call("/v1/chat/completions", _GPT, stream=False), key) == 200
        forwarded: Final = rig.upstream.drain()
        assert len(forwarded) == 1, forwarded
        samples: Final = eventually(
            lambda: scrape(rig.gateway),
            lambda found: (
                _total(found, _LATENCY_COUNT, hashed_api_key=hashed) == 1
                and _total(found, _TOTAL_REQUESTS, hashed_api_key=hashed) == 1
            ),
            seconds=30,
        )
        requests: Final = _series(samples, _TOTAL_REQUESTS, hashed_api_key=hashed)
        assert [(_shared_labels(sample), sample.value) for sample in requests] == [
            (("None", "None", _GPT, "None"), 1.0)
        ], requests


def test_a_mixed_burst_through_an_upstream_outage_counts_every_request_once_and_never_fails_the_callback(
    rig: _Rig,
) -> None:
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        owner: Final = _project_key(scenario, _GPT)
        before: Final = _total(scrape(rig.gateway), _CALLBACK_FAILURES, callback_name="PrometheusLogger")
        calls: Final = tuple(call for call in _CALLS if call.model == _GPT) * 2

        def burst() -> list[tuple[str, int]]:
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                return list(pool.map(lambda call: (call.route, _send(rig.gateway, call, owner.key)), calls))

        healthy: Final = burst()
        rig.outage.set()
        try:
            down: Final = burst()
        finally:
            rig.outage.clear()
        recovered: Final = burst()
        assert {status for _, status in healthy + recovered} == {200}, (healthy, recovered)
        assert all(status >= 500 for _, status in down), down
        rig.upstream.drain()
        observed: Final = Counter((route, str(status)) for route, status in healthy + down + recovered)
        samples: Final = eventually(
            lambda: scrape(rig.gateway),
            lambda found: (
                sum(sample.value for sample in _series(found, _TOTAL_REQUESTS, hashed_api_key=owner.hashed))
                == len(calls) * 3
            ),
            seconds=60,
        )
        counted: Final = Counter(
            {
                (sample.labels["route"], sample.labels["status_code"]): int(sample.value)
                for sample in _series(samples, _TOTAL_REQUESTS, hashed_api_key=owner.hashed)
            }
        )
        assert counted == observed, (counted, observed)
        assert _total(samples, _LATENCY_COUNT, hashed_api_key=owner.hashed) == len(healthy) + len(recovered)
        assert _total(samples, _FAILED_REQUESTS, hashed_api_key=owner.hashed) == len(down)
        assert _total(samples, _CALLBACK_FAILURES, callback_name="PrometheusLogger") == before


@pytest.mark.parametrize(
    ("custom_labels", "settings"),
    [
        pytest.param(("project-alias",), {}, id="a-spelling-that-sanitizes-to-the-built-in-name"),
        pytest.param(("project_alias",), {"prometheus_metrics_max_series_per_metric": 500}, id="series-cap-on"),
    ],
)
def test_a_project_key_call_under_other_label_configurations_keeps_one_project_alias(
    tmp_path: Path, custom_labels: tuple[str, ...], settings: dict[str, JsonValue]
) -> None:
    with _owned_rig(tmp_path, custom_labels, **settings) as owned, owned.gateway.scenario() as scenario:
        owner: Final = _project_key(scenario, _GPT)
        assert _send(owned.gateway, _Call("/v1/chat/completions", _GPT, stream=False), owner.key) == 200
        forwarded: Final = owned.upstream.drain()
        assert len(forwarded) == 1, forwarded
        samples: Final = eventually(
            lambda: scrape(owned.gateway),
            lambda found: (
                _total(found, _LATENCY_COUNT, hashed_api_key=owner.hashed) == 1
                and _total(found, _TOTAL_REQUESTS, hashed_api_key=owner.hashed) == 1
                and len(_project_gauges(found, owner.project)) == 4
            ),
            seconds=30,
        )
        requests: Final = _series(samples, _TOTAL_REQUESTS, hashed_api_key=owner.hashed)
        assert [(sample.labels["project_alias"], sample.value) for sample in requests] == [(owner.alias, 1.0)]
        gauges: Final = _project_gauges(samples, owner.project)
        assert {sample.labels["project_alias"] for sample in gauges} == {owner.alias}, gauges
