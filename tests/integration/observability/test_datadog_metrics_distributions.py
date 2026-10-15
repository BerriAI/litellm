"""Datadog custom metrics: every request's latency reaches Datadog as a distribution sample.

Datadog keeps one gauge value per second per tag set, so concurrent requests with the same tags collapse
into one point and `p95:litellm.request.total_latency` is not a per-request percentile. The
``datadog_metrics`` callback therefore also sends each latency sample to ``/api/v1/distribution_points``.
The owned proxy points ``DD_SITE`` at a TLS double for ``api.localhost`` that inflates every upload, answers
like the real intake (202, or 503 while an outage is scripted) and records what each flush carried, so the
cases count samples per request instead of reading Datadog's aggregates.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import threading
import time
import uuid
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from _s3_v2_support import SURFACES, call_surface, surface_reply
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

SERIES: Final = "/api/v2/series"
DISTRIBUTIONS: Final = "/api/v1/distribution_points"
LATENCIES: Final = ("litellm.request.total_latency", "litellm.llm_api.latency", "litellm.overhead.latency")
REQUEST_COUNT_METRIC: Final = "litellm.llm_api.request_count"
SITE_HOSTS: Final = ("api.localhost",)
DD_API_KEY: Final = "synthetic-datadog-api-key"
WORKERS: Final = 2
FLUSH_WAIT_SECONDS: Final = 60
SERIES_LIST: Final = TypeAdapter(list[dict[str, JsonValue]])


@dataclass(frozen=True, slots=True)
class Upload:
    target: str
    encoding: str | None
    api_key: str | None
    status: int
    series: tuple[dict[str, JsonValue], ...]


class DatadogIntake:
    """Answers `/api/v2/series` and `/api/v1/distribution_points` like Datadog and keeps every inflated body.

    `failing` scripts an outage as the number of uploads to answer 503 per path; `delay` holds every answer
    so a flush stays in flight while the proxy keeps serving requests."""

    def __init__(self, failing: Mapping[str, int] | None = None, delay: float = 0) -> None:
        self._failing: Final = dict(failing or {})
        self._delay: Final = delay
        self._lock: Final = threading.Lock()
        self._uploads: Final[list[Upload]] = []

    def respond(self, request: Request) -> Reply:
        assert request.method == "POST" and request.target in (SERIES, DISTRIBUTIONS), request.target
        encoding: Final = request.headers.get("content-encoding")
        body: Final = (
            gzip.decompress(request.body)
            if encoding == "gzip"
            else zlib.decompress(request.body)
            if encoding == "deflate"
            else request.body
        )
        payload: Final = json.loads(body)
        with self._lock:
            remaining: Final = self._failing.get(request.target, 0)
            if remaining:
                self._failing[request.target] = remaining - 1
            status: Final = 503 if remaining else 202
            self._uploads.append(
                Upload(
                    request.target,
                    encoding,
                    request.headers.get("dd-api-key"),
                    status,
                    tuple(SERIES_LIST.validate_python(payload["series"])),
                )
            )
        if self._delay:
            time.sleep(self._delay)
        if status == 503:
            return Reply(status=503, body=b'{"errors": ["Service Unavailable"]}')
        return Reply(status=202, body=b'{"status": "ok"}' if request.target == DISTRIBUTIONS else b'{"errors": []}')

    def uploads(self, target: str | None = None) -> tuple[Upload, ...]:
        with self._lock:
            return tuple(upload for upload in self._uploads if target is None or upload.target == target)

    def accepted(self, target: str) -> tuple[Upload, ...]:
        return tuple(upload for upload in self.uploads(target) if upload.status == 202)

    def rejected(self, target: str) -> tuple[Upload, ...]:
        return tuple(upload for upload in self.uploads(target) if upload.status != 202)


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    intake: DatadogIntake
    provider_url: str


def _tags(series: Mapping[str, JsonValue]) -> tuple[str, ...]:
    tags: Final = series["tags"]
    assert isinstance(tags, list), series
    return tuple(str(tag) for tag in tags)


def _for_model(series: Mapping[str, JsonValue], metric: str, model: str) -> bool:
    return series["metric"] == metric and f"model_group:{model}" in _tags(series)


def distribution_samples(
    intake: DatadogIntake, metric: str, model: str, status_code: str | None = None
) -> tuple[float, ...]:
    """Every sample value Datadog accepted for `<metric>.distribution` from requests to `model`."""
    samples: Final[list[float]] = []
    for upload in intake.accepted(DISTRIBUTIONS):
        for series in upload.series:
            if not _for_model(series, f"{metric}.distribution", model):
                continue
            if status_code is not None and f"status_code:{status_code}" not in _tags(series):
                continue
            assert series["type"] == "distribution", series
            points: Final = series["points"]
            assert isinstance(points, list), series
            for point in points:
                assert isinstance(point, list) and len(point) == 2 and isinstance(point[1], list), series
                samples.extend(float(str(value)) for value in point[1])
    return tuple(samples)


def series_points(intake: DatadogIntake, metric: str, model: str, status_code: str | None = None) -> tuple[float, ...]:
    """Every point value Datadog accepted on `/api/v2/series` for `metric` from requests to `model`."""
    values: Final[list[float]] = []
    for upload in intake.accepted(SERIES):
        for series in upload.series:
            if not _for_model(series, metric, model):
                continue
            if status_code is not None and f"status_code:{status_code}" not in _tags(series):
                continue
            points: Final = series["points"]
            assert isinstance(points, list), series
            for point in points:
                assert isinstance(point, dict), series
                values.append(float(str(point["value"])))
    return tuple(values)


def every_series(intake: DatadogIntake) -> tuple[dict[str, JsonValue], ...]:
    return tuple(series for upload in intake.uploads() for series in upload.series)


def _config(path: Path, callback_setting: str | None) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    if callback_setting is not None:
        config["litellm_settings"].update({callback_setting: ["datadog_metrics"]})
    target: Final = path / "datadog_metrics.yaml"
    target.write_text(yaml.safe_dump(config))
    return target


def upstream_failing_marked_requests(request: Request) -> Reply:
    body: Final = json.loads(request.body) if request.body else {}
    messages: Final = body.get("messages") if isinstance(body, dict) else None
    content: Final = messages[0].get("content") if isinstance(messages, list) and messages else None
    if isinstance(content, str) and content.startswith("fail-"):
        return Reply(status=500, body=json.dumps({"error": {"message": f"upstream rejected {content}"}}).encode())
    return surface_reply(request)


RigFactory = Callable[..., AbstractContextManager[Rig]]


@pytest.fixture
def rig_factory(gateway: Gateway, tmp_path: Path) -> RigFactory:
    """Boots one owned proxy wired to a fresh Datadog intake double per case."""

    @contextmanager
    def build(
        intake: DatadogIntake,
        *,
        hostname: str | None = None,
        callback_setting: str | None = "callbacks",
        upstream: Callable[[Request], Reply] = surface_reply,
    ) -> Iterator[Rig]:
        cert, key = write_self_signed_cert(tmp_path, SITE_HOSTS)
        with (
            wire_server(upstream) as provider,
            wire_server(intake.respond, tls=server_context(cert, key), keep_alive=True) as site,
        ):
            environment: Final = {
                "DD_API_KEY": DD_API_KEY,
                "DD_SITE": f"localhost:{urlsplit(site.url).port}",
                "DD_ENV": "integration",
                "DD_SERVICE": "litellm-integration",
                "SSL_CERT_FILE": str(cert),
                **({"HOSTNAME": hostname} if hostname is not None else {}),
            }
            removed: Final = ("DD_APP_KEY", "POD_NAME", *(("HOSTNAME",) if hostname is None else ()))
            with owned_proxy(
                gateway,
                tmp_path,
                environment,
                config=_config(tmp_path, callback_setting),
                remove_environment=removed,
                workers=WORKERS,
            ) as candidate:
                yield Rig(candidate, intake, provider.url)
            provider.drain()

    return build


def _chat(base_url: str, key: str, model: str, markers: tuple[str, ...]) -> tuple[int, ...]:
    async def send_all() -> tuple[int, ...]:
        async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
            responses: Final = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json={"model": model, "messages": [{"role": "user", "content": marker}]},
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    for marker in markers
                )
            )
        return tuple(response.status_code for response in responses)

    return asyncio.run(send_all())


def _wait_for_samples(intake: DatadogIntake, model: str, count: int, metrics: tuple[str, ...] = LATENCIES) -> None:
    eventually(
        lambda: tuple(len(distribution_samples(intake, metric, model)) for metric in metrics),
        lambda counts: all(total >= count for total in counts),
        seconds=FLUSH_WAIT_SECONDS,
        return_last_on_timeout=True,
    )


def test_concurrent_chat_requests_each_reach_datadog_as_a_distribution_sample(rig_factory: RigFactory) -> None:
    """The reporter's flow: a burst of concurrent `/v1/chat/completions` to one model, all 200. Every request
    must add one sample to each latency distribution, while the legacy gauges and the request count keep
    arriving on `/api/v2/series` unchanged, and an unset HOSTNAME adds no empty `HOSTNAME:` tag."""
    burst: Final = 8
    intake: Final = DatadogIntake()
    with rig_factory(intake) as rig:
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model(api_base=rig.provider_url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            marker: Final = f"dd-{uuid.uuid4().hex[:8]}"
            statuses: Final = _chat(
                str(rig.proxy.client.base_url), key, model, tuple(f"{marker}-{index}" for index in range(burst))
            )
            assert statuses == (200,) * burst, statuses
            _wait_for_samples(intake, model, burst)

            for metric in LATENCIES:
                samples = distribution_samples(intake, metric, model)
                assert len(samples) == burst, (
                    f"{metric}.distribution got {len(samples)} samples for {burst} requests: "
                    f"{[(upload.target, upload.status) for upload in intake.uploads()]}"
                )
                assert all(value >= 0 for value in samples), samples
            for metric in LATENCIES:
                assert len(series_points(intake, metric, model)) == burst, metric
            assert sum(series_points(intake, REQUEST_COUNT_METRIC, model, "200")) == burst

            assert {(upload.encoding, upload.api_key) for upload in intake.uploads(DISTRIBUTIONS)} == {
                ("deflate", DD_API_KEY)
            }
            assert {(upload.encoding, upload.api_key) for upload in intake.uploads(SERIES)} == {("gzip", DD_API_KEY)}
            hostname_tags: Final = {
                tag for series in every_series(intake) for tag in _tags(series) if "HOSTNAME" in tag
            }
            assert hostname_tags == set(), hostname_tags


def test_every_surface_reports_its_latency_as_distribution_samples(rig_factory: RigFactory) -> None:
    """Chat, Anthropic messages and Responses, streaming and not, through the sync and async SDKs, each on
    its own model: every call adds one total and one provider latency sample, every overhead gauge point
    the surface reports gets its distribution twin, each with the same provider, model and status tags as
    its gauge, and the overhead distribution carries no status code, like the overhead gauge."""
    rounds: Final = 2
    intake: Final = DatadogIntake()
    with rig_factory(intake) as rig:
        with rig.proxy.scenario() as scenario:
            models: Final = {
                surface: (
                    scenario.model(
                        model="anthropic/claude-sonnet-4-5-20250929",
                        api_base=rig.provider_url,
                        api_key="synthetic-provider-key",
                    )
                    if surface.startswith("messages")
                    else scenario.model(api_base=rig.provider_url + "/v1", api_key="synthetic-provider-key")
                )
                for surface in SURFACES
            }
            key: Final = scenario.key(models=list(models.values()))
            marker: Final = f"dd-{uuid.uuid4().hex[:8]}"
            for index in range(rounds):
                for surface, model in models.items():
                    call_surface(rig.proxy, surface, model, model, key, f"{marker}-{surface}-{index}")
            for model in models.values():
                _wait_for_samples(intake, model, rounds, LATENCIES[:2])

            for surface, model in models.items():
                for metric in LATENCIES[:2]:
                    assert len(distribution_samples(intake, metric, model, "200")) == rounds, (surface, metric)
                    assert len(series_points(intake, metric, model, "200")) == rounds, (surface, metric)
                overhead_gauges = series_points(intake, LATENCIES[2], model)
                assert len(distribution_samples(intake, LATENCIES[2], model)) == len(overhead_gauges), surface

            for upload in intake.accepted(DISTRIBUTIONS):
                for series in upload.series:
                    tags = _tags(series)
                    provider = (
                        "anthropic"
                        if any(
                            f"model_group:{models[surface]}" in tags for surface in SURFACES if "messages" in surface
                        )
                        else "openai"
                    )
                    assert f"provider:{provider}" in tags, series
                    assert {"env:integration", "service:litellm-integration"} <= set(tags), series
                    if series["metric"] == "litellm.overhead.latency.distribution":
                        assert not any(tag.startswith("status_code:") for tag in tags), series
            gauge_tags: Final = {
                (str(series["metric"]), _tags(series))
                for upload in intake.accepted(SERIES)
                for series in upload.series
                if series["metric"] in LATENCIES
            }
            distribution_tags: Final = {
                (str(series["metric"]).removesuffix(".distribution"), _tags(series))
                for upload in intake.accepted(DISTRIBUTIONS)
                for series in upload.series
            }
            assert distribution_tags == gauge_tags


def test_a_set_hostname_tags_every_series_and_distribution(rig_factory: RigFactory) -> None:
    intake: Final = DatadogIntake()
    with rig_factory(intake, hostname="pod-lit6804") as rig:
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model(api_base=rig.provider_url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            assert _chat(str(rig.proxy.client.base_url), key, model, (f"dd-{uuid.uuid4().hex[:8]}",)) == (200,)
            _wait_for_samples(intake, model, 1)

            for metric in LATENCIES:
                assert len(distribution_samples(intake, metric, model)) == 1, metric
            uploaded: Final = tuple(
                series for series in every_series(intake) if f"model_group:{model}" in _tags(series)
            )
            assert {str(series["metric"]) for series in uploaded} == {
                *LATENCIES,
                *(f"{metric}.distribution" for metric in LATENCIES),
                REQUEST_COUNT_METRIC,
            }
            assert {tuple(tag for tag in _tags(series) if tag.startswith("HOSTNAME")) for series in uploaded} == {
                ("HOSTNAME:pod-lit6804",)
            }


@pytest.mark.parametrize(("callback_setting", "reports_failures"), (("callbacks", True), ("success_callback", False)))
def test_failed_requests_report_latency_under_their_status_code(
    rig_factory: RigFactory, callback_setting: str, reports_failures: bool
) -> None:
    """A provider 500 reaches the caller as an error; with the callback on both events its latency lands as
    distribution samples tagged `status_code:500` next to the successes, and in success-only mode nothing
    of the failures is sent."""
    per_outcome: Final = 3
    intake: Final = DatadogIntake()
    with rig_factory(intake, callback_setting=callback_setting, upstream=upstream_failing_marked_requests) as rig:
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model(api_base=rig.provider_url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            marker: Final = uuid.uuid4().hex[:8]
            statuses: Final = _chat(
                str(rig.proxy.client.base_url),
                key,
                model,
                (*(f"ok-{marker}-{index}" for index in range(per_outcome)),)
                + tuple(f"fail-{marker}-{index}" for index in range(per_outcome)),
            )
            assert sorted(statuses) == [200] * per_outcome + [500] * per_outcome, statuses
            eventually(
                lambda: len(distribution_samples(intake, LATENCIES[0], model, "500")),
                lambda total: total >= (per_outcome if reports_failures else 0),
                seconds=FLUSH_WAIT_SECONDS,
                return_last_on_timeout=True,
            )
            _wait_for_samples(intake, model, per_outcome)

            assert len(distribution_samples(intake, LATENCIES[0], model, "200")) == per_outcome
            failed: Final = distribution_samples(intake, LATENCIES[0], model, "500")
            assert len(failed) == (per_outcome if reports_failures else 0), failed
            assert sum(series_points(intake, REQUEST_COUNT_METRIC, model, "500")) == (
                per_outcome if reports_failures else 0
            )
            assert len(series_points(intake, LATENCIES[0], model, "500")) == len(failed)


@pytest.mark.parametrize(
    ("outage", "rejected_path"),
    (({DISTRIBUTIONS: 2}, DISTRIBUTIONS), ({SERIES: 2}, SERIES)),
    ids=("distribution-endpoint-down", "series-endpoint-down"),
)
def test_an_endpoint_outage_delivers_every_sample_exactly_once_after_recovery(
    rig_factory: RigFactory, outage: Mapping[str, int], rejected_path: str
) -> None:
    """Datadog answers 503 to the first uploads of one endpoint. After it recovers, every request's gauges,
    count and distribution samples have been accepted exactly once: the retry neither drops the rejected
    half nor resends the half Datadog already took."""
    burst: Final = 6
    intake: Final = DatadogIntake(failing=outage)
    with rig_factory(intake) as rig:
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model(api_base=rig.provider_url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            marker: Final = f"dd-{uuid.uuid4().hex[:8]}"
            statuses: Final = _chat(
                str(rig.proxy.client.base_url), key, model, tuple(f"{marker}-{index}" for index in range(burst))
            )
            assert statuses == (200,) * burst, statuses
            _wait_for_samples(intake, model, burst)
            eventually(
                lambda: sum(series_points(intake, REQUEST_COUNT_METRIC, model)),
                lambda total: total >= burst,
                seconds=FLUSH_WAIT_SECONDS,
                return_last_on_timeout=True,
            )

            assert intake.rejected(rejected_path), "the scripted outage never answered an upload"
            for metric in LATENCIES:
                assert len(distribution_samples(intake, metric, model)) == burst, metric
                assert len(series_points(intake, metric, model)) == burst, metric
            assert sum(series_points(intake, REQUEST_COUNT_METRIC, model)) == burst


def test_requests_finishing_while_a_slow_flush_is_in_flight_are_not_lost(rig_factory: RigFactory) -> None:
    """Datadog takes 1.5 s to answer each upload while requests keep finishing. Samples queued during an
    upload must ride the next flush rather than be cleared with the batch that was being sent."""
    requests: Final = 40
    intake: Final = DatadogIntake(delay=1.5)
    with rig_factory(intake) as rig:
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model(api_base=rig.provider_url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            marker: Final = f"dd-{uuid.uuid4().hex[:8]}"
            for index in range(requests):
                assert _chat(str(rig.proxy.client.base_url), key, model, (f"{marker}-{index}",)) == (200,)
                time.sleep(0.3)
            _wait_for_samples(intake, model, requests)
            eventually(
                lambda: len(series_points(intake, LATENCIES[0], model)),
                lambda total: total >= requests,
                seconds=FLUSH_WAIT_SECONDS,
                return_last_on_timeout=True,
            )

            assert len(intake.accepted(DISTRIBUTIONS)) >= 2, "every sample fit in one flush, nothing overlapped"
            for metric in LATENCIES:
                assert len(distribution_samples(intake, metric, model)) == requests, metric
                assert len(series_points(intake, metric, model)) == requests, metric


def _mock_health_request_series(intake: DatadogIntake, target: str, metric: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        series
        for upload in intake.accepted(target)
        for series in upload.series
        if series["metric"] == metric and any("litellm-mock-response-model" in tag for tag in _tags(series))
    )


def test_the_health_check_test_request_reaches_datadog_as_a_distribution_sample(rig_factory: RigFactory) -> None:
    """With the callback configured, `GET /health/services?service=datadog_metrics` answers with a mock
    completion, and that completion's latency reaches Datadog like any request: one gauge point and one
    distribution sample with the same tags."""
    intake: Final = DatadogIntake()
    with rig_factory(intake) as rig:
        response: Final = rig.proxy.request("GET", "/health/services", params={"service": "datadog_metrics"})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success", response.text
        metric: Final = f"{LATENCIES[0]}.distribution"
        eventually(
            lambda: len(_mock_health_request_series(intake, DISTRIBUTIONS, metric)),
            lambda total: total >= 1,
            seconds=FLUSH_WAIT_SECONDS,
            return_last_on_timeout=True,
        )

        distributions: Final = _mock_health_request_series(intake, DISTRIBUTIONS, metric)
        gauges: Final = _mock_health_request_series(intake, SERIES, LATENCIES[0])
        assert len(distributions) == 1 and len(gauges) == 1, intake.uploads()
        assert _tags(distributions[0]) == _tags(gauges[0])


def test_the_health_check_probe_without_the_callback_still_posts_only_to_the_series_endpoint(
    rig_factory: RigFactory,
) -> None:
    """Without the callback configured, the same health check sends its own `litellm.health_check` probe,
    which still goes only to `/api/v2/series`, so a distribution intake outage does not show up there."""
    intake: Final = DatadogIntake()
    with rig_factory(intake, callback_setting=None) as rig:
        response: Final = rig.proxy.request("GET", "/health/services", params={"service": "datadog_metrics"})
        assert response.status_code == 200, response.text
        assert response.json() == {"status": "healthy", "message": "Datadog Metrics is healthy"}, response.text
        probes: Final = tuple(
            (upload.target, upload.encoding, str(series["metric"]))
            for upload in intake.uploads()
            for series in upload.series
        )
        assert probes == ((SERIES, "gzip", "litellm.health_check"),), intake.uploads()
