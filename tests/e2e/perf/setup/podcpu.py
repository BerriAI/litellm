"""CPU seconds the gateway pods burned during the traffic phase, from metrics-server (`metrics.k8s.io`
PodMetrics), so a case can put a ceiling on the CPU the gateway costs per request: a slower gateway shows
up here at 5 rps, long before it saturates. The runner samples the pods matching
`E2E_PERF_GATEWAY_POD_SELECTOR` every few seconds with its own service account; `usage.cpu` is the
pods' average core use over metrics-server's window, integrated over the sampling interval. Without the
selector or the in-cluster token the reading is None and a Limit on it reports as missing."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import quote

from e2e_http import AuthHeaders, Success, get_external
from pydantic import BaseModel

from setup.series import Recorder, Series

SELECTOR_ENV: Final = "E2E_PERF_GATEWAY_POD_SELECTOR"
_SERVICE_ACCOUNT: Final = Path("/var/run/secrets/kubernetes.io/serviceaccount")
_INTERVAL_S: Final = 5.0
_SUFFIX_CORES: Final = {"n": 1e-9, "u": 1e-6, "m": 1e-3}


class _Usage(BaseModel):
    cpu: str


class _Container(BaseModel):
    usage: _Usage


class _PodMetrics(BaseModel):
    containers: list[_Container]


class _PodMetricsList(BaseModel):
    items: list[_PodMetrics]


def cores(quantity: str) -> float:
    """A Kubernetes CPU quantity (`1500m`, `250000000n`, `2`) as cores."""
    if quantity and quantity[-1] in _SUFFIX_CORES:
        return float(quantity[:-1]) * _SUFFIX_CORES[quantity[-1]]
    return float(quantity)


class GatewayCpuSampler:
    """Integrates the matching pods' core use over the time it is running; `seconds` is CPU seconds and
    `series` the CPU seconds of each sampling interval, stamped when the interval ended."""

    def __init__(self, selector: str, namespace: str, token: str, ca_bundle: str) -> None:
        self._url: Final = (
            f"https://kubernetes.default.svc/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods"
            f"?labelSelector={quote(selector)}"
        )
        self._headers: Final = AuthHeaders(authorization=f"Bearer {token}")
        self._ca_bundle: Final = ca_bundle
        self._stop: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, name="gateway-cpu-sampler", daemon=True)
        self._intervals: Final[Recorder[float]] = Recorder()

    @property
    def seconds(self) -> float | None:
        return sum(sample.value for sample in self._intervals.series.samples) if len(self._intervals) else None

    @property
    def series(self) -> Series[float]:
        return self._intervals.series

    def _read_cores(self) -> float | None:
        result: Final = get_external(
            self._url, response_type=_PodMetricsList, headers=self._headers, timeout=10.0, ca_bundle=self._ca_bundle
        )
        if not isinstance(result, Success):
            return None
        return sum(cores(container.usage.cpu) for pod in result.data.items for container in pod.containers)

    def _run(self) -> None:
        last: float = time.monotonic()  # rebind-ok: the sampling clock advances every tick
        while not self._stop.wait(_INTERVAL_S):
            now: float = time.monotonic()
            current: float | None = self._read_cores()
            if current is not None:
                self._intervals.record(current * (now - last))
            last = now  # rebind-ok: the sampling clock advances every tick

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=_INTERVAL_S * 3)


class NoSampler:
    seconds: Final[None] = None
    series: Final[None] = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def in_cluster() -> tuple[str, str, str] | None:
    token: Final = _SERVICE_ACCOUNT / "token"
    ca: Final = _SERVICE_ACCOUNT / "ca.crt"
    namespace: Final = _SERVICE_ACCOUNT / "namespace"
    if not (token.is_file() and ca.is_file() and namespace.is_file()):
        return None
    return token.read_text().strip(), str(ca), namespace.read_text().strip()


@contextmanager
def gateway_cpu_sampler() -> Generator[GatewayCpuSampler | NoSampler]:
    selector: Final = os.environ.get(SELECTOR_ENV, "")
    cluster: Final = in_cluster()
    sampler: Final = (
        GatewayCpuSampler(selector, cluster[2], cluster[0], cluster[1]) if selector and cluster else NoSampler()
    )
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()
