"""An SDK app under a prefork worker pool (Celery, gunicorn) with ``PROMETHEUS_MULTIPROC_DIR`` set: the Redis
circuit-breaker state gauge must collect as one series per state across every worker, and it must stay off
the registry entirely for an app that never turned on LiteLLM's own Prometheus logging."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from integration._support.redis_breaker_worker import STATE_GAUGE, WorkerReport, run_breaker_worker
from integration._support.wire import Reply, Request, wire_server
from prometheus_client import CollectorRegistry, multiprocess

_ANSWER: Final = "answered through the redis cache"
_CHAT_RESPONSE: Final = json.dumps(
    {
        "id": "chatcmpl-redis-breaker-worker",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": _ANSWER}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    }
).encode()
_CLOSED: Final = (("state", "closed"),)


def _reply(request: Request) -> Reply:
    assert request.method == "POST", request.method
    return Reply(body=_CHAT_RESPONSE)


def _state_samples(prom_dir: Path) -> Iterator[tuple[tuple[tuple[str, str], ...], float]]:
    registry: Final = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(prom_dir))
    for family in registry.collect():
        if family.name != STATE_GAUGE:
            continue
        for sample in family.samples:
            yield tuple(sorted(sample.labels.items())), sample.value


def _collected_state_series(prom_dir: Path) -> tuple[tuple[tuple[tuple[str, str], ...], float], ...]:
    return tuple(sorted(_state_samples(prom_dir)))


def _worked(report: WorkerReport) -> WorkerReport:
    assert report.answer == _ANSWER, report
    return report


def test_two_worker_processes_collect_as_one_state_series_per_state(tmp_path: Path) -> None:
    with wire_server(_reply) as wire:
        reports: Final = tuple(
            _worked(run_breaker_worker("callbacks_before_cache", wire.url, tmp_path)) for _ in range(2)
        )
    assert len({report.pid for report in reports}) == 2, reports
    assert _collected_state_series(tmp_path) == ((_CLOSED, 2.0),), reports


def test_prometheus_logging_turned_on_after_the_cache_still_reports_the_closed_breaker(tmp_path: Path) -> None:
    with wire_server(_reply) as wire:
        report: Final = _worked(run_breaker_worker("callbacks_after_cache", wire.url, tmp_path))
    assert report.state_series == ((_CLOSED, 1.0),), report
    assert _collected_state_series(tmp_path) == ((_CLOSED, 1.0),), report


def test_app_without_litellm_prometheus_logging_gets_no_breaker_series(tmp_path: Path) -> None:
    with wire_server(_reply) as wire:
        report: Final = _worked(run_breaker_worker("no_callbacks", wire.url, tmp_path))
    assert report.state_series == (), report
    assert _collected_state_series(tmp_path) == (), report


def test_opt_out_keeps_breaker_series_off_with_prometheus_logging_on(tmp_path: Path) -> None:
    with wire_server(_reply) as wire:
        report: Final = _worked(run_breaker_worker("opted_out", wire.url, tmp_path))
    assert report.state_series == (), report
    assert _collected_state_series(tmp_path) == (), report
