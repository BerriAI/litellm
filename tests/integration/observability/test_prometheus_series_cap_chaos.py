"""Prometheus series cap under load: a concurrent burst across every endpoint while /metrics is scraped, a
provider outage between bursts, a worker killed mid-burst, and restarts that wipe or keep the multiprocess
directory."""

from __future__ import annotations

import json
import re
import signal
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import cycle, product
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import httpx
import psutil
import pytest
from integration._support.client import eventually, object_value, string_value
from integration._support.prometheus_series import (
    AGENT_HEADERS,
    PROXY_FAILURES,
    PROXY_REQUESTS,
    REQUESTS,
    Call,
    CapRig,
    Key,
    Sample,
    SpendRow,
    alias_values,
    expect_spend_rows,
    families_over,
    label_values,
    overflow_total,
    scrape,
    series_cap_rig,
    spend_rows,
    sse_data,
    worker_samples,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(240)

CAP: Final = 3
CHAT: Final = "/v1/chat/completions"
MESSAGES: Final = "/v1/messages"
RESPONSES: Final = "/v1/responses"
ROUTES: Final = (CHAT, MESSAGES, RESPONSES)
EXTRA_KEYS: Final = 7
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Served:
    key: Key
    call: Call
    route: str
    streamed: bool
    status: int
    text: str

    @property
    def response_id(self) -> str:
        assert self.status == 200, self.text
        if not self.streamed:
            return string_value(object_value(json.loads(self.text))["id"])
        events: Final = sse_data(self.text)
        match self.route:
            case "/v1/messages":
                starts: Final = tuple(event for event in events if event.get("type") == "message_start")
                return string_value(object_value(starts[0]["message"])["id"])
            case "/v1/responses":
                completed: Final = tuple(event for event in events if event.get("type") == "response.completed")
                return string_value(object_value(completed[0]["response"])["id"])
            case _:
                ids: Final = frozenset(string_value(event["id"]) for event in events)
                assert len(ids) == 1, ids
                return next(iter(ids))


def _body(route: str, model: str, call: Call, streamed: bool) -> dict[str, JsonValue]:
    match route:
        case "/v1/messages":
            return {"model": model, "max_tokens": 64, "messages": [call.message], "stream": streamed}
        case "/v1/responses":
            return {"model": model, "input": call.text, "stream": streamed}
        case _:
            return {"model": model, "messages": [call.message], "stream": streamed}


def _send(rig: CapRig, key: Key, route: str, streamed: bool, tolerate_transport_errors: bool = False) -> Served:
    call: Final = Call.new()
    try:
        with httpx.Client(base_url=rig.base_url, timeout=60, trust_env=False) as client:
            response: Final = client.post(
                route,
                json=_body(route, rig.model, call, streamed),
                headers={**AGENT_HEADERS, **call.headers, "Authorization": f"Bearer {key.token}"},
            )
    except httpx.TransportError as error:
        if not tolerate_transport_errors:
            raise
        return Served(key, call, route, streamed, 0, repr(error))
    return Served(key, call, route, streamed, response.status_code, response.text)


@dataclass(frozen=True, slots=True)
class Plan:
    key: Key
    route: str
    streamed: bool


def _plans(keys: Sequence[Key]) -> tuple[Plan, ...]:
    streaming: Final = cycle((False, True))
    return tuple(Plan(key, route, next(streaming)) for key, route in product(keys, ROUTES))


def _burst(rig: CapRig, plans: Sequence[Plan], tolerate_transport_errors: bool = False) -> tuple[Served, ...]:
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        return tuple(
            pool.map(lambda plan: _send(rig, plan.key, plan.route, plan.streamed, tolerate_transport_errors), plans)
        )


def _scrape_until(rig: CapRig, stop: threading.Event, sizes: SimpleQueue[int]) -> None:
    while not stop.is_set():
        try:
            sizes.put(len(scrape(rig.gateway)))
        except (AssertionError, httpx.HTTPError):
            sizes.put(-1)


def _rows_by_alias(keys: Sequence[Key]) -> Mapping[str, tuple[SpendRow, ...]]:
    return MappingProxyType({key.alias: spend_rows(key.alias) for key in keys})


class TestBurst:
    def test_concurrent_burst_across_every_endpoint_while_scraping(self, capped: CapRig) -> None:
        """C1: 30 concurrent calls from ten keys across chat, messages, and responses, streamed and not, with
        /metrics scraped throughout: every call answers, the warm keys keep their series, every other call counts
        on `other`, and every call writes one spend row."""
        extra: Final = tuple(capped.key(f"c1-{index}") for index in range(EXTRA_KEYS))
        keys: Final = (*capped.warm, *extra)
        earlier: Final = _rows_by_alias(keys)
        before: Final = scrape(capped.gateway)
        stop: Final = threading.Event()
        sizes: Final[SimpleQueue[int]] = SimpleQueue()
        scraper: Final = threading.Thread(target=_scrape_until, args=(capped, stop, sizes))
        scraper.start()
        try:
            served: Final = _burst(capped, _plans(keys))
        finally:
            stop.set()
            scraper.join()
        scrapes: Final = tuple(sizes.get_nowait() for _ in range(sizes.qsize()))
        assert scrapes and all(count > 0 for count in scrapes), scrapes
        assert all(item.status == 200 and item.call.answer in item.text for item in served), [
            (item.route, item.status, item.text[:200]) for item in served if item.status != 200
        ]
        extra_requests: Final = EXTRA_KEYS * len(ROUTES)
        samples: Final = eventually(
            lambda: scrape(capped.gateway),
            lambda after: (
                overflow_total(after, REQUESTS) - overflow_total(before, REQUESTS) >= extra_requests
                or any(key.alias in alias_values(after, REQUESTS) for key in extra)
            ),
            seconds=90,
        )
        assert alias_values(samples, REQUESTS) == capped.warm_aliases
        assert not families_over(samples, CAP), families_over(samples, CAP)
        assert overflow_total(samples, REQUESTS) - overflow_total(before, REQUESTS) == extra_requests
        assert alias_values(samples, PROXY_REQUESTS) == capped.warm_aliases
        off_route: Final = len(capped.warm) * (len(ROUTES) - 1)
        assert overflow_total(samples, PROXY_REQUESTS) - overflow_total(before, PROXY_REQUESTS) == (
            extra_requests + off_route
        )
        for key in keys:
            expect_spend_rows(
                key.alias,
                tuple(item.response_id for item in served if item.key == key),
                earlier=earlier[key.alias],
            )

    def test_outage_between_bursts_counts_every_failure_once(self, capped: CapRig) -> None:
        """C2: a burst answers, the provider goes down for the next burst, and comes back for the last: the warm
        keys keep their failure series, the fourth key's failures count on `other`, and every call writes one row."""
        extra: Final = capped.key("c2")
        keys: Final = (*capped.warm, extra)
        earlier: Final = _rows_by_alias(keys)
        before: Final = scrape(capped.gateway)
        plans: Final = tuple(Plan(key, CHAT, streamed) for key, streamed in product(keys, (False, True)))
        first: Final = _burst(capped, plans)
        capped.outage.set()
        try:
            prefill: Final = tuple(_send(capped, warm, CHAT, False) for warm in capped.warm)
            down: Final = _burst(capped, plans)
        finally:
            capped.outage.clear()
        last: Final = _burst(capped, plans)
        failed: Final = (*prefill, *down)
        assert all(item.status == 200 for item in (*first, *last)), [item.status for item in (*first, *last)]
        assert all(item.status == 500 for item in failed), [item.status for item in failed]
        samples: Final = eventually(
            lambda: scrape(capped.gateway),
            lambda after: (
                overflow_total(after, PROXY_FAILURES) - overflow_total(before, PROXY_FAILURES) >= 2
                or extra.alias in alias_values(after, PROXY_FAILURES)
            ),
            seconds=90,
        )
        assert alias_values(samples, PROXY_FAILURES) == capped.warm_aliases
        assert not families_over(samples, CAP), families_over(samples, CAP)
        assert overflow_total(samples, PROXY_FAILURES) - overflow_total(before, PROXY_FAILURES) == 2
        assert overflow_total(samples, REQUESTS) - overflow_total(before, REQUESTS) == 4
        for key in keys:
            expect_spend_rows(
                key.alias,
                tuple(item.response_id for item in (*first, *last) if item.key == key),
                tuple(item.call.call_id for item in failed if item.key == key),
                earlier=earlier[key.alias],
            )


def _worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text()
    return tuple(int(pid) for pid in _STARTED_WORKER.findall(text)), text.count("Application startup complete.")


@pytest.mark.timeout(420)
def test_killed_worker_is_replaced_by_one_that_reads_the_same_admissions(tmp_path: Path) -> None:
    """C3: SIGKILL one of two workers mid-burst: the sibling keeps answering, and the replacement worker puts a
    fourth key on `other` because the admitted series live in the shared directory, not in the dead process."""
    with series_cap_rig(tmp_path, {"prometheus_metrics_max_series_per_metric": CAP}, workers=2, warm_keys=3) as rig:
        workers, _ = eventually(
            lambda: _worker_startups(rig.proxy.log), lambda found: len(found[0]) == 2 and found[1] == 2, seconds=120
        )
        extra: Final = tuple(rig.key(f"c3-{index}") for index in range(4))
        plans: Final = _plans(extra)
        with ThreadPoolExecutor(max_workers=1) as pool:
            burst: Final = pool.submit(_burst, rig, plans, True)
            victim: Final = psutil.Process(workers[0])
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            served: Final = burst.result()
        answered: Final = tuple(item for item in served if item.status == 200)
        assert answered, [(item.status, item.text[:200]) for item in served]
        assert all(item.call.answer in item.text for item in answered)
        replacement: Final = eventually(
            lambda: _worker_startups(rig.proxy.log),
            lambda found: len(frozenset(found[0]) - frozenset(workers)) == 1,
            seconds=120,
        )
        (new_pid,) = frozenset(replacement[0]) - frozenset(workers)
        late: Final = rig.key("c3-late")

        def send_until_the_replacement_counts() -> tuple[Sample, ...]:
            assert rig.chat(late, Call.new()).status_code == 200
            return scrape(rig.gateway)

        samples: Final = eventually(
            send_until_the_replacement_counts,
            lambda after: (
                any(
                    sample.pid == new_pid and (sample.overflow > 0 or late.alias in sample.aliases)
                    for sample in worker_samples(rig.prom_dir, REQUESTS)
                )
                or late.alias in alias_values(after, REQUESTS)
            ),
            seconds=90,
        )
        assert late.alias not in label_values(samples)
        by_pid: Final = {sample.pid: sample for sample in worker_samples(rig.prom_dir, REQUESTS)}
        assert by_pid[new_pid].overflow > 0 and by_pid[new_pid].aliases <= rig.warm_aliases, by_pid[new_pid]


def test_restart_with_two_workers_starts_the_cap_over(tmp_path: Path) -> None:
    """C4: a second boot on the same multiprocess directory wipes it: the old keys are gone, three new keys get
    their series, and a fourth lands on `other`."""
    shared_dir: Final = tmp_path / "prom-shared"
    settings: Final = {"prometheus_metrics_max_series_per_metric": CAP}
    with series_cap_rig(tmp_path, settings, workers=2, warm_keys=3, multiproc_dir=shared_dir) as first_boot:
        old_aliases: Final = first_boot.warm_aliases
        assert alias_values(scrape(first_boot.gateway), REQUESTS) == old_aliases
    with series_cap_rig(tmp_path, settings, workers=2, warm_keys=3, multiproc_dir=shared_dir) as second_boot:
        samples: Final = scrape(second_boot.gateway)
        assert alias_values(samples, REQUESTS) == second_boot.warm_aliases
        assert not old_aliases & label_values(samples)
        extra: Final = second_boot.key("c4")
        before: Final = scrape(second_boot.gateway)
        assert second_boot.chat(extra, Call.new()).status_code == 200
        after: Final = eventually(
            lambda: scrape(second_boot.gateway),
            lambda now: (
                overflow_total(now, REQUESTS) - overflow_total(before, REQUESTS) >= 1
                or extra.alias in alias_values(now, REQUESTS)
            ),
            seconds=60,
        )
        assert extra.alias not in label_values(after)


def test_restart_with_one_worker_and_an_operator_directory_starts_the_cap_over(tmp_path: Path) -> None:
    """C5: one worker, no metrics port, PROMETHEUS_MULTIPROC_DIR set by the operator and kept across a restart:
    the second boot's three keys get their series and a fourth lands on `other`, because the admitted series
    files are dropped at boot even though the operator's sample files are left alone."""
    operator_dir: Final = tmp_path / "prom-operator"
    settings: Final = {"prometheus_metrics_max_series_per_metric": CAP}
    with series_cap_rig(tmp_path, settings, workers=1, warm_keys=3, multiproc_dir=operator_dir) as first_boot:
        old_aliases: Final = first_boot.warm_aliases
        assert alias_values(scrape(first_boot.gateway), REQUESTS) == old_aliases
    old_pids: Final = frozenset(sample.pid for sample in worker_samples(operator_dir, REQUESTS))
    with series_cap_rig(tmp_path, settings, workers=1, warm_keys=3, multiproc_dir=operator_dir) as second_boot:
        fresh: Final = tuple(sample for sample in worker_samples(operator_dir, REQUESTS) if sample.pid not in old_pids)
        assert len(fresh) == 1 and fresh[0].aliases == second_boot.warm_aliases, fresh
        extra: Final = second_boot.key("c5")
        before: Final = scrape(second_boot.gateway)
        assert second_boot.chat(extra, Call.new()).status_code == 200
        after: Final = eventually(
            lambda: scrape(second_boot.gateway),
            lambda now: (
                overflow_total(now, REQUESTS) - overflow_total(before, REQUESTS) >= 1
                or extra.alias in alias_values(now, REQUESTS)
            ),
            seconds=60,
        )
        assert extra.alias not in label_values(after)
