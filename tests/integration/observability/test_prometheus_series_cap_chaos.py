"""Prometheus series cap under load: a concurrent burst across every endpoint while /metrics is scraped, a
provider outage between bursts, a worker killed mid-burst, and restarts that wipe or keep the multiprocess
directory."""

from __future__ import annotations

import json
import re
import signal
import threading
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
from integration._support.process import owned_gateway_image, setup_only_proxy_run
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
    alias_total,
    alias_values,
    expect_spend_rows,
    families_over,
    label_values,
    overflow_total,
    scrape,
    series_cap_config,
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


def _landed(before: Sequence[Sample], after: Sequence[Sample], growth: Mapping[str, int]) -> bool:
    """Every counter the cell asserts on has counted its calls on `other`: the request and failure counters of
    one call increment at different points of the logging callback, so a scrape between them is not the end
    state."""
    return all(overflow_total(after, name) - overflow_total(before, name) >= by for name, by in growth.items())


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
        off_route: Final = len(capped.warm) * (len(ROUTES) - 1)
        growth: Final = {REQUESTS: extra_requests, PROXY_REQUESTS: extra_requests + off_route}
        samples: Final = eventually(
            lambda: scrape(capped.gateway),
            lambda after: (
                _landed(before, after, growth) or any(key.alias in alias_values(after, REQUESTS) for key in extra)
            ),
            seconds=90,
        )
        assert alias_values(samples, REQUESTS) == capped.warm_aliases
        assert not families_over(samples, CAP), families_over(samples, CAP)
        assert overflow_total(samples, REQUESTS) - overflow_total(before, REQUESTS) == extra_requests
        assert alias_values(samples, PROXY_REQUESTS) == capped.warm_aliases
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
                _landed(before, after, {PROXY_FAILURES: 2, REQUESTS: 4})
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


@pytest.mark.timeout(420)
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


@pytest.mark.timeout(420)
def test_restart_with_one_worker_and_an_operator_directory_starts_the_cap_over(tmp_path: Path) -> None:
    """C5: one worker, no metrics port, PROMETHEUS_MULTIPROC_DIR set by the operator and kept across a restart:
    the directory is wiped at boot the way the multi-worker path wipes it, so the merged scrape shows only the
    second boot's three keys and a fourth lands on `other`."""
    operator_dir: Final = tmp_path / "prom-operator"
    settings: Final = {"prometheus_metrics_max_series_per_metric": CAP}
    with series_cap_rig(tmp_path, settings, workers=1, warm_keys=3, multiproc_dir=operator_dir) as first_boot:
        old_aliases: Final = first_boot.warm_aliases
        assert alias_values(scrape(first_boot.gateway), REQUESTS) == old_aliases
    with series_cap_rig(tmp_path, settings, workers=1, warm_keys=3, multiproc_dir=operator_dir) as second_boot:
        samples: Final = scrape(second_boot.gateway)
        assert alias_values(samples, REQUESTS) == second_boot.warm_aliases
        assert not old_aliases & label_values(samples)
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


def test_setup_only_run_leaves_a_live_proxy_samples_alone(tmp_path: Path) -> None:
    """P1: a `--skip_server_startup` run of the proxy CLI (the image's setup step) pointed at a live two-worker
    proxy's operator-set `PROMETHEUS_MULTIPROC_DIR` leaves the live samples alone: the warm keys keep their series
    and their totals, and a fourth key still lands on `other`."""
    operator_dir: Final = tmp_path / "prom-operator"
    settings: Final = {"prometheus_metrics_max_series_per_metric": CAP}
    with series_cap_rig(tmp_path, settings, workers=2, warm_keys=3, multiproc_dir=operator_dir) as rig:
        before: Final = scrape(rig.gateway)
        assert alias_values(before, REQUESTS) == rig.warm_aliases
        completed: Final = setup_only_proxy_run(
            rig.gateway,
            {"PROMETHEUS_MULTIPROC_DIR": str(operator_dir)},
            config=series_cap_config(tmp_path, settings),
            workers=2,
        )
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
        assert "Skipping server startup" in completed.stdout, completed.stdout[-2000:]
        after_setup: Final = scrape(rig.gateway)
        assert alias_values(after_setup, REQUESTS) == rig.warm_aliases
        assert all(
            alias_total(after_setup, REQUESTS, alias) == alias_total(before, REQUESTS, alias)
            for alias in rig.warm_aliases
        )
        extra: Final = rig.key("p1")
        assert rig.chat(extra, Call.new()).status_code == 200
        after: Final = eventually(
            lambda: scrape(rig.gateway),
            lambda now: (
                overflow_total(now, REQUESTS) - overflow_total(after_setup, REQUESTS) >= 1
                or extra.alias in alias_values(now, REQUESTS)
            ),
            seconds=60,
        )
        assert extra.alias not in label_values(after)
        assert alias_values(after, REQUESTS) == rig.warm_aliases


IMAGE_MODEL: Final = "series-cap-image"


def _image_deployment(provider_url: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        {
            "model_name": IMAGE_MODEL,
            "litellm_params": {
                "model": f"openai/gpt-{IMAGE_MODEL}",
                "api_base": provider_url + "/v1",
                "api_key": "synthetic-provider-key",
            },
        },
    )


@contextmanager
def _gateway_image(control: CapRig, config: Path, prom_dir: Path) -> Iterator[CapRig]:
    """One container life of the gateway image on `prom_dir`: two workers serving the keys the control plane proxy
    mints in the database both read."""
    with owned_gateway_image(
        control.gateway, config.parent, {"PROMETHEUS_MULTIPROC_DIR": str(prom_dir)}, config=config, workers=2
    ) as image:
        yield CapRig(image, control.scenario, IMAGE_MODEL, control.provider, control.outage, (), prom_dir)


def _fill_the_cap(image: CapRig, cell: str) -> frozenset[str]:
    """Three new keys call once each on a boot that has counted nothing yet, and each gets its own series."""
    keys: Final = tuple(image.key(cell) for _ in range(CAP))
    assert all(image.chat(key, Call.new()).status_code == 200 for key in keys)
    aliases: Final = frozenset(key.alias for key in keys)
    samples: Final = eventually(
        lambda: scrape(image.gateway),
        lambda now: sum(sample.value for sample in now if sample.name == REQUESTS) >= CAP,
        seconds=60,
    )
    assert alias_values(samples, REQUESTS) == aliases, (alias_values(samples, REQUESTS), aliases)
    return aliases


@pytest.mark.timeout(420)
def test_gateway_image_restart_on_a_kept_directory_starts_the_cap_over(tmp_path: Path) -> None:
    """D1: the gateway image's launcher (`docker/component_entrypoint.sh` running `python -m gateway.launch`, two
    workers) restarted on a kept PROMETHEUS_MULTIPROC_DIR: the entrypoint removes the previous container's samples
    and admitted series before the workers fork, so the second boot shows only its own three keys and a fourth
    lands on `other`."""
    control_dir: Final = tmp_path / "control"
    image_dir: Final = tmp_path / "image"
    control_dir.mkdir()
    image_dir.mkdir()
    prom_dir: Final = tmp_path / "prom-image"
    with series_cap_rig(control_dir, {}, workers=1, warm_keys=0) as control:
        config: Final = series_cap_config(
            image_dir,
            {"prometheus_metrics_max_series_per_metric": CAP},
            model_list=_image_deployment(control.provider.url),
        )
        with _gateway_image(control, config, prom_dir) as first_boot:
            old_aliases: Final = _fill_the_cap(first_boot, "d1-old")
        with _gateway_image(control, config, prom_dir) as second_boot:
            assert not old_aliases & label_values(scrape(second_boot.gateway))
            new_aliases: Final = _fill_the_cap(second_boot, "d1-new")
            extra: Final = second_boot.key("d1")
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
            assert alias_values(after, REQUESTS) == new_aliases
            assert not old_aliases & label_values(after)
