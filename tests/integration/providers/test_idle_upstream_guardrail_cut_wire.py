import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.idle_gateway import Arrival
from integration._support.process import graceful_stop_seconds
from integration._support.wire import Reply, Request
from integration.providers._idle_upstream_support import (
    REUSE_IDLE_SECONDS,
    ProxyTarget,
    Row,
    Sent,
    UpstreamProxy,
    answer,
    answered,
    chat_with_guardrail,
    expected_answers,
    guardrail_deployment,
    landed,
    on_each_worker,
    send,
    spend_statuses,
    stay_idle,
    upstream_proxy,
)

CUT_AFTER_SECONDS: Final = 3.0
POOLED_PER_WORKER: Final = 2
RETRIED: Final = "guardrail-cut-num-retries-1"
UNSET: Final = "guardrail-cut-num-retries-unset"

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 180 + REUSE_IDLE_SECONDS))


@dataclass(frozen=True, slots=True)
class PairedWarmup:
    """While warming, each check waits for a second one so every worker pools two connections to the guardrail."""

    warming: threading.Event = field(default_factory=threading.Event)
    pair: threading.Barrier = field(default_factory=lambda: threading.Barrier(POOLED_PER_WORKER))

    def respond(self, request: Request) -> Reply:
        if self.warming.is_set():
            self.pair.wait(timeout=30)
        return answer(request)


@dataclass(frozen=True, slots=True)
class StaleGuardrails:
    proxy: UpstreamProxy
    pooled: Mapping[str, frozenset[int]]


WARMUPS: Final = MappingProxyType({RETRIED: PairedWarmup(), UNSET: PairedWarmup()})
ROWS: Final = (
    Row(
        RETRIED,
        partial(guardrail_deployment, guardrail=RETRIED, num_retries=1),
        partial(chat_with_guardrail, guardrail=RETRIED),
        respond=WARMUPS[RETRIED].respond,
    ),
    Row(
        UNSET,
        partial(guardrail_deployment, guardrail=UNSET),
        partial(chat_with_guardrail, guardrail=UNSET),
        respond=WARMUPS[UNSET].respond,
    ),
)


def _pair(row: Row, target: ProxyTarget) -> tuple[Sent, ...]:
    with ThreadPoolExecutor(max_workers=POOLED_PER_WORKER) as pool:
        return tuple(pool.map(send, (row,) * POOLED_PER_WORKER, (target,) * POOLED_PER_WORKER))


def _warm(proxy: UpstreamProxy, row: Row) -> frozenset[int]:
    warmup: Final = WARMUPS[row.name]
    warmup.warming.set()
    sents: Final = on_each_worker(proxy.owned, proxy.workers, lambda: _pair(row, proxy.target))
    warmup.warming.clear()
    assert answered(sents) == expected_answers(row, sents), sents
    return frozenset(arrival.connection for arrival in proxy.gateways[row.name].drain())


@pytest.fixture(scope="module")
def stale(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[StaleGuardrails]:
    directory: Final = tmp_path_factory.mktemp("idle-upstream-guardrail-cut")
    with upstream_proxy(session_gateway, directory, ROWS, {}, drop_after=CUT_AFTER_SECONDS) as proxy:
        pooled: Final = MappingProxyType({row.name: _warm(proxy, row) for row in ROWS})
        assert {name: len(connections) for name, connections in pooled.items()} == {
            row.name: 2 * POOLED_PER_WORKER for row in ROWS
        }, pooled
        stay_idle(tuple(proxy.gateways.values()), REUSE_IDLE_SECONDS)
        yield StaleGuardrails(proxy, pooled)


def _cut_once(sents: tuple[Sent, ...]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted((sent.marker, True) for sent in sents))


def _cut_once_then_landed(sents: tuple[Sent, ...]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted(chain.from_iterable(((sent.marker, False), (sent.marker, True)) for sent in sents)))


def _landed_on(arrivals: tuple[Arrival, ...]) -> frozenset[int]:
    return frozenset(arrival.connection for arrival in arrivals if not arrival.dropped)


def test_retried_check_cut_on_a_stale_pooled_connection_lands_on_a_connection_that_was_never_pooled(
    stale: StaleGuardrails,
) -> None:
    sents, arrivals = stale.proxy.probe(RETRIED)

    assert answered(sents) == expected_answers(stale.proxy.rows[RETRIED], sents), (sents, arrivals)
    assert landed(arrivals) == _cut_once_then_landed(sents), arrivals
    assert _landed_on(arrivals).isdisjoint(stale.pooled[RETRIED]), (arrivals, stale.pooled)
    assert spend_statuses(sents) == ("success",) * len(sents), sents


def test_check_cut_on_a_stale_pooled_connection_is_not_resent_when_num_retries_is_unset(
    stale: StaleGuardrails,
) -> None:
    sents, arrivals = stale.proxy.probe(UNSET)

    assert [sent.answer.status == 200 for sent in sents] == [False] * len(sents), sents
    assert all("Generic Guardrail API failed" in sent.answer.text for sent in sents), sents
    assert landed(arrivals) == _cut_once(sents), arrivals
