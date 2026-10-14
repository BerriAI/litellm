from contextlib import ExitStack
from dataclasses import replace
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.idle_gateway import idle_gateway
from integration._support.process import graceful_stop_seconds
from integration.providers._idle_upstream_support import (
    DROP_AFTER_SECONDS,
    IDLE_SECONDS,
    REUSE_IDLE_SECONDS,
    Row,
    answer,
    answered,
    anthropic_deployment,
    burst_answers,
    burst_expected,
    chat_sdk,
    chat_sdk_stream,
    drain_all,
    expected_answers,
    landed,
    landed_once_undropped,
    messages_sdk,
    messages_sdk_stream,
    on_each_worker,
    openai_deployment,
    proxy_for,
    responses_sdk,
    responses_sdk_stream,
    send,
    spend_statuses,
    stale_pools,
    stay_idle,
    two_workers,
    upstream_proxy,
    warm_and_idle,
)

BURST_EACH: Final = 4
BURST_ROWS: Final = (
    Row("burst-chat-openai-sdk-sync", openai_deployment, chat_sdk),
    Row("burst-chat-openai-sdk-async-stream", openai_deployment, chat_sdk_stream),
    Row("burst-messages-anthropic-sdk-sync", anthropic_deployment, messages_sdk),
    Row("burst-messages-anthropic-sdk-async-stream", anthropic_deployment, messages_sdk_stream),
    Row("burst-responses-openai-sdk-sync", openai_deployment, responses_sdk),
    Row("burst-responses-openai-sdk-async-stream", openai_deployment, responses_sdk_stream),
)
KILL_ROWS: Final = (
    Row("kill-chat-openai-sdk-sync", openai_deployment, chat_sdk),
    Row("kill-messages-anthropic-sdk-sync", anthropic_deployment, messages_sdk),
)
RESTART_ROWS: Final = (
    Row("restart-chat-openai-sdk-sync", openai_deployment, chat_sdk),
    Row("restart-messages-anthropic-sdk-sync", anthropic_deployment, messages_sdk),
)


@pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + IDLE_SECONDS))
def test_concurrent_mixed_burst_after_upstream_idle_past_keepalive_lands_every_request_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    with stale_pools(gateway, tmp_path, BURST_ROWS, {}) as pools:
        sent: Final = pools.proxy.burst(tuple(row.name for row in BURST_ROWS), BURST_EACH)
        arrivals: Final = drain_all(tuple(pools.proxy.gateways.values()))
        sents: Final = tuple(one for _, one in sent)
        statuses: Final = spend_statuses(sents)
    assert burst_answers(sent) == burst_expected(sent), sent
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
    assert statuses == ("success",) * len(sents), sents


@pytest.mark.timeout(int(3 * graceful_stop_seconds() + 180 + IDLE_SECONDS))
def test_worker_killed_while_the_other_serves_then_both_open_fresh_connections_after_upstream_idle(
    gateway: Gateway, tmp_path: Path
) -> None:
    with upstream_proxy(gateway, tmp_path, KILL_ROWS, {}) as proxy:
        victim, _ = proxy.workers
        with proxy.owned.replacing_worker(victim) as kill:
            kill()
            during: Final = send(KILL_ROWS[0], proxy.target)
        stale: Final = warm_and_idle(replace(proxy, workers=two_workers(proxy.owned)))
        probes: Final = tuple(stale.probe(row.name) for row in KILL_ROWS)
        sents: Final = tuple(chain.from_iterable(probe_sents for probe_sents, _ in probes))
        arrivals: Final = tuple(chain.from_iterable(probe_arrivals for _, probe_arrivals in probes))
        statuses: Final = spend_statuses(sents)
    expected: Final = tuple(
        chain.from_iterable(
            expected_answers(row, probe_sents) for row, (probe_sents, _) in zip(KILL_ROWS, probes, strict=True)
        )
    )
    assert answered((during,)) == expected_answers(KILL_ROWS[0], (during,)), during
    assert answered(sents) == expected, sents
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
    assert statuses == ("success",) * len(sents), sents


@pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + IDLE_SECONDS))
def test_upstream_restarted_on_the_same_port_with_connections_pooled_serves_a_burst_on_fresh_connections(
    gateway: Gateway, tmp_path: Path
) -> None:
    with ExitStack() as stack:
        steady: Final = stack.enter_context(idle_gateway(answer, drop_after=None))
        before_restart: Final = stack.enter_context(ExitStack())
        first: Final = MappingProxyType(
            {
                row.name: before_restart.enter_context(idle_gateway(answer, drop_after=DROP_AFTER_SECONDS))
                for row in RESTART_ROWS
            }
        )
        proxy: Final = stack.enter_context(proxy_for(gateway, tmp_path, RESTART_ROWS, first, steady.url, {}))
        target: Final = proxy.target
        warm: Final = on_each_worker(
            proxy.owned, proxy.workers, lambda: tuple(send(row, target) for row in RESTART_ROWS)
        )
        before_restart.close()
        second: Final = MappingProxyType(
            {
                name: stack.enter_context(idle_gateway(answer, drop_after=DROP_AFTER_SECONDS, port=upstream.port))
                for name, upstream in first.items()
            }
        )
        stay_idle(tuple(second.values()), REUSE_IDLE_SECONDS)
        sent: Final = replace(proxy, gateways=second).burst(tuple(row.name for row in RESTART_ROWS), BURST_EACH)
        arrivals: Final = drain_all(tuple(second.values()))
        sents: Final = tuple(one for _, one in sent)
        statuses: Final = spend_statuses(sents)
    assert tuple(one.answer.status for one in warm) == (200,) * 2 * len(RESTART_ROWS), warm
    assert burst_answers(sent) == burst_expected(sent), sent
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
    assert statuses == ("success",) * len(sents), sents
