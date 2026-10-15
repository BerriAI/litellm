from itertools import chain
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.idle_gateway import Arrival
from integration._support.process import graceful_stop_seconds
from integration.providers._idle_upstream_support import (
    REUSE_IDLE_SECONDS,
    Row,
    Sent,
    answered,
    chat_with_guardrail,
    expected_answers,
    guardrail_deployment,
    landed,
    spend_statuses,
    stay_idle,
    upstream_proxy,
)

CUT_AFTER_SECONDS: Final = 3.0

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + REUSE_IDLE_SECONDS))

GUARDRAIL_ROW: Final = Row("guardrail-short-lb-cut", guardrail_deployment, chat_with_guardrail)


def _cut_once_then_landed(sents: tuple[Sent, ...]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted(chain.from_iterable(((sent.marker, False), (sent.marker, True)) for sent in sents)))


def _connections_per_check(sents: tuple[Sent, ...], arrivals: tuple[Arrival, ...]) -> tuple[int, ...]:
    return tuple(len({arrival.connection for arrival in arrivals if arrival.marker == sent.marker}) for sent in sents)


def test_guardrail_check_cut_on_an_idle_pooled_connection_is_resent_on_each_worker(
    gateway: Gateway, tmp_path: Path
) -> None:
    with upstream_proxy(gateway, tmp_path, (GUARDRAIL_ROW,), {}, drop_after=CUT_AFTER_SECONDS) as proxy:
        warm, _ = proxy.probe(GUARDRAIL_ROW.name)
        assert answered(warm) == expected_answers(GUARDRAIL_ROW, warm), warm
        stay_idle((proxy.gateways[GUARDRAIL_ROW.name],), REUSE_IDLE_SECONDS)
        sents, arrivals = proxy.probe(GUARDRAIL_ROW.name)
        assert answered(sents) == expected_answers(GUARDRAIL_ROW, sents), (sents, arrivals)
        assert landed(arrivals) == _cut_once_then_landed(sents), arrivals
        assert _connections_per_check(sents, arrivals) == (2,) * len(sents), arrivals
        assert spend_statuses(sents) == ("success",) * len(sents), sents
