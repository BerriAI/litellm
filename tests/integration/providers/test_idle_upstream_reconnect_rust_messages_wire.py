from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.idle_gateway import Drop
from integration._support.process import graceful_stop_seconds
from integration._support.routing import assert_routing_matches_policy, rule_aware_overrides, trace_path
from integration.providers._idle_upstream_support import (
    IDLE_SECONDS,
    Row,
    StalePools,
    answered,
    anthropic_deployment,
    expected_answers,
    landed,
    landed_once_undropped,
    messages_sdk,
    messages_sdk_stream,
    spend_statuses,
    stale_pools,
)

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + IDLE_SECONDS))

ROWS: Final = (
    Row("messages-anthropic-sdk-sync", anthropic_deployment, messages_sdk),
    Row("messages-anthropic-sdk-async-stream", anthropic_deployment, messages_sdk_stream),
    Row(
        "messages-upstream-resets-stale-connection",
        anthropic_deployment,
        messages_sdk,
        drop=Drop.RESET,
    ),
)


@pytest.fixture(scope="module")
def pools(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[StalePools]:
    directory: Final = tmp_path_factory.mktemp("idle-upstream-rust")
    with stale_pools(session_gateway, directory, ROWS, rule_aware_overrides(directory, rust=True)) as value:
        yield value


@pytest.mark.parametrize("name", [row.name for row in ROWS])
def test_messages_after_upstream_idle_follow_route_policy_and_open_fresh_worker_connections(
    pools: StalePools, name: str
) -> None:
    row: Final = pools.rows[name]
    sents, arrivals = pools.probe(name)
    assert answered(sents) == expected_answers(row, sents), sents
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
    for sent in sents:
        assert_routing_matches_policy(
            {
                "x-integration-routing-id": sent.answer.routing_id,
                "x-litellm-rust": "true" if sent.answer.rust else "false",
            },
            trace_path(pools.proxy.trace_dir, sent.answer.routing_id),
        )
    assert spend_statuses(sents) == ("success",) * len(sents), sents
