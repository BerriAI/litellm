from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.process import graceful_stop_seconds
from integration.providers._idle_upstream_support import (
    QUIET_STREAM_SECONDS,
    REUSE_IDLE_SECONDS,
    Row,
    UpstreamProxy,
    answered,
    anthropic_deployment,
    chat_sdk,
    chat_sdk_stream,
    connections,
    expected_answers,
    landed,
    landed_once_undropped,
    messages_sdk,
    messages_sdk_stream,
    openai_deployment,
    quiet_stream_answer,
    served_by_rust,
    spend_statuses,
    stale_pools,
    stay_idle,
    upstream_proxy,
)

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + 2 * QUIET_STREAM_SECONDS))

REUSE_ROWS: Final = (
    Row("reuse-chat-openai-sdk-sync", openai_deployment, chat_sdk),
    Row("reuse-messages-rust-anthropic-sdk-sync", anthropic_deployment, messages_sdk, rust=True),
)
QUIET_ROWS: Final = (
    Row("quiet-chat-openai-sdk-async-stream", openai_deployment, chat_sdk_stream, respond=quiet_stream_answer),
    Row(
        "quiet-messages-rust-anthropic-sdk-async-stream",
        anthropic_deployment,
        messages_sdk_stream,
        rust=True,
        respond=quiet_stream_answer,
    ),
)
MALFORMED_ROW: Final = Row("malformed-chat-openai-sdk-sync", openai_deployment, chat_sdk)


@pytest.fixture(scope="module")
def edges(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[UpstreamProxy]:
    with upstream_proxy(
        session_gateway,
        tmp_path_factory.mktemp("idle-upstream-edges"),
        (*REUSE_ROWS, *QUIET_ROWS),
        {"LITELLM_RUST": "1"},
        drop_after=None,
    ) as proxy:
        yield proxy


@pytest.mark.parametrize("name", [row.name for row in REUSE_ROWS])
def test_request_idle_under_the_keepalive_reuses_the_pooled_connection_on_each_worker(
    edges: UpstreamProxy, name: str
) -> None:
    row: Final = edges.rows[name]
    warm, warmed = edges.probe(name)
    stay_idle((edges.gateways[name],), REUSE_IDLE_SECONDS)
    probe, arrivals = edges.probe(name)
    sents: Final = (*warm, *probe)
    assert answered(sents) == expected_answers(row, sents), sents
    assert landed((*warmed, *arrivals)) == landed_once_undropped(sents), (warmed, arrivals)
    assert connections(arrivals) == connections(warmed), (warmed, arrivals)
    assert served_by_rust(sents) == (row.rust,) * len(sents), sents


@pytest.mark.parametrize("name", [row.name for row in QUIET_ROWS])
def test_stream_quiet_for_longer_than_the_keepalive_completes_on_each_worker(edges: UpstreamProxy, name: str) -> None:
    row: Final = edges.rows[name]
    sents, arrivals = edges.probe(name)
    assert answered(sents) == expected_answers(row, sents), sents
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
    assert served_by_rust(sents) == (row.rust,) * len(sents), sents
    assert spend_statuses(sents) == ("success",) * len(sents), sents


@pytest.mark.parametrize("value", ["fifteen", ""], ids=["word", "empty"])
def test_malformed_keepalive_timeout_boots_with_the_default_keepalive(
    gateway: Gateway, tmp_path: Path, value: str
) -> None:
    with stale_pools(gateway, tmp_path, (MALFORMED_ROW,), {"AIOHTTP_KEEPALIVE_TIMEOUT": value}) as pools:
        sents, arrivals = pools.probe(MALFORMED_ROW.name)
    assert answered(sents) == expected_answers(MALFORMED_ROW, sents), sents
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
