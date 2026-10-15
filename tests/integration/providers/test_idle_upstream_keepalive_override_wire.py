from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.process import graceful_stop_seconds
from integration.providers._idle_upstream_support import (
    IDLE_SECONDS,
    Row,
    StalePools,
    answered,
    anthropic_deployment,
    chat_sdk,
    connections,
    expected_answers,
    landed,
    landed_once_undropped,
    messages_sdk,
    ocr,
    ocr_deployment,
    openai_deployment,
    served_by_rust,
    stale_pools,
)

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + IDLE_SECONDS))

ROWS: Final = (
    Row("override-chat-openai-sdk-sync", openai_deployment, chat_sdk),
    Row("override-messages-rust-anthropic-sdk-sync", anthropic_deployment, messages_sdk, rust=True),
    Row("override-ocr-mistral-rust-bridge", ocr_deployment, ocr, rust=True),
)


@pytest.fixture(scope="module")
def pools(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[StalePools]:
    with stale_pools(
        session_gateway,
        tmp_path_factory.mktemp("idle-upstream-override"),
        ROWS,
        {"AIOHTTP_KEEPALIVE_TIMEOUT": "120", "LITELLM_RUST": "1"},
        drop_after=None,
    ) as value:
        yield value


@pytest.mark.parametrize("name", [row.name for row in ROWS])
def test_explicit_keepalive_timeout_keeps_reusing_the_pooled_connection_past_the_default(
    pools: StalePools, name: str
) -> None:
    row: Final = pools.rows[name]
    sents, arrivals = pools.probe(name)
    assert answered(sents) == expected_answers(row, sents), sents
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
    assert connections(arrivals) == connections(pools.warmed[name]), (arrivals, pools.warmed[name])
    assert served_by_rust(sents) == (row.rust,) * len(sents), sents
