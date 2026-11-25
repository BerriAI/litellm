from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.idle_gateway import Drop
from integration._support.process import graceful_stop_seconds
from integration.providers._idle_upstream_support import (
    IDLE_SECONDS,
    Row,
    StalePools,
    answered,
    anthropic_deployment,
    chat_httpx,
    chat_sdk,
    chat_sdk_stream,
    chat_sdk_with_default_retries,
    embedding_deployment,
    embeddings_sdk,
    expected_answers,
    expected_embedding,
    landed,
    landed_once_undropped,
    openai_deployment,
    responses_sdk,
    responses_sdk_stream,
    served_by_rust,
    spend_statuses,
    stale_pools,
)

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + IDLE_SECONDS))

ROWS: Final = (
    Row("chat-openai-sdk-sync", openai_deployment, chat_sdk),
    Row("chat-openai-sdk-async-stream", openai_deployment, chat_sdk_stream),
    Row("chat-raw-httpx", openai_deployment, chat_httpx),
    Row("chat-to-anthropic-deployment", anthropic_deployment, chat_sdk),
    Row("embeddings-openai-sdk", embedding_deployment, embeddings_sdk, expected=expected_embedding),
    Row("responses-openai-sdk-sync", openai_deployment, responses_sdk),
    Row("responses-openai-sdk-async-stream", openai_deployment, responses_sdk_stream),
    Row("chat-upstream-resets-stale-connection", openai_deployment, chat_sdk, drop=Drop.RESET),
    Row("chat-with-default-router-retries", openai_deployment, chat_sdk_with_default_retries),
)


@pytest.fixture(scope="module")
def pools(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[StalePools]:
    with stale_pools(session_gateway, tmp_path_factory.mktemp("idle-upstream"), ROWS, {}) as value:
        yield value


@pytest.mark.parametrize("name", [row.name for row in ROWS])
def test_request_after_upstream_idle_past_keepalive_opens_a_fresh_connection_on_each_worker(
    pools: StalePools, name: str
) -> None:
    row: Final = pools.rows[name]
    sents, arrivals = pools.probe(name)
    assert answered(sents) == expected_answers(row, sents), sents
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
    assert served_by_rust(sents) == (row.rust,) * len(sents), sents
    assert spend_statuses(sents) == ("success",) * len(sents), sents
