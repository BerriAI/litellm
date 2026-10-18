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
    chat_with_guardrail,
    expected_answers,
    guardrail_deployment,
    landed,
    landed_once_undropped,
    messages_sdk,
    messages_sdk_stream,
    ocr,
    ocr_deployment,
    pass_through,
    pass_through_deployment,
    served_by_rust,
    spend_statuses,
    stale_pools,
    transcription,
    transcription_deployment,
)

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + IDLE_SECONDS))

ROWS: Final = (
    Row("messages-anthropic-sdk-sync", anthropic_deployment, messages_sdk),
    Row("messages-anthropic-sdk-async-stream", anthropic_deployment, messages_sdk_stream),
    Row("messages-upstream-resets-stale-connection", anthropic_deployment, messages_sdk, drop=Drop.RESET),
    Row("custom-pass-through-raw-httpx", pass_through_deployment, pass_through),
    Row("generic-guardrail-api-pre-call", guardrail_deployment, chat_with_guardrail),
    Row("ocr-mistral-rust-bridge", ocr_deployment, ocr, rust=True),
    Row("audio-transcription-bedrock-rust-bridge", transcription_deployment, transcription, rust=True),
)


@pytest.fixture(scope="module")
def pools(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[StalePools]:
    with stale_pools(session_gateway, tmp_path_factory.mktemp("idle-upstream-routes"), ROWS, {}) as value:
        yield value


@pytest.mark.parametrize("name", [row.name for row in ROWS])
def test_route_after_upstream_idle_past_keepalive_opens_a_fresh_connection_on_each_worker(
    pools: StalePools, name: str
) -> None:
    row: Final = pools.rows[name]
    sents, arrivals = pools.probe(name)
    assert answered(sents) == expected_answers(row, sents), sents
    assert landed(arrivals) == landed_once_undropped(sents), arrivals
    assert served_by_rust(sents) == (row.rust,) * len(sents), sents
    assert spend_statuses(sents) == ("success",) * len(sents), sents
