"""Every response.create turn on a native Responses WebSocket is its own spend row.

OpenAI bills each turn as a request, so one socket carrying two turns has to leave two
rows, each keyed by the response id the client saw and priced from that turn's own usage,
instead of one row at socket close that sums both (LIT-8949). A turn the provider fails
leaves a failure row of its own and does not swallow the completed turn before it.
"""

import asyncio
import hashlib
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_stream import RATE_LIMIT_MESSAGE, openai_error_event
from integration._support.responses_ws_peer import PROVIDER_MODEL, USAGE, completed, events, responses_peer, session
from integration._support.tls import write_self_signed_cert
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 180), pytest.mark.shares_worker_state]

INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
TURN_SPEND: Final = USAGE["input_tokens"] * INPUT_RATE + USAGE["output_tokens"] * OUTPUT_RATE
SPEND_ROWS_FOR_KEY: Final = (
    'SELECT request_id, spend, total_tokens, call_type, status, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s'
)


def _second_turn_rate_limited(response_id: str) -> tuple[dict[str, JsonValue], ...]:
    if response_id == "resp_peer_1":
        return events(response_id, "seven")
    return (events(response_id, "")[0], dict(openai_error_event("rate_limit_exceeded", RATE_LIMIT_MESSAGE)))


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("responses-ws-rows-cert"))


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        with owned_proxy(base, tmp_path_factory.mktemp("responses-ws-rows"), {"SSL_CERT_FILE": str(cert[0])}) as proxy:
            yield proxy


def test_two_turns_on_one_socket_leave_two_rows_keyed_by_the_response_ids_the_client_saw(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{PROVIDER_MODEL}",
            api_base=peer.url,
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
        )
        key: Final = scenario.key(models=[model])
        texts: Final = (f"first {uuid.uuid4().hex}", f"second {uuid.uuid4().hex}")

        first, second = asyncio.run(session(str(candidate.client.base_url), key, model, texts))

        response_ids: Final = {str(completed(first)["id"]), str(completed(second)["id"])}
        assert len(response_ids) == 2, response_ids
        rows: Final = eventually(
            lambda: read_rows(SPEND_ROWS_FOR_KEY, (_key_hash(key),)),
            lambda found: len(found) >= 2,
            seconds=70,
        )
        assert {str(row["request_id"]) for row in rows} == response_ids, rows
        assert [float(row["spend"]) for row in rows] == [pytest.approx(TURN_SPEND)] * 2, rows
        assert {(row["call_type"], row["status"], row["total_tokens"]) for row in rows} == {
            ("_aresponses_websocket", "success", USAGE["total_tokens"])
        }, rows


def test_a_turn_the_provider_rejects_leaves_its_own_failure_row_next_to_the_completed_turn(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert, _second_turn_rate_limited) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{PROVIDER_MODEL}",
            api_base=peer.url,
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
        )
        key: Final = scenario.key(models=[model])
        texts: Final = (f"first {uuid.uuid4().hex}", f"second {uuid.uuid4().hex}")

        first, second = asyncio.run(session(str(candidate.client.base_url), key, model, texts))

        assert second[-1]["type"] == "error", second
        rows: Final = eventually(
            lambda: read_rows(SPEND_ROWS_FOR_KEY, (_key_hash(key),)),
            lambda found: len(found) >= 2,
            seconds=70,
        )
        by_status: Final = {str(row["status"]): row for row in rows}
        assert set(by_status) == {"success", "failure"}, rows
        assert str(by_status["success"]["request_id"]) == str(completed(first)["id"]), rows
        assert float(by_status["success"]["spend"]) == pytest.approx(TURN_SPEND), rows
        assert float(by_status["failure"]["spend"]) == 0.0, rows
        error: Final = object_value(object_value(by_status["failure"]["metadata"])["error_information"])
        assert str(error["error_code"]) == "429", error
        assert RATE_LIMIT_MESSAGE in string_value(error["error_message"]), error
