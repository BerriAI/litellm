import json
from typing import Final

import pytest
from integration._support.client import (  # pyright: ignore[reportUnknownVariableType]  # integration fixture module
    Gateway,
    eventually,
)
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "gemini/gemini-4-neon"
_KEY: Final = "synthetic-gemini-interactions-long-decode-key"
_JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_FIRST_HOP: Final = {
    "id": "v1_long_decode_first_hop",
    "object": "interaction",
    "status": "requires_continuation",
    "continuation_token": "token-after-first-hop",
    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "The first half, "}]}],
    "usage": {"total_input_tokens": 10, "total_output_tokens": 100, "total_tokens": 110},
}
_SECOND_HOP: Final = {
    "id": "v1_long_decode_second_hop",
    "object": "interaction",
    "status": "completed",
    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "then the second half."}]}],
    "usage": {"total_input_tokens": 10, "total_output_tokens": 50, "total_tokens": 60},
}


def _spend_of(request_id: str) -> list[dict[str, JsonValue]]:  # mutable-ok: read_rows return contract
    return read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,))


def _single_row(rows: list[dict[str, JsonValue]]) -> bool:  # mutable-ok: predicate over read_rows list
    return len(rows) == 1


def _scripted(request: Request) -> Reply:
    assert request.target == "/v1beta/interactions"
    hop: Final = _SECOND_HOP if "continuation_token" in json.loads(request.body) else _FIRST_HOP
    return Reply(body=json.dumps(hop).encode())


def test_gemini_4_chat_answer_google_ends_early_is_continued_into_one_answer_billed_once(gateway: Gateway) -> None:
    with wire_server(_scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base=wire.url,
            api_key=_KEY,
            input_cost_per_token=_INPUT_COST,
            output_cost_per_token=_OUTPUT_COST,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "write it all"}], "max_tokens": 1000},
            key=scenario.key(models=[model]),
        )
        assert response.status_code == 200, response.text
        first_body, second_body = (json.loads(wire.received.get(timeout=5).body) for _ in range(2))
        assert "continuation_token" not in first_body
        assert second_body == {**first_body, "continuation_token": "token-after-first-hop"}
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        choice: Final = payload["choices"][0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["message"]["content"] == "The first half, then the second half."  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["finish_reason"] == "stop"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert payload["id"] == "v1_long_decode_first_hop"
        assert payload["usage"]["completion_tokens"] == 150  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        billed: Final = float(response.headers["x-litellm-response-cost"])
        assert billed == pytest.approx(20 * _INPUT_COST + 150 * _OUTPUT_COST)  # pyright: ignore[reportUnknownMemberType]  # pytest.approx untyped overload
        rows: Final[list[dict[str, JsonValue]]] = eventually(  # mutable-ok: read_rows return contract
            lambda: _spend_of("v1_long_decode_first_hop"), _single_row, seconds=70
        )
        spend: Final = rows[0]["spend"]
        assert isinstance(spend, (int, float, str))
        assert float(spend) == pytest.approx(billed), rows  # pyright: ignore[reportUnknownMemberType]  # pytest.approx untyped overload


def test_gemini_4_chat_answer_with_no_output_cap_is_not_continued_and_returns_its_token(gateway: Gateway) -> None:
    with wire_server(_scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "write it all"}]},
            key=scenario.key(models=[model]),
        )
        assert response.status_code == 200, response.text
        wire.received.get(timeout=5)
        assert wire.received.empty()
        choice: Final = _JSON_OBJECT.validate_json(response.content)["choices"][0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["finish_reason"] == "length"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["message"]["content"] == "The first half, "  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["message"]["provider_specific_fields"]["continuation_token"] == "token-after-first-hop"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
