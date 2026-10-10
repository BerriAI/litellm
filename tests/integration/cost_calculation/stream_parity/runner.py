import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.cost_calculation.stream_parity.case import MODEL, RESPONSE_ID, StreamParityTestCase
from pydantic import JsonValue


def _fill(value: JsonValue, substitutions: Mapping[str, str]) -> JsonValue:
    if isinstance(value, str):
        return next((filled for pattern, filled in substitutions.items() if value == pattern), value)
    if isinstance(value, dict):
        return {name: _fill(item, substitutions) for name, item in value.items()}
    if isinstance(value, list):
        return [_fill(item, substitutions) for item in value]
    return value


def _fill_object(value: Mapping[str, JsonValue], substitutions: Mapping[str, str]) -> dict[str, JsonValue]:
    return {name: _fill(item, substitutions) for name, item in value.items()}


def _sse(chunk: Mapping[str, JsonValue]) -> bytes:
    event: Final = f"event: {chunk['type']}\n" if "type" in chunk else ""
    return f"{event}data: {json.dumps(chunk)}\n\n".encode()


def _stream_reply(case: StreamParityTestCase, substitutions: Mapping[str, str]) -> Reply:
    chunks: Final = tuple(_sse(_fill_object(chunk, substitutions)) for chunk in case.mock_provider_stream)
    named_events: Final = any("type" in chunk for chunk in case.mock_provider_stream)
    return Reply(content_type="text/event-stream", chunks=chunks if named_events else (*chunks, b"data: [DONE]\n\n"))


def _call(
    case: StreamParityTestCase, gateway: Gateway, provider: SharedProvider, model: str, key: str, *, streamed: bool
) -> str:
    substitutions: Final = {MODEL: model, RESPONSE_ID: uuid4().hex}
    provider.expect(
        _stream_reply(case, substitutions)
        if streamed
        else Reply(body=json.dumps(_fill_object(case.mock_provider_response, substitutions)).encode())
    )
    body: Final = _fill_object({**case.litellm_request, **(case.stream_parameters if streamed else {})}, substitutions)
    response: Final = gateway.request("POST", case.litellm_endpoint, body, key=key)
    assert response.status_code == 200, response.text
    return response.headers["x-litellm-call-id"]


def _spend_row(rows: list[dict[str, JsonValue]], call_id: str, columns: tuple[str, ...]) -> dict[str, JsonValue]:
    matching: Final = [row for row in rows if row["litellm_call_id"] == call_id]
    assert len(matching) == 1, (call_id, rows)
    return {column: float(str(matching[0][column])) if column == "spend" else matching[0][column] for column in columns}


def assert_stream_parity(case: StreamParityTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(**case.deployment)
        key: Final = scenario.key(models=[model])
        plain_call: Final = _call(case, gateway, provider, model, key, streamed=False)
        streamed_call: Final = _call(case, gateway, provider, model, key, streamed=True)
        received: Final = [(sent.method, sent.target) for sent in provider.received()]
        assert received == [("POST", case.expected_provider_endpoint)] * 2, received

        key_hash: Final = sha256(key.encode()).hexdigest()
        expected: Final = {**_fill_object(case.expected_spend_row, {MODEL: model}), "api_key": key_hash}
        columns: Final = tuple(expected)
        rows: Final = eventually(
            lambda: read_rows(
                f'SELECT litellm_call_id, {", ".join(columns)} FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (key_hash,)
            ),
            lambda found: {str(row["litellm_call_id"]) for row in found} >= {plain_call, streamed_call},
            seconds=70,
        )
        approximate: Final = {**expected, "spend": pytest.approx(expected["spend"])}
        assert _spend_row(rows, plain_call, columns) == approximate, "plain request"
        assert _spend_row(rows, streamed_call, columns) == approximate, "streamed request"
