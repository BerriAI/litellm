import json
from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.tracing.ledger_envelope import committed_lens_export

START: Final = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
ACCOUNTING: Final = {
    "request_id": "request",
    "spend": 0.25,
    "startTime": START.isoformat(),
    "team_id": "team",
    "api_key": "hashed-key",
    "user": "user",
    "prompt_tokens": 12,
    "metadata": {"secret": "never exported"},
}
ENVELOPE: Final = {
    "record": {"request_id": "request", "spend": 0.25, "messages": "preview"},
    "payloads": [
        {
            "field": "messages",
            "team_id": "team",
            "api_key": "hashed-key",
            "event_time_ms": 1,
            "content": "[]",
            "bytes": 2,
            "sha256": "0" * 64,
        }
    ],
}


@pytest.mark.parametrize("encoded", [False, True], ids=["native_mapping", "queue_json_text"])
def test_the_envelope_reads_the_same_from_a_mapping_or_queue_json(encoded: bool) -> None:
    value = json.dumps(ENVELOPE) if encoded else ENVELOPE

    export = committed_lens_export(value, request_id="request", accounting=ACCOUNTING)

    assert export is not None
    assert export["record"] == ENVELOPE["record"]
    assert [dict(body) for body in export["payloads"]] == ENVELOPE["payloads"]


@pytest.mark.parametrize(
    "malformed",
    ["{not json", json.dumps({"record": {}}), {"record": {}, "payloads": [{"field": "other"}]}],
    ids=["broken_json", "missing_payloads", "invalid_body"],
)
def test_a_malformed_envelope_falls_back_to_the_committed_cost_without_content(malformed: object) -> None:
    export = committed_lens_export(malformed, request_id="request", accounting=ACCOUNTING)

    assert export is not None
    assert export["payloads"] == ()
    assert export["record"] == {
        "api_key": "hashed-key",
        "team_id": "team",
        "user": "user",
        "prompt_tokens": 12,
        "messages": "",
        "response": "",
        "metadata": "",
        "error_str": "",
        "request_id": "request",
        "spend": 0.25,
        "start_time": int(START.timestamp() * 1000),
        "end_time": int(START.timestamp() * 1000),
    }


@pytest.mark.parametrize(
    ("change", "expected_spend"),
    [({"spend": True}, None), ({"spend": float("inf")}, None), ({"spend": 0}, 0.0)],
    ids=["boolean", "infinite", "zero"],
)
def test_fallback_spend_keeps_only_finite_numbers(change: dict[str, object], expected_spend: float | None) -> None:
    export = committed_lens_export("{", request_id="request", accounting={**ACCOUNTING, **change})

    assert export is not None
    assert export["record"]["spend"] == expected_spend


@pytest.mark.parametrize("start", [None, "yesterday", "1970-01-01T00:00:00+00:00"], ids=["missing", "invalid", "epoch"])
def test_fallback_without_a_usable_event_time_exports_nothing(start: object) -> None:
    assert committed_lens_export("{", request_id="request", accounting={**ACCOUNTING, "startTime": start}) is None
