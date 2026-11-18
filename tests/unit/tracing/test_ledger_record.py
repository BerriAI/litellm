import hashlib
import json
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm.tracing.ledger_record import lens_export, safe_lens_export
from litellm.tracing.types import LedgerCapabilities, LensExport

START: Final = datetime(2026, 11, 18, 12, 0, 0, tzinfo=timezone.utc)
END: Final = datetime(2026, 11, 18, 12, 0, 1, tzinfo=timezone.utc)
INLINE: Final = LedgerCapabilities(versions=(1,), payload_store=False, inline_bytes=1024, preview_bytes=256)
STORE: Final = LedgerCapabilities(versions=(1,), payload_store=True, inline_bytes=1024, preview_bytes=256)


def _accounting(spend: float = 0.25) -> dict[str, object]:
    return {"request_id": "accounting-request", "spend": spend, "call_type": "acompletion", "team_id": "team"}


def _kwargs(messages: object = "hello", response: object = "world", estimate: float = 9.0) -> dict[str, object]:
    return {
        "response_cost": estimate,
        "standard_logging_object": {
            "id": "chatcmpl-1",
            "litellm_call_id": "call-1",
            "call_type": "acompletion",
            "response_cost": estimate,
            "status": "success",
            "metadata": {"user_api_key_hash": "hash", "user_api_key_team_id": "team"},
            "messages": messages,
            "response": response,
        },
    }


def _export(
    kwargs: dict[str, object],
    capabilities: LedgerCapabilities = INLINE,
    *,
    keep_content: bool = True,
    payload_limit: int = 1024 * 1024,
) -> LensExport:
    return lens_export(
        _accounting(),  # pyright: ignore[reportArgumentType]  # minimal accounting payload for the fields used
        kwargs,
        start_time=START,
        end_time=END,
        keep_content=keep_content,
        capabilities=capabilities,
        payload_limit=payload_limit,
    )


def test_amount_and_request_id_come_from_the_accepted_delivery_not_the_callback_estimate() -> None:
    record = _export(_kwargs(estimate=9.0))["record"]

    assert record["spend"] == 0.25
    assert record["request_id"] == "accounting-request"
    assert record["litellm_call_id"] == "call-1"
    assert record["response_id"] == "chatcmpl-1"
    assert record["start_time"] == int(START.timestamp() * 1000)
    assert record["end_time"] == int(END.timestamp() * 1000)


def test_disabled_logging_exports_the_cost_without_request_content() -> None:
    export = _export(_kwargs(messages="secret prompt", response="secret answer"), STORE, keep_content=False)

    assert export["record"]["messages"] == ""
    assert export["record"]["response"] == ""
    assert export["record"]["metadata"] == ""
    assert export["record"]["spend"] == 0.25
    assert export["payloads"] == ()


def test_small_bodies_stay_inline_without_uploads() -> None:
    export = _export(_kwargs(messages="hello", response="world"), STORE)

    assert export["record"]["messages"] == "hello"
    assert export["record"]["response"] == "world"
    assert export["payloads"] == ()


def test_large_body_with_a_store_uploads_the_full_bytes_and_keeps_only_a_preview() -> None:
    body = "雪" * 2000
    export = _export(_kwargs(messages=body), STORE)

    [payload] = export["payloads"]
    stored = str(export["record"]["messages"])
    assert payload["field"] == "messages"
    assert payload["content"] == body
    assert payload["bytes"] == len(body.encode())
    assert payload["sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert payload["team_id"] == "team"
    assert payload["api_key"] == "hash"
    assert payload["event_time_ms"] == int(START.timestamp() * 1000)
    assert len(stored.encode()) <= STORE.preview_bytes
    assert "bytes omitted]" in stored
    assert body.startswith(stored.split("…[Lens preview: ")[0])


@pytest.mark.parametrize(
    ("capabilities", "payload_limit"),
    [(INLINE, 1024 * 1024), (STORE, 100)],
    ids=["spillover_disabled", "above_upload_limit"],
)
def test_large_body_without_an_upload_becomes_a_marked_preview_within_the_inline_limit(
    capabilities: LedgerCapabilities, payload_limit: int
) -> None:
    body = json.dumps([{"role": "user", "content": "雪" * 2000}], ensure_ascii=False)
    export = _export(
        _kwargs(messages=[{"role": "user", "content": "雪" * 2000}]), capabilities, payload_limit=payload_limit
    )

    stored = str(export["record"]["messages"])
    assert export["payloads"] == ()
    assert len(stored.encode()) <= capabilities.inline_bytes
    assert json.loads(stored)[0]["role"] == "user"
    assert "bytes omitted]" in stored
    assert len(body.encode()) > capabilities.inline_bytes


def test_nul_characters_cannot_roll_back_the_charge_transaction() -> None:
    record = _export(_kwargs(messages="a\x00b"))["record"]

    assert "\x00" not in str(record["messages"])


def test_trace_ingest_requests_fall_back_to_the_accounting_row() -> None:
    kwargs = _kwargs()
    kwargs["standard_logging_object"] = {**dict(kwargs["standard_logging_object"]), "call_type": "/v1/traces"}  # pyright: ignore[reportArgumentType]  # test-built mapping

    record = _export(kwargs)["record"]

    assert record["request_id"] == "accounting-request"
    assert record["messages"] == ""
    assert "litellm_call_id" not in record


def test_disabled_logging_also_drops_error_strings() -> None:
    kwargs = _kwargs()
    kwargs["standard_logging_object"] = {**dict(kwargs["standard_logging_object"]), "error_str": "input: secret"}  # pyright: ignore[reportArgumentType]  # test-built mapping

    record = _export(kwargs, keep_content=False)["record"]

    assert record["error_str"] == ""


def test_message_logging_off_redacts_echoed_input_from_error_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    leaked = "Invalid request: [{'type': 'missing', 'input': [{'role': 'user', 'content': 'secret prompt'}]}]"
    kwargs = _kwargs()
    kwargs["standard_logging_object"] = {**dict(kwargs["standard_logging_object"]), "error_str": leaked}  # pyright: ignore[reportArgumentType]  # test-built mapping

    record = _export(kwargs)["record"]

    assert record["messages"] == ""
    assert "secret prompt" not in str(record["error_str"])


def test_oversized_metadata_is_bounded_to_a_marked_preview() -> None:
    kwargs = _kwargs()
    metadata = {**dict(dict(kwargs["standard_logging_object"])["metadata"]), "huge": "x" * 10_000}  # pyright: ignore[reportArgumentType]  # test-built mapping
    kwargs["standard_logging_object"] = {**dict(kwargs["standard_logging_object"]), "metadata": metadata}  # pyright: ignore[reportArgumentType]  # test-built mapping

    record = _export(kwargs, STORE)["record"]

    assert len(str(record["metadata"]).encode()) <= STORE.inline_bytes
    assert "bytes omitted]" in str(record["metadata"])


def test_upload_limit_counts_json_escaping_not_raw_bytes() -> None:
    body = '"' * 2000
    raw_and_escaped = (len(body.encode()), len(json.dumps(body, ensure_ascii=False).encode()))

    export = _export(_kwargs(messages=body), STORE, payload_limit=3000)

    assert raw_and_escaped[0] < 3000 < raw_and_escaped[1]
    assert export["payloads"] == ()
    assert "bytes omitted]" in str(export["record"]["messages"])


def test_a_request_the_row_builder_cannot_read_still_exports_its_cost() -> None:
    kwargs = {**_kwargs(), "litellm_params": 5}

    export = safe_lens_export(
        _accounting(),  # pyright: ignore[reportArgumentType]  # minimal accounting payload for the fields used
        kwargs,
        start_time=START,
        end_time=END,
        keep_content=True,
        capabilities=STORE,
    )

    assert export["record"]["spend"] == 0.25
    assert export["record"]["request_id"] == "accounting-request"
    assert export["record"]["messages"] == ""
    assert export["payloads"] == ()
