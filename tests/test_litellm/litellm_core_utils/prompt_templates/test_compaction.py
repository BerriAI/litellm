import json
from copy import deepcopy
from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.compaction import (
    CompactionHistoryError,
    NativeHistory,
    NativeProtocol,
    compaction_headers,
    native_compaction_payload,
    native_compaction_result,
    split_native_history,
)

_USER: Final = {"role": "user", "content": "latest"}
_HISTORY: Final = NativeHistory("input", (), (_USER,))


@pytest.mark.parametrize("protocol", ("responses", "messages"))
def test_native_payload_preserves_source_and_only_forwards_supported_parameters(protocol: NativeProtocol) -> None:
    field: Final = "input" if protocol == "responses" else "messages"
    older: Final = {"role": "user", "content": [{"type": "input_file", "file_id": "opaque-file"}]}
    state: Final = {"type": "compaction", "encrypted_content": "opaque"}
    payload: Final = {
        field: [older, state, _USER],
        "instructions": "rules",
        "system": "system",
        "tools": [{"name": "tool"}],
        "temperature": 1,
        "max_tokens": 1,
        "extra_headers": {"Anthropic-Beta": "other", "x-test": "keep"},
    }
    original: Final = deepcopy(payload)
    history: Final = split_native_history(payload, protocol)
    assert isinstance(history, NativeHistory)
    assert (history.field, history.prefix, history.tail) == (field, (older, state), (_USER,))
    request: Final = native_compaction_payload(history, payload, protocol)
    assert request[field] == [older, state]
    if protocol == "responses":
        assert request == {"input": [older, state], "instructions": "rules"}
    else:
        assert request == {
            "messages": [older, state],
            "system": "system",
            "tools": [{"name": "tool"}],
            "compaction": {"type": "summarize"},
            "max_tokens": 4096,
            "extra_headers": compaction_headers(payload["extra_headers"]),
        }
    request[field][0]["content"][0]["file_id"] = "changed-private-copy"
    assert payload == original and history.prefix[0] == older


@pytest.mark.parametrize("mixed", (False, True))
def test_messages_tool_result_is_not_a_new_request(mixed: bool) -> None:
    older: Final = {"role": "user", "content": "old"}
    call: Final = {"role": "assistant", "content": [{"type": "tool_use", "id": "call"}]}
    block: Final = {"type": "tool_result", "tool_use_id": "call", "content": "ok"}
    result: Final = {"role": "user", "content": [block, *([{"type": "text", "text": "next"}] if mixed else [])]}
    history: Final = split_native_history({"messages": [older, _USER, call, result]}, "messages")
    assert isinstance(history, NativeHistory)
    assert history.prefix == ((older, _USER, call) if mixed else (older,))
    assert history.tail == ((result,) if mixed else (_USER, call, result))


@pytest.mark.parametrize("items", (None, [], ["invalid"], [{"role": "assistant", "content": "none"}], [_USER]))
@pytest.mark.parametrize("protocol", ("responses", "messages"))
def test_history_requires_older_prefix(items: object, protocol: NativeProtocol) -> None:
    result: Final = split_native_history({"input" if protocol == "responses" else "messages": items}, protocol)
    assert isinstance(result, CompactionHistoryError) and result.message


@pytest.mark.parametrize("field", ("previous_response_id", "conversation"))
def test_unresolved_history_rejected(field: str) -> None:
    result: Final = split_native_history({"input": [_USER, _USER], field: "server-state"}, "responses")
    assert isinstance(result, CompactionHistoryError) and "Server-side" in result.message


def test_responses_retains_canonical_window_and_only_removes_null_compaction_created_by() -> None:
    retained: Final = {"type": "message", "role": "user", "content": "retained", "created_by": None}
    compacted: Final = {"type": "compaction", "encrypted_content": " opaque ", "created_by": None, "other": None}
    extra: Final = {**compacted, "created_by": "keep"}
    unknown: Final = {"type": "future_item", "metadata": {"created_by": None}}
    response: Final = {"output": [retained, compacted, extra, unknown]}
    original: Final = deepcopy(response)
    result: Final = native_compaction_result(response, _HISTORY, "responses")
    assert result == (
        retained,
        {"type": "compaction", "encrypted_content": " opaque ", "other": None},
        extra,
        unknown,
        _USER,
    )
    assert json.loads(json.dumps(result)) == [dict(item) for item in result]
    assert response == original


def test_messages_preserves_signed_block_and_tail_without_aliasing() -> None:
    block: Final = {"type": "compaction", "content": " Summary\n", "signature": "exact-signature", "extra": None}
    response: Final = {"stop_reason": "compaction", "content": [block]}
    original: Final = deepcopy(response)
    result: Final = native_compaction_result(response, _HISTORY, "messages")
    assert result == ({"role": "assistant", "content": [block]}, _USER)
    assert json.loads(json.dumps(result)) == [dict(item) for item in result]
    result[0]["content"][0]["signature"] = "changed-private-copy"
    assert response == original


@pytest.mark.parametrize(
    "protocol,field", (("responses", "encrypted_content"), ("messages", "signature"), ("messages", "content"))
)
@pytest.mark.parametrize("state", (None, "", " ", 42))
def test_invalid_compaction_state_rejected(protocol: NativeProtocol, field: str, state: object) -> None:
    valid: Final = {"type": "compaction", "content": "summary", "signature": "signed", "encrypted_content": "opaque"}
    block: Final = {**valid, field: state}
    response: Final = {"stop_reason": "compaction", "content": [block], "output": [block]}
    assert isinstance(native_compaction_result(response, _HISTORY, protocol), CompactionHistoryError)


@pytest.mark.parametrize("blocks", (None, [], ["invalid"], [{"type": "text", "text": "plain summary"}]))
@pytest.mark.parametrize("protocol", ("responses", "messages"))
def test_missing_native_output_rejected(blocks: object, protocol: NativeProtocol) -> None:
    response: Final = {"stop_reason": "compaction", "content": blocks, "output": blocks}
    assert isinstance(native_compaction_result(response, _HISTORY, protocol), CompactionHistoryError)


@pytest.mark.parametrize(("stop_reason", "count"), (("end_turn", 1), ("compaction", 2)))
def test_messages_requires_one_block_and_compaction_stop(stop_reason: str, count: int) -> None:
    block: Final = {"type": "compaction", "content": "summary", "signature": "signed"}
    response: Final = {"stop_reason": stop_reason, "content": [block] * count}
    assert isinstance(native_compaction_result(response, _HISTORY, "messages"), CompactionHistoryError)


def test_beta_headers_merge_case_insensitively_and_idempotently() -> None:
    required: Final = compaction_headers(None)["anthropic-beta"]
    existing: Final = {"Anthropic-Beta": f"other, {required}", "anthropic-beta": "other,third", "x-test": "keep"}
    original: Final = dict(existing)
    merged: Final = compaction_headers(existing)
    assert merged == {"anthropic-beta": f"other,{required},third", "x-test": "keep"}
    assert compaction_headers(merged) == merged and existing == original


@pytest.mark.parametrize("headers", ([], {"x-test": 1}, {"x-test": b"bytes"}, {"anthropic-beta": None}))
def test_invalid_headers_fail_at_payload_boundary(headers: object) -> None:
    with pytest.raises(ValueError, match="validation error"):
        native_compaction_payload(_HISTORY, {"extra_headers": headers}, "messages")
