"""Tests for ``apply_guardrail`` text-side BLOCK/MASK/DETECT/NO_ACTION + core scanning path.

Tool-call / tool-definition / legacy functions[] scanning lives in
``test_apply_guardrail_tools.py``; fail-open/closed, missing secrets, and helpers
live in ``test_apply_guardrail_failmodes.py``.
"""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException

# ----------------------------- BLOCK -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_block_action(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    detection = Mock()
    detection.model_dump = Mock(return_value={"policy_name": "x", "confidence": 0.9})
    result_obj.detections = [detection]
    result_obj.correlation_id = "corr-1"
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"
    assert exc.value.detail["wonderfence_correlation_id"] == "corr-1"
    assert exc.value.detail["error"] == ("Content violates our policies and has been blocked")
    assert exc.value.detail["detections"][0]["policy_name"] == "x"


@pytest.mark.asyncio
async def test_apply_guardrail_block_uses_custom_block_message(make_guardrail, make_request_data):
    guardrail, client = make_guardrail(block_message="custom blocked text")
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.detail["error"] == "custom blocked text"


@pytest.mark.asyncio
async def test_block_not_bypassed_by_fail_open(make_guardrail, make_request_data):
    guardrail, client = make_guardrail(fail_open=True)
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["bad"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400


# ----------------------------- MASK -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_mask_replaces_scanned_text(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = "[REDACTED]"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["sensitive"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["[REDACTED]"]


@pytest.mark.asyncio
async def test_apply_guardrail_mask_reconstructs_only_the_flagged_message_part(guardrail_and_client, make_request_data):
    """Request side joins the message parts into one document, scans once, and on
    MASK reconstructs per-part masked text by aligning the join against the
    masked document. Only the flagged part changes; the others survive."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        # One joined call: mask just the sensitive part inside the joined doc.
        r.action = "MASK"
        r.action_text = prompt.replace("sensitive content", "[REDACTED]")
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["first", "ack", "sensitive content"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["first", "ack", "[REDACTED]"]
    client.evaluate_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_guardrail_scans_non_user_role_segments(guardrail_and_client, make_request_data):
    """Bypass regression: blocked content in a system/assistant/tool message
    must still BLOCK. The translation layer already strips system/tool when the
    guardrail is configured to skip them, so whatever remains in ``texts`` is
    scanned regardless of role; the hook must not re-filter to user-only."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "disallowed system instruction" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "structured_messages": [
            {"role": "system", "content": "disallowed system instruction"},
            {"role": "user", "content": "hello"},
        ],
        "texts": ["disallowed system instruction", "hello"],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"


@pytest.mark.asyncio
async def test_apply_guardrail_mask_replaces_scanned_text_response(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = "[REDACTED]"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_response.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["model output"]},
        request_data=make_request_data(),
        input_type="response",
    )
    assert out["texts"] == ["[REDACTED]"]


@pytest.mark.asyncio
async def test_apply_guardrail_mask_fallback_when_action_text_is_none(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = None
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["a"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["[MASKED]"]


# ----------------------------- DETECT / NO_ACTION -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_no_action_passthrough(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["safe"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["safe"]
    client.evaluate_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_guardrail_detect_action_passes_through(guardrail_and_client, make_request_data):
    """DETECT action logs a warning but does not block or mutate inputs."""
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "DETECT"
    result_obj.detections = []
    result_obj.correlation_id = "corr-detect"
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["watch me"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["watch me"]
    client.evaluate_prompt.assert_awaited_once()


# ----------------------------- core path / app_id passthrough -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_passes_app_id_per_call(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=make_request_data(metadata={"user_api_key_metadata": {"alice_wonderfence_app_id": "tenant-A"}}),
        input_type="request",
    )
    kwargs = client.evaluate_prompt.call_args.kwargs
    assert kwargs["app_id"] == "tenant-A"
    assert kwargs["prompt"] == "hi"
    assert kwargs["custom_fields"] is None


@pytest.mark.asyncio
async def test_apply_guardrail_response_path_passes_app_id(make_guardrail, make_request_data):
    guardrail, client = make_guardrail()
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_response.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data=make_request_data(metadata={"user_api_key_metadata": {"alice_wonderfence_app_id": "tenant-B"}}),
        input_type="response",
    )
    kwargs = client.evaluate_response.call_args.kwargs
    assert kwargs["app_id"] == "tenant-B"
    assert kwargs["response"] == "resp"


@pytest.mark.asyncio
async def test_apply_guardrail_joins_all_message_parts_into_one_call(guardrail_and_client, make_request_data):
    """Every message part is scanned, but as a single joined document in ONE
    Alice call (call volume scales with size, not message count). The join uses
    a plain newline so cross-part content is seen whole."""
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["t1", "t2", "t3"]},
        request_data=make_request_data(),
        input_type="request",
    )
    client.evaluate_prompt.assert_awaited_once()
    assert client.evaluate_prompt.call_args.kwargs["prompt"] == "t1\nt2\nt3"


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_earlier_user_turn(guardrail_and_client, make_request_data):
    """Bypass regression: disallowed content in an earlier user turn followed by
    a benign final turn must still BLOCK. The old last-only path only saw the
    benign final message and let the request through."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "disallowed" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "structured_messages": [
            {"role": "user", "content": "disallowed"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "hello"},
        ],
        "texts": ["disallowed", "ok", "hello"],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_when_oversized_message_trips_in_late_chunk(
    guardrail_and_client, make_request_data
):
    """A single user message over the prompt limit is chunked; a BLOCK in a
    non-first chunk still blocks the request."""
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
        chunked_evaluation,
    )

    guardrail, client = guardrail_and_client
    long_prompt = ("safe " * 5000) + "TRIPWIRE"
    assert len(long_prompt) > chunked_evaluation.MAX_PROMPT_CHARS

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "TRIPWIRE" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": [long_prompt]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert client.evaluate_prompt.call_count > 1


@pytest.mark.asyncio
async def test_apply_guardrail_no_text_short_circuits(guardrail_and_client, make_request_data):
    """Empty inputs must skip the SDK call and return inputs unchanged."""
    guardrail, client = guardrail_and_client
    out = await guardrail.apply_guardrail(
        inputs={"texts": []},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out == {"texts": []}
    client.evaluate_prompt.assert_not_awaited()
    client.evaluate_response.assert_not_awaited()


# ----------------------------- join: cross-part visibility -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_phrase_split_across_message_parts(guardrail_and_client, make_request_data):
    """Two content parts that individually look benign are joined into one
    document, so a phrase split across the part boundary is seen in a single
    scan and still BLOCKs (the join replaces the old cross-segment windows)."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        # Neither part alone contains the whole phrase; the joined document does.
        r.action = "BLOCK" if ("make a b" in prompt and "omb" in prompt) else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["how to make a b", "omb please"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    client.evaluate_prompt.assert_awaited_once()


# ----------------------------- join: MASK reconstruction write-back -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_mask_writes_back_to_the_correct_message(guardrail_and_client, make_request_data):
    """A real PII MASK on the joined document reconstructs per-part masked text
    and writes it back to the message that carried the PII, leaving the others
    intact."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "MASK"
        r.action_text = prompt.replace("john@example.com", "[EMAIL]")
        r.detections = []
        r.correlation_id = "corr-mask"
        return r

    client.evaluate_prompt.side_effect = evaluate

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["hello there", "my email is john@example.com", "thanks"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["hello there", "my email is [EMAIL]", "thanks"]
    client.evaluate_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_guardrail_mask_reconstruction_failure_fails_closed(guardrail_and_client, make_request_data):
    """If masking destroys a joiner (parts would merge), reconstruction cannot
    safely attribute the redaction, so the request is blocked rather than
    silently misassigned or passed through unmasked."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "MASK"
        r.action_text = prompt.replace("\n", "")  # destroys the joiner -> parts merge
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["alpha", "beta"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400


# ----------------------------- total-work cap -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_rejects_over_segment_cap_without_scanning(make_guardrail, make_request_data):
    guardrail, client = make_guardrail(max_scan_segments=3)
    guardrail._client_cache["default-api-key"] = client

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["a", "b", "c", "d"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["limit"] == "max_scan_segments"
    client.evaluate_prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_guardrail_cap_is_not_bypassed_by_fail_open(make_guardrail, make_request_data):
    """The cap is a config/abuse guard, never fail-open: an oversized request
    is rejected 400 even with fail_open=True and the SDK is never called."""
    guardrail, client = make_guardrail(max_scan_chars=10, fail_open=True)
    guardrail._client_cache["default-api-key"] = client

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["x" * 50]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["limit"] == "max_scan_chars"
    client.evaluate_prompt.assert_not_awaited()
