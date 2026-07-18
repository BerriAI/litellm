"""Unit tests for the /v1/messages/batches Bedrock<->Anthropic mapping."""

import json

import pytest

from litellm.proxy.anthropic_endpoints.messages_batches import (
    BEDROCK_MSGBATCH_PREFIX,
    _bedrock_error_to_anthropic,
    _CUSTOM_ID_PATTERN,
    _map_job_to_message_batch,
)

BASE = "https://llm.example.com"


def _job(status, total=100, processed=0, success=0, error=0, **extra):
    return {
        "status": status,
        "totalRecordCount": total,
        "processedRecordCount": processed,
        "successRecordCount": success,
        "errorRecordCount": error,
        "submitTime": "2026-07-17T00:00:00Z",
        "jobExpirationTime": "2026-07-18T00:00:00Z",
        **extra,
    }


def test_in_progress_statuses_map_to_in_progress():
    for status in ("Submitted", "Validating", "Scheduled", "InProgress"):
        mapped = _map_job_to_message_batch(_job(status), "msgbatch_bedrock_x", BASE)
        assert mapped["processing_status"] == "in_progress"
        assert mapped["results_url"] is None
        assert mapped["ended_at"] is None
        assert mapped["type"] == "message_batch"


def test_completed_maps_to_ended_with_results_url():
    mapped = _map_job_to_message_batch(
        _job("Completed", processed=100, success=98, error=2, endTime="2026-07-17T01:00:00Z"),
        "msgbatch_bedrock_x",
        BASE,
    )
    assert mapped["processing_status"] == "ended"
    assert mapped["request_counts"] == {
        "processing": 0,
        "succeeded": 98,
        "errored": 2,
        "canceled": 0,
        "expired": 0,
    }
    assert mapped["results_url"] == f"{BASE}/v1/messages/batches/msgbatch_bedrock_x/results"
    assert mapped["ended_at"] == "2026-07-17T01:00:00Z"


def test_partially_completed_marks_remainder_expired():
    mapped = _map_job_to_message_batch(
        _job("PartiallyCompleted", processed=60, success=55, error=5), "msgbatch_bedrock_x", BASE
    )
    assert mapped["processing_status"] == "ended"
    assert mapped["request_counts"]["expired"] == 40
    assert mapped["request_counts"]["succeeded"] == 55


def test_stopped_marks_remainder_canceled():
    mapped = _map_job_to_message_batch(_job("Stopped", processed=30, success=30), "msgbatch_bedrock_x", BASE)
    assert mapped["request_counts"]["canceled"] == 70
    assert mapped["processing_status"] == "ended"


def test_stopping_maps_to_canceling():
    mapped = _map_job_to_message_batch(_job("Stopping", processed=30, success=30), "msgbatch_bedrock_x", BASE)
    assert mapped["processing_status"] == "canceling"
    assert mapped["cancel_initiated_at"] is not None


def test_expired_marks_all_expired():
    mapped = _map_job_to_message_batch(_job("Expired"), "msgbatch_bedrock_x", BASE)
    assert mapped["request_counts"]["expired"] == 100
    assert mapped["processing_status"] == "ended"


def test_failed_marks_unsucceeded_errored():
    mapped = _map_job_to_message_batch(_job("Failed", processed=10, success=10), "msgbatch_bedrock_x", BASE)
    assert mapped["request_counts"]["errored"] == 90
    assert mapped["request_counts"]["succeeded"] == 10


def test_counts_always_sum_to_total_on_ended():
    for status in ("Completed", "PartiallyCompleted", "Stopped", "Expired", "Failed"):
        mapped = _map_job_to_message_batch(
            _job(status, processed=70, success=60, error=10), "msgbatch_bedrock_x", BASE
        )
        assert sum(mapped["request_counts"].values()) == 100, status


def test_bedrock_error_mapping():
    assert (
        _bedrock_error_to_anthropic({"errorCode": 400, "errorMessage": "bad"})["error"]["error"]["type"]
        == "invalid_request_error"
    )
    assert (
        _bedrock_error_to_anthropic({"errorCode": 429, "errorMessage": "slow"})["error"]["error"]["type"]
        == "rate_limit_error"
    )
    assert (
        _bedrock_error_to_anthropic({"errorCode": 500, "errorMessage": "boom"})["error"]["error"]["type"]
        == "api_error"
    )
    payload = _bedrock_error_to_anthropic({"errorCode": 500, "errorMessage": "boom"})
    assert payload["type"] == "errored"
    assert payload["error"]["type"] == "error"


def test_custom_id_pattern():
    assert _CUSTOM_ID_PATTERN.match("my-custom-id_1")
    assert not _CUSTOM_ID_PATTERN.match("")
    assert not _CUSTOM_ID_PATTERN.match("a" * 65)
    assert not _CUSTOM_ID_PATTERN.match("bad id!")


def test_prefix_constant():
    assert BEDROCK_MSGBATCH_PREFIX == "msgbatch_bedrock_"


def test_pre_end_counts_hide_live_counters():
    """Anthropic contract: terminal counters stay 0 until the batch ends."""
    from litellm.proxy.anthropic_endpoints.messages_batches import _map_job_to_message_batch as m

    for status in ("Submitted", "Validating", "Scheduled", "InProgress", "Stopping"):
        mapped = m(_job(status, processed=50, success=40, error=10), "msgbatch_bedrock_x", BASE)
        assert mapped["request_counts"]["processing"] == 100, status
        assert mapped["request_counts"]["succeeded"] == 0, status
        assert mapped["request_counts"]["errored"] == 0, status


def test_bedrock_batch_id_owner_split():
    from litellm.proxy.anthropic_endpoints.messages_batches import _split_bedrock_batch_id

    assert _split_bedrock_batch_id("msgbatch_bedrock_abc123_deadbeef") == ("abc123", "deadbeef")
    # Legacy ids (pre-ownership) have no tag and stay readable.
    assert _split_bedrock_batch_id("msgbatch_bedrock_abc123") == ("abc123", None)
