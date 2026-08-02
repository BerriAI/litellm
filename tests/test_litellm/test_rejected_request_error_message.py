"""Regression: ``RejectedRequestError`` keeps the unprefixed text for completion content."""

from __future__ import annotations

from litellm.exceptions import RejectedRequestError


def test_rejected_request_error_keeps_raw_message():
    err = RejectedRequestError(message="366", model="gpt-4o", llm_provider="", request_data={"model": "gpt-4o"})
    assert err.raw_message == "366"
    assert "RejectedRequestError" in err.message
