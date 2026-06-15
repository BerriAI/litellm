"""
Unit tests for the structured `guardrail_response` surfaced by the
`/apply_guardrail` endpoint on both the success and the blocked (failure) paths.

Covers the helpers added on top of PR #28970:
    - `_collect_guardrail_info_from_data`
    - `_enrich_guardrail_block_exception`
and the new `ApplyGuardrailResponse.guardrail_response` field.

These are mocked unit tests (no real LLM / guardrail API calls).
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from fastapi import HTTPException

from litellm.proxy._types import ProxyException
from litellm.proxy.guardrails.guardrail_endpoints import (
    _collect_guardrail_info_from_data,
    _enrich_guardrail_block_exception,
)
from litellm.types.guardrails import ApplyGuardrailResponse


def _sample_slg(status="success", guardrail_response=None):
    return {
        "guardrail_name": "content-safety-multi",
        "guardrail_status": status,
        "guardrail_mode": "pre_call",
        "guardrail_provider": "litellm_content_filter",
        "guardrail_response": [] if guardrail_response is None else guardrail_response,
        "duration": 0.001,
    }


def test_collect_guardrail_info_from_metadata():
    data = {"metadata": {"standard_logging_guardrail_information": [_sample_slg()]}}
    info = _collect_guardrail_info_from_data(data)
    assert len(info) == 1
    assert info[0]["guardrail_name"] == "content-safety-multi"
    assert info[0]["guardrail_status"] == "success"
    assert info[0]["guardrail_mode"] == "pre_call"
    assert info[0]["guardrail_provider"] == "litellm_content_filter"


def test_collect_guardrail_info_from_litellm_metadata():
    data = {
        "litellm_metadata": {"standard_logging_guardrail_information": [_sample_slg()]}
    }
    info = _collect_guardrail_info_from_data(data)
    assert len(info) == 1
    assert info[0]["guardrail_name"] == "content-safety-multi"


def test_collect_guardrail_info_empty_when_absent():
    assert _collect_guardrail_info_from_data({}) == []
    assert _collect_guardrail_info_from_data({"metadata": {}}) == []


def test_enrich_block_exception_adds_structured_classification():
    classification = [
        {
            "type": "category_keyword",
            "category": "denied_medical_advice",
            "keyword": "medicine",
            "severity": "high",
            "action": "BLOCK",
        }
    ]
    data = {
        "metadata": {
            "standard_logging_guardrail_information": [
                _sample_slg(
                    status="guardrail_intervened", guardrail_response=classification
                )
            ]
        }
    }
    original = HTTPException(
        status_code=403,
        detail={
            "error": "Content blocked: denied_medical_advice category keyword 'medicine' detected (severity: high)",
            "category": "denied_medical_advice",
        },
    )
    enriched = _enrich_guardrail_block_exception(original, data)

    assert isinstance(enriched, ProxyException)
    assert str(enriched.code) == "403"
    assert "Content blocked" in enriched.message
    assert enriched.provider_specific_fields is not None
    surfaced = enriched.provider_specific_fields["guardrail_response"]
    assert surfaced[0]["guardrail_status"] == "guardrail_intervened"
    assert surfaced[0]["guardrail_response"] == classification


def test_enrich_block_exception_noop_without_guardrail_info():
    original = HTTPException(status_code=403, detail="blocked")
    # No guardrail info in data -> the original exception is returned unchanged.
    assert _enrich_guardrail_block_exception(original, {"metadata": {}}) is original


def test_enrich_block_exception_noop_for_non_http_exception():
    err = ValueError("boom")
    data = {"metadata": {"standard_logging_guardrail_information": [_sample_slg()]}}
    # Non-HTTPException errors are passed through untouched.
    assert _enrich_guardrail_block_exception(err, data) is err


def test_apply_guardrail_response_model_carries_guardrail_response():
    # defaults to None (field omitted when no hook/endpoint attaches it)
    assert ApplyGuardrailResponse(response_text="hi").guardrail_response is None

    populated = ApplyGuardrailResponse(
        response_text="hi", guardrail_response=[{"guardrail_name": "x"}]
    )
    assert populated.guardrail_response[0]["guardrail_name"] == "x"
