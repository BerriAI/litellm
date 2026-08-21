import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.upstream_usage_headers import (
    UpstreamReportedUsage,
    apply_upstream_reported_usage,
    parse_upstream_reported_usage,
)
from litellm.types.utils import Usage


def _headers(**values: str) -> httpx.Headers:
    return httpx.Headers(values)


def test_parse_reads_both_totals():
    reported = parse_upstream_reported_usage(
        _headers(
            **{
                "x-litellm-response-cost": "0.000415",
                "x-litellm-total-tokens": "1874",
            }
        )
    )

    assert reported == UpstreamReportedUsage(response_cost=0.000415, total_tokens=1874)


def test_parse_returns_none_when_upstream_does_not_speak_the_contract():
    assert parse_upstream_reported_usage(_headers(**{"content-type": "application/json"})) is None


def test_parse_accepts_explicit_zero_totals():
    reported = parse_upstream_reported_usage(
        _headers(**{"x-litellm-response-cost": "0", "x-litellm-total-tokens": "0"})
    )

    assert reported == UpstreamReportedUsage(response_cost=0.0, total_tokens=0)


@pytest.mark.parametrize(
    "raw_cost",
    ["not-a-number", "-0.5", "nan", "inf", ""],
)
def test_parse_rejects_unusable_cost_but_keeps_tokens(raw_cost: str):
    reported = parse_upstream_reported_usage(
        _headers(**{"x-litellm-response-cost": raw_cost, "x-litellm-total-tokens": "12"})
    )

    assert reported == UpstreamReportedUsage(response_cost=None, total_tokens=12)


@pytest.mark.parametrize("raw_tokens", ["1.5", "twelve", "-3", ""])
def test_parse_rejects_unusable_tokens_but_keeps_cost(raw_tokens: str):
    reported = parse_upstream_reported_usage(
        _headers(**{"x-litellm-response-cost": "1.25", "x-litellm-total-tokens": raw_tokens})
    )

    assert reported == UpstreamReportedUsage(response_cost=1.25, total_tokens=None)


def test_parse_reports_missing_counterpart_header():
    assert parse_upstream_reported_usage(_headers(**{"x-litellm-response-cost": "2.5"})) == UpstreamReportedUsage(
        response_cost=2.5, total_tokens=None
    )
    assert parse_upstream_reported_usage(_headers(**{"x-litellm-total-tokens": "7"})) == UpstreamReportedUsage(
        response_cost=None, total_tokens=7
    )


def _logging_obj() -> LiteLLMLoggingObj:
    logging_obj = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "x"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=None,
        litellm_call_id="test-call-id",
        function_id="1",
    )
    return logging_obj


def test_apply_records_reported_totals():
    logging_obj = _logging_obj()

    reported = apply_upstream_reported_usage(
        logging_obj=logging_obj,
        headers=_headers(
            **{
                "x-litellm-response-cost": "0.000415",
                "x-litellm-total-tokens": "1874",
            }
        ),
    )

    assert reported is not None
    assert logging_obj.model_call_details["response_cost"] == 0.000415
    assert logging_obj.model_call_details["combined_usage_object"] == Usage(total_tokens=1874)


def test_apply_leaves_litellm_derived_values_alone_when_upstream_is_silent():
    logging_obj = _logging_obj()
    logging_obj.model_call_details["response_cost"] = 9.99
    logging_obj.model_call_details["combined_usage_object"] = Usage(total_tokens=42)

    assert apply_upstream_reported_usage(logging_obj=logging_obj, headers=_headers()) is None
    assert logging_obj.model_call_details["response_cost"] == 9.99
    assert logging_obj.model_call_details["combined_usage_object"] == Usage(total_tokens=42)


def test_apply_only_overwrites_what_upstream_reported():
    """A target that reports cost but not tokens must not zero out the token
    count LiteLLM derived from the response body itself."""
    logging_obj = _logging_obj()
    logging_obj.model_call_details["response_cost"] = 9.99
    logging_obj.model_call_details["combined_usage_object"] = Usage(total_tokens=42)

    apply_upstream_reported_usage(
        logging_obj=logging_obj,
        headers=_headers(**{"x-litellm-response-cost": "0.5"}),
    )

    assert logging_obj.model_call_details["response_cost"] == 0.5
    assert logging_obj.model_call_details["combined_usage_object"] == Usage(total_tokens=42)
