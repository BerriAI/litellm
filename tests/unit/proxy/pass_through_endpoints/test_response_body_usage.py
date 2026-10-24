from datetime import datetime, timezone

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.response_body_usage import (
    apply_response_body_usage,
    parse_response_body_usage,
)
from litellm.types.utils import Usage


def _logging_obj() -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="decider",
        messages=[],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        litellm_call_id="body-usage-1",
        function_id="body-usage",
    )


def test_parse_reads_openai_shape():
    usage = parse_response_body_usage({"usage": {"prompt_tokens": 1874, "completion_tokens": 12, "total_tokens": 1886}})

    assert usage == Usage(prompt_tokens=1874, completion_tokens=12, total_tokens=1886)


def test_parse_reads_anthropic_shape_and_sums_the_total():
    usage = parse_response_body_usage({"answers": {}, "usage": {"input_tokens": 1874, "output_tokens": 0}})

    assert usage == Usage(prompt_tokens=1874, completion_tokens=0, total_tokens=1874)


def test_parse_counts_a_missing_side_as_zero():
    usage = parse_response_body_usage({"usage": {"input_tokens": 40}})

    assert usage == Usage(prompt_tokens=40, completion_tokens=0, total_tokens=40)


def test_parse_keeps_an_explicit_total_over_the_sum():
    usage = parse_response_body_usage({"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 20}})

    assert usage == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=20)


@pytest.mark.parametrize(
    "response_body",
    [
        None,
        "not json",
        [{"usage": {"input_tokens": 3}}],
        {"answer": "ok"},
        {"usage": None},
        {"usage": "1874"},
        {"usage": {}},
        {"usage": {"input_tokens": "1874", "output_tokens": "0"}},
        {"usage": {"input_tokens": -1}},
        {"usage": {"input_tokens": True}},
        {"usage": {"input_tokens": 12.5}},
    ],
)
def test_parse_returns_none_when_the_body_reports_no_usable_count(response_body: object):
    assert parse_response_body_usage(response_body) is None


def test_parse_skips_an_unusable_value_but_keeps_the_usable_side():
    usage = parse_response_body_usage({"usage": {"input_tokens": "oops", "output_tokens": 7}})

    assert usage == Usage(prompt_tokens=0, completion_tokens=7, total_tokens=7)


def test_apply_records_the_body_usage_on_the_logging_object():
    logging_obj = _logging_obj()

    applied = apply_response_body_usage(logging_obj, {"usage": {"input_tokens": 1874, "output_tokens": 0}})

    assert applied == Usage(prompt_tokens=1874, completion_tokens=0, total_tokens=1874)
    assert logging_obj.model_call_details["combined_usage_object"] == applied


def test_apply_leaves_an_upstream_reported_total_untouched():
    logging_obj = _logging_obj()
    logging_obj.model_call_details["combined_usage_object"] = Usage(total_tokens=930)

    assert apply_response_body_usage(logging_obj, {"usage": {"input_tokens": 1874, "output_tokens": 0}}) is None
    assert logging_obj.model_call_details["combined_usage_object"] == Usage(total_tokens=930)


def test_apply_writes_nothing_without_usage_in_the_body():
    logging_obj = _logging_obj()

    assert apply_response_body_usage(logging_obj, {"answer": "ok"}) is None
    assert "combined_usage_object" not in logging_obj.model_call_details
