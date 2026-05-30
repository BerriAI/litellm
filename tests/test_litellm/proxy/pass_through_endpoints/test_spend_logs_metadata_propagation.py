"""Tests for `x-litellm-spend-logs-metadata` propagation through pass-through endpoints.

OpenAI-compatible endpoints parse this header in
`LiteLLMProxyRequestSetup._get_spend_logs_metadata_from_request_headers` and
stuff it into `data["metadata"]["spend_logs_metadata"]`. Pass-through endpoints
used to skip this extraction, so `/spend/logs` entries for
`call_type=pass_through_endpoint` always had
`metadata.spend_logs_metadata = None`.

These tests pin the fix:

1. Header extraction handles present/missing/malformed input.
2. `BasePassthroughLoggingHandler._apply_spend_logs_metadata` writes the
   metadata to the same key path the OpenAI-compatible flow uses.
3. `_seed_streaming_kwargs_from_logging_obj` threads the pass-through payload
   from `logging_obj.model_call_details` into fresh streaming kwargs, so
   streaming responses retain the metadata.
"""

import json
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"x-litellm-spend-logs-metadata": json.dumps({"task": "t1"})}, {"task": "t1"}),
        ({}, None),
        ({"x-litellm-spend-logs-metadata": "not-json"}, None),
    ],
)
def test_header_extraction(headers, expected):
    """Helper handles valid JSON, missing header, and malformed JSON."""
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    assert (
        LiteLLMProxyRequestSetup._get_spend_logs_metadata_from_request_headers(headers)
        == expected
    )


def test_apply_spend_logs_metadata_writes_matching_key_path():
    """The helper must write to `kwargs["litellm_params"]["metadata"]["spend_logs_metadata"]`
    — the same key path OpenAI-compatible flows use, so
    `get_standard_logging_object_payload` picks it up identically.
    """
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
        BasePassthroughLoggingHandler,
    )

    kwargs: dict = {}
    metadata = {"task": "cost-report", "issue": "#42"}

    BasePassthroughLoggingHandler._apply_spend_logs_metadata(
        kwargs,
        {"url": "x", "spend_logs_metadata": metadata},
    )

    assert kwargs["litellm_params"]["metadata"]["spend_logs_metadata"] == metadata


def test_apply_spend_logs_metadata_preserves_existing_metadata():
    """Existing metadata keys must not be clobbered when attaching
    `spend_logs_metadata`.
    """
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
        BasePassthroughLoggingHandler,
    )

    kwargs: dict = {"litellm_params": {"metadata": {"tags": ["prod"]}}}

    BasePassthroughLoggingHandler._apply_spend_logs_metadata(
        kwargs,
        {"url": "x", "spend_logs_metadata": {"task": "t1"}},
    )

    assert kwargs["litellm_params"]["metadata"] == {
        "tags": ["prod"],
        "spend_logs_metadata": {"task": "t1"},
    }


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"url": "x"}, {"url": "x", "spend_logs_metadata": None}],
)
def test_apply_spend_logs_metadata_is_noop_when_missing(payload):
    """When the payload is missing or carries no metadata, the helper must
    leave kwargs untouched — this is the path for existing callers that don't
    send the header, and it must remain backward-compatible.
    """
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
        BasePassthroughLoggingHandler,
    )

    kwargs: dict = {}
    BasePassthroughLoggingHandler._apply_spend_logs_metadata(kwargs, payload)
    assert kwargs == {}


def test_seed_streaming_kwargs_copies_payload_from_logging_obj():
    """Streaming logging paths build fresh kwargs; the seed helper must lift
    the pass-through payload stashed by `success_handler` onto that fresh dict
    so `_create_*_response_logging_payload` can still see it.
    """
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
        BasePassthroughLoggingHandler,
    )

    payload = {"url": "x", "spend_logs_metadata": {"task": "t1"}}
    logging_obj = MagicMock(model_call_details={"passthrough_logging_payload": payload})

    seeded = BasePassthroughLoggingHandler._seed_streaming_kwargs_from_logging_obj(
        logging_obj
    )

    assert seeded == {"passthrough_logging_payload": payload}


def test_seed_streaming_kwargs_empty_when_no_payload():
    """No pass-through payload on `logging_obj` → empty dict (prior behavior)."""
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
        BasePassthroughLoggingHandler,
    )

    logging_obj = MagicMock(model_call_details={})

    assert (
        BasePassthroughLoggingHandler._seed_streaming_kwargs_from_logging_obj(
            logging_obj
        )
        == {}
    )
