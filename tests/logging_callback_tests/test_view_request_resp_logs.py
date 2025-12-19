import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import json
import logging
import tempfile
from litellm._uuid import uuid

import json
from datetime import datetime, timedelta, timezone
from datetime import datetime

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.gcs_bucket.gcs_bucket import (
    GCSBucketLogger,
    StandardLoggingPayload,
)
from litellm.types.utils import StandardCallbackDynamicParams


# This is the response payload that GCS would return.
mock_response_data = {
    "id": "chatcmpl-9870a859d6df402795f75dc5fca5b2e0",
    "trace_id": None,
    "call_type": "acompletion",
    "cache_hit": None,
    "stream": True,
    "status": "success",
    "custom_llm_provider": "openai",
    "saved_cache_cost": 0.0,
    "startTime": 1739235379.683053,
    "endTime": 1739235379.84533,
    "completionStartTime": 1739235379.84533,
    "response_time": 0.1622769832611084,
    "model": "my-fake-model",
    "metadata": {
        "user_api_key_hash": "sk-test-mock-api-key-123",
        "user_api_key_alias": None,
        "user_api_key_team_id": None,
        "user_api_key_org_id": None,
        "user_api_key_user_id": "default_user_id",
        "user_api_key_team_alias": None,
        "spend_logs_metadata": None,
        "requester_ip_address": "127.0.0.1",
        "requester_metadata": {},
        "user_api_key_end_user_id": None,
        "prompt_management_metadata": None,
    },
    "cache_key": None,
    "response_cost": 3.7500000000000003e-05,
    "total_tokens": 21,
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "request_tags": [],
    "end_user": "",
    "api_base": "https://exampleopenaiendpoint-production.up.railway.app",
    "model_group": "fake-openai-endpoint",
    "model_id": "b68d56d76b0c24ac9462ab69541e90886342508212210116e300441155f37865",
    "requester_ip_address": "127.0.0.1",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "very gm to u"}]}
    ],
    "response": {
        "id": "chatcmpl-9870a859d6df402795f75dc5fca5b2e0",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "\n\nHello there, how may I assist you today?",
                    "role": "assistant",
                    "tool_calls": None,
                    "function_call": None,
                    "refusal": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 12,
            "prompt_tokens": 9,
            "total_tokens": 21,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
        "service_tier": None,
    },
    "model_parameters": {"stream": False, "max_retries": 0, "extra_body": {}},
    "hidden_params": {
        "model_id": "b68d56d76b0c24ac9462ab69541e90886342508212210116e300441155f37865",
        "cache_key": None,
        "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
        "response_cost": 3.7500000000000003e-05,
        "additional_headers": {},
        "litellm_overhead_time_ms": 2.126,
    },
    "model_map_information": {
        "model_map_key": "gpt-3.5-turbo-0301",
        "model_map_value": {},
    },
    "error_str": None,
    "error_information": {"error_code": "", "error_class": "", "llm_provider": ""},
    "response_cost_failure_debug_info": None,
    "guardrail_information": None,
}


@pytest.mark.asyncio
async def test_get_payload_current_day():
    """
    Verify that the payload is returned when it is found on the current day.
    """
    gcs_logger = GCSBucketLogger()
    # Use January 1, 2024 as the current day
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id = mock_response_data["id"]

    async def fake_download(object_name: str, **kwargs) -> bytes | None:
        if "2024-01-01" in object_name:
            return json.dumps(mock_response_data).encode("utf-8")
        return None

    gcs_logger.download_gcs_object = fake_download

    payload = await gcs_logger.get_request_response_payload(
        request_id, start_time, None
    )
    assert payload is not None
    assert payload["id"] == request_id


@pytest.mark.asyncio
async def test_get_payload_next_day():
    """
    Verify that if the payload is not found on the current day,
    but is available on the next day, it is returned.
    """
    gcs_logger = GCSBucketLogger()
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id = mock_response_data["id"]

    async def fake_download(object_name: str, **kwargs) -> bytes | None:
        if "2024-01-02" in object_name:
            return json.dumps(mock_response_data).encode("utf-8")
        return None

    gcs_logger.download_gcs_object = fake_download

    payload = await gcs_logger.get_request_response_payload(
        request_id, start_time, None
    )
    assert payload is not None
    assert payload["id"] == request_id


@pytest.mark.asyncio
async def test_get_payload_previous_day():
    """
    Verify that if the payload is not found on the current or next day,
    but is available on the previous day, it is returned.
    """
    gcs_logger = GCSBucketLogger()
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id = mock_response_data["id"]

    async def fake_download(object_name: str, **kwargs) -> bytes | None:
        if "2023-12-31" in object_name:
            return json.dumps(mock_response_data).encode("utf-8")
        return None

    gcs_logger.download_gcs_object = fake_download

    payload = await gcs_logger.get_request_response_payload(
        request_id, start_time, None
    )
    assert payload is not None
    assert payload["id"] == request_id


@pytest.mark.asyncio
async def test_get_payload_not_found():
    """
    Verify that if none of the three days contain the payload, None is returned.
    """
    gcs_logger = GCSBucketLogger()
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id = mock_response_data["id"]

    async def fake_download(object_name: str, **kwargs) -> bytes | None:
        return None

    gcs_logger.download_gcs_object = fake_download

    payload = await gcs_logger.get_request_response_payload(
        request_id, start_time, None
    )
    assert payload is None
