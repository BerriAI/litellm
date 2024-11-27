import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.datadog.datadog import DataDogLogger, DataDogStatus
from datetime import datetime, timedelta
from litellm.types.integrations.datadog_llm_obs import *
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingModelInformation,
    StandardLoggingMetadata,
    StandardLoggingHiddenParams,
)

verbose_logger.setLevel(logging.DEBUG)


def create_standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        call_type="completion",
        response_cost=0.1,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="gpt-3.5-turbo", model_map_value=None
        ),
        model="gpt-3.5-turbo",
        model_id="model-123",
        model_group="openai-gpt",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_org_id=None,
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_team_alias="test_team_alias",
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.1",
            additional_headers=None,
        ),
    )


@pytest.mark.asyncio
async def test_create_datadog_logging_payload():
    """Test creating a DataDog logging payload from a standard logging object"""
    dd_logger = DataDogLogger()
    standard_payload = create_standard_logging_payload()

    # Create mock kwargs with the standard logging object
    kwargs = {"standard_logging_object": standard_payload}

    # Test payload creation
    dd_payload = dd_logger.create_datadog_logging_payload(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    # Verify payload structure
    assert dd_payload["ddsource"] == os.getenv("DD_SOURCE", "litellm")
    assert dd_payload["service"] == "litellm-server"
    assert dd_payload["status"] == DataDogStatus.INFO

    # verify the message field == standard_payload
    dict_payload = json.loads(dd_payload["message"])
    assert dict_payload == standard_payload


@pytest.mark.asyncio
async def test_datadog_failure_logging():
    """Test logging a failure event to DataDog"""
    dd_logger = DataDogLogger()
    standard_payload = create_standard_logging_payload()
    standard_payload["status"] = "failure"  # Set status to failure
    standard_payload["error_str"] = "Test error"

    kwargs = {"standard_logging_object": standard_payload}

    dd_payload = dd_logger.create_datadog_logging_payload(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert (
        dd_payload["status"] == DataDogStatus.ERROR
    )  # Verify failure maps to warning status

    # verify the message field == standard_payload
    dict_payload = json.loads(dd_payload["message"])
    assert dict_payload == standard_payload

    # verify error_str is in the message field
    assert "error_str" in dict_payload
    assert dict_payload["error_str"] == "Test error"
