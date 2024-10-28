"""
Test the DataDogLLMObsLogger
"""

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
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
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
async def test_datadog_llm_obs_logging():
    datadog_llm_obs_logger = DataDogLLMObsLogger()
    litellm.callbacks = [datadog_llm_obs_logger]
    litellm.set_verbose = True

    for _ in range(2):
        response = await litellm.acompletion(
            model="gpt-4o", messages=["Hello testing dd llm obs!"], mock_response="hi"
        )

        print(response)

    await asyncio.sleep(6)


@pytest.mark.asyncio
async def test_create_llm_obs_payload():
    datadog_llm_obs_logger = DataDogLLMObsLogger()
    standard_logging_payload = create_standard_logging_payload()
    payload = datadog_llm_obs_logger.create_llm_obs_payload(
        kwargs={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "standard_logging_object": standard_logging_payload,
        },
        response_obj=litellm.ModelResponse(
            id="test_id",
            choices=[{"message": {"content": "Hi there!"}}],
            created=12,
            model="gpt-4",
        ),
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(seconds=1),
    )

    print("dd created payload", payload)

    assert payload["name"] == "litellm_llm_call"
    assert payload["meta"]["kind"] == "llm"
    assert payload["meta"]["input"]["messages"] == [
        {"role": "user", "content": "Hello, world!"}
    ]
    assert payload["meta"]["output"]["messages"] == [
        {
            "content": "Hi there!",
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
        }
    ]
    assert payload["metrics"]["input_tokens"] == 20
    assert payload["metrics"]["output_tokens"] == 10
    assert payload["metrics"]["total_tokens"] == 30
