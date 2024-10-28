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

verbose_logger.setLevel(logging.DEBUG)


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
    payload = datadog_llm_obs_logger.create_llm_obs_payload(
        kwargs={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        response_obj=litellm.ModelResponse(
            id="test_id",
            choices=[{"message": {"content": "Hi there!"}}],
            created=12,
            model="gpt-4",
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(seconds=1),
    )

    assert payload["name"] == "test_span"
    assert payload["meta"]["kind"] == "llm"
    assert payload["meta"]["input"]["messages"] == [
        {"role": "user", "content": "Hello"}
    ]
    assert payload["meta"]["output"]["messages"] == [{"content": "Hi there!"}]
    assert payload["metrics"]["input_tokens"] == 10
    assert payload["metrics"]["output_tokens"] == 5
    assert payload["metrics"]["total_tokens"] == 15
