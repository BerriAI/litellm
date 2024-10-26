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
from datetime import datetime

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
