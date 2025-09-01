import asyncio
import json
import logging
import os
import sys
import time

import pytest

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger

verbose_logger.setLevel(logging.DEBUG)

sys.path.insert(0, os.path.abspath("../.."))

# Testing scenarios for logfire logging:
# 1. Test logfire logging for completion
# 2. Test logfire logging for acompletion
# 3. Test logfire logging for completion while streaming is enabled
# 4. Test logfire logging for completion while streaming is enabled


@pytest.mark.skip(reason="Breaks on ci/cd but works locally")
@pytest.mark.parametrize("stream", [False, True])
def test_completion_logfire_logging(stream):
    from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig

    litellm.callbacks = ["logfire"]
    litellm.set_verbose = True
    messages = [{"role": "user", "content": "what llm are u"}]
    temperature = 0.3
    max_tokens = 10
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )
    print(response)

    if stream:
        for chunk in response:
            print(chunk)

    time.sleep(5)


@pytest.mark.skip(reason="Breaks on ci/cd but works locally")
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_acompletion_logfire_logging(stream):
    from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig

    litellm.callbacks = ["logfire"]
    litellm.set_verbose = True
    messages = [{"role": "user", "content": "what llm are u"}]
    temperature = 0.3
    max_tokens = 10
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )
    print(response)
    if stream:
        async for chunk in response:
            print(chunk)

    await asyncio.sleep(5)
