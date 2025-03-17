import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
import litellm
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig, Span
import asyncio


@pytest.mark.asyncio
async def test_dynamic_arize_callback_params():
    litellm.callbacks = ["arize"]

    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hi",
        temperature=0.1,
        user="OTEL_USER",
        arize_space_key="arize_space_key_1",
        arize_api_key="arize_api_key_1",
    )

    print("done with request 1......")

    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hi",
        temperature=0.1,
        user="OTEL_USER",
        arize_space_key="arize_space_key_2",
        arize_api_key="arize_api_key_2",
    )

    print("running request 2.....")

    await asyncio.sleep(1)

    print("done with request 2......")
