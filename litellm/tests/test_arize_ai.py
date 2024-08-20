import asyncio
import logging
import os
import time

import pytest
from dotenv import load_dotenv
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig

load_dotenv()
import logging


@pytest.mark.asyncio()
async def test_async_otel_callback():
    litellm.set_verbose = True
    litellm.success_callback = ["arize"]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi test from local arize"}],
        mock_response="hello",
        temperature=0.1,
        user="OTEL_USER",
    )
