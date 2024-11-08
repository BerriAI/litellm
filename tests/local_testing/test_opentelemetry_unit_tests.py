# What is this?
## Unit tests for opentelemetry integration

# What is this?
## Unit test for presidio pii masking
import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_opentelemetry_integration():
    """
    Unit test to confirm the parent otel span is ended
    """

    parent_otel_span = MagicMock()
    litellm.callbacks = ["otel"]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Hey!",
        metadata={"litellm_parent_otel_span": parent_otel_span},
    )

    await asyncio.sleep(1)

    parent_otel_span.end.assert_called_once()
