import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import logging
import uuid

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.gcs_bucket import GCSBucketLogger

verbose_logger.setLevel(logging.DEBUG)


gcs_logger = GCSBucketLogger()
print("GCSBucketLogger", gcs_logger)


@pytest.mark.asyncio
async def test_basic_gcs_logger():
    litellm.callbacks = [gcs_logger]
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[{"role": "user", "content": "This is a test"}],
        max_tokens=10,
        user="ishaan-2",
        mock_response="Hi!",
    )

    print("response", response)

    await asyncio.sleep(5)
