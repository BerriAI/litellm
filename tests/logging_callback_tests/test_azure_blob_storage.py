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
from litellm.integrations.datadog.datadog import *
from datetime import datetime, timedelta
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingModelInformation,
    StandardLoggingMetadata,
    StandardLoggingHiddenParams,
)
from litellm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger

verbose_logger.setLevel(logging.DEBUG)


@pytest.mark.asyncio
async def test_azure_blob_storage():
    azure_storage_logger = AzureBlobStorageLogger(flush_interval=1)
    litellm.callbacks = [azure_storage_logger]

    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)

    await asyncio.sleep(3)
    pass
