# What is this?
## This tests the braintrust integration

import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def test_braintrust_logging():
    import litellm

    http_client = HTTPHandler()

    setattr(
        litellm.integrations.braintrust_logging,
        "global_braintrust_sync_http_handler",
        http_client,
    )

    with patch.object(http_client, "post", new=MagicMock()) as mock_client:

        # set braintrust as a callback, litellm will send the data to braintrust
        litellm.callbacks = ["braintrust"]

        # openai call
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
        )

        mock_client.assert_called()
