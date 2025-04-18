import os
import sys
import pytest
import asyncio
from typing import Optional
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponseTextConfig,
    ResponseAPIUsage,
    IncompleteDetails,
)
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from base_responses_api import BaseResponsesAPITest

class TestAnthropicResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        #litellm._turn_on_debug()
        return {
            "model": "anthropic/claude-3-5-sonnet-latest",
        }
