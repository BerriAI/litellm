import json
import os
import sys
from typing import Optional
from unittest.mock import MagicMock

import pytest

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
from litellm.integrations.SlackAlerting.utils import _add_langfuse_trace_id_to_alert
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager


@pytest.mark.asyncio
async def test_langfuse_not_initialized_returns_none_early():
    """
    Test that when no LangfusePromptManagement is initialized,
    the function returns None immediately without executing further logic
    """
    # Ensure no Langfuse logger is in the callback manager
    litellm.logging_callback_manager = LoggingCallbackManager()

    # Create request data that would normally trigger processing
    request_data = {"litellm_logging_obj": MagicMock(), "trace_id": "test-trace-id"}

    # Call the function
    result = await _add_langfuse_trace_id_to_alert(request_data)

    # Should return None early without processing request_data
    assert result is None

    # Verify the litellm_logging_obj was never accessed (early return)
    request_data["litellm_logging_obj"].assert_not_called()
