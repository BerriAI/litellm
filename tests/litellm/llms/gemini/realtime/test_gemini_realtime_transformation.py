import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.gemini.realtime.transformation import GeminiRealtimeConfig
from litellm.types.llms.openai import OpenAIRealtimeStreamSessionEvents


def test_gemini_realtime_transformation_session_created():
    config = GeminiRealtimeConfig()
    assert config is not None

    session_configuration_request = {
        "model": "gemini-1.5-flash",
        "generationConfig": {"responseModalities": ["TEXT"]},
    }
    session_configuration_request_str = json.dumps(session_configuration_request)
    session_created_message = {"setupComplete": {}}

    session_created_message_str = json.dumps(session_created_message)
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id.return_value = "123"

    transformed_message = config.transform_realtime_response(
        session_created_message_str,
        "gemini-1.5-flash",
        logging_obj,
        session_configuration_request_str,
    )
    assert transformed_message["type"] == "session.created"
