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
    assert transformed_message["response"]["type"] == "session.created"


def test_gemini_realtime_transformation_content_delta():
    config = GeminiRealtimeConfig()
    assert config is not None

    session_configuration_request = {
        "model": "gemini-1.5-flash",
        "generationConfig": {"responseModalities": ["TEXT"]},
    }
    session_configuration_request_str = json.dumps(session_configuration_request)
    session_created_message = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"text": "Hello, world!"},
                    {"text": "How are you?"},
                ]
            }
        }
    }

    session_created_message_str = json.dumps(session_created_message)
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id.return_value = "123"

    returned_object = config.transform_realtime_response(
        session_created_message_str,
        "gemini-1.5-flash",
        logging_obj,
        session_configuration_request_str,
    )
    transformed_message = returned_object["response"]
    assert isinstance(transformed_message, list)
    print(transformed_message)
    transformed_message_str = json.dumps(transformed_message)
    assert "Hello, world" in transformed_message_str
    assert "How are you?" in transformed_message_str
    print(transformed_message)

    ## assert all instances of 'event_id' are unique
    event_ids = [
        event["event_id"] for event in transformed_message if "event_id" in event
    ]
    assert len(event_ids) == len(set(event_ids))
    ## assert all instances of 'response_id' are the same
    response_ids = [
        event["response_id"] for event in transformed_message if "response_id" in event
    ]
    assert len(set(response_ids)) == 1
    ## assert all instances of 'output_item_id' are the same
    output_item_ids = [
        event["item_id"] for event in transformed_message if "item_id" in event
    ]
    assert len(set(output_item_ids)) == 1
