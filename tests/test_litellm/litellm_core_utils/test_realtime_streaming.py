import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.types.llms.openai import (
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamSessionEvents,
)


def test_realtime_streaming_store_message():
    # Setup
    websocket = MagicMock()
    backend_ws = MagicMock()
    logging_obj = MagicMock()
    streaming = RealTimeStreaming(websocket, backend_ws, logging_obj)

    # Test 1: Session created event (string input)
    session_created_msg = json.dumps(
        {"type": "session.created", "session": {"id": "test-session"}}
    )
    streaming.store_message(session_created_msg)
    assert len(streaming.messages) == 1
    assert "session" in streaming.messages[0]
    assert streaming.messages[0]["type"] == "session.created"

    # Test 2: Response object (bytes input)
    response_msg = json.dumps(
        {
            "type": "response.create",
            "event_id": "test-event",
            "response": {"text": "test response"},
        }
    ).encode("utf-8")
    streaming.store_message(response_msg)
    assert len(streaming.messages) == 2
    assert "response" in streaming.messages[1]
    assert streaming.messages[1]["type"] == "response.create"

    # Test 3: Invalid message format
    invalid_msg = "invalid json"
    with pytest.raises(Exception):
        streaming.store_message(invalid_msg)

    # Test 4: Message type not in logged events
    streaming.logged_real_time_event_types = [
        "session.created"
    ]  # Only log session.created
    other_msg = json.dumps(
        {
            "type": "response.done",
            "event_id": "test-event",
            "response": {"text": "test response"},
        }
    )
    streaming.store_message(other_msg)
    assert len(streaming.messages) == 2  # Should not store the new message
