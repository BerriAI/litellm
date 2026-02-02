"""
Test that audio delta events are buffered until response.audio_transcript.done is emitted.

When a response.content_part.added event with part.type == "audio" is received,
no response.audio_transcript.delta or response.audio.delta events should be
emitted to the client until a response.audio_transcript.done event is emitted.

This test file tests the audio buffering functionality in:
    litellm/litellm_core_utils/realtime_streaming.py

The tests verify that:
1. When content_part.added with audio type is received, buffering starts
2. response.audio_transcript.delta events are buffered (not sent to client)
3. response.audio.delta events are buffered (not sent to client)
4. When response.audio_transcript.done is received, all buffered events are released
5. Text events (response.text.delta) are NOT affected by audio buffering
6. response.content_part.added events are always emitted immediately
"""

import json
import os
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming


def create_mock_logging_obj():
    """Create a mock logging object with proper async methods."""
    logging_obj = MagicMock()
    logging_obj.pre_call = MagicMock()
    logging_obj.success_handler = MagicMock()

    async def async_success_handler(*args, **kwargs):
        pass

    logging_obj.async_success_handler = async_success_handler
    return logging_obj


class MockClientWebSocket:
    """Mock client WebSocket that tracks all sent messages."""

    def __init__(self):
        self.sent_messages: List[str] = []

    async def send_text(self, message: str):
        self.sent_messages.append(message)

    def get_sent_event_types(self) -> List[str]:
        """Extract event types from all sent messages."""
        event_types = []
        for msg in self.sent_messages:
            try:
                data = json.loads(msg)
                if "type" in data:
                    event_types.append(data["type"])
            except json.JSONDecodeError:
                pass
        return event_types

    def get_sent_events(self) -> List[dict]:
        """Get all sent events as dictionaries."""
        events = []
        for msg in self.sent_messages:
            try:
                events.append(json.loads(msg))
            except json.JSONDecodeError:
                pass
        return events


class MockConnectionClosed(Exception):
    """Mock exception to simulate websocket connection closed."""

    pass


class MockBackendWebSocket:
    """Mock backend WebSocket that yields predefined messages."""

    def __init__(self, messages: List[str]):
        self.messages = messages
        self.index = 0

    async def recv(self, decode: bool = True) -> str:
        if self.index >= len(self.messages):
            # Simulate connection close by raising an exception
            from websockets.exceptions import ConnectionClosed

            raise ConnectionClosed(None, None)
        message = self.messages[self.index]
        self.index += 1
        return message


def create_content_part_added_audio_event(
    event_id: str = "event_1",
    response_id: str = "resp_1",
    item_id: str = "item_1",
) -> dict:
    """Create a response.content_part.added event with audio type."""
    return {
        "type": "response.content_part.added",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "audio", "transcript": ""},
    }


def create_content_part_added_text_event(
    event_id: str = "event_1",
    response_id: str = "resp_1",
    item_id: str = "item_1",
) -> dict:
    """Create a response.content_part.added event with text type."""
    return {
        "type": "response.content_part.added",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "text", "text": ""},
    }


def create_audio_transcript_delta_event(
    delta: str,
    event_id: str = "event_2",
    response_id: str = "resp_1",
    item_id: str = "item_1",
) -> dict:
    """Create a response.audio_transcript.delta event."""
    return {
        "type": "response.audio_transcript.delta",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


def create_audio_delta_event(
    delta: str,
    event_id: str = "event_3",
    response_id: str = "resp_1",
    item_id: str = "item_1",
) -> dict:
    """Create a response.audio.delta event."""
    return {
        "type": "response.audio.delta",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


def create_audio_transcript_done_event(
    transcript: str,
    event_id: str = "event_4",
    response_id: str = "resp_1",
    item_id: str = "item_1",
) -> dict:
    """Create a response.audio_transcript.done event."""
    return {
        "type": "response.audio_transcript.done",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "transcript": transcript,
    }


def create_text_delta_event(
    delta: str,
    event_id: str = "event_5",
    response_id: str = "resp_1",
    item_id: str = "item_1",
) -> dict:
    """Create a response.text.delta event."""
    return {
        "type": "response.text.delta",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


class TestRealtimeStreamingAudioBuffering:
    """
    Test suite for verifying that audio delta events are properly buffered
    until response.audio_transcript.done is emitted.

    These tests verify the behavior of the backend_to_client_send_messages method
    which is responsible for forwarding events from the backend to the client.
    """

    @pytest.mark.asyncio
    async def test_should_not_emit_audio_transcript_delta_before_transcript_done(self):
        """
        Test that response.audio_transcript.delta events are not emitted to the client
        when response.content_part.added has part.type == 'audio',
        until response.audio_transcript.done is received.

        The expected behavior is:
        1. When content_part.added with audio type is received, start buffering
        2. Buffer all audio_transcript.delta events
        3. Only emit them after audio_transcript.done is received
        """
        # Prepare backend messages - sequence without transcript.done
        backend_messages = [
            json.dumps(create_content_part_added_audio_event()),
            json.dumps(create_audio_transcript_delta_event("That", event_id="event_2")),
            json.dumps(create_audio_transcript_delta_event("'s", event_id="event_3")),
            json.dumps(create_audio_transcript_delta_event(" the", event_id="event_4")),
        ]

        client_ws = MockClientWebSocket()
        backend_ws = MockBackendWebSocket(backend_messages)
        logging_obj = create_mock_logging_obj()

        streaming = RealTimeStreaming(
            websocket=client_ws,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
        )
        streaming.logged_real_time_event_types = "*"

        # Run the backend_to_client_send_messages until it completes
        await streaming.backend_to_client_send_messages()

        # Check what was sent to the client websocket
        sent_types = client_ws.get_sent_event_types()

        # response.audio_transcript.delta should NOT be in sent messages
        # before transcript.done is received
        assert (
            "response.audio_transcript.delta" not in sent_types
        ), f"response.audio_transcript.delta should not be emitted before response.audio_transcript.done. Sent types: {sent_types}"

    @pytest.mark.asyncio
    async def test_should_not_emit_audio_delta_before_transcript_done(self):
        """
        Test that response.audio.delta events are not emitted to the client
        when response.content_part.added has part.type == 'audio',
        until response.audio_transcript.done is received.
        """
        # Prepare backend messages - sequence without transcript.done
        backend_messages = [
            json.dumps(create_content_part_added_audio_event()),
            json.dumps(create_audio_delta_event("audio_data_1", event_id="event_2")),
            json.dumps(create_audio_delta_event("audio_data_2", event_id="event_3")),
        ]

        client_ws = MockClientWebSocket()
        backend_ws = MockBackendWebSocket(backend_messages)
        logging_obj = create_mock_logging_obj()

        streaming = RealTimeStreaming(
            websocket=client_ws,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
        )
        streaming.logged_real_time_event_types = "*"

        # Run the backend_to_client_send_messages until it completes
        await streaming.backend_to_client_send_messages()

        # Check what was sent to the client websocket
        sent_types = client_ws.get_sent_event_types()

        # response.audio.delta should NOT be in sent messages before transcript.done
        assert (
            "response.audio.delta" not in sent_types
        ), f"response.audio.delta should not be emitted before response.audio_transcript.done. Sent types: {sent_types}"

    @pytest.mark.asyncio
    async def test_should_emit_audio_events_after_transcript_done(self):
        """
        Test that after response.audio_transcript.done is received,
        all buffered audio delta events are emitted to the client.
        """
        # Prepare backend messages - complete sequence with transcript.done
        backend_messages = [
            json.dumps(create_content_part_added_audio_event()),
            json.dumps(
                create_audio_transcript_delta_event("Hello", event_id="event_2")
            ),
            json.dumps(create_audio_delta_event("audio_data_1", event_id="event_3")),
            json.dumps(
                create_audio_transcript_done_event("Hello world", event_id="event_4")
            ),
        ]

        client_ws = MockClientWebSocket()
        backend_ws = MockBackendWebSocket(backend_messages)
        logging_obj = create_mock_logging_obj()

        streaming = RealTimeStreaming(
            websocket=client_ws,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
        )
        streaming.logged_real_time_event_types = "*"

        # Run the backend_to_client_send_messages until it completes
        await streaming.backend_to_client_send_messages()

        # Check what was sent to the client websocket
        sent_types = client_ws.get_sent_event_types()

        # After transcript.done, the buffered events should be emitted
        assert (
            "response.audio_transcript.done" in sent_types
        ), f"response.audio_transcript.done should be emitted. Sent types: {sent_types}"

        # The delta events should also be emitted after transcript.done
        assert (
            "response.audio_transcript.delta" in sent_types
        ), f"Buffered response.audio_transcript.delta should be emitted after transcript.done. Sent types: {sent_types}"

        assert (
            "response.audio.delta" in sent_types
        ), f"Buffered response.audio.delta should be emitted after transcript.done. Sent types: {sent_types}"

    @pytest.mark.asyncio
    async def test_should_emit_content_part_added_immediately(self):
        """
        Test that response.content_part.added is emitted immediately
        regardless of whether it's audio or text type.
        """
        backend_messages = [
            json.dumps(create_content_part_added_audio_event()),
        ]

        client_ws = MockClientWebSocket()
        backend_ws = MockBackendWebSocket(backend_messages)
        logging_obj = create_mock_logging_obj()

        streaming = RealTimeStreaming(
            websocket=client_ws,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
        )
        streaming.logged_real_time_event_types = "*"

        await streaming.backend_to_client_send_messages()

        sent_types = client_ws.get_sent_event_types()

        # content_part.added should be emitted immediately
        assert (
            "response.content_part.added" in sent_types
        ), f"response.content_part.added should be emitted immediately. Sent types: {sent_types}"

    @pytest.mark.asyncio
    async def test_should_not_buffer_text_events(self):
        """
        Test that text events (response.text.delta) are not affected
        by the audio buffering logic and are emitted immediately.
        """
        backend_messages = [
            json.dumps(create_content_part_added_text_event()),
            json.dumps(create_text_delta_event("Hello", event_id="event_2")),
            json.dumps(create_text_delta_event(" world", event_id="event_3")),
        ]

        client_ws = MockClientWebSocket()
        backend_ws = MockBackendWebSocket(backend_messages)
        logging_obj = create_mock_logging_obj()

        streaming = RealTimeStreaming(
            websocket=client_ws,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
        )
        streaming.logged_real_time_event_types = "*"

        await streaming.backend_to_client_send_messages()

        sent_types = client_ws.get_sent_event_types()

        # Text events should be emitted immediately
        assert (
            "response.text.delta" in sent_types
        ), f"response.text.delta should be emitted immediately for text content. Sent types: {sent_types}"

    @pytest.mark.asyncio
    async def test_should_preserve_event_order_after_transcript_done(self):
        """
        Test that when events are released after transcript.done,
        they maintain the correct order: buffered events first, then transcript.done.
        """
        backend_messages = [
            json.dumps(create_content_part_added_audio_event()),
            json.dumps(
                create_audio_transcript_delta_event("First", event_id="event_2")
            ),
            json.dumps(
                create_audio_transcript_delta_event("Second", event_id="event_3")
            ),
            json.dumps(
                create_audio_transcript_done_event("First Second", event_id="event_4")
            ),
        ]

        client_ws = MockClientWebSocket()
        backend_ws = MockBackendWebSocket(backend_messages)
        logging_obj = create_mock_logging_obj()

        streaming = RealTimeStreaming(
            websocket=client_ws,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
        )
        streaming.logged_real_time_event_types = "*"

        await streaming.backend_to_client_send_messages()

        sent_events = client_ws.get_sent_events()
        sent_types = [e.get("type") for e in sent_events]

        # Find indices of events
        content_part_idx = (
            sent_types.index("response.content_part.added")
            if "response.content_part.added" in sent_types
            else -1
        )

        # Find all delta indices
        delta_indices = [
            i
            for i, t in enumerate(sent_types)
            if t == "response.audio_transcript.delta"
        ]

        # Find transcript.done index
        done_idx = (
            sent_types.index("response.audio_transcript.done")
            if "response.audio_transcript.done" in sent_types
            else -1
        )

        # content_part.added should come first
        assert (
            content_part_idx == 0
        ), f"content_part.added should be first. Order: {sent_types}"

        # All deltas should come before transcript.done
        if delta_indices and done_idx >= 0:
            for delta_idx in delta_indices:
                assert (
                    delta_idx < done_idx
                ), f"Delta events should come before transcript.done. Order: {sent_types}"


class TestRealtimeStreamingAudioBufferingIntegration:
    """
    Integration tests that simulate realistic event sequences
    as seen in production logs.
    """

    @pytest.mark.asyncio
    async def test_should_handle_realistic_audio_event_sequence(self):
        """
        Test with a realistic sequence of events as seen in production logs.
        """
        # Realistic sequence based on the log.txt
        backend_messages = [
            # Content part added with audio type
            json.dumps(
                {
                    "type": "response.content_part.added",
                    "event_id": "event_D4tCFZ6S4Q7JO9kP70G0P",
                    "response_id": "resp_D4tCF4ARnzIifWdlz2MgJ",
                    "item_id": "item_D4tCFMTAAGMeuP92L2EgC",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "audio", "transcript": ""},
                }
            ),
            # Audio transcript deltas (should be buffered)
            json.dumps(
                {
                    "type": "response.audio_transcript.delta",
                    "event_id": "event_D4tCFWMl5yGEgTuvhskQt",
                    "response_id": "resp_D4tCF4ARnzIifWdlz2MgJ",
                    "item_id": "item_D4tCFMTAAGMeuP92L2EgC",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "That",
                }
            ),
            json.dumps(
                {
                    "type": "response.audio_transcript.delta",
                    "event_id": "event_D4tCFvyNuI8DfoTiRwBVe",
                    "response_id": "resp_D4tCF4ARnzIifWdlz2MgJ",
                    "item_id": "item_D4tCFMTAAGMeuP92L2EgC",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "'s",
                }
            ),
            json.dumps(
                {
                    "type": "response.audio_transcript.delta",
                    "event_id": "event_D4tCFblb0eTnMWPPyALYX",
                    "response_id": "resp_D4tCF4ARnzIifWdlz2MgJ",
                    "item_id": "item_D4tCFMTAAGMeuP92L2EgC",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": " the",
                }
            ),
            # Audio delta (should be buffered)
            json.dumps(
                {
                    "type": "response.audio.delta",
                    "event_id": "event_D4tCFYO9NtUswUHbN17Yb",
                    "response_id": "resp_D4tCF4ARnzIifWdlz2MgJ",
                    "item_id": "item_D4tCFMTAAGMeuP92L2EgC",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "AQAAAAQAAQD6/wYAAQAEAP7/AgD+/wYA...",
                }
            ),
        ]

        client_ws = MockClientWebSocket()
        backend_ws = MockBackendWebSocket(backend_messages)
        logging_obj = create_mock_logging_obj()

        streaming = RealTimeStreaming(
            websocket=client_ws,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
        )
        streaming.logged_real_time_event_types = "*"

        await streaming.backend_to_client_send_messages()

        sent_types = client_ws.get_sent_event_types()

        # Before transcript.done, these should not be emitted
        assert (
            "response.audio_transcript.delta" not in sent_types
        ), f"Audio transcript deltas should be buffered. Sent types: {sent_types}"

        assert (
            "response.audio.delta" not in sent_types
        ), f"Audio deltas should be buffered. Sent types: {sent_types}"

    @pytest.mark.asyncio
    async def test_should_handle_interleaved_audio_and_text_events(self):
        """
        Test that when both audio and text content parts exist,
        only audio events are buffered while text events flow through.
        """
        backend_messages = [
            # Text content part
            json.dumps(
                create_content_part_added_text_event(
                    event_id="event_1", item_id="text_item"
                )
            ),
            json.dumps(
                create_text_delta_event(
                    "Hello", event_id="event_2", item_id="text_item"
                )
            ),
            # Audio content part
            json.dumps(
                create_content_part_added_audio_event(
                    event_id="event_3", item_id="audio_item"
                )
            ),
            json.dumps(
                create_audio_transcript_delta_event(
                    "World", event_id="event_4", item_id="audio_item"
                )
            ),
            # More text
            json.dumps(
                create_text_delta_event(
                    " there", event_id="event_5", item_id="text_item"
                )
            ),
        ]

        client_ws = MockClientWebSocket()
        backend_ws = MockBackendWebSocket(backend_messages)
        logging_obj = create_mock_logging_obj()

        streaming = RealTimeStreaming(
            websocket=client_ws,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
        )
        streaming.logged_real_time_event_types = "*"

        await streaming.backend_to_client_send_messages()

        sent_types = client_ws.get_sent_event_types()

        # Text events should be emitted
        assert (
            "response.text.delta" in sent_types
        ), f"Text deltas should be emitted immediately. Sent types: {sent_types}"

        # Audio transcript delta should be buffered
        assert (
            "response.audio_transcript.delta" not in sent_types
        ), f"Audio transcript deltas should be buffered. Sent types: {sent_types}"
