import os

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

import litellm


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY", None) is None,
    reason="No OpenAI API key provided",
)
async def test_openai_realtime_direct_call_no_intent():
    """
    End-to-end test calling the actual OpenAI realtime endpoint via LiteLLM SDK
    without intent parameter. This should succeed without "Invalid intent" error.
    Uses real websocket connection to OpenAI.
    """
    import asyncio
    import json

    class RealTimeWebSocketClient:
        def __init__(self):
            self.messages_sent = []
            self.messages_received = []
            self.received_session_created = False
            self.connection_successful = False
            self._receive_called = False
            self.close_code = None
            self.close_reason = None

        async def accept(self):
            pass

        async def send_text(self, message):
            self.messages_sent.append(message)
            try:
                if isinstance(message, bytes):
                    message_str = message.decode("utf-8")
                else:
                    message_str = message

                msg_data = json.loads(message_str)
                msg_type = msg_data.get("type", "unknown")

                if msg_type == "error":
                    error_info = msg_data.get("error", {})
                    error_code = error_info.get("code", "unknown")
                    error_message = error_info.get("message", "unknown")
                    # Don't fail on error, just record it - some errors are expected
                    self.messages_received.append(msg_data)
                    return

                if msg_type == "session.created" and not self.received_session_created:
                    self.messages_received.append(msg_data)
                    self.received_session_created = True
                    self.connection_successful = True
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Non-JSON messages are acceptable
                pass

        async def receive_text(self):
            if not self._receive_called:
                self._receive_called = True
                max_wait = 60.0
                check_interval = 0.1
                waited = 0.0

                while waited < max_wait:
                    if self.connection_successful:
                        break
                    await asyncio.sleep(check_interval)
                    waited += check_interval

                if not self.connection_successful:
                    await asyncio.sleep(3.0)

            raise ConnectionClosedOK(None, None)

        async def close(self, code=1000, reason=""):
            self.close_code = code
            self.close_reason = reason

        @property
        def headers(self):
            return {}

    websocket_client = RealTimeWebSocketClient()
    caught_exception = None

    try:
        await litellm._arealtime(
            # OpenAI shut down the gpt-4o-realtime-preview family (incl. the
            # undated alias) on 2026-05-07; gpt-realtime is the GA successor.
            model="openai/gpt-realtime",
            websocket=websocket_client,
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60,
        )
    except (ConnectionClosedOK, ConnectionClosedError):
        pass
    except Exception as e:
        caught_exception = e
        if "invalid_intent" in str(e).lower():
            pytest.fail(f"Still getting invalid intent error: {e}")
        # Other exceptions are recorded but don't fail immediately

    # Build detailed error message for debugging
    error_details = []
    error_details.append(f"messages_sent count: {len(websocket_client.messages_sent)}")
    error_details.append(
        f"messages_received count: {len(websocket_client.messages_received)}"
    )
    error_details.append(f"close_code: {websocket_client.close_code}")
    error_details.append(f"close_reason: {websocket_client.close_reason}")
    if caught_exception:
        error_details.append(
            f"exception: {type(caught_exception).__name__}: {caught_exception}"
        )

    assert (
        websocket_client.connection_successful
    ), f"Failed to establish connection. Debug info: {'; '.join(error_details)}"
    assert (
        websocket_client.received_session_created
    ), "Did not receive session.created response"
    assert len(websocket_client.messages_received) > 0, "No messages received"

    session_message = websocket_client.messages_received[0]
    assert (
        session_message["type"] == "session.created"
    ), f"Expected session.created, got {session_message.get('type')}"
    assert (
        "session" in session_message
    ), "session.created response missing session object"
    assert "id" in session_message["session"], "Session object missing id field"
    assert "model" in session_message["session"], "Session object missing model field"
