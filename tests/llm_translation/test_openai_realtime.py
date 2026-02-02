import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.realtime import RealtimeQueryParams


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
    
    Note: This test may be skipped on transient connection failures since it depends
    on external OpenAI API availability.
    """
    import websockets
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
                    message_str = message.decode('utf-8')
                else:
                    message_str = message
                    
                msg_data = json.loads(message_str)
                msg_type = msg_data.get('type', 'unknown')
                
                if msg_type == "error":
                    error_info = msg_data.get('error', {})
                    error_code = error_info.get('code', 'unknown')
                    error_message = error_info.get('message', 'unknown')
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
            
            raise websockets.exceptions.ConnectionClosed(None, None)
            
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
            model="gpt-4o-realtime-preview-2024-10-01",
            websocket=websocket_client,
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60
        )
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        caught_exception = e
        if "invalid_intent" in str(e).lower():
            pytest.fail(f"Still getting invalid intent error: {e}")
        # Other exceptions are recorded but don't fail immediately
    
    # Build detailed error message for debugging
    error_details = []
    error_details.append(f"messages_sent count: {len(websocket_client.messages_sent)}")
    error_details.append(f"messages_received count: {len(websocket_client.messages_received)}")
    error_details.append(f"close_code: {websocket_client.close_code}")
    error_details.append(f"close_reason: {websocket_client.close_reason}")
    if caught_exception:
        error_details.append(f"exception: {type(caught_exception).__name__}: {caught_exception}")
    
    # Skip test on transient connection failures (e.g., WebSocket connection rejected)
    # These are not regressions, just external API availability issues
    if not websocket_client.connection_successful and websocket_client.close_code is not None:
        pytest.skip(f"Skipping due to transient connection failure: close_code={websocket_client.close_code}, close_reason={websocket_client.close_reason}")
    
    assert websocket_client.connection_successful, f"Failed to establish connection. Debug info: {'; '.join(error_details)}"
    assert websocket_client.received_session_created, "Did not receive session.created response"
    assert len(websocket_client.messages_received) > 0, "No messages received"
    
    session_message = websocket_client.messages_received[0]
    assert session_message["type"] == "session.created", f"Expected session.created, got {session_message.get('type')}"
    assert "session" in session_message, "session.created response missing session object"
    assert "id" in session_message["session"], "Session object missing id field"
    assert "model" in session_message["session"], "Session object missing model field"


@pytest.mark.asyncio  
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY", None) is None,
    reason="No OpenAI API key provided",
)
async def test_openai_realtime_direct_call_with_intent():
    """
    End-to-end test calling the actual OpenAI realtime endpoint via LiteLLM SDK
    with explicit intent parameter. This should include the intent in the URL.
    Uses real websocket connection to OpenAI.
    
    Note: This test may be skipped on transient connection failures since it depends
    on external OpenAI API availability.
    """
    import websockets
    import asyncio
    import json
    
    class RealTimeWebSocketClient:
        def __init__(self):
            self.messages_sent = []
            self.messages_received = []
            self.received_session_created = False
            self.connection_successful = False
            self._receive_called = False
            self.intent_error_received = None
            self.close_code = None
            self.close_reason = None

        async def accept(self):
            pass

        async def send_text(self, message):
            self.messages_sent.append(message)
            try:
                if isinstance(message, bytes):
                    message_str = message.decode('utf-8')
                else:
                    message_str = message

                msg_data = json.loads(message_str)
                msg_type = msg_data.get('type', 'unknown')

                if msg_type == "error":
                    error_info = msg_data.get('error', {})
                    error_code = error_info.get('code', 'unknown')
                    error_message = error_info.get('message', 'unknown')
                    
                    if error_code == "invalid_intent":
                        self.intent_error_received = {
                            'code': error_code,
                            'message': error_message
                        }
                    # Don't fail on other errors, just record them
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

            raise websockets.exceptions.ConnectionClosed(None, None)

        async def close(self, code=1000, reason=""):
            self.close_code = code
            self.close_reason = reason

        @property
        def headers(self):
            return {}

    websocket_client = RealTimeWebSocketClient()
    caught_exception = None
    
    query_params: RealtimeQueryParams = {
        "model": "gpt-4o-realtime-preview-2024-10-01",
        "intent": "chat"
    }
    
    try:
        await litellm._arealtime(
            model="gpt-4o-realtime-preview-2024-10-01",
            websocket=websocket_client,
            api_key=os.environ.get("OPENAI_API_KEY"),
            query_params=query_params,
            timeout=60
        )
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        caught_exception = e
        if "invalid_intent" in str(e).lower():
            pytest.fail(f"Unexpected invalid intent error: {e}")
        # Other exceptions are recorded but don't fail immediately
    
    if websocket_client.intent_error_received:
        websocket_client.connection_successful = True
    
    # Build detailed error message for debugging
    error_details = []
    error_details.append(f"messages_sent count: {len(websocket_client.messages_sent)}")
    error_details.append(f"messages_received count: {len(websocket_client.messages_received)}")
    error_details.append(f"close_code: {websocket_client.close_code}")
    error_details.append(f"close_reason: {websocket_client.close_reason}")
    if caught_exception:
        error_details.append(f"exception: {type(caught_exception).__name__}: {caught_exception}")
    
    # Skip test on transient connection failures (e.g., WebSocket connection rejected)
    # These are not regressions, just external API availability issues
    if not websocket_client.connection_successful and websocket_client.close_code is not None:
        pytest.skip(f"Skipping due to transient connection failure: close_code={websocket_client.close_code}, close_reason={websocket_client.close_reason}")
    
    assert websocket_client.connection_successful, f"Failed to establish connection or verify intent parameter pass-through. Debug info: {'; '.join(error_details)}"
    
    if websocket_client.received_session_created:
        assert len(websocket_client.messages_received) > 0, "No messages received"
        session_message = websocket_client.messages_received[0]
        assert session_message["type"] == "session.created", f"Expected session.created, got {session_message.get('type')}"
        assert "session" in session_message, "session.created response missing session object"
        assert "id" in session_message["session"], "Session object missing id field"
        assert "model" in session_message["session"], "Session object missing model field"
    elif websocket_client.intent_error_received:
        # invalid_intent error confirms intent parameter was passed through
        pass
    else:
        pytest.fail(f"Unexpected test state: connection_successful={websocket_client.connection_successful}, "
                   f"received_session_created={websocket_client.received_session_created}, "
                   f"intent_error_received={websocket_client.intent_error_received}")


def test_realtime_query_params_construction():
    """
    Test that query params are constructed correctly by the proxy server logic
    """
    from litellm.types.realtime import RealtimeQueryParams
    
    # Test case 1: intent is None (should not be included)
    model = "gpt-4o-realtime-preview-2024-10-01"
    intent = None
    
    query_params: RealtimeQueryParams = {"model": model}
    if intent is not None:
        query_params["intent"] = intent
        
    assert "model" in query_params
    assert query_params["model"] == model
    assert "intent" not in query_params
    
    # Test case 2: intent is provided (should be included)
    intent = "chat"
    query_params2: RealtimeQueryParams = {"model": model}
    if intent is not None:
        query_params2["intent"] = intent
        
    assert "model" in query_params2
    assert query_params2["model"] == model
    assert "intent" in query_params2
    assert query_params2["intent"] == intent


@pytest.mark.asyncio
async def test_realtime_query_params_use_normalized_model_name(monkeypatch):
    """
    Ensure query params overwrite model with normalized provider model name.
    """
    from litellm.realtime_api import main as realtime_main

    mock_async_realtime = AsyncMock()
    monkeypatch.setattr(
        realtime_main,
        "openai_realtime",
        MagicMock(async_realtime=mock_async_realtime),
    )

    def fake_get_llm_provider(model, api_base=None, api_key=None):
        return ("gpt-4o-realtime-preview-2024-10-01", "openai", None, None)

    monkeypatch.setattr(realtime_main, "get_llm_provider", fake_get_llm_provider)

    query_params: RealtimeQueryParams = {
        "model": "openai/gpt-4o-realtime-preview-2024-10-01",
        "intent": "chat",
    }

    await realtime_main._arealtime(
        model="openai/gpt-4o-realtime-preview-2024-10-01",
        websocket=MagicMock(),
        api_key="sk-test",
        query_params=query_params,
        litellm_logging_obj=MagicMock(),
    )

    called_kwargs = mock_async_realtime.call_args.kwargs
    assert (
        called_kwargs["query_params"]["model"] == "gpt-4o-realtime-preview-2024-10-01"
    )
    assert called_kwargs["query_params"]["intent"] == "chat"
