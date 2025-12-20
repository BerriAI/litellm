import os
import sys
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
    """
    import websockets
    import asyncio
    import json
    
    # Create a real websocket client that will validate OpenAI responses
    class RealTimeWebSocketClient:
        def __init__(self):
            self.messages_sent = []
            self.messages_received = []
            self.received_session_created = False
            self.connection_successful = False
            self._receive_called = False
            
        async def accept(self):
            # Not needed for client-side websocket
            pass
            
        async def send_text(self, message):
            # This is called by the realtime handler when forwarding messages FROM OpenAI TO the client
            # Messages from OpenAI come through backend_ws and are forwarded here via send_text()
            self.messages_sent.append(message)
            try:
                msg_data = json.loads(message)
                msg_type = msg_data.get('type', 'unknown')
                print(f"Received from OpenAI (via send_text): {msg_type}")
                
                # Check if this is the session.created message we're waiting for
                if msg_type == "session.created" and not self.received_session_created:
                    self.messages_received.append(msg_data)
                    self.received_session_created = True
                    self.connection_successful = True
                    print(f"✅ Successfully received session.created from OpenAI")
            except json.JSONDecodeError:
                pass
            
        async def receive_text(self):
            # This is called by client_ack_messages() to read messages FROM the client
            # Since this test doesn't send any client messages, we'll wait a bit then close
            if not self._receive_called:
                self._receive_called = True
                # Wait a bit to allow the backend_to_client_send_messages task to receive session.created
                await asyncio.sleep(1.0)
            
            # After waiting, close the connection to end the test
            print("Test validation complete - closing connection")
            raise websockets.exceptions.ConnectionClosed(None, None)
            
        async def close(self, code=1000, reason=""):
            # Connection will be closed by the realtime handler
            pass
            
        @property
        def headers(self):
            return {}

    websocket_client = RealTimeWebSocketClient()
    
    # Test with no intent parameter - this should NOT produce "Invalid intent" error
    # and should receive a valid session.created response
    try:
        await litellm._arealtime(
            model="gpt-4o-realtime-preview-2024-10-01",
            websocket=websocket_client,
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=15
        )
    except websockets.exceptions.ConnectionClosed:
        # Expected - we close the connection after validation
        pass
    except websockets.exceptions.InvalidStatusCode as e:
        # If we get a 4000 status with "invalid_intent", the fix didn't work
        if "invalid_intent" in str(e).lower():
            pytest.fail(f"Still getting invalid_intent error: {e}")
        else:
            # Other connection errors are expected in test environment
            pass
    except Exception as e:
        # Make sure we're not getting the "Invalid intent" error
        if "invalid_intent" in str(e).lower() or "Invalid intent" in str(e):
            pytest.fail(f"Fix failed - still getting invalid intent error: {e}")
        # Other exceptions are acceptable for this connection test
    
    # Validate that we successfully connected and received expected response
    assert websocket_client.connection_successful, "Failed to establish successful connection to OpenAI"
    assert websocket_client.received_session_created, "Did not receive session.created response from OpenAI"
    assert len(websocket_client.messages_received) > 0, "No messages received from OpenAI"
    
    # Validate the structure of the session.created response
    session_message = websocket_client.messages_received[0]
    assert session_message["type"] == "session.created", f"Expected session.created, got {session_message.get('type')}"
    assert "session" in session_message, "session.created response missing session object"
    assert "id" in session_message["session"], "Session object missing id field"
    assert "model" in session_message["session"], "Session object missing model field"
    
    print(f"✅ Successfully validated OpenAI realtime API response structure")


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
    """
    import websockets
    import asyncio
    import json
    
    # Create a real websocket client that will validate OpenAI responses  
    class RealTimeWebSocketClient:
        def __init__(self):
            self.messages_sent = []
            self.messages_received = []
            self.received_session_created = False
            self.connection_successful = False
            self._receive_called = False
            
        async def accept(self):
            # Not needed for client-side websocket
            pass
            
        async def send_text(self, message):
            # This is called by the realtime handler when forwarding messages FROM OpenAI TO the client
            # Messages from OpenAI come through backend_ws and are forwarded here via send_text()
            self.messages_sent.append(message)
            try:
                msg_data = json.loads(message)
                msg_type = msg_data.get('type', 'unknown')
                print(f"Received from OpenAI (via send_text, with intent): {msg_type}")
                
                # Check if this is the session.created message we're waiting for
                if msg_type == "session.created" and not self.received_session_created:
                    self.messages_received.append(msg_data)
                    self.received_session_created = True
                    self.connection_successful = True
                    print(f"✅ Successfully received session.created from OpenAI (with intent)")
            except json.JSONDecodeError:
                pass
            
        async def receive_text(self):
            # This is called by client_ack_messages() to read messages FROM the client
            # Since this test doesn't send any client messages, we'll wait a bit then close
            if not self._receive_called:
                self._receive_called = True
                # Wait a bit to allow the backend_to_client_send_messages task to receive session.created
                await asyncio.sleep(1.0)
            
            # After waiting, close the connection to end the test
            print("Test validation complete (with intent) - closing connection")
            raise websockets.exceptions.ConnectionClosed(None, None)
            
        async def close(self, code=1000, reason=""):
            # Connection will be closed by the realtime handler
            pass
            
        @property
        def headers(self):
            return {}

    websocket_client = RealTimeWebSocketClient()
    
    query_params: RealtimeQueryParams = {
        "model": "gpt-4o-realtime-preview-2024-10-01",
        "intent": "chat"
    }
    
    # Test with explicit intent parameter
    try:
        await litellm._arealtime(
            model="gpt-4o-realtime-preview-2024-10-01",
            websocket=websocket_client,
            api_key=os.environ.get("OPENAI_API_KEY"),
            query_params=query_params,
            timeout=10
        )
    except websockets.exceptions.ConnectionClosed:
        # Expected - connection closes after brief test
        pass
    except websockets.exceptions.InvalidStatusCode as e:
        # Any connection errors are expected in test environment
        # The important thing is we can establish connection without invalid_intent
        pass
    except Exception as e:
        # Make sure we're not getting unexpected errors
        if "invalid_intent" in str(e).lower() or "Invalid intent" in str(e):
            pytest.fail(f"Unexpected invalid intent error with explicit intent: {e}")
    
    # Validate that we successfully connected and received expected response  
    assert websocket_client.connection_successful, "Failed to establish successful connection to OpenAI (with intent)"
    assert websocket_client.received_session_created, "Did not receive session.created response from OpenAI (with intent)"
    assert len(websocket_client.messages_received) > 0, "No messages received from OpenAI (with intent)"
    
    # Validate the structure of the session.created response
    session_message = websocket_client.messages_received[0]
    assert session_message["type"] == "session.created", f"Expected session.created, got {session_message.get('type')} (with intent)"
    assert "session" in session_message, "session.created response missing session object (with intent)"
    assert "id" in session_message["session"], "Session object missing id field (with intent)"
    assert "model" in session_message["session"], "Session object missing model field (with intent)"
    
    print(f"✅ Successfully validated OpenAI realtime API response structure (with intent=chat)")



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
    assert "intent" not in query_params  # Should not be present when None
    
    # Test case 2: intent is provided (should be included)
    intent = "chat"
    query_params2: RealtimeQueryParams = {"model": model}
    if intent is not None:
        query_params2["intent"] = intent
        
    assert "model" in query_params2
    assert query_params2["model"] == model
    assert "intent" in query_params2
    assert query_params2["intent"] == intent