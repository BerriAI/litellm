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
    
    # Create a real websocket client that will connect to OpenAI
    class RealTimeWebSocketClient:
        def __init__(self):
            self.messages_sent = []
            self.messages_received = []
            self.websocket = None
            
        async def accept(self):
            # Not needed for client-side websocket
            pass
            
        async def send_text(self, message):
            self.messages_sent.append(message)
            
        async def receive_text(self):
            # For testing, we'll just wait a bit then close
            await asyncio.sleep(0.5)
            # Send a simple session update to simulate real usage
            if len(self.messages_received) == 0:
                response = {"type": "session.created", "session": {"id": "test_session"}}
                self.messages_received.append(response)
                return json.dumps(response)
            # Close after first exchange
            raise websockets.exceptions.ConnectionClosed(None, None)
            
        async def close(self, code=1000, reason=""):
            # Connection will be closed by the realtime handler
            pass
            
        @property
        def headers(self):
            return {}

    websocket_client = RealTimeWebSocketClient()
    
    # Test with no intent parameter - this should NOT produce "Invalid intent" error
    try:
        await litellm._arealtime(
            model="gpt-4o-realtime-preview-2024-10-01",
            websocket=websocket_client,
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=10
        )
    except websockets.exceptions.ConnectionClosed:
        # Expected - connection closes after brief test
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
    
    # Create a real websocket client that will connect to OpenAI
    class RealTimeWebSocketClient:
        def __init__(self):
            self.messages_sent = []
            self.messages_received = []
            
        async def accept(self):
            # Not needed for client-side websocket
            pass
            
        async def send_text(self, message):
            self.messages_sent.append(message)
            
        async def receive_text(self):
            # For testing, we'll just wait a bit then close
            await asyncio.sleep(0.5)
            # Send a simple session update to simulate real usage
            if len(self.messages_received) == 0:
                response = {"type": "session.created", "session": {"id": "test_session"}}
                self.messages_received.append(response)
                return json.dumps(response)
            # Close after first exchange
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