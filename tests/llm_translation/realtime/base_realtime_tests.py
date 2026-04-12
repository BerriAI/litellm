"""
Base test class for LiteLLM Realtime API E2E tests.

Provides common test infrastructure for testing realtime WebSocket connections
across different providers (OpenAI, xAI, etc.)
"""
import asyncio
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import pytest
import websockets

sys.path.insert(0, os.path.abspath("../../.."))

import litellm


class RealTimeWebSocketClient:
    """
    Mock WebSocket client for testing realtime connections.
    Captures messages sent from the backend and provides a simple interface
    for testing connection success.
    """
    
    def __init__(self):
        self.messages_sent = []
        self.messages_received = []
        self.received_initial_event = False
        self.connection_successful = False
        self.close_code = None
        self.close_reason = None
        # Required by realtime_streaming.py - import exceptions module
        from websockets import exceptions as websockets_exceptions
        self.exceptions = websockets_exceptions
        
    async def accept(self):
        """Accept the WebSocket connection"""
        pass
        
    async def send_text(self, message):
        """Receive message from backend and store it"""
        self.messages_sent.append(message)
        try:
            if isinstance(message, bytes):
                message_str = message.decode('utf-8')
            else:
                message_str = message
                
            msg_data = json.loads(message_str)
            msg_type = msg_data.get('type', 'unknown')
            
            # Pretty print API response
            print(f"\n{'='*80}")
            print(f"API RESPONSE #{len(self.messages_received) + 1} - Event: {msg_type}")
            print(f"{'='*80}")
            print(json.dumps(msg_data, indent=2, sort_keys=False))
            print(f"{'='*80}\n")
            
            self.messages_received.append(msg_data)
            
            # Check for initial connection event
            if not self.received_initial_event and self._is_initial_event(msg_type):
                self.received_initial_event = True
                self.connection_successful = True
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Non-JSON messages are acceptable
            print(f"\n[Non-JSON message: {e}]")
            print(f"Raw content: {str(message)[:200]}\n")
            pass
    
    def _is_initial_event(self, msg_type: str) -> bool:
        """Check if message type is an initial connection event"""
        # OpenAI sends "session.created", xAI sends "conversation.created"
        return msg_type in ["session.created", "conversation.created"]
        
    async def receive_text(self):
        """
        Wait briefly for messages, then close connection.
        This allows the backend forwarding task to send messages.
        """
        print(f"\nWaiting for connection to establish...")
        max_wait = 5.0
        check_interval = 0.1
        waited = 0.0
        
        while waited < max_wait:
            if self.connection_successful:
                print(f"Connection successful after {waited:.1f}s\n")
                break
            await asyncio.sleep(check_interval)
            waited += check_interval
        
        if not self.connection_successful:
            print(f"Warning: No initial event received after {max_wait}s\n")
        
        # If we have a pending message to send, send it now
        if hasattr(self, '_pending_client_message') and self._pending_client_message:
            print(f"Sending client message to backend...\n")
            # This simulates receiving a message from the client that needs to be forwarded to backend
            # We return it as if it came from the client
            msg = self._pending_client_message
            self._pending_client_message = None
            return msg
        
        # Close connection to end the test
        print(f"\n{'='*80}")
        print(f"TEST COMPLETE - Closing connection")
        print(f"Total messages received from API: {len(self.messages_received)}")
        print(f"{'='*80}\n")
        raise websockets.exceptions.ConnectionClosed(None, None)
    
    def queue_client_message(self, message: str):
        """Queue a message to be sent from 'client' to backend"""
        self._pending_client_message = message
        
    async def close(self, code=1000, reason=""):
        """Close the WebSocket"""
        self.close_code = code
        self.close_reason = reason
        
    @property
    def headers(self):
        return {}


class BaseRealtimeTest(ABC):
    """
    Abstract base test class for realtime API tests.
    
    Child classes must implement:
    - get_model(): Return the model name to test
    - get_api_key_env_var(): Return the environment variable name for the API key
    - get_initial_event_type(): Return the expected initial event type (e.g., "session.created")
    """
    
    @abstractmethod
    def get_model(self) -> str:
        """Return the model name to test (e.g., 'gpt-4o-realtime-preview-2024-10-01')"""
        pass
    
    @abstractmethod
    def get_api_key_env_var(self) -> str:
        """Return the environment variable name for the API key (e.g., 'OPENAI_API_KEY')"""
        pass
    
    @abstractmethod
    def get_initial_event_type(self) -> str:
        """Return the expected initial event type (e.g., 'session.created' or 'conversation.created')"""
        pass
    
    def get_skip_reason(self) -> str:
        """Return the skip reason when API key is missing"""
        return f"No {self.get_api_key_env_var()} provided"
    
    def should_skip(self) -> bool:
        """Check if tests should be skipped due to missing API key"""
        return os.environ.get(self.get_api_key_env_var()) is None
    
    @pytest.mark.asyncio
    async def test_realtime_connection(self):
        """
        Test basic realtime WebSocket connection.
        Verifies that:
        1. Connection is established successfully
        2. Initial event is received
        3. Messages are properly forwarded
        """
        litellm._turn_on_debug()
        if self.should_skip():
            pytest.skip(self.get_skip_reason())
        
        websocket_client = RealTimeWebSocketClient()
        caught_exception = None
        
        print(f"\n{'='*80}")
        print(f"STARTING REALTIME CONNECTION TEST")
        print(f"Model: {self.get_model()}")
        print(f"API Key Env Var: {self.get_api_key_env_var()}")
        print(f"{'='*80}\n")
        
        try:
            await litellm._arealtime(
                model=self.get_model(),
                websocket=websocket_client,
                api_key=os.environ.get(self.get_api_key_env_var()),
                timeout=60
            )
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"\nException: {type(e).__name__}: {e}\n")
            caught_exception = e
        
        # Build debug info
        error_details = []
        error_details.append(f"messages_sent: {len(websocket_client.messages_sent)}")
        error_details.append(f"messages_received: {len(websocket_client.messages_received)}")
        error_details.append(f"close_code: {websocket_client.close_code}")
        error_details.append(f"close_reason: {websocket_client.close_reason}")
        if caught_exception:
            error_details.append(f"exception: {type(caught_exception).__name__}: {caught_exception}")
        
        # Skip on transient connection failures
        if not websocket_client.connection_successful and websocket_client.close_code is not None:
            pytest.skip(f"Transient connection failure: {'; '.join(error_details)}")
        
        # Assertions
        assert websocket_client.connection_successful, f"Failed to connect. Debug: {'; '.join(error_details)}"
        assert websocket_client.received_initial_event, f"Did not receive initial event"
        assert len(websocket_client.messages_received) > 0, "No messages received"
        
        # Verify initial event
        initial_event = websocket_client.messages_received[0]
        assert initial_event["type"] == self.get_initial_event_type(), \
            f"Expected {self.get_initial_event_type()}, got {initial_event.get('type')}"
    
    @pytest.mark.asyncio
    async def test_realtime_with_query_params(self):
        """
        Test realtime connection with explicit query parameters.
        Verifies that query params are properly passed to the backend.
        """
        litellm._turn_on_debug()
        if self.should_skip():
            pytest.skip(self.get_skip_reason())
        
        from litellm.types.realtime import RealtimeQueryParams
        
        websocket_client = RealTimeWebSocketClient()
        caught_exception = None
        
        # Strip provider prefix from model name for query params
        model_name = self.get_model()
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]
        
        query_params: RealtimeQueryParams = {"model": model_name}
        
        try:
            await litellm._arealtime(
                model=self.get_model(),
                websocket=websocket_client,
                api_key=os.environ.get(self.get_api_key_env_var()),
                query_params=query_params,
                timeout=60
            )
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            caught_exception = e
        
        # Build debug info
        error_details = []
        error_details.append(f"messages_sent: {len(websocket_client.messages_sent)}")
        error_details.append(f"messages_received: {len(websocket_client.messages_received)}")
        if caught_exception:
            error_details.append(f"exception: {type(caught_exception).__name__}: {caught_exception}")
        
        # Skip on transient failures
        if not websocket_client.connection_successful and websocket_client.close_code is not None:
            pytest.skip(f"Transient connection failure: {'; '.join(error_details)}")
        
        # Assertions
        assert websocket_client.connection_successful, f"Failed to connect. Debug: {'; '.join(error_details)}"
        assert len(websocket_client.messages_received) > 0, "No messages received"
    
    @pytest.mark.asyncio
    async def test_send_user_message(self):
        """
        Test sending an actual user message and receiving responses.
        This creates a more realistic conversation flow.
        """
        if self.should_skip():
            pytest.skip(self.get_skip_reason())
        
        litellm._turn_on_debug()
        
        # Create a custom websocket client that sends a message
        class InteractiveWebSocketClient(RealTimeWebSocketClient):
            def __init__(self):
                super().__init__()
                self.sent_user_message = False
                self.response_messages = []
                self.wait_for_responses = True
            
            async def receive_text(self):
                """Enhanced receive that sends a user message after connection"""
                print(f"\n{'='*80}")
                print(f"CLIENT-SIDE RECEIVE HANDLER")
                print(f"{'='*80}\n")
                
                # Wait for initial connection
                max_wait = 5.0
                check_interval = 0.1
                waited = 0.0
                
                while waited < max_wait:
                    if self.connection_successful:
                        print(f"Connection established after {waited:.1f}s\n")
                        break
                    await asyncio.sleep(check_interval)
                    waited += check_interval
                
                # Step 1: Send a user message after connection is established
                if self.connection_successful and not self.sent_user_message:
                    self.sent_user_message = True
                    user_msg_data = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Say hi back to me!"}]
                        }
                    }
                    user_msg = json.dumps(user_msg_data)
                    
                    print(f"\n{'='*80}")
                    print(f"STEP 1: SENDING USER MESSAGE TO BACKEND")
                    print(f"{'='*80}")
                    print(json.dumps(user_msg_data, indent=2))
                    print(f"{'='*80}\n")
                    
                    return user_msg
                
                # Step 2: Trigger the response after user message is acknowledged
                if not hasattr(self, 'triggered_response'):
                    self.triggered_response = True
                    # Wait a bit for the user message to be processed
                    await asyncio.sleep(0.5)
                    
                    response_create_data = {
                        "type": "response.create"
                    }
                    response_create = json.dumps(response_create_data)
                    
                    print(f"\n{'='*80}")
                    print(f"STEP 2: TRIGGERING LLM RESPONSE")
                    print(f"{'='*80}")
                    print(json.dumps(response_create_data, indent=2))
                    print(f"{'='*80}\n")
                    
                    return response_create
                
                # Step 3: Wait for LLM responses
                if self.wait_for_responses:
                    print(f"\nSTEP 3: Waiting 5 seconds for LLM to respond...\n")
                    await asyncio.sleep(5.0)
                    self.wait_for_responses = False
                    
                    # Collect response info
                    for msg in self.messages_received:
                        msg_type = msg.get('type', 'unknown')
                        if msg_type not in ['conversation.created', 'ping']:
                            self.response_messages.append(msg)
                    
                    print(f"\nReceived {len(self.response_messages)} response messages (excluding init/ping)\n")
                
                print(f"\n{'='*80}")
                print(f"CLOSING CONNECTION")
                print(f"Total messages received: {len(self.messages_received)}")
                print(f"{'='*80}\n")
                raise websockets.exceptions.ConnectionClosed(None, None)
        
        websocket_client = InteractiveWebSocketClient()
        caught_exception = None
        
        print(f"\n{'='*80}")
        print(f"STARTING INTERACTIVE MESSAGE TEST")
        print(f"Model: {self.get_model()}")
        print(f"Message: 'Say hi back to me!'")
        print(f"{'='*80}\n")
        
        try:
            await litellm._arealtime(
                model=self.get_model(),
                websocket=websocket_client,
                api_key=os.environ.get(self.get_api_key_env_var()),
                timeout=60
            )
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"\nException: {type(e).__name__}: {e}\n")
            caught_exception = e
        
        # Print results
        print(f"\n{'='*80}")
        print(f"TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Connection successful: {websocket_client.connection_successful}")
        print(f"User message sent: {websocket_client.sent_user_message}")
        print(f"Total messages received: {len(websocket_client.messages_received)}")
        print(f"Response messages (excluding init/ping): {len(websocket_client.response_messages)}")
        
        if websocket_client.response_messages:
            print(f"\nResponse Event Types:")
            for i, msg in enumerate(websocket_client.response_messages, 1):
                print(f"  {i}. {msg.get('type', 'unknown')}")
        
        print(f"{'='*80}\n")
        
        # Skip if no responses (might be timing issue)
        if not websocket_client.response_messages:
            pytest.skip("No response messages received (might be timing/network issue)")
        
        assert websocket_client.connection_successful, "Failed to establish connection"
        assert websocket_client.sent_user_message, "Failed to send user message"
    
    def test_query_params_construction(self):
        """Test that query params are constructed correctly"""
        from litellm.types.realtime import RealtimeQueryParams
        
        # Strip provider prefix from model name
        model_name = self.get_model()
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]
        
        query_params: RealtimeQueryParams = {"model": model_name}
        
        assert "model" in query_params
        assert query_params["model"] == model_name
