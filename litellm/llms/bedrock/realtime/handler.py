"""
Bedrock Nova Sonic realtime handler using native HTTP/2 bidirectional streaming.

This bridges the user's WebSocket connection to LiteLLM with Bedrock's HTTP/2
bidirectional streaming protocol using AWS SigV4 signing.
"""

import asyncio
import json
from typing import Any, Optional

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

from ..base_aws_llm import BaseAWSLLM
from .transformation import BedrockRealtimeConfig


class BedrockRealtime(BaseAWSLLM):
    """Handler for Bedrock Nova Sonic realtime using native HTTP/2 bidirectional streaming."""

    def __init__(self):
        super().__init__()
        self.config = BedrockRealtimeConfig()

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        logging_obj: LiteLLMLogging,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
        aws_region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Bridge user's WebSocket to Bedrock HTTP/2 bidirectional stream.
        
        Args:
            model: Model name (e.g., "bedrock/amazon.nova-sonic-v1:0")
            websocket: FastAPI WebSocket connection from user
            logging_obj: LiteLLM logging object
            api_base: Optional API base URL
            api_key: AWS access key ID (or None to use environment)
            aws_region_name: AWS region
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
        """
        try:
            # Import required libraries for HTTP/2 and AWS signing
            import httpx
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError as e:
            error_msg = f"Required libraries not available: {str(e)}. Install with: pip install httpx botocore"
            verbose_logger.error(error_msg)
            await websocket.close(code=1011, reason=error_msg)
            return

        # Determine AWS region
        if aws_region_name is None:
            # Try to get region from model ARN
            region = self._get_aws_region_from_model_arn(model)
            if region is None:
                # Default to us-west-2 if no region specified
                region = "us-west-2"
        else:
            region = aws_region_name

        # Get AWS credentials
        credentials = self.get_credentials(
            aws_access_key_id=aws_access_key_id or api_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=region,
        )

        model_id = model.replace("bedrock/", "")
        endpoint = api_base or f"https://bedrock-runtime.{region}.amazonaws.com"

        verbose_logger.debug(f"Connecting to Bedrock Nova Sonic: model={model_id}, region={region}")

        try:
            # Log the request
            logging_obj.pre_call(
                input=None,
                api_key=credentials.access_key,
                additional_args={
                    "api_base": endpoint,
                    "model": model_id,
                    "region": region,
                },
            )

            # Create HTTP/2 bidirectional stream to Bedrock
            await self._bridge_streams_http2(
                websocket=websocket,
                endpoint=endpoint,
                model_id=model_id,
                credentials=credentials,
                region=region,
                logging_obj=logging_obj,
                model=model,
            )

        except Exception as e:
            verbose_logger.exception(f"Error in Bedrock realtime: {e}")
            try:
                await websocket.close(
                    code=1011, reason=f"Internal server error: {str(e)}"
                )
            except RuntimeError as close_error:
                if "already completed" in str(close_error) or "websocket.close" in str(
                    close_error
                ):
                    pass
                else:
                    raise

    async def _bridge_streams_http2(
        self,
        websocket: Any,
        endpoint: str,
        model_id: str,
        credentials: Any,
        region: str,
        logging_obj: LiteLLMLogging,
        model: str,
    ):
        """
        Bridge messages between user's WebSocket and Bedrock's HTTP/2 bidirectional stream.
        
        Args:
            websocket: User's WebSocket connection
            endpoint: Bedrock endpoint URL
            model_id: Model ID
            credentials: AWS credentials
            region: AWS region
            logging_obj: Logging object
            model: Model name
        """
        import httpx
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        # Construct the URL for bidirectional streaming
        url = f"{endpoint}/model/{model_id}/invoke-with-bidirectional-stream"
        
        # Create a signed request
        headers = {
            "Content-Type": "application/vnd.amazon.eventstream",
            "Accept": "application/vnd.amazon.eventstream",
        }
        
        # Sign the request using AWS SigV4
        request = AWSRequest(method="POST", url=url, headers=headers, data=b"")
        SigV4Auth(credentials, "bedrock", region).add_auth(request)
        
        # Extract signed headers
        signed_headers = dict(request.headers)
        
        verbose_logger.debug(f"Connecting to Bedrock HTTP/2 stream: {url}")
        
        # Create a queue for request data
        import asyncio
        request_queue = asyncio.Queue()
        
        # Create HTTP/2 client
        async with httpx.AsyncClient(http2=True, timeout=None) as http_client:
            # Create bidirectional stream using HTTP/2
            async with http_client.stream(
                "POST",
                url,
                headers=signed_headers,
                content=self._generate_request_stream_from_queue(request_queue),
            ) as response:
                verbose_logger.debug(f"Bedrock stream established: status={response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"Failed to establish stream: {response.status_code}"
                    try:
                        error_body = await response.aread()
                        verbose_logger.error(f"Error response: {error_body}")
                    except:
                        pass
                    verbose_logger.error(error_msg)
                    await websocket.close(code=1011, reason=error_msg[:100])  # Limit reason length
                    return
                
                # Run request handling and response handling concurrently
                await asyncio.gather(
                    self._handle_websocket_to_bedrock(websocket, model, request_queue),
                    self._forward_bedrock_to_user(
                        response=response,
                        websocket=websocket,
                        logging_obj=logging_obj,
                        model=model,
                    ),
                    return_exceptions=True,
                )

    async def _generate_request_stream(self, websocket: Any, model: str):
        """
        Generate request stream from user WebSocket messages.
        
        Args:
            websocket: User's WebSocket connection
            model: Model name
            
        Yields:
            Encoded event stream data
        """
        try:
            while True:
                # Receive message from user's WebSocket
                message = await websocket.receive_text()
                verbose_logger.debug(f"Received from user: {message[:200]}...")
                
                # Transform OpenAI format to Bedrock format
                bedrock_messages = self.config.transform_realtime_request(
                    message=message,
                    model=model,
                    session_configuration_request=None,
                )
                
                # Yield each transformed message as event stream data
                for bedrock_msg in bedrock_messages:
                    # Encode as event stream format
                    event_data = self._encode_event_stream_message(bedrock_msg)
                    yield event_data
                    verbose_logger.debug(f"Sent to Bedrock: {bedrock_msg[:200]}...")
                    
        except Exception as e:
            verbose_logger.debug(f"Request stream ended: {e}")

    async def _generate_request_stream_from_queue(self, request_queue):
        """
        Generate request stream from a queue.
        
        Args:
            request_queue: asyncio.Queue containing encoded event stream data
            
        Yields:
            Encoded event stream data
        """
        try:
            while True:
                # Get data from queue
                data = await request_queue.get()
                if data is None:  # Sentinel value to end stream
                    break
                yield data
        except Exception as e:
            verbose_logger.debug(f"Request stream from queue ended: {e}")

    async def _handle_websocket_to_bedrock(self, websocket: Any, model: str, request_queue):
        """
        Handle messages from WebSocket and put them in the request queue.
        
        Args:
            websocket: User's WebSocket connection
            model: Model name
            request_queue: asyncio.Queue to put encoded messages
        """
        try:
            while True:
                # Receive message from user's WebSocket
                message = await websocket.receive_text()
                verbose_logger.debug(f"Received from user: {message[:200]}...")
                
                # Transform OpenAI format to Bedrock format
                bedrock_messages = self.config.transform_realtime_request(
                    message=message,
                    model=model,
                    session_configuration_request=None,
                )
                
                # Put each transformed message in the queue
                for bedrock_msg in bedrock_messages:
                    # Encode as event stream format
                    event_data = self._encode_event_stream_message(bedrock_msg)
                    await request_queue.put(event_data)
                    verbose_logger.debug(f"Sent to Bedrock: {bedrock_msg[:200]}...")
                    
        except Exception as e:
            verbose_logger.debug(f"WebSocket to Bedrock handler ended: {e}")
        finally:
            # Signal end of stream
            await request_queue.put(None)

    def _encode_event_stream_message(self, message: str) -> bytes:
        """
        Encode a message in AWS event stream format.
        
        Args:
            message: JSON message string
            
        Returns:
            Encoded event stream bytes
        """
        import struct
        import binascii
        
        # Convert message to bytes
        payload = message.encode('utf-8')
        
        # AWS event stream format:
        # Prelude (12 bytes):
        #   - Total byte length (4 bytes, big-endian uint32)
        #   - Headers byte length (4 bytes, big-endian uint32)  
        #   - Prelude CRC (4 bytes, big-endian uint32)
        # Headers (variable, can be 0 bytes)
        # Payload (variable)
        # Message CRC (4 bytes, big-endian uint32)
        
        headers_bytes = b""  # No headers for now
        headers_length = len(headers_bytes)
        
        # Calculate total length (prelude + headers + payload + message CRC)
        total_length = 12 + headers_length + len(payload) + 4
        
        # Build prelude (without CRC yet)
        prelude_without_crc = struct.pack('>II', total_length, headers_length)
        
        # Calculate prelude CRC
        prelude_crc = binascii.crc32(prelude_without_crc) & 0xFFFFFFFF
        prelude = prelude_without_crc + struct.pack('>I', prelude_crc)
        
        # Build message (without final CRC)
        message_without_crc = prelude + headers_bytes + payload
        
        # Calculate message CRC
        message_crc = binascii.crc32(message_without_crc) & 0xFFFFFFFF
        
        # Build complete message
        complete_message = message_without_crc + struct.pack('>I', message_crc)
        
        return complete_message

    async def _forward_bedrock_to_user(
        self,
        response: Any,
        websocket: Any,
        logging_obj: LiteLLMLogging,
        model: str,
    ):
        """
        Forward responses from Bedrock to user's WebSocket.
        
        Args:
            response: HTTP/2 streaming response
            websocket: User's WebSocket connection
            logging_obj: Logging object
            model: Model name
        """
        from botocore.eventstream import EventStreamBuffer
        
        try:
            current_state = {
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": None,
                "current_item_chunks": None,
                "current_delta_type": None,
                "session_configuration_request": None,
            }
            
            # Create event stream buffer for decoding
            event_buffer = EventStreamBuffer()
            
            # Read event stream from response
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                
                verbose_logger.debug(f"Received chunk from Bedrock: {len(chunk)} bytes")
                    
                # Add chunk to event stream buffer
                event_buffer.add_data(chunk)
                
                # Decode messages from buffer
                messages = self._decode_event_stream_buffer(event_buffer)
                
                for raw_response in messages:
                    verbose_logger.debug(f"Received from Bedrock: {raw_response[:200]}...")
                    
                    # Transform Bedrock format to OpenAI format
                    result = self.config.transform_realtime_response(
                        message=raw_response,
                        model=model,
                        logging_obj=logging_obj,
                        realtime_response_transform_input=current_state,
                    )
                    
                    # Update state
                    current_state.update({
                        "current_output_item_id": result["current_output_item_id"],
                        "current_response_id": result["current_response_id"],
                        "current_conversation_id": result["current_conversation_id"],
                        "current_delta_chunks": result["current_delta_chunks"],
                        "current_item_chunks": result["current_item_chunks"],
                        "current_delta_type": result["current_delta_type"],
                        "session_configuration_request": result["session_configuration_request"],
                    })
                    
                    # Send transformed events to user
                    response_data = result["response"]
                    if isinstance(response_data, list):
                        for event_obj in response_data:
                            event_str = json.dumps(event_obj)
                            await websocket.send_text(event_str)
                            verbose_logger.debug(f"Sent to user: {event_str[:200]}...")
                    else:
                        event_str = json.dumps(response_data)
                        await websocket.send_text(event_str)
                        verbose_logger.debug(f"Sent to user: {event_str[:200]}...")
                        
        except Exception as e:
            verbose_logger.exception(f"Forward to user ended: {e}")

    def _decode_event_stream_buffer(self, buffer: Any) -> list[str]:
        """
        Decode AWS event stream using EventStreamBuffer.
        
        Args:
            buffer: EventStreamBuffer instance
            
        Returns:
            List of decoded JSON message strings
        """
        messages = []
        
        for event in buffer:
            try:
                # Get the payload from the event
                if hasattr(event, 'payload'):
                    payload = event.payload
                elif hasattr(event, 'to_response_dict'):
                    response_dict = event.to_response_dict()
                    if 'body' in response_dict:
                        payload = response_dict['body']
                    else:
                        continue
                else:
                    continue
                
                # Decode payload
                if isinstance(payload, bytes):
                    message = payload.decode('utf-8')
                    messages.append(message)
                elif isinstance(payload, str):
                    messages.append(payload)
                    
            except Exception as e:
                verbose_logger.warning(f"Failed to decode event: {e}")
                
        return messages
