"""
This file contains the handler for AWS Bedrock Nova Sonic realtime API.

This uses aws_sdk_bedrock_runtime for bidirectional streaming with Nova Sonic.
"""

import asyncio
import json
from typing import Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

from ..base_aws_llm import BaseAWSLLM
from .transformation import BedrockRealtimeConfig


class BedrockRealtime(BaseAWSLLM):
    """Handler for Bedrock Nova Sonic realtime speech-to-speech API."""

    def __init__(self):
        super().__init__()

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        logging_obj: LiteLLMLogging,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        aws_region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_role_name: Optional[str] = None,
        aws_session_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        aws_web_identity_token: Optional[str] = None,
        aws_sts_endpoint: Optional[str] = None,
        aws_bedrock_runtime_endpoint: Optional[str] = None,
        aws_external_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Establish bidirectional streaming connection with Bedrock Nova Sonic.

        Args:
            model: Model ID (e.g., 'amazon.nova-sonic-v1:0')
            websocket: Client WebSocket connection
            logging_obj: LiteLLM logging object
            aws_region_name: AWS region
            Various AWS authentication parameters
        """
        try:
            from aws_sdk_bedrock_runtime.client import (
                BedrockRuntimeClient,
                InvokeModelWithBidirectionalStreamOperationInput,
            )
            from aws_sdk_bedrock_runtime.config import Config
            from smithy_aws_core.identity.environment import (
                EnvironmentCredentialsResolver,
            )
        except ImportError:
            raise ImportError(
                "Missing aws_sdk_bedrock_runtime. Install with: pip install aws-sdk-bedrock-runtime"
            )

        # Get AWS region
        if aws_region_name is None:
            optional_params = {
                "aws_region_name": aws_region_name,
            }
            aws_region_name = self._get_aws_region_name(optional_params, model)

        # Get endpoint URL
        if api_base is not None:
            endpoint_uri = api_base
        elif aws_bedrock_runtime_endpoint is not None:
            endpoint_uri = aws_bedrock_runtime_endpoint
        else:
            endpoint_uri = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"

        verbose_proxy_logger.debug(
            f"Bedrock Realtime: Connecting to {endpoint_uri} with model {model}"
        )

        # Initialize Bedrock client with aws_sdk_bedrock_runtime
        config = Config(
            endpoint_uri=endpoint_uri,
            region=aws_region_name,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        bedrock_client = BedrockRuntimeClient(config=config)

        transformation_config = BedrockRealtimeConfig()

        try:
            # Initialize the bidirectional stream
            bedrock_stream = await bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=model)
            )

            verbose_proxy_logger.debug(
                "Bedrock Realtime: Bidirectional stream established"
            )

            # Track state for transformation
            session_state = {
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": None,
                "current_item_chunks": None,
                "current_delta_type": None,
                "session_configuration_request": None,
            }

            # Create tasks for bidirectional forwarding
            client_to_bedrock_task = asyncio.create_task(
                self._forward_client_to_bedrock(
                    websocket,
                    bedrock_stream,
                    transformation_config,
                    model,
                    session_state,
                )
            )

            bedrock_to_client_task = asyncio.create_task(
                self._forward_bedrock_to_client(
                    bedrock_stream,
                    websocket,
                    transformation_config,
                    model,
                    logging_obj,
                    session_state,
                )
            )

            # Wait for both tasks to complete
            await asyncio.gather(
                client_to_bedrock_task,
                bedrock_to_client_task,
                return_exceptions=True,
            )

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in BedrockRealtime.async_realtime: {e}"
            )
            try:
                await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
            except Exception:
                pass
            raise

    async def _forward_client_to_bedrock(
        self,
        client_ws: Any,
        bedrock_stream: Any,
        transformation_config: BedrockRealtimeConfig,
        model: str,
        session_state: dict,
    ):
        """Forward messages from client WebSocket to Bedrock stream."""
        try:
            from aws_sdk_bedrock_runtime.models import (
                BidirectionalInputPayloadPart,
                InvokeModelWithBidirectionalStreamInputChunk,
            )

            while True:
                # Receive message from client
                message = await client_ws.receive_text()
                verbose_proxy_logger.debug(
                    f"Bedrock Realtime: Received from client: {message[:200]}"
                )

                # Transform OpenAI format to Bedrock format
                transformed_messages = transformation_config.transform_realtime_request(
                    message=message,
                    model=model,
                    session_configuration_request=session_state.get(
                        "session_configuration_request"
                    ),
                )

                # Send transformed messages to Bedrock
                for bedrock_message in transformed_messages:
                    event = InvokeModelWithBidirectionalStreamInputChunk(
                        value=BidirectionalInputPayloadPart(
                            bytes_=bedrock_message.encode("utf-8")
                        )
                    )
                    await bedrock_stream.input_stream.send(event)
                    verbose_proxy_logger.debug(
                        f"Bedrock Realtime: Sent to Bedrock: {bedrock_message[:200]}"
                    )

        except Exception as e:
            verbose_proxy_logger.debug(
                f"Client to Bedrock forwarding ended: {e}", exc_info=True
            )
            # Close the Bedrock stream input
            try:
                await bedrock_stream.input_stream.close()
            except Exception:
                pass

    async def _forward_bedrock_to_client(
        self,
        bedrock_stream: Any,
        client_ws: Any,
        transformation_config: BedrockRealtimeConfig,
        model: str,
        logging_obj: LiteLLMLogging,
        session_state: dict,
    ):
        """Forward messages from Bedrock stream to client WebSocket."""
        try:
            while True:
                # Receive from Bedrock
                output = await bedrock_stream.await_output()
                result = await output[1].receive()

                if result.value and result.value.bytes_:
                    bedrock_response = result.value.bytes_.decode("utf-8")
                    verbose_proxy_logger.debug(
                        f"Bedrock Realtime: Received from Bedrock: {bedrock_response[:200]}"
                    )

                    # Transform Bedrock format to OpenAI format
                    from litellm.types.realtime import RealtimeResponseTransformInput
                    
                    realtime_response_transform_input: RealtimeResponseTransformInput = {
                        "current_output_item_id": session_state.get(
                            "current_output_item_id"
                        ),
                        "current_response_id": session_state.get("current_response_id"),
                        "current_conversation_id": session_state.get(
                            "current_conversation_id"
                        ),
                        "current_delta_chunks": session_state.get(
                            "current_delta_chunks"
                        ),
                        "current_item_chunks": session_state.get("current_item_chunks"),
                        "current_delta_type": session_state.get("current_delta_type"),
                        "session_configuration_request": session_state.get(
                            "session_configuration_request"
                        ),
                    }

                    transformed_response = (
                        transformation_config.transform_realtime_response(
                            message=bedrock_response,
                            model=model,
                            logging_obj=logging_obj,
                            realtime_response_transform_input=realtime_response_transform_input,
                        )
                    )

                    # Update session state
                    session_state.update(
                        {
                            "current_output_item_id": transformed_response.get(
                                "current_output_item_id"
                            ),
                            "current_response_id": transformed_response.get(
                                "current_response_id"
                            ),
                            "current_conversation_id": transformed_response.get(
                                "current_conversation_id"
                            ),
                            "current_delta_chunks": transformed_response.get(
                                "current_delta_chunks"
                            ),
                            "current_item_chunks": transformed_response.get(
                                "current_item_chunks"
                            ),
                            "current_delta_type": transformed_response.get(
                                "current_delta_type"
                            ),
                            "session_configuration_request": transformed_response.get(
                                "session_configuration_request"
                            ),
                        }
                    )

                    # Send transformed messages to client
                    openai_messages = transformed_response.get("response", [])
                    for openai_message in openai_messages:
                        message_json = json.dumps(openai_message)
                        await client_ws.send_text(message_json)
                        verbose_proxy_logger.debug(
                            f"Bedrock Realtime: Sent to client: {message_json[:200]}"
                        )

        except Exception as e:
            verbose_proxy_logger.debug(
                f"Bedrock to client forwarding ended: {e}", exc_info=True
            )
            # Close the client WebSocket
            try:
                await client_ws.close()
            except Exception:
                pass
