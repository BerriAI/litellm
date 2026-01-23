"""
This file contains the transformation logic for the Bedrock Nova Sonic realtime API.

Bedrock Nova Sonic uses bidirectional streaming with the InvokeModelWithBidirectionalStream API.
"""

import json
import uuid as uuid_module
from typing import Any, Dict, List, Optional, Union, cast

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.llms.openai import (
    OpenAIRealtimeContentPartDone,
    OpenAIRealtimeConversationItemCreated,
    OpenAIRealtimeDoneEvent,
    OpenAIRealtimeEvents,
    OpenAIRealtimeEventTypes,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseAudioDone,
    OpenAIRealtimeResponseContentPartAdded,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeResponseDoneObject,
    OpenAIRealtimeResponseTextDone,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeStreamSession,
    OpenAIRealtimeStreamSessionEvents,
    OpenAIRealtimeTurnDetection,
)
from litellm.types.realtime import (
    ALL_DELTA_TYPES,
    RealtimeModalityResponseTransformOutput,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)
from litellm.utils import get_empty_usage

from ..base_aws_llm import BaseAWSLLM

# Map OpenAI voice names to Bedrock Nova Sonic voice IDs
OPENAI_TO_BEDROCK_VOICE_MAP = {
    "alloy": "matthew",
    "echo": "matthew",
    "fable": "ruth",
    "onyx": "matthew",
    "nova": "ruth",
    "shimmer": "ruth",
}


class BedrockRealtimeConfig(BaseRealtimeConfig, BaseAWSLLM):
    """
    Configuration for Bedrock Nova Sonic realtime API.
    
    Transforms between OpenAI realtime API format and Bedrock's bidirectional streaming format.
    """

    def __init__(self):
        super().__init__()
        self.prompt_name = str(uuid_module.uuid4())
        self.content_name = str(uuid_module.uuid4())

    def validate_environment(
        self, headers: dict, model: str, api_key: Optional[str] = None
    ) -> dict:
        """Validate AWS credentials are available."""
        return headers

    def get_complete_url(
        self, api_base: Optional[str], model: str, api_key: Optional[str] = None
    ) -> str:
        """
        Get the API base URL for Bedrock (not WebSocket - AWS SDK handles the connection).
        
        Example output:
        "https://bedrock-runtime.us-west-1.amazonaws.com"
        """
        # If api_base is not provided, get region from credentials
        if api_base is None:
            from botocore.credentials import Credentials
            
            credentials = self.get_credentials(
                aws_access_key_id=api_key,
                aws_secret_access_key=None,
                aws_session_token=None,
                aws_region_name=None,
            )
            # credentials can be either Boto3CredentialsInfo or Credentials (from cache)
            if isinstance(credentials, Credentials):
                # If cached, it's just a Credentials object - use default region
                region = "us-west-1"
            else:
                # It's a Boto3CredentialsInfo object
                region = credentials.aws_region_name
            api_base = f"https://bedrock-runtime.{region}.amazonaws.com"
        
        return api_base

    def map_openai_voice_to_bedrock(self, voice: Optional[str]) -> str:
        """Map OpenAI voice names to Bedrock voice IDs."""
        if voice is None:
            return "matthew"  # Default voice
        return OPENAI_TO_BEDROCK_VOICE_MAP.get(voice.lower(), "matthew")

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: Optional[str] = None,
    ) -> List[str]:
        """
        Transform OpenAI realtime request format to Bedrock format.
        
        Bedrock expects events in this sequence:
        1. sessionStart
        2. promptStart
        3. contentStart
        4. textInput (or audioInput)
        5. contentEnd
        6. promptEnd
        7. sessionEnd (when done)
        """
        try:
            json_message = json.loads(message)
        except json.JSONDecodeError:
            if isinstance(message, bytes):
                message_str = message.decode("utf-8", errors="replace")
            else:
                message_str = str(message)
            raise ValueError(f"Invalid JSON message: {message_str}")

        messages: List[str] = []
        
        # Handle session.update - this sets up the session configuration
        if "type" in json_message and json_message["type"] == "session.update":
            session = json_message.get("session", {})
            
            # Extract configuration
            temperature = session.get("temperature", 0.7)
            max_tokens = session.get("max_response_output_tokens", 1024)
            voice = session.get("voice", "alloy")
            bedrock_voice = self.map_openai_voice_to_bedrock(voice)
            
            # Store configuration for later use
            self.session_config = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "voice": bedrock_voice,
            }
            
            # Don't send anything yet - wait for actual content
            return messages
        
        # Handle input_audio_buffer.append - audio input
        elif "type" in json_message and json_message["type"] == "input_audio_buffer.append":
            audio_data = json_message.get("audio", "")
            
            # Create full event sequence for audio input
            events = self._create_audio_input_events(audio_data)
            messages.append(json.dumps({"events": events}))
        
        # Handle response.create - text input
        elif "type" in json_message and json_message["type"] == "response.create":
            # Extract text from the conversation
            response_data = json_message.get("response", {})
            # For now, we'll handle this as a trigger to start processing
            # The actual text should come from previous messages
            pass
        
        # Handle conversation.item.create - text message
        elif "type" in json_message and json_message["type"] == "conversation.item.create":
            item = json_message.get("item", {})
            content = item.get("content", [])
            
            # Extract text from content
            text_content = ""
            for part in content:
                if part.get("type") == "input_text":
                    text_content = part.get("text", "")
                elif part.get("type") == "text":
                    text_content = part.get("text", "")
            
            if text_content:
                events = self._create_text_input_events(text_content)
                messages.append(json.dumps({"events": events}))
        
        return messages

    def _create_text_input_events(self, text: str) -> List[Dict[str, Any]]:
        """Create the event sequence for text input."""
        config = getattr(self, "session_config", {})
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1024)
        voice = config.get("voice", "matthew")
        
        return [
            {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": max_tokens,
                            "topP": 0.9,
                            "temperature": temperature,
                        }
                    }
                }
            },
            {
                "event": {
                    "promptStart": {
                        "promptName": self.prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 24000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": voice,
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            },
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                        "type": "TEXT",
                        "interactive": False,
                        "role": "USER",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            },
            {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                        "content": text,
                    }
                }
            },
            {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                    }
                }
            },
            {
                "event": {
                    "promptEnd": {
                        "promptName": self.prompt_name,
                    }
                }
            },
            {"event": {"sessionEnd": {}}},
        ]

    def _create_audio_input_events(self, audio_data: str) -> List[Dict[str, Any]]:
        """Create the event sequence for audio input."""
        config = getattr(self, "session_config", {})
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1024)
        voice = config.get("voice", "matthew")
        
        return [
            {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": max_tokens,
                            "topP": 0.9,
                            "temperature": temperature,
                        }
                    }
                }
            },
            {
                "event": {
                    "promptStart": {
                        "promptName": self.prompt_name,
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 16000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "encoding": "base64",
                        },
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 24000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": voice,
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            },
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                        "type": "AUDIO",
                        "interactive": False,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 16000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "encoding": "base64",
                        },
                    }
                }
            },
            {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                        "content": audio_data,
                    }
                }
            },
            {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                    }
                }
            },
            {
                "event": {
                    "promptEnd": {
                        "promptName": self.prompt_name,
                    }
                }
            },
            {"event": {"sessionEnd": {}}},
        ]

    def transform_realtime_response(
        self,
        message: Union[str, bytes],
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        """
        Transform Bedrock realtime response to OpenAI format.
        
        Bedrock sends events like:
        - sessionStarted
        - promptStarted
        - contentStarted
        - textOutput / audioOutput (streaming chunks)
        - contentEnded
        - promptEnded
        - sessionEnded
        """
        try:
            json_message = json.loads(message)
        except json.JSONDecodeError:
            if isinstance(message, bytes):
                message_str = message.decode("utf-8", errors="replace")
            else:
                message_str = str(message)
            raise ValueError(f"Invalid JSON message: {message_str}")

        logging_session_id = logging_obj.litellm_trace_id
        
        current_output_item_id = realtime_response_transform_input["current_output_item_id"]
        current_response_id = realtime_response_transform_input["current_response_id"]
        current_conversation_id = realtime_response_transform_input["current_conversation_id"]
        current_delta_chunks = realtime_response_transform_input["current_delta_chunks"]
        session_configuration_request = realtime_response_transform_input["session_configuration_request"]
        current_item_chunks = realtime_response_transform_input["current_item_chunks"]
        current_delta_type = realtime_response_transform_input["current_delta_type"]
        
        returned_message: List[OpenAIRealtimeEvents] = []
        
        # Handle different Bedrock event types
        if "sessionStarted" in json_message:
            # Create session.created event
            session_event = self._create_session_event(model, logging_session_id)
            returned_message.append(session_event)
            session_configuration_request = json.dumps(session_event)
        
        elif "textOutput" in json_message:
            # Handle text output chunks
            text_chunk = json_message["textOutput"].get("content", "")
            
            if not current_response_id:
                current_response_id = f"resp_{uuid.uuid4()}"
            if not current_output_item_id:
                current_output_item_id = f"item_{uuid.uuid4()}"
            if not current_conversation_id:
                current_conversation_id = f"conv_{uuid.uuid4()}"
            
            # Create initial events if this is the first chunk
            if current_delta_chunks is None:
                current_delta_chunks = []
                current_delta_type = "text"
                initial_events = self._create_initial_response_events(
                    current_response_id,
                    current_output_item_id,
                    current_conversation_id,
                    "text"
                )
                returned_message.extend(initial_events)
            
            # Create text delta event
            delta_event = OpenAIRealtimeResponseDelta(
                type="response.text.delta",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
                delta=text_chunk,
            )
            returned_message.append(delta_event)
            current_delta_chunks.append(delta_event)
        
        elif "audioOutput" in json_message:
            # Handle audio output chunks
            audio_chunk = json_message["audioOutput"].get("content", "")
            
            if not current_response_id:
                current_response_id = f"resp_{uuid.uuid4()}"
            if not current_output_item_id:
                current_output_item_id = f"item_{uuid.uuid4()}"
            if not current_conversation_id:
                current_conversation_id = f"conv_{uuid.uuid4()}"
            
            # Create initial events if this is the first chunk
            if current_delta_chunks is None:
                current_delta_chunks = []
                current_delta_type = "audio"
                initial_events = self._create_initial_response_events(
                    current_response_id,
                    current_output_item_id,
                    current_conversation_id,
                    "audio"
                )
                returned_message.extend(initial_events)
            
            # Create audio delta event
            delta_event = OpenAIRealtimeResponseDelta(
                type="response.audio.delta",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
                delta=audio_chunk,
            )
            returned_message.append(delta_event)
            # Don't accumulate audio chunks to avoid memory issues
        
        elif "contentEnded" in json_message or "promptEnded" in json_message or "sessionEnded" in json_message:
            # Create done events
            if current_delta_type and current_output_item_id and current_response_id:
                done_events = self._create_done_events(
                    current_output_item_id,
                    current_response_id,
                    current_conversation_id,
                    current_delta_chunks,
                    current_delta_type,
                )
                returned_message.extend(done_events)
                
                # Reset state
                current_delta_chunks = None
                current_output_item_id = None
                current_response_id = None
                current_delta_type = None
        
        return {
            "response": returned_message,
            "current_output_item_id": current_output_item_id,
            "current_response_id": current_response_id,
            "current_delta_chunks": current_delta_chunks,
            "current_conversation_id": current_conversation_id,
            "current_item_chunks": current_item_chunks,
            "current_delta_type": current_delta_type,
            "session_configuration_request": session_configuration_request,
        }

    def _create_session_event(
        self, model: str, session_id: str
    ) -> OpenAIRealtimeStreamSessionEvents:
        """Create a session.created event."""
        return OpenAIRealtimeStreamSessionEvents(
            type="session.created",
            session=OpenAIRealtimeStreamSession(
                id=session_id,
                model=model.replace("bedrock/", ""),
                modalities=["text", "audio"],
            ),
            event_id=f"event_{uuid.uuid4()}",
        )

    def _create_initial_response_events(
        self,
        response_id: str,
        output_item_id: str,
        conversation_id: str,
        delta_type: ALL_DELTA_TYPES,
    ) -> List[OpenAIRealtimeEvents]:
        """Create initial events when starting a new response."""
        events: List[OpenAIRealtimeEvents] = []
        
        # response.created
        events.append(
            OpenAIRealtimeStreamResponseBaseObject(
                type="response.created",
                event_id=f"event_{uuid.uuid4()}",
                response={
                    "object": "realtime.response",
                    "id": response_id,
                    "status": "in_progress",
                    "output": [],
                    "conversation_id": conversation_id,
                    "modalities": [delta_type],
                },
            )
        )
        
        # response.output_item.added
        events.append(
            OpenAIRealtimeStreamResponseOutputItemAdded(
                type="response.output_item.added",
                response_id=response_id,
                output_index=0,
                item={
                    "id": output_item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            )
        )
        
        # conversation.item.created
        events.append(
            OpenAIRealtimeConversationItemCreated(
                type="conversation.item.created",
                event_id=f"event_{uuid.uuid4()}",
                item={
                    "id": output_item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            )
        )
        
        # response.content_part.added
        events.append(
            OpenAIRealtimeResponseContentPartAdded(
                type="response.content_part.added",
                content_index=0,
                output_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=output_item_id,
                part={
                    "type": delta_type,
                    "text": "" if delta_type == "text" else None,
                    "transcript": "" if delta_type == "audio" else None,
                },
                response_id=response_id,
            )
        )
        
        return events

    def _create_done_events(
        self,
        output_item_id: str,
        response_id: str,
        conversation_id: Optional[str],
        delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]],
        delta_type: ALL_DELTA_TYPES,
    ) -> List[OpenAIRealtimeEvents]:
        """Create done events when response is complete."""
        events: List[OpenAIRealtimeEvents] = []
        
        # Accumulate text if available
        text_content = ""
        if delta_chunks and delta_type == "text":
            text_content = "".join([chunk["delta"] for chunk in delta_chunks])
        
        # response.text.done or response.audio.done
        if delta_type == "text":
            events.append(
                OpenAIRealtimeResponseTextDone(
                    type="response.text.done",
                    content_index=0,
                    event_id=f"event_{uuid.uuid4()}",
                    item_id=output_item_id,
                    output_index=0,
                    response_id=response_id,
                    text=text_content,
                )
            )
        else:
            events.append(
                OpenAIRealtimeResponseAudioDone(
                    type="response.audio.done",
                    content_index=0,
                    event_id=f"event_{uuid.uuid4()}",
                    item_id=output_item_id,
                    output_index=0,
                    response_id=response_id,
                )
            )
        
        # response.content_part.done
        events.append(
            OpenAIRealtimeContentPartDone(
                type="response.content_part.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=output_item_id,
                output_index=0,
                part={
                    "type": delta_type,
                    "text": text_content if delta_type == "text" else None,
                    "transcript": "" if delta_type == "audio" else None,
                },
                response_id=response_id,
            )
        )
        
        # response.output_item.done
        events.append(
            OpenAIRealtimeOutputItemDone(
                type="response.output_item.done",
                event_id=f"event_{uuid.uuid4()}",
                output_index=0,
                response_id=response_id,
                item={
                    "id": output_item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": delta_type,
                            "text": text_content if delta_type == "text" else None,
                            "transcript": "" if delta_type == "audio" else None,
                        }
                    ],
                },
            )
        )
        
        # response.done
        events.append(
            OpenAIRealtimeDoneEvent(
                type="response.done",
                event_id=f"event_{uuid.uuid4()}",
                response=OpenAIRealtimeResponseDoneObject(
                    object="realtime.response",
                    id=response_id,
                    status="completed",
                    output=[
                        {
                            "id": output_item_id,
                            "object": "realtime.item",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": delta_type,
                                    "text": text_content if delta_type == "text" else None,
                                    "transcript": "" if delta_type == "audio" else None,
                                }
                            ],
                        }
                    ],
                    conversation_id=conversation_id or f"conv_{uuid.uuid4()}",
                    modalities=[delta_type],
                    usage=get_empty_usage().model_dump(),
                ),
            )
        )
        
        return events

    def requires_session_configuration(self) -> bool:
        """Bedrock requires session configuration."""
        return True

    def session_configuration_request(self, model: str) -> Optional[str]:
        """Return default session configuration."""
        session_event = self._create_session_event(model, str(uuid.uuid4()))
        return json.dumps(session_event)
