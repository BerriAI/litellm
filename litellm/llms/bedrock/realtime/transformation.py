"""
This file contains the transformation logic for Bedrock Nova Sonic realtime API.

Transforms between OpenAI Realtime API format and Bedrock Nova Sonic format.
"""

import json
import uuid as uuid_lib
from typing import Any, List, Optional, Union

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.llms.openai import (
    OpenAIRealtimeContentPartDone,
    OpenAIRealtimeDoneEvent,
    OpenAIRealtimeEvents,
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
)
from litellm.types.realtime import (
    ALL_DELTA_TYPES,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)
from litellm.utils import get_empty_usage


class BedrockRealtimeConfig(BaseRealtimeConfig):
    """Configuration for Bedrock Nova Sonic realtime transformations."""

    def __init__(self):
        # Track session state
        self.prompt_name = str(uuid_lib.uuid4())
        self.content_name = str(uuid_lib.uuid4())
        self.audio_content_name = str(uuid_lib.uuid4())
        
        # Default configuration values
        # Inference configuration
        self.max_tokens = 1024
        self.top_p = 0.9
        self.temperature = 0.7
        
        # Audio output configuration
        self.output_sample_rate_hertz = 24000
        self.output_sample_size_bits = 16
        self.output_channel_count = 1
        self.voice_id = "matthew"
        self.output_encoding = "base64"
        self.output_audio_type = "SPEECH"
        self.output_media_type = "audio/lpcm"
        
        # Audio input configuration
        self.input_sample_rate_hertz = 16000
        self.input_sample_size_bits = 16
        self.input_channel_count = 1
        self.input_encoding = "base64"
        self.input_audio_type = "SPEECH"
        self.input_media_type = "audio/lpcm"
        
        # Text configuration
        self.text_media_type = "text/plain"

    def validate_environment(
        self, headers: dict, model: str, api_key: Optional[str] = None
    ) -> dict:
        """Validate environment - no special validation needed for Bedrock."""
        return headers

    def get_complete_url(
        self, api_base: Optional[str], model: str, api_key: Optional[str] = None
    ) -> str:
        """Get complete URL - handled by aws_sdk_bedrock_runtime."""
        return api_base or ""

    def requires_session_configuration(self) -> bool:
        """Bedrock requires session configuration."""
        return True

    def session_configuration_request(self, model: str, tools: Optional[List[dict]] = None) -> str:
        """
        Create initial session configuration for Bedrock Nova Sonic.

        Args:
            model: Model ID
            tools: Optional list of tool definitions

        Returns JSON string with session start and prompt start events.
        """
        session_start = {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": self.max_tokens,
                        "topP": self.top_p,
                        "temperature": self.temperature,
                    }
                }
            }
        }

        prompt_start_config = {
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": self.text_media_type},
            "audioOutputConfiguration": {
                "mediaType": self.output_media_type,
                "sampleRateHertz": self.output_sample_rate_hertz,
                "sampleSizeBits": self.output_sample_size_bits,
                "channelCount": self.output_channel_count,
                "voiceId": self.voice_id,
                "encoding": self.output_encoding,
                "audioType": self.output_audio_type,
            },
        }

        # Add tool configuration if tools are provided
        if tools:
            prompt_start_config["toolUseOutputConfiguration"] = {
                "mediaType": "application/json"
            }
            prompt_start_config["toolConfiguration"] = {
                "tools": self._transform_tools_to_bedrock_format(tools)
            }

        prompt_start = {"event": {"promptStart": prompt_start_config}}

        # Return as a marker that we've sent the configuration
        return json.dumps(
            {"session_start": session_start, "prompt_start": prompt_start}
        )

    def _transform_tools_to_bedrock_format(self, tools: List[dict]) -> List[dict]:
        """
        Transform OpenAI tool format to Bedrock tool format.

        Args:
            tools: List of OpenAI format tools

        Returns:
            List of Bedrock format tools
        """
        bedrock_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                bedrock_tool = {
                    "toolSpec": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "inputSchema": {
                            "json": json.dumps(function.get("parameters", {}))
                        }
                    }
                }
                bedrock_tools.append(bedrock_tool)
        return bedrock_tools

    def _map_audio_format_to_sample_rate(self, audio_format: str, is_output: bool = True) -> int:
        """
        Map OpenAI audio format to sample rate.
        
        Args:
            audio_format: OpenAI audio format (pcm16, g711_ulaw, g711_alaw)
            is_output: Whether this is for output (True) or input (False)
            
        Returns:
            Sample rate in Hz
        """
        # OpenAI uses 24kHz for output and can vary for input
        # Bedrock Nova Sonic uses 24kHz for output and 16kHz for input by default
        if audio_format == "pcm16":
            return 24000 if is_output else 16000
        elif audio_format in ["g711_ulaw", "g711_alaw"]:
            return 8000  # G.711 typically uses 8kHz
        return 24000 if is_output else 16000

    def transform_session_update_event(self, json_message: dict) -> List[str]:
        """
        Transform session.update event to Bedrock session configuration.

        Args:
            json_message: OpenAI session.update message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling session.update")
        messages: List[str] = []
        
        session_config = json_message.get("session", {})
        
        # Update inference configuration from session if provided
        if "max_response_output_tokens" in session_config:
            self.max_tokens = session_config["max_response_output_tokens"]
        if "temperature" in session_config:
            self.temperature = session_config["temperature"]
        
        # Update audio output configuration from session if provided
        if "voice" in session_config:
            self.voice_id = session_config["voice"]
        if "output_audio_format" in session_config:
            output_format = session_config["output_audio_format"]
            self.output_sample_rate_hertz = self._map_audio_format_to_sample_rate(
                output_format, is_output=True
            )
        
        # Update audio input configuration from session if provided
        if "input_audio_format" in session_config:
            input_format = session_config["input_audio_format"]
            self.input_sample_rate_hertz = self._map_audio_format_to_sample_rate(
                input_format, is_output=False
            )
        
        # Allow direct override of sample rates if provided (custom extension)
        if "output_sample_rate_hertz" in session_config:
            self.output_sample_rate_hertz = session_config["output_sample_rate_hertz"]
        if "input_sample_rate_hertz" in session_config:
            self.input_sample_rate_hertz = session_config["input_sample_rate_hertz"]

        # Send session start
        session_start = {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": self.max_tokens,
                        "topP": self.top_p,
                        "temperature": self.temperature,
                    }
                }
            }
        }
        messages.append(json.dumps(session_start))

        # Send prompt start
        prompt_start_config = {
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": self.text_media_type},
            "audioOutputConfiguration": {
                "mediaType": self.output_media_type,
                "sampleRateHertz": self.output_sample_rate_hertz,
                "sampleSizeBits": self.output_sample_size_bits,
                "channelCount": self.output_channel_count,
                "voiceId": self.voice_id,
                "encoding": self.output_encoding,
                "audioType": self.output_audio_type,
            },
        }

        # Add tool configuration if tools are provided
        tools = session_config.get("tools")
        if tools:
            prompt_start_config["toolUseOutputConfiguration"] = {
                "mediaType": "application/json"
            }
            prompt_start_config["toolConfiguration"] = {
                "tools": self._transform_tools_to_bedrock_format(tools)
            }

        prompt_start = {"event": {"promptStart": prompt_start_config}}
        messages.append(json.dumps(prompt_start))

        # Send system prompt if provided
        instructions = session_config.get("instructions")
        if instructions:
            text_content_name = str(uuid_lib.uuid4())

            # Content start
            text_content_start = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name,
                        "type": "TEXT",
                        "interactive": False,
                        "role": "SYSTEM",
                        "textInputConfiguration": {"mediaType": self.text_media_type},
                    }
                }
            }
            messages.append(json.dumps(text_content_start))

            # Text input
            text_input = {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name,
                        "content": instructions,
                    }
                }
            }
            messages.append(json.dumps(text_input))

            # Content end
            text_content_end = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": text_content_name,
                    }
                }
            }
            messages.append(json.dumps(text_content_end))

        return messages

    def transform_input_audio_buffer_append_event(self, json_message: dict) -> List[str]:
        """
        Transform input_audio_buffer.append event to Bedrock audio input.

        Args:
            json_message: OpenAI input_audio_buffer.append message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling input_audio_buffer.append")
        messages: List[str] = []

        # Check if we need to start audio content
        if not hasattr(self, "_audio_content_started"):
            audio_content_start = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": self.input_media_type,
                            "sampleRateHertz": self.input_sample_rate_hertz,
                            "sampleSizeBits": self.input_sample_size_bits,
                            "channelCount": self.input_channel_count,
                            "audioType": self.input_audio_type,
                            "encoding": self.input_encoding,
                        },
                    }
                }
            }
            messages.append(json.dumps(audio_content_start))
            self._audio_content_started = True

        # Send audio chunk
        audio_data = json_message.get("audio", "")
        audio_event = {
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": audio_data,
                }
            }
        }
        messages.append(json.dumps(audio_event))

        return messages

    def transform_input_audio_buffer_commit_event(self, json_message: dict) -> List[str]:
        """
        Transform input_audio_buffer.commit event to Bedrock audio content end.

        Args:
            json_message: OpenAI input_audio_buffer.commit message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling input_audio_buffer.commit")
        messages: List[str] = []

        if hasattr(self, "_audio_content_started"):
            audio_content_end = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                    }
                }
            }
            messages.append(json.dumps(audio_content_end))
            delattr(self, "_audio_content_started")

        return messages

    def transform_conversation_item_create_event(self, json_message: dict) -> List[str]:
        """
        Transform conversation.item.create event to Bedrock text input or tool result.

        Args:
            json_message: OpenAI conversation.item.create message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling conversation.item.create")
        messages: List[str] = []

        item = json_message.get("item", {})
        item_type = item.get("type")

        # Handle tool result
        if item_type == "function_call_output":
            return self.transform_conversation_item_create_tool_result_event(json_message)

        # Handle regular message
        if item_type == "message":
            content = item.get("content", [])
            for content_part in content:
                if content_part.get("type") == "input_text":
                    text_content_name = str(uuid_lib.uuid4())

                    # Content start
                    text_content_start = {
                        "event": {
                            "contentStart": {
                                "promptName": self.prompt_name,
                                "contentName": text_content_name,
                                "type": "TEXT",
                                "interactive": True,
                                "role": "USER",
                                "textInputConfiguration": {
                                    "mediaType": self.text_media_type
                                },
                            }
                        }
                    }
                    messages.append(json.dumps(text_content_start))

                    # Text input
                    text_input = {
                        "event": {
                            "textInput": {
                                "promptName": self.prompt_name,
                                "contentName": text_content_name,
                                "content": content_part.get("text", ""),
                            }
                        }
                    }
                    messages.append(json.dumps(text_input))

                    # Content end
                    text_content_end = {
                        "event": {
                            "contentEnd": {
                                "promptName": self.prompt_name,
                                "contentName": text_content_name,
                            }
                        }
                    }
                    messages.append(json.dumps(text_content_end))

        return messages

    def transform_response_create_event(self, json_message: dict) -> List[str]:
        """
        Transform response.create event to Bedrock format.

        Args:
            json_message: OpenAI response.create message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling response.create")
        # Bedrock starts generating automatically, no explicit trigger needed
        return []

    def transform_response_cancel_event(self, json_message: dict) -> List[str]:
        """
        Transform response.cancel event to Bedrock format.

        Args:
            json_message: OpenAI response.cancel message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling response.cancel")
        # Send interrupt signal if needed
        return []

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: Optional[str] = None,
    ) -> List[str]:
        """
        Transform OpenAI realtime request to Bedrock Nova Sonic format.

        Args:
            message: OpenAI format message (JSON string)
            model: Model ID
            session_configuration_request: Previous session config

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        try:
            json_message = json.loads(message)
        except json.JSONDecodeError:
            verbose_logger.warning(f"Invalid JSON message: {message[:200]}")
            return []

        message_type = json_message.get("type")

        # Route to appropriate transformation method
        if message_type == "session.update":
            return self.transform_session_update_event(json_message)
        elif message_type == "input_audio_buffer.append":
            return self.transform_input_audio_buffer_append_event(json_message)
        elif message_type == "input_audio_buffer.commit":
            return self.transform_input_audio_buffer_commit_event(json_message)
        elif message_type == "conversation.item.create":
            return self.transform_conversation_item_create_event(json_message)
        elif message_type == "response.create":
            return self.transform_response_create_event(json_message)
        elif message_type == "response.cancel":
            return self.transform_response_cancel_event(json_message)
        else:
            verbose_logger.warning(f"Unknown message type: {message_type}")
            return []

    def transform_session_start_event(
        self,
        event: dict,
        model: str,
        logging_obj: LiteLLMLoggingObj,
    ) -> OpenAIRealtimeStreamSessionEvents:
        """
        Transform Bedrock sessionStart event to OpenAI session.created.

        Args:
            event: Bedrock sessionStart event
            model: Model ID
            logging_obj: Logging object

        Returns:
            OpenAI session.created event
        """
        verbose_logger.debug("Handling sessionStart")
        
        session = OpenAIRealtimeStreamSession(
            id=logging_obj.litellm_trace_id,
            modalities=["text", "audio"],
        )
        if model is not None and isinstance(model, str):
            session["model"] = model
        
        return OpenAIRealtimeStreamSessionEvents(
            type="session.created",
            session=session,
            event_id=str(uuid.uuid4()),
        )

    def transform_content_start_event(
        self,
        event: dict,
        current_response_id: Optional[str],
        current_output_item_id: Optional[str],
        current_conversation_id: Optional[str],
    ) -> tuple[
        List[OpenAIRealtimeEvents],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[ALL_DELTA_TYPES],
    ]:
        """
        Transform Bedrock contentStart event to OpenAI response events.

        Args:
            event: Bedrock contentStart event
            current_response_id: Current response ID
            current_output_item_id: Current output item ID
            current_conversation_id: Current conversation ID

        Returns:
            Tuple of (events, response_id, output_item_id, conversation_id, delta_type)
        """
        content_start = event["contentStart"]
        role = content_start.get("role")

        if role != "ASSISTANT":
            return [], current_response_id, current_output_item_id, current_conversation_id, None

        verbose_logger.debug("Handling ASSISTANT contentStart")

        # Initialize IDs if needed
        if not current_response_id:
            current_response_id = f"resp_{uuid.uuid4()}"
        if not current_output_item_id:
            current_output_item_id = f"item_{uuid.uuid4()}"
        if not current_conversation_id:
            current_conversation_id = f"conv_{uuid.uuid4()}"

        # Determine content type
        content_type = content_start.get("type", "TEXT")
        current_delta_type: ALL_DELTA_TYPES = "text" if content_type == "TEXT" else "audio"

        returned_messages: List[OpenAIRealtimeEvents] = []

        # Send response.created
        response_created = OpenAIRealtimeStreamResponseBaseObject(
            type="response.created",
            event_id=f"event_{uuid.uuid4()}",
            response={
                "object": "realtime.response",
                "id": current_response_id,
                "status": "in_progress",
                "output": [],
                "conversation_id": current_conversation_id,
            },
        )
        returned_messages.append(response_created)

        # Send response.output_item.added
        output_item_added = OpenAIRealtimeStreamResponseOutputItemAdded(
            type="response.output_item.added",
            response_id=current_response_id,
            output_index=0,
            item={
                "id": current_output_item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        )
        returned_messages.append(output_item_added)

        # Send response.content_part.added
        content_part_added = OpenAIRealtimeResponseContentPartAdded(
            type="response.content_part.added",
            content_index=0,
            output_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            part=(
                {"type": "text", "text": ""}
                if current_delta_type == "text"
                else {"type": "audio", "transcript": ""}
            ),
            response_id=current_response_id,
        )
        returned_messages.append(content_part_added)

        return (
            returned_messages,
            current_response_id,
            current_output_item_id,
            current_conversation_id,
            current_delta_type,
        )

    def transform_text_output_event(
        self,
        event: dict,
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
        current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]],
    ) -> tuple[List[OpenAIRealtimeEvents], Optional[List[OpenAIRealtimeResponseDelta]]]:
        """
        Transform Bedrock textOutput event to OpenAI response.text.delta.

        Args:
            event: Bedrock textOutput event
            current_output_item_id: Current output item ID
            current_response_id: Current response ID
            current_delta_chunks: Current delta chunks

        Returns:
            Tuple of (events, updated_delta_chunks)
        """
        verbose_logger.debug("Handling textOutput")
        text_content = event["textOutput"].get("content", "")

        if not current_output_item_id or not current_response_id:
            return [], current_delta_chunks

        text_delta = OpenAIRealtimeResponseDelta(
            type="response.text.delta",
            content_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            output_index=0,
            response_id=current_response_id,
            delta=text_content,
        )

        # Track delta chunks
        if current_delta_chunks is None:
            current_delta_chunks = []
        current_delta_chunks.append(text_delta)

        return [text_delta], current_delta_chunks

    def transform_audio_output_event(
        self,
        event: dict,
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
    ) -> List[OpenAIRealtimeEvents]:
        """
        Transform Bedrock audioOutput event to OpenAI response.audio.delta.

        Args:
            event: Bedrock audioOutput event
            current_output_item_id: Current output item ID
            current_response_id: Current response ID

        Returns:
            List of OpenAI events
        """
        verbose_logger.debug("Handling audioOutput")
        audio_content = event["audioOutput"].get("content", "")

        if not current_output_item_id or not current_response_id:
            return []

        audio_delta = OpenAIRealtimeResponseDelta(
            type="response.audio.delta",
            content_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            output_index=0,
            response_id=current_response_id,
            delta=audio_content,
        )

        return [audio_delta]

    def transform_content_end_event(
        self,
        event: dict,
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
        current_delta_type: Optional[str],
        current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]],
    ) -> tuple[List[OpenAIRealtimeEvents], Optional[List[OpenAIRealtimeResponseDelta]]]:
        """
        Transform Bedrock contentEnd event to OpenAI response done events.

        Args:
            event: Bedrock contentEnd event
            current_output_item_id: Current output item ID
            current_response_id: Current response ID
            current_delta_type: Current delta type (text or audio)
            current_delta_chunks: Current delta chunks

        Returns:
            Tuple of (events, reset_delta_chunks)
        """
        content_end = event["contentEnd"]
        verbose_logger.debug(f"Handling contentEnd: {content_end}")

        if not current_output_item_id or not current_response_id:
            return [], current_delta_chunks

        returned_messages: List[OpenAIRealtimeEvents] = []

        # Send appropriate done event based on type
        if current_delta_type == "text":
            # Accumulate text
            accumulated_text = ""
            if current_delta_chunks:
                accumulated_text = "".join(
                    [chunk.get("delta", "") for chunk in current_delta_chunks]
                )

            text_done = OpenAIRealtimeResponseTextDone(
                type="response.text.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
                text=accumulated_text,
            )
            returned_messages.append(text_done)

            # Send content_part.done
            content_part_done = OpenAIRealtimeContentPartDone(
                type="response.content_part.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                part={"type": "text", "text": accumulated_text},
                response_id=current_response_id,
            )
            returned_messages.append(content_part_done)

        elif current_delta_type == "audio":
            audio_done = OpenAIRealtimeResponseAudioDone(
                type="response.audio.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
            )
            returned_messages.append(audio_done)

            # Send content_part.done
            content_part_done = OpenAIRealtimeContentPartDone(
                type="response.content_part.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                part={"type": "audio", "transcript": ""},
                response_id=current_response_id,
            )
            returned_messages.append(content_part_done)

        # Send output_item.done
        output_item_done = OpenAIRealtimeOutputItemDone(
            type="response.output_item.done",
            event_id=f"event_{uuid.uuid4()}",
            output_index=0,
            response_id=current_response_id,
            item={
                "id": current_output_item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [],
            },
        )
        returned_messages.append(output_item_done)

        # Reset delta chunks
        return returned_messages, None

    def transform_prompt_end_event(
        self,
        event: dict,
        current_response_id: Optional[str],
        current_conversation_id: Optional[str],
    ) -> tuple[List[OpenAIRealtimeEvents], Optional[str], Optional[str], Optional[ALL_DELTA_TYPES]]:
        """
        Transform Bedrock promptEnd event to OpenAI response.done.

        Args:
            event: Bedrock promptEnd event
            current_response_id: Current response ID
            current_conversation_id: Current conversation ID

        Returns:
            Tuple of (events, reset_output_item_id, reset_response_id, reset_delta_type)
        """
        verbose_logger.debug("Handling promptEnd")

        if not current_response_id or not current_conversation_id:
            return [], None, None, None

        usage_obj = get_empty_usage()
        response_done = OpenAIRealtimeDoneEvent(
            type="response.done",
            event_id=f"event_{uuid.uuid4()}",
            response=OpenAIRealtimeResponseDoneObject(
                object="realtime.response",
                id=current_response_id,
                status="completed",
                output=[],
                conversation_id=current_conversation_id,
                usage={
                    "prompt_tokens": usage_obj.prompt_tokens,
                    "completion_tokens": usage_obj.completion_tokens,
                    "total_tokens": usage_obj.total_tokens,
                },
            ),
        )

        # Reset state for next response
        return [response_done], None, None, None

    def transform_tool_use_event(
        self,
        event: dict,
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
    ) -> tuple[List[OpenAIRealtimeEvents], str, str]:
        """
        Transform Bedrock toolUse event to OpenAI format.

        Args:
            event: Bedrock toolUse event
            current_output_item_id: Current output item ID
            current_response_id: Current response ID

        Returns:
            Tuple of (events, tool_call_id, tool_name) for tracking
        """
        verbose_logger.debug("Handling toolUse")
        tool_use = event["toolUse"]

        if not current_output_item_id or not current_response_id:
            return [], "", ""

        # Parse the tool input
        tool_input = {}
        if "input" in tool_use:
            try:
                tool_input = json.loads(tool_use["input"]) if isinstance(tool_use["input"], str) else tool_use["input"]
            except json.JSONDecodeError:
                tool_input = {}

        tool_call_id = tool_use.get("toolUseId", "")
        tool_name = tool_use.get("toolName", "")

        # Create a function call arguments done event
        # This is a custom event format that matches what clients expect
        from typing import cast
        function_call_event: dict[str, Any] = {
            "type": "response.function_call_arguments.done",
            "event_id": f"event_{uuid.uuid4()}",
            "response_id": current_response_id,
            "item_id": current_output_item_id,
            "output_index": 0,
            "call_id": tool_call_id,
            "name": tool_name,
            "arguments": json.dumps(tool_input),
        }

        return [cast(OpenAIRealtimeEvents, function_call_event)], tool_call_id, tool_name

    def transform_conversation_item_create_tool_result_event(self, json_message: dict) -> List[str]:
        """
        Transform conversation.item.create with tool result to Bedrock format.

        Args:
            json_message: OpenAI conversation.item.create message with tool result

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling conversation.item.create for tool result")
        messages: List[str] = []

        item = json_message.get("item", {})
        if item.get("type") == "function_call_output":
            tool_content_name = str(uuid_lib.uuid4())
            call_id = item.get("call_id", "")
            output = item.get("output", "")

            # Content start for tool result
            tool_content_start = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": call_id,
                            "type": "TEXT",
                            "textInputConfiguration": {
                                "mediaType": "text/plain"
                            }
                        }
                    }
                }
            }
            messages.append(json.dumps(tool_content_start))

            # Tool result
            tool_result = {
                "event": {
                    "toolResult": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name,
                        "content": output if isinstance(output, str) else json.dumps(output)
                    }
                }
            }
            messages.append(json.dumps(tool_result))

            # Content end
            tool_content_end = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name,
                    }
                }
            }
            messages.append(json.dumps(tool_content_end))

        return messages

    def transform_realtime_response(
        self,
        message: Union[str, bytes],
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        """
        Transform Bedrock Nova Sonic response to OpenAI realtime format.

        Args:
            message: Bedrock format message (JSON string)
            model: Model ID
            logging_obj: Logging object
            realtime_response_transform_input: Current state

        Returns:
            Transformed response with updated state
        """
        try:
            json_message = json.loads(message)
        except json.JSONDecodeError:
            message_preview = message[:200].decode('utf-8', errors='replace') if isinstance(message, bytes) else message[:200]
            verbose_logger.warning(f"Invalid JSON message: {message_preview}")
            return {
                "response": [],
                "current_output_item_id": realtime_response_transform_input.get(
                    "current_output_item_id"
                ),
                "current_response_id": realtime_response_transform_input.get(
                    "current_response_id"
                ),
                "current_delta_chunks": realtime_response_transform_input.get(
                    "current_delta_chunks"
                ),
                "current_conversation_id": realtime_response_transform_input.get(
                    "current_conversation_id"
                ),
                "current_item_chunks": realtime_response_transform_input.get(
                    "current_item_chunks"
                ),
                "current_delta_type": realtime_response_transform_input.get(
                    "current_delta_type"
                ),
                "session_configuration_request": realtime_response_transform_input.get(
                    "session_configuration_request"
                ),
            }

        # Extract state
        current_output_item_id = realtime_response_transform_input.get(
            "current_output_item_id"
        )
        current_response_id = realtime_response_transform_input.get(
            "current_response_id"
        )
        current_conversation_id = realtime_response_transform_input.get(
            "current_conversation_id"
        )
        current_delta_chunks = realtime_response_transform_input.get(
            "current_delta_chunks"
        )
        current_delta_type = realtime_response_transform_input.get("current_delta_type")
        session_configuration_request = realtime_response_transform_input.get(
            "session_configuration_request"
        )

        returned_messages: List[OpenAIRealtimeEvents] = []

        # Parse Bedrock event
        event = json_message.get("event", {})

        # Route to appropriate transformation method
        if "sessionStart" in event:
            session_created = self.transform_session_start_event(
                event, model, logging_obj
            )
            returned_messages.append(session_created)
            session_configuration_request = json.dumps({"configured": True})

        elif "contentStart" in event:
            (
                events,
                current_response_id,
                current_output_item_id,
                current_conversation_id,
                current_delta_type,
            ) = self.transform_content_start_event(
                event,
                current_response_id,
                current_output_item_id,
                current_conversation_id,
            )
            returned_messages.extend(events)

        elif "textOutput" in event:
            events, current_delta_chunks = self.transform_text_output_event(
                event,
                current_output_item_id,
                current_response_id,
                current_delta_chunks,
            )
            returned_messages.extend(events)

        elif "audioOutput" in event:
            events = self.transform_audio_output_event(
                event, current_output_item_id, current_response_id
            )
            returned_messages.extend(events)

        elif "contentEnd" in event:
            events, current_delta_chunks = self.transform_content_end_event(
                event,
                current_output_item_id,
                current_response_id,
                current_delta_type,
                current_delta_chunks,
            )
            returned_messages.extend(events)

        elif "toolUse" in event:
            events, tool_call_id, tool_name = self.transform_tool_use_event(
                event, current_output_item_id, current_response_id
            )
            returned_messages.extend(events)
            # Store tool call info for potential use
            verbose_logger.debug(f"Tool use event: {tool_name} (ID: {tool_call_id})")

        elif "promptEnd" in event:
            (
                events,
                current_output_item_id,
                current_response_id,
                current_delta_type,
            ) = self.transform_prompt_end_event(
                event, current_response_id, current_conversation_id
            )
            returned_messages.extend(events)

        return {
            "response": returned_messages,
            "current_output_item_id": current_output_item_id,
            "current_response_id": current_response_id,
            "current_delta_chunks": current_delta_chunks,
            "current_conversation_id": current_conversation_id,
            "current_item_chunks": realtime_response_transform_input.get(
                "current_item_chunks"
            ),
            "current_delta_type": current_delta_type,
            "session_configuration_request": session_configuration_request,
        }
