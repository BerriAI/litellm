"""
This file contains the transformation logic for Bedrock Nova Sonic realtime API.

Transforms between OpenAI Realtime API format and Bedrock Nova Sonic format.
"""

import base64
import dataclasses
import json
import uuid as uuid_lib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import ConfigDict, JsonValue

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.realtime.trigger_audio import ready_trigger_pcm
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import (
    OpenAIRealtimeContentPartDone,
    OpenAIRealtimeDoneEvent,
    OpenAIRealtimeEvents,
    OpenAIRealtimeFunctionCallArgumentsDone,
    OpenAIRealtimeInputAudioBufferSpeechEvent,
    OpenAIRealtimeInputAudioTranscriptionCompleted,
    OpenAIRealtimeInputAudioTranscriptionDelta,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseAudioDone,
    OpenAIRealtimeResponseContentPartAdded,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeResponseDoneObject,
    OpenAIRealtimeResponseTextDone,
    OpenAIRealtimeResponseUsage,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamResponseOutputItem,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeStreamResponseOutputItemContent,
    OpenAIRealtimeStreamSession,
    OpenAIRealtimeStreamSessionEvents,
    OpenAIRealtimeUsageTokenDetails,
)
from litellm.types.realtime import (
    ALL_DELTA_TYPES,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)


class BedrockContentEnd(LiteLLMBaseModel):
    stopReason: str | None = None
    type: str | None = None


class BedrockToolUse(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    toolUseId: str
    toolName: str
    content: str = "{}"


class BedrockUsageTokenDetails(LiteLLMBaseModel):
    speechTokens: int = 0
    textTokens: int = 0


class BedrockUsageDetailsTotal(LiteLLMBaseModel):
    input: BedrockUsageTokenDetails = BedrockUsageTokenDetails()
    output: BedrockUsageTokenDetails = BedrockUsageTokenDetails()


class BedrockUsageDetails(LiteLLMBaseModel):
    total: BedrockUsageDetailsTotal = BedrockUsageDetailsTotal()


class BedrockUsageEvent(LiteLLMBaseModel):
    totalInputTokens: int = 0
    totalOutputTokens: int = 0
    totalTokens: int = 0
    details: BedrockUsageDetails = BedrockUsageDetails()


TRIGGER_AUDIO_SAMPLE_RATE_HERTZ: Final = 16000
TRIGGER_AUDIO_BYTES_PER_SECOND: Final = TRIGGER_AUDIO_SAMPLE_RATE_HERTZ * 2
TRIGGER_LEADING_SILENCE: Final = bytes(TRIGGER_AUDIO_BYTES_PER_SECOND // 2)
TRIGGER_TRAILING_SILENCE: Final = bytes(TRIGGER_AUDIO_BYTES_PER_SECOND * 3)
TRIGGER_AUDIO_CHUNK_SIZE: Final = 1024

DEFAULT_SYSTEM_PROMPT: Final = "You are a helpful assistant."


@dataclasses.dataclass(frozen=True, slots=True)
class BedrockLockedSessionSettings:
    """The session settings Bedrock Nova Sonic locks in once the prompt starts."""

    tools: JsonValue
    system_prompt: str
    voice_id: str
    output_sample_rate_hertz: int
    temperature: float
    max_tokens: int | None


class BedrockRealtimeConfig(BaseRealtimeConfig):
    """Configuration for Bedrock Nova Sonic realtime transformations."""

    def __init__(self):
        # Track session state
        self.prompt_name = str(uuid_lib.uuid4())
        self.content_name = str(uuid_lib.uuid4())
        self.audio_content_name = str(uuid_lib.uuid4())
        self.prompt_started = False
        self.client_audio_streamed = False

        # Default configuration values
        # Inference configuration
        self.max_tokens: int | None = 1024
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

        # Response-stream state (Bedrock events carry no role on textOutput,
        # so the USER/ASSISTANT split from contentStart is tracked here)
        self._user_transcript_active = False
        self._user_transcript_generation_stage: str | None = None
        self._assistant_final_text_active = False
        self._user_item_id: str | None = None
        self._user_transcript_buffer = ""
        self._cumulative_usage = BedrockUsageEvent()
        self._reported_usage = BedrockUsageEvent()
        self._response_done_items: Mapping[int, OpenAIRealtimeStreamResponseOutputItem] = MappingProxyType({})
        self._response_item_count = 0
        self._message_output_index = 0
        self._assistant_block_open = False
        self._response_done_pending = False
        self._started_settings = BedrockLockedSessionSettings(
            tools=[],
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            voice_id=self.voice_id,
            output_sample_rate_hertz=self.output_sample_rate_hertz,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: base passes response headers as a dict
    ) -> BedrockError:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def validate_environment(self, headers: dict, model: str, api_key: str | None = None) -> dict:
        """Validate environment - no special validation needed for Bedrock."""
        return headers

    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        """Get complete URL - handled by aws_sdk_bedrock_runtime."""
        return api_base or ""

    def requires_session_configuration(self) -> bool:
        """Bedrock requires session configuration."""
        return True

    def session_configuration_request(self, model: str, tools: list[dict] | None = None) -> str:
        """
        Create initial session configuration for Bedrock Nova Sonic.

        Args:
            model: Model ID
            tools: Optional list of tool definitions

        Returns JSON string with session start and prompt start events.
        """
        session_start: Final = {"event": {"sessionStart": {"inferenceConfiguration": self._inference_configuration()}}}

        prompt_start_config: Final[Mapping[str, object]] = {
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
            **(
                {
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {"tools": self._transform_tools_to_bedrock_format(tools)},
                }
                if tools
                else {}
            ),
        }

        prompt_start: Final = {"event": {"promptStart": prompt_start_config}}

        # Return as a marker that we've sent the configuration
        return json.dumps({"session_start": session_start, "prompt_start": prompt_start})

    def _transform_tools_to_bedrock_format(self, tools: list[dict]) -> list[dict]:
        """
        Transform OpenAI tool format to Bedrock tool format.

        Args:
            tools: List of OpenAI format tools

        Returns:
            List of Bedrock format tools
        """
        bedrock_tools: Final = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function") or tool
                bedrock_tool = {
                    "toolSpec": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "inputSchema": {"json": json.dumps(function.get("parameters", {}))},
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

    def transform_session_update_event(self, json_message: dict) -> list[str]:
        """
        Transform session.update event to Bedrock session configuration.

        Args:
            json_message: OpenAI session.update message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling session.update")
        messages: Final[list[str]] = []

        session_config: Final = json_message.get("session", {})
        locked_settings: Final = self._apply_locked_settings(session_config, self._started_settings)

        # Settings Bedrock locks at prompt start may only change before it
        if not self.prompt_started:
            self.max_tokens = locked_settings.max_tokens
            self.temperature = locked_settings.temperature
            self.voice_id = locked_settings.voice_id
            self.output_sample_rate_hertz = locked_settings.output_sample_rate_hertz

        # Update audio input configuration from session if provided
        if "input_audio_format" in session_config:
            input_format: Final = session_config["input_audio_format"]
            self.input_sample_rate_hertz = self._map_audio_format_to_sample_rate(input_format, is_output=False)

        # Allow direct override of the input sample rate if provided (custom extension)
        if "input_sample_rate_hertz" in session_config:
            self.input_sample_rate_hertz = session_config["input_sample_rate_hertz"]

        if self.prompt_started:
            return messages

        # Send session start
        session_start: Final = {"event": {"sessionStart": {"inferenceConfiguration": self._inference_configuration()}}}
        messages.append(json.dumps(session_start))

        # Send prompt start
        tools: Final = session_config.get("tools")
        prompt_start_config: Final[Mapping[str, object]] = {
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
            **(
                {
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {"tools": self._transform_tools_to_bedrock_format(tools)},
                }
                if tools
                else {}
            ),
        }

        prompt_start: Final = {"event": {"promptStart": prompt_start_config}}
        messages.append(json.dumps(prompt_start))
        self.prompt_started = True
        self._started_settings = locked_settings

        # Bedrock requires the first content block in a prompt to be SYSTEM role
        instructions: Final = locked_settings.system_prompt
        text_content_name: Final = str(uuid_lib.uuid4())

        # Content start
        text_content_start: Final = {
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
        text_input: Final = {
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
        text_content_end: Final = {
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": text_content_name,
                }
            }
        }
        messages.append(json.dumps(text_content_end))

        return messages

    def _inference_configuration(self) -> Mapping[str, int | float]:
        return {
            "topP": self.top_p,
            "temperature": self.temperature,
            **({"maxTokens": self.max_tokens} if self.max_tokens is not None else {}),
        }

    def _apply_locked_settings(
        self, session_config: Mapping[str, JsonValue], base: BedrockLockedSessionSettings
    ) -> BedrockLockedSessionSettings:
        output_rate_hertz: Final = session_config.get("output_sample_rate_hertz")
        temperature: Final = session_config.get("temperature")
        max_output_tokens: Final = session_config.get("max_response_output_tokens")
        tools: Final = (session_config["tools"] or []) if "tools" in session_config else base.tools
        system_prompt: Final = (
            str(session_config["instructions"] or DEFAULT_SYSTEM_PROMPT)
            if "instructions" in session_config
            else base.system_prompt
        )
        voice_id: Final = str(session_config["voice"]) if "voice" in session_config else base.voice_id
        format_sample_rate: Final = (
            self._map_audio_format_to_sample_rate(str(session_config["output_audio_format"]), is_output=True)
            if "output_audio_format" in session_config
            else base.output_sample_rate_hertz
        )
        output_sample_rate_hertz: Final = (
            output_rate_hertz if isinstance(output_rate_hertz, int) else format_sample_rate
        )
        locked_temperature: Final = float(temperature) if isinstance(temperature, (int, float)) else base.temperature
        explicit_max_tokens: Final = max_output_tokens if isinstance(max_output_tokens, int) else base.max_tokens
        max_tokens: Final = None if max_output_tokens == "inf" else explicit_max_tokens
        return dataclasses.replace(
            base,
            tools=tools,
            system_prompt=system_prompt,
            voice_id=voice_id,
            output_sample_rate_hertz=output_sample_rate_hertz,
            temperature=locked_temperature,
            max_tokens=max_tokens,
        )

    def session_update_conflict(self, json_message: Mapping[str, JsonValue]) -> str | None:
        """Error message when a session.update changes what Bedrock locked at prompt start."""
        if not self.prompt_started:
            return None
        session_config: Final = json_message.get("session")
        if not isinstance(session_config, Mapping):
            return None
        if self._apply_locked_settings(session_config, self._started_settings) != self._started_settings:
            return "Bedrock Nova Sonic cannot change tools, instructions, voice, output audio format, temperature or max output tokens after the session has started, so this session.update was not applied"
        return None

    def transform_input_audio_buffer_append_event(self, json_message: dict) -> list[str]:
        """
        Transform input_audio_buffer.append event to Bedrock audio input.

        Args:
            json_message: OpenAI input_audio_buffer.append message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling input_audio_buffer.append")
        self.client_audio_streamed = True
        messages: Final[list[str]] = []

        if hasattr(self, "_audio_content_started") and self._audio_content_sample_rate != self.input_sample_rate_hertz:
            mismatched_content_end: Final = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                    }
                }
            }
            messages.append(json.dumps(mismatched_content_end))
            delattr(self, "_audio_content_started")
            self.audio_content_name = str(uuid_lib.uuid4())

        # Check if we need to start audio content
        if not hasattr(self, "_audio_content_started"):
            audio_content_start: Final = {
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
            self._audio_content_sample_rate = self.input_sample_rate_hertz

        # Send audio chunk
        audio_data: Final = json_message.get("audio", "")
        audio_event: Final = {
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

    def transform_input_audio_buffer_commit_event(self, json_message: dict) -> list[str]:
        """
        Transform input_audio_buffer.commit event to Bedrock audio content end.

        Args:
            json_message: OpenAI input_audio_buffer.commit message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling input_audio_buffer.commit")
        messages: Final[list[str]] = []

        if hasattr(self, "_audio_content_started"):
            audio_content_end: Final = {
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

    def transform_conversation_item_create_event(self, json_message: dict) -> list[str]:
        """
        Transform conversation.item.create event to Bedrock text input or tool result.

        Args:
            json_message: OpenAI conversation.item.create message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling conversation.item.create")

        item: Final = json_message.get("item", {})
        item_type: Final = item.get("type")

        # Handle tool result
        if item_type == "function_call_output":
            return self.transform_conversation_item_create_tool_result_event(json_message)

        messages: Final[list[str]] = []

        # Handle regular message
        if item_type == "message":
            content: Final = item.get("content", [])
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

    def transform_response_create_event(self, json_message: dict) -> list[str]:
        """
        Transform response.create event to Bedrock format.

        Nova Sonic only starts generating after it detects user speech, so text-only
        sessions never get a response on their own. Injecting a short spoken "ready"
        utterance (followed by silence) makes the model respond to the pending
        interactive text input. Sessions where the client streams its own audio rely
        on Nova Sonic's built-in turn detection instead.

        Args:
            json_message: OpenAI response.create message

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling response.create")
        if not self.prompt_started or self.client_audio_streamed:
            return []

        messages: Final[list[str]] = []
        if not hasattr(self, "_audio_content_started"):
            trigger_content_start: Final = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": self.input_media_type,
                            "sampleRateHertz": TRIGGER_AUDIO_SAMPLE_RATE_HERTZ,
                            "sampleSizeBits": self.input_sample_size_bits,
                            "channelCount": self.input_channel_count,
                            "audioType": self.input_audio_type,
                            "encoding": self.input_encoding,
                        },
                    }
                }
            }
            messages.append(json.dumps(trigger_content_start))
            self._audio_content_started = True
            self._audio_content_sample_rate = TRIGGER_AUDIO_SAMPLE_RATE_HERTZ

        messages.extend(self._response_trigger_audio_messages())
        return messages

    def _response_trigger_audio_messages(self) -> list[str]:
        pcm: Final = TRIGGER_LEADING_SILENCE + ready_trigger_pcm() + TRIGGER_TRAILING_SILENCE
        return [
            json.dumps(
                {
                    "event": {
                        "audioInput": {
                            "promptName": self.prompt_name,
                            "contentName": self.audio_content_name,
                            "content": base64.b64encode(pcm[offset : offset + TRIGGER_AUDIO_CHUNK_SIZE]).decode(),
                        }
                    }
                }
            )
            for offset in range(0, len(pcm), TRIGGER_AUDIO_CHUNK_SIZE)
        ]

    def transform_response_cancel_event(self, json_message: dict) -> list[str]:
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

    def session_close_messages(self) -> list[str]:
        """
        Build the Bedrock events that gracefully close the session
        (contentEnd for any open audio content, promptEnd, sessionEnd).

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        if not self.prompt_started:
            return []

        messages: Final[list[str]] = []
        if hasattr(self, "_audio_content_started"):
            audio_content_end: Final = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                    }
                }
            }
            messages.append(json.dumps(audio_content_end))
            delattr(self, "_audio_content_started")

        messages.append(json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}}))
        messages.append(json.dumps({"event": {"sessionEnd": {}}}))
        self.prompt_started = False
        return messages

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: str | None = None,
    ) -> list[str]:
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
            json_message: Final[dict[str, object]] = json.loads(message)
        except json.JSONDecodeError:
            verbose_logger.warning("Invalid JSON message: %s", message[:200])
            return []

        message_type: Final = json_message.get("type")

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
            verbose_logger.warning("Unknown message type: %s", message_type)
            return []

    def _session_object(
        self,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        modalities: list[str] | None = None,
    ) -> OpenAIRealtimeStreamSession:
        session: Final = OpenAIRealtimeStreamSession(
            id=logging_obj.litellm_trace_id,
            modalities=modalities if modalities is not None else ["text", "audio"],
        )
        if model is not None and isinstance(model, str):
            session["model"] = model
        return session

    def session_created_event(
        self,
        model: str,
        logging_obj: LiteLLMLoggingObj,
    ) -> OpenAIRealtimeStreamSessionEvents:
        """Build the OpenAI session.created event for this realtime session."""
        return OpenAIRealtimeStreamSessionEvents(
            type="session.created",
            session=self._session_object(model, logging_obj),
            event_id=str(uuid.uuid4()),
        )

    def session_updated_event(
        self,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        modalities: list[str] | None = None,
    ) -> OpenAIRealtimeStreamSessionEvents:
        """Build the OpenAI session.updated ack reflecting the client's requested modalities."""
        return OpenAIRealtimeStreamSessionEvents(
            type="session.updated",
            session=self._session_object(model, logging_obj, modalities),
            event_id=str(uuid.uuid4()),
        )

    def transform_content_start_event(
        self,
        event: dict,
        current_response_id: str | None,
        current_output_item_id: str | None,
        current_conversation_id: str | None,
        current_delta_type: ALL_DELTA_TYPES | None,
    ) -> tuple[
        list[OpenAIRealtimeEvents],
        str | None,
        str | None,
        str | None,
        ALL_DELTA_TYPES | None,
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
        content_start: Final = event["contentStart"]
        role: Final = content_start.get("role")

        if role != "ASSISTANT":
            if role == "USER" and content_start.get("type") == "TEXT":
                self._user_transcript_active = True
                self._user_transcript_generation_stage = self._parse_generation_stage(
                    content_start.get("additionalModelFields")
                )
            return (
                [],
                current_response_id,
                current_output_item_id,
                current_conversation_id,
                current_delta_type,
            )

        generation_stage: Final = self._parse_generation_stage(content_start.get("additionalModelFields"))
        if content_start.get("type") == "TEXT" and generation_stage == "FINAL":
            verbose_logger.debug(
                "Dropping FINAL ASSISTANT text block: the SPECULATIVE block already carried its sentences"
            )
            self._assistant_final_text_active = True
            return (
                [],
                current_response_id,
                current_output_item_id,
                current_conversation_id,
                None,
            )

        verbose_logger.debug("Handling ASSISTANT contentStart")
        is_new_response: Final = current_response_id is None

        # Initialize IDs if needed
        if not current_response_id:
            current_response_id = f"resp_{uuid.uuid4()}"
        if not current_output_item_id:
            current_output_item_id = f"item_{uuid.uuid4()}"
            self._message_output_index = self._response_item_count
            self._response_item_count += 1
        if not current_conversation_id:
            current_conversation_id = f"conv_{uuid.uuid4()}"

        # Determine content type
        content_type: Final = content_start.get("type", "TEXT")
        next_delta_type: Final[ALL_DELTA_TYPES] = "text" if content_type == "TEXT" else "audio"

        returned_messages: Final[list[OpenAIRealtimeEvents]] = []

        # Send response.created only once per response (a response can contain
        # multiple content blocks, e.g. TEXT then AUDIO)
        response_created: Final = OpenAIRealtimeStreamResponseBaseObject(
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
        if is_new_response:
            returned_messages.append(response_created)

        # Send response.output_item.added
        output_item_added: Final = OpenAIRealtimeStreamResponseOutputItemAdded(
            type="response.output_item.added",
            response_id=current_response_id,
            output_index=self._message_output_index,
            item={
                "id": current_output_item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        )
        self._assistant_block_open = True
        returned_messages.append(output_item_added)

        # Send response.content_part.added
        content_part_added: Final = OpenAIRealtimeResponseContentPartAdded(
            type="response.content_part.added",
            content_index=0,
            output_index=self._message_output_index,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            part=({"type": "text", "text": ""} if next_delta_type == "text" else {"type": "audio", "transcript": ""}),
            response_id=current_response_id,
        )
        returned_messages.append(content_part_added)

        return (
            returned_messages,
            current_response_id,
            current_output_item_id,
            current_conversation_id,
            next_delta_type,
        )

    @staticmethod
    def _parse_generation_stage(additional_model_fields: object) -> str | None:
        if not isinstance(additional_model_fields, str):
            return None
        try:
            parsed: Final = json.loads(additional_model_fields)
        except json.JSONDecodeError:
            return None
        stage: Final = parsed.get("generationStage") if isinstance(parsed, dict) else None
        return stage if isinstance(stage, str) else None

    def _current_user_item_id(self, new_utterance: bool = False) -> str:
        """Item id shared by all events of one user utterance (speech boundaries and transcript)."""
        if new_utterance or self._user_item_id is None:
            self._user_item_id = f"item_{uuid.uuid4()}"
        return self._user_item_id

    def transform_user_speech_event(self, is_speech_start: bool) -> tuple[OpenAIRealtimeEvents, ...]:
        """Transform Bedrock userSpeechStart/userSpeechEnd to OpenAI speech boundary events."""
        verbose_logger.debug("Handling userSpeech%s", "Start" if is_speech_start else "End")
        speech_event: Final[OpenAIRealtimeInputAudioBufferSpeechEvent] = {
            "type": "input_audio_buffer.speech_started" if is_speech_start else "input_audio_buffer.speech_stopped",
            "event_id": f"event_{uuid.uuid4()}",
            "item_id": self._current_user_item_id(new_utterance=is_speech_start),
        }
        return (speech_event,)

    def transform_usage_event(self, usage_event: BedrockUsageEvent) -> None:
        """Record Bedrock's session-cumulative usage totals for the next response.done."""
        verbose_logger.debug("Handling usageEvent")
        self._cumulative_usage = usage_event

    def _take_usage_delta(self) -> OpenAIRealtimeResponseUsage:
        """Usage for the response now completing: cumulative totals minus what prior response.done events reported."""
        prior: Final = self._reported_usage
        latest: Final = self._cumulative_usage
        self._reported_usage = latest
        input_details: Final[OpenAIRealtimeUsageTokenDetails] = {
            "audio_tokens": latest.details.total.input.speechTokens - prior.details.total.input.speechTokens,
            "text_tokens": latest.details.total.input.textTokens - prior.details.total.input.textTokens,
            "cached_tokens": 0,
        }
        output_details: Final[OpenAIRealtimeUsageTokenDetails] = {
            "audio_tokens": latest.details.total.output.speechTokens - prior.details.total.output.speechTokens,
            "text_tokens": latest.details.total.output.textTokens - prior.details.total.output.textTokens,
        }
        usage_delta: Final[OpenAIRealtimeResponseUsage] = {
            "input_tokens": latest.totalInputTokens - prior.totalInputTokens,
            "output_tokens": latest.totalOutputTokens - prior.totalOutputTokens,
            "total_tokens": latest.totalTokens - prior.totalTokens,
            "input_token_details": input_details,
            "output_token_details": output_details,
        }
        return usage_delta

    def leftover_usage_done_events(self) -> tuple[OpenAIRealtimeEvents, ...]:
        """Logged-only response.done for usage Bedrock reports after the final turn's contentEnd."""
        if self._cumulative_usage == self._reported_usage:
            return ()
        usage: Final = self._take_usage_delta()
        leftover_done: Final = OpenAIRealtimeDoneEvent(
            type="response.done",
            event_id=f"event_{uuid.uuid4()}",
            response=OpenAIRealtimeResponseDoneObject(
                object="realtime.response",
                id=f"resp_{uuid.uuid4()}",
                status="completed",
                conversation_id=f"conv_{uuid.uuid4()}",
                usage=dict(usage),
            ),
        )
        return (leftover_done,)

    def transform_user_transcript_event(self, transcript: str) -> tuple[OpenAIRealtimeEvents, ...]:
        """Transform a USER-role Bedrock textOutput (ASR transcript) to an OpenAI transcription delta."""
        verbose_logger.debug("Handling USER textOutput (ASR transcript)")
        delta_event: Final[OpenAIRealtimeInputAudioTranscriptionDelta] = {
            "type": "conversation.item.input_audio_transcription.delta",
            "event_id": f"event_{uuid.uuid4()}",
            "item_id": self._current_user_item_id(),
            "content_index": 0,
            "delta": transcript,
        }
        if self._user_transcript_generation_stage != "SPECULATIVE":
            self._user_transcript_buffer += transcript
        return (delta_event,)

    def user_transcript_completed_events(self) -> tuple[OpenAIRealtimeEvents, ...]:
        """One completed event with the full transcript once the FINAL user content block ends."""
        transcript: Final = self._user_transcript_buffer
        if not transcript:
            return ()
        self._user_transcript_buffer = ""
        completed_event: Final[OpenAIRealtimeInputAudioTranscriptionCompleted] = {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": f"event_{uuid.uuid4()}",
            "item_id": self._current_user_item_id(),
            "content_index": 0,
            "transcript": transcript,
        }
        return (completed_event,)

    def transform_text_output_event(
        self,
        event: dict,
        current_output_item_id: str | None,
        current_response_id: str | None,
        current_delta_chunks: list[OpenAIRealtimeResponseDelta] | None,
    ) -> tuple[list[OpenAIRealtimeEvents], list[OpenAIRealtimeResponseDelta] | None]:
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
        text_content: Final = event["textOutput"].get("content", "")

        if self._assistant_final_text_active:
            return [], current_delta_chunks
        if not current_output_item_id or not current_response_id:
            return [], current_delta_chunks

        text_delta: Final = OpenAIRealtimeResponseDelta(
            type="response.text.delta",
            content_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            output_index=self._message_output_index,
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
        current_output_item_id: str | None,
        current_response_id: str | None,
    ) -> list[OpenAIRealtimeEvents]:
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
        audio_content: Final = event["audioOutput"].get("content", "")

        if not current_output_item_id or not current_response_id:
            return []

        audio_delta: Final = OpenAIRealtimeResponseDelta(
            type="response.audio.delta",
            content_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            output_index=self._message_output_index,
            response_id=current_response_id,
            delta=audio_content,
        )

        return [audio_delta]

    def transform_content_end_event(
        self,
        event: dict,
        current_output_item_id: str | None,
        current_response_id: str | None,
        current_delta_type: str | None,
        current_delta_chunks: list[OpenAIRealtimeResponseDelta] | None,
    ) -> tuple[list[OpenAIRealtimeEvents], list[OpenAIRealtimeResponseDelta] | None]:
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
        content_end: Final = event["contentEnd"]
        verbose_logger.debug("Handling contentEnd: %s", content_end)

        if self._assistant_final_text_active:
            self._assistant_final_text_active = False
            return [], current_delta_chunks
        if not current_output_item_id or not current_response_id:
            return [], current_delta_chunks
        if not self._assistant_block_open:
            return [], current_delta_chunks

        self._assistant_block_open = False
        returned_messages: Final[list[OpenAIRealtimeEvents]] = []
        accumulated_text: Final = "".join(chunk.get("delta", "") for chunk in current_delta_chunks or [])

        # Send appropriate done event based on type
        if current_delta_type == "text":
            text_done: Final = OpenAIRealtimeResponseTextDone(
                type="response.text.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=self._message_output_index,
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
                output_index=self._message_output_index,
                part={"type": "text", "text": accumulated_text},
                response_id=current_response_id,
            )
            returned_messages.append(content_part_done)

        elif current_delta_type == "audio":
            audio_done: Final = OpenAIRealtimeResponseAudioDone(
                type="response.audio.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=self._message_output_index,
                response_id=current_response_id,
            )
            returned_messages.append(audio_done)

            # Send content_part.done
            content_part_done = OpenAIRealtimeContentPartDone(
                type="response.content_part.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=self._message_output_index,
                part={"type": "audio", "transcript": ""},
                response_id=current_response_id,
            )
            returned_messages.append(content_part_done)

        completed_parts: Final = self._completed_parts(current_delta_type, accumulated_text)
        existing_item: Final = self._response_done_items.get(self._message_output_index)
        existing_content: Final = existing_item.get("content", []) if existing_item is not None else []
        completed_message_item: Final[OpenAIRealtimeStreamResponseOutputItem] = {
            "id": current_output_item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [*existing_content, *completed_parts],
        }
        output_item_done: Final = OpenAIRealtimeOutputItemDone(
            type="response.output_item.done",
            event_id=f"event_{uuid.uuid4()}",
            output_index=self._message_output_index,
            response_id=current_response_id,
            item=completed_message_item,
        )
        returned_messages.append(output_item_done)
        self._response_done_items = MappingProxyType(
            {**self._response_done_items, self._message_output_index: completed_message_item}
        )

        # Reset delta chunks
        return returned_messages, None

    def transform_prompt_end_event(
        self,
        event: dict,
        current_response_id: str | None,
        current_conversation_id: str | None,
    ) -> tuple[
        list[OpenAIRealtimeEvents],
        str | None,
        str | None,
        ALL_DELTA_TYPES | None,
    ]:
        """
        Transform a Bedrock end-of-response event (promptEnd, completionEnd, or an
        END_TURN contentEnd) to OpenAI response.done.

        Args:
            event: Bedrock event that ends the response
            current_response_id: Current response ID
            current_conversation_id: Current conversation ID

        Returns:
            Tuple of (events, reset_output_item_id, reset_response_id, reset_delta_type)
        """
        verbose_logger.debug("Handling promptEnd")
        return self._response_done_events(current_response_id, current_conversation_id)

    def _response_done_events(
        self,
        current_response_id: str | None,
        current_conversation_id: str | None,
    ) -> tuple[
        list[OpenAIRealtimeEvents],
        str | None,
        str | None,
        ALL_DELTA_TYPES | None,
    ]:
        if not current_response_id or not current_conversation_id:
            return [], None, None, None

        usage: Final = self._take_usage_delta()
        output_items: Final = tuple(item for _, item in sorted(self._response_done_items.items()))
        self._response_done_items = MappingProxyType({})
        self._response_item_count = 0
        self._message_output_index = 0
        self._assistant_block_open = False
        self._response_done_pending = False
        response_done: Final = OpenAIRealtimeDoneEvent(
            type="response.done",
            event_id=f"event_{uuid.uuid4()}",
            response=OpenAIRealtimeResponseDoneObject(
                object="realtime.response",
                id=current_response_id,
                status="completed",
                output=list(output_items),
                conversation_id=current_conversation_id,
                usage=dict(usage),
            ),
        )

        # Reset state for next response
        return [response_done], None, None, None

    def transform_tool_use_event(
        self,
        event: dict,
        current_response_id: str | None,
        current_conversation_id: str | None,
    ) -> tuple[list[OpenAIRealtimeEvents], str, str]:
        """
        Transform Bedrock toolUse event to OpenAI format.

        Args:
            event: Bedrock toolUse event
            current_response_id: Current response ID
            current_conversation_id: Current conversation ID

        Returns:
            Tuple of (events, response_id, conversation_id)
        """
        verbose_logger.debug("Handling toolUse")
        tool_use: Final = BedrockToolUse.model_validate(event["toolUse"])
        verbose_logger.debug("Tool use event: %s (ID: %s)", tool_use.toolName, tool_use.toolUseId)

        response_id: Final = current_response_id or f"resp_{uuid.uuid4()}"
        conversation_id: Final = current_conversation_id or f"conv_{uuid.uuid4()}"
        item_id: Final = f"item_{uuid.uuid4()}"
        output_index: Final = self._response_item_count
        self._response_item_count += 1

        response_created_events: Final[tuple[OpenAIRealtimeEvents, ...]] = (
            (
                OpenAIRealtimeStreamResponseBaseObject(
                    type="response.created",
                    event_id=f"event_{uuid.uuid4()}",
                    response={
                        "object": "realtime.response",
                        "id": response_id,
                        "status": "in_progress",
                        "output": [],
                        "conversation_id": conversation_id,
                    },
                ),
            )
            if current_response_id is None
            else ()
        )

        output_item_added: Final = OpenAIRealtimeStreamResponseOutputItemAdded(
            type="response.output_item.added",
            response_id=response_id,
            output_index=output_index,
            item={
                "id": item_id,
                "object": "realtime.item",
                "type": "function_call",
                "status": "in_progress",
                "call_id": tool_use.toolUseId,
                "name": tool_use.toolName,
                "arguments": "",
            },
        )

        function_call_event: Final = OpenAIRealtimeFunctionCallArgumentsDone(
            type="response.function_call_arguments.done",
            event_id=f"event_{uuid.uuid4()}",
            response_id=response_id,
            item_id=item_id,
            output_index=output_index,
            call_id=tool_use.toolUseId,
            name=tool_use.toolName,
            arguments=tool_use.content,
        )

        completed_item: Final[OpenAIRealtimeStreamResponseOutputItem] = {
            "id": item_id,
            "object": "realtime.item",
            "type": "function_call",
            "status": "completed",
            "call_id": tool_use.toolUseId,
            "name": tool_use.toolName,
            "arguments": tool_use.content,
        }
        output_item_done: Final = OpenAIRealtimeOutputItemDone(
            type="response.output_item.done",
            event_id=f"event_{uuid.uuid4()}",
            output_index=output_index,
            response_id=response_id,
            item=completed_item,
        )

        self._response_done_items = MappingProxyType({**self._response_done_items, output_index: completed_item})

        return (
            [
                *response_created_events,
                output_item_added,
                function_call_event,
                output_item_done,
            ],
            response_id,
            conversation_id,
        )

    def _tool_content_end_events(
        self,
        current_output_item_id: str | None,
        current_response_id: str | None,
        current_conversation_id: str | None,
        current_delta_type: ALL_DELTA_TYPES | None,
    ) -> tuple[
        tuple[OpenAIRealtimeEvents, ...],
        str | None,
        str | None,
        ALL_DELTA_TYPES | None,
    ]:
        if self._assistant_block_open:
            self._response_done_pending = True
            return (), current_output_item_id, current_response_id, current_delta_type
        done_events, output_item_id, response_id, delta_type = self._response_done_events(
            current_response_id, current_conversation_id
        )
        return tuple(done_events), output_item_id, response_id, delta_type

    @staticmethod
    def _completed_parts(
        delta_type: str | None, accumulated_text: str
    ) -> tuple[OpenAIRealtimeStreamResponseOutputItemContent, ...]:
        if delta_type == "text":
            return ({"type": "text", "text": accumulated_text},)
        elif delta_type == "audio":
            return ({"type": "audio", "transcript": ""},)
        return ()

    def _should_close_response(self, stop_reason: str | None) -> bool:
        return stop_reason == "END_TURN" or self._response_done_pending

    def transform_conversation_item_create_tool_result_event(self, json_message: dict) -> list[str]:
        """
        Transform conversation.item.create with tool result to Bedrock format.

        Args:
            json_message: OpenAI conversation.item.create message with tool result

        Returns:
            List of Bedrock format messages (JSON strings)
        """
        verbose_logger.debug("Handling conversation.item.create for tool result")
        messages: Final[list[str]] = []

        item: Final = json_message.get("item", {})
        if item.get("type") == "function_call_output":
            tool_content_name: Final = str(uuid_lib.uuid4())
            call_id: Final = item.get("call_id", "")
            output: Final = item.get("output", "")

            # Content start for tool result
            tool_content_start: Final = {
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
                            "textInputConfiguration": {"mediaType": "text/plain"},
                        },
                    }
                }
            }
            messages.append(json.dumps(tool_content_start))

            # Tool result
            tool_result: Final = {
                "event": {
                    "toolResult": {
                        "promptName": self.prompt_name,
                        "contentName": tool_content_name,
                        "content": (output if isinstance(output, str) else json.dumps(output)),
                    }
                }
            }
            messages.append(json.dumps(tool_result))

            # Content end
            tool_content_end: Final = {
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
        message: str | bytes,
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
            json_message: Final = json.loads(message)
        except json.JSONDecodeError:
            message_preview: Final = (
                message[:200].decode("utf-8", errors="replace") if isinstance(message, bytes) else message[:200]
            )
            verbose_logger.warning("Invalid JSON message: %s", message_preview)
            return {
                "response": [],
                "current_output_item_id": realtime_response_transform_input.get("current_output_item_id"),
                "current_response_id": realtime_response_transform_input.get("current_response_id"),
                "current_delta_chunks": realtime_response_transform_input.get("current_delta_chunks"),
                "current_conversation_id": realtime_response_transform_input.get("current_conversation_id"),
                "current_item_chunks": realtime_response_transform_input.get("current_item_chunks"),
                "current_delta_type": realtime_response_transform_input.get("current_delta_type"),
                "session_configuration_request": realtime_response_transform_input.get("session_configuration_request"),
            }

        # Extract state
        current_output_item_id = realtime_response_transform_input.get("current_output_item_id")
        current_response_id = realtime_response_transform_input.get("current_response_id")
        current_conversation_id = realtime_response_transform_input.get("current_conversation_id")
        current_delta_chunks = realtime_response_transform_input.get("current_delta_chunks")
        current_delta_type = realtime_response_transform_input.get("current_delta_type")
        session_configuration_request = realtime_response_transform_input.get("session_configuration_request")

        returned_messages: Final[list[OpenAIRealtimeEvents]] = []

        # Parse Bedrock event
        event: Final = json_message.get("event", {})

        # Route to appropriate transformation method
        if "sessionStart" in event:
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
                current_delta_type,
            )
            returned_messages.extend(events)

        elif "textOutput" in event:
            if self._user_transcript_active:
                returned_messages.extend(self.transform_user_transcript_event(event["textOutput"].get("content", "")))
            else:
                events, current_delta_chunks = self.transform_text_output_event(
                    event,
                    current_output_item_id,
                    current_response_id,
                    current_delta_chunks,
                )
                returned_messages.extend(events)

        elif "audioOutput" in event:
            events = self.transform_audio_output_event(event, current_output_item_id, current_response_id)
            returned_messages.extend(events)

        elif "contentEnd" in event and BedrockContentEnd.model_validate(event["contentEnd"]).type == "TOOL":
            (
                done_events,
                current_output_item_id,  # rebind-ok: same state-threading pattern as the adjacent event branches
                current_response_id,  # rebind-ok: same state-threading pattern as the adjacent event branches
                current_delta_type,  # rebind-ok: same state-threading pattern as the adjacent event branches
            ) = self._tool_content_end_events(
                current_output_item_id, current_response_id, current_conversation_id, current_delta_type
            )
            returned_messages.extend(done_events)

        elif "contentEnd" in event and self._user_transcript_active:
            self._user_transcript_active = False
            self._user_transcript_generation_stage = None
            returned_messages.extend(self.user_transcript_completed_events())

        elif "contentEnd" in event:
            events, current_delta_chunks = self.transform_content_end_event(
                event,
                current_output_item_id,
                current_response_id,
                current_delta_type,
                current_delta_chunks,
            )
            returned_messages.extend(events)
            content_end: Final = BedrockContentEnd.model_validate(event["contentEnd"])
            if self._should_close_response(content_end.stopReason):
                (
                    done_events,
                    current_output_item_id,
                    current_response_id,
                    current_delta_type,
                ) = self._response_done_events(current_response_id, current_conversation_id)
                returned_messages.extend(done_events)

        elif "userSpeechStart" in event or "userSpeechEnd" in event:
            returned_messages.extend(self.transform_user_speech_event("userSpeechStart" in event))

        elif "usageEvent" in event:
            self.transform_usage_event(BedrockUsageEvent.model_validate(event["usageEvent"]))

        elif "toolUse" in event:
            events, current_response_id, current_conversation_id = self.transform_tool_use_event(
                event, current_response_id, current_conversation_id
            )
            returned_messages.extend(events)

        elif "promptEnd" in event or "completionEnd" in event:
            (
                events,
                current_output_item_id,
                current_response_id,
                current_delta_type,
            ) = self.transform_prompt_end_event(event, current_response_id, current_conversation_id)
            returned_messages.extend(events)

        return {
            "response": returned_messages,
            "current_output_item_id": current_output_item_id,
            "current_response_id": current_response_id,
            "current_delta_chunks": current_delta_chunks,
            "current_conversation_id": current_conversation_id,
            "current_item_chunks": realtime_response_transform_input.get("current_item_chunks"),
            "current_delta_type": current_delta_type,
            "session_configuration_request": session_configuration_request,
        }
