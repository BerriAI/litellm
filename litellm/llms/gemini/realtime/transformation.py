"""
This file contains the transformation logic for the Gemini realtime API.
"""

import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, cast

import litellm
from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.gemini import (
    AutomaticActivityDetection,
    BidiGenerateContentRealtimeInput,
    BidiGenerateContentRealtimeInputConfig,
    BidiGenerateContentServerContent,
    BidiGenerateContentServerMessage,
    BidiGenerateContentSetup,
)
from litellm.types.llms.openai import (
    OpenAIRealtimeContentPartDone,
    OpenAIRealtimeDoneEvent,
    OpenAIRealtimeEvents,
    OpenAIRealtimeEventTypes,
    OpenAIRealtimeFunctionCallArgumentsDone,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseAudioDone,
    OpenAIRealtimeResponseContentPartAdded,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeResponseDoneObject,
    OpenAIRealtimeResponseTextDone,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamResponseOutputItem,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeStreamSession,
    OpenAIRealtimeStreamSessionEvents,
    OpenAIRealtimeTurnDetection,
    ResponsesAPIStreamEvents,
)
from litellm.types.llms.vertex_ai import (
    GeminiResponseModalities,
    HttpxBlobType,
    HttpxContentType,
)
from litellm.types.realtime import (
    ALL_DELTA_TYPES,
    RealtimeModalityResponseTransformOutput,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)
from litellm.utils import get_empty_usage

from ..common_utils import encode_unserializable_types, get_api_key_from_env

MAP_GEMINI_FIELD_TO_OPENAI_EVENT: Dict[
    str, Union[OpenAIRealtimeEventTypes, ResponsesAPIStreamEvents]
] = {
    "setupComplete": OpenAIRealtimeEventTypes.SESSION_CREATED,
    "serverContent.generationComplete": OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE,
    "serverContent.turnComplete": OpenAIRealtimeEventTypes.RESPONSE_DONE,
    "serverContent.interrupted": OpenAIRealtimeEventTypes.RESPONSE_DONE,
    "toolCall": ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
}

# Top-level keys in a Gemini realtime message that map_openai_event knows how
# to handle. Other keys (e.g. ``usageMetadata``) can appear alongside these as
# siblings and must be skipped by the main transform loop — otherwise
# map_openai_event raises ``ValueError`` and the WebSocket session terminates.
_KNOWN_GEMINI_TOP_LEVEL_KEYS: set = {
    map_key.split(".", 1)[0] for map_key in MAP_GEMINI_FIELD_TO_OPENAI_EVENT
}

# Gemini Live native-audio model ids carry this marker (e.g.
# ``gemini-2.5-flash-native-audio-preview-09-2025``). These models reject a
# ``speechConfig`` on ``setup`` with a 1007 invalid-argument error, so it is
# stripped in ``_finalize_gemini_live_setup``.
_GEMINI_NATIVE_AUDIO_MODEL_MARKER = "native-audio"


class GeminiRealtimeConfig(BaseRealtimeConfig):
    # Cap the LRU of in-flight tool calls so long sessions with many tool
    # calls don't grow the dict without bound. Sized large enough to cover
    # bursts of pending tool responses; the oldest entry is evicted when a
    # new call beyond the cap arrives.
    _TOOL_CALL_ID_TO_NAME_MAX = 256

    def __init__(self):
        super().__init__()
        # Store call_id → function_name mapping for tool call round-trip
        self._tool_call_id_to_name: "OrderedDict[str, str]" = OrderedDict()
        # Buffer ``usageMetadata`` that Gemini Live emits as a standalone
        # frame (between turns) so the next ``response.done`` attributes the
        # tokens consumed. Without this an authenticated client can drive
        # tool-call or normal turns whose token usage is recorded as zero,
        # bypassing spend and budget accounting.
        self._pending_usage_metadata: Optional[dict] = None

    @staticmethod
    def _usage_detail_alias(details: Any, defaults: Dict[str, int]) -> Dict[str, Any]:
        if not isinstance(details, dict):
            return dict(defaults)
        return {
            **defaults,
            **{key: value for key, value in details.items() if value is not None},
        }

    @staticmethod
    def _add_pipecat_usage_detail_aliases(usage_dict: Dict[str, Any]) -> Dict[str, Any]:
        usage_dict.setdefault(
            "input_token_details",
            GeminiRealtimeConfig._usage_detail_alias(
                usage_dict.get("input_tokens_details"),
                {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0},
            ),
        )
        usage_dict.setdefault(
            "output_token_details",
            GeminiRealtimeConfig._usage_detail_alias(
                usage_dict.get("output_tokens_details"),
                {"text_tokens": 0, "audio_tokens": 0},
            ),
        )
        return usage_dict

    def validate_environment(
        self, headers: dict, model: str, api_key: Optional[str] = None
    ) -> dict:
        return headers

    def get_complete_url(
        self, api_base: Optional[str], model: str, api_key: Optional[str] = None
    ) -> str:
        """
        Example output:
        "BACKEND_WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"";
        """
        if api_base is None:
            api_base = "wss://generativelanguage.googleapis.com"
        if api_key is None:
            api_key = get_api_key_from_env()
        if api_key is None:
            raise ValueError("api_key is required for Gemini API calls")
        api_base = api_base.replace("https://", "wss://")
        api_base = api_base.replace("http://", "ws://")
        # WebSocket connections do not support custom HTTP headers in all clients,
        # so the API key must remain as a query parameter here. This is an accepted
        # limitation; httpx is not used for WebSocket so MaskedHTTPStatusError
        # already covers the main leak vector.
        return f"{api_base}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"

    def map_model_turn_event(
        self, model_turn: HttpxContentType
    ) -> OpenAIRealtimeEventTypes:
        """
        Map the model turn event to the OpenAI realtime events.

        Returns either:
        - response.text.delta - model_turn: {"parts": [{"text": "..."}]}
        - response.audio.delta - model_turn: {"parts": [{"inlineData": {"mimeType": "audio/pcm", "data": "..."}}]}

        Assumes parts is a single element list.
        """
        if "parts" in model_turn:
            parts = model_turn["parts"]
            if len(parts) != 1:
                verbose_logger.warning(
                    f"Realtime: Expected 1 part, got {len(parts)} for Gemini model turn event."
                )
            part = parts[0]
            if "text" in part:
                return OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA
            elif "inlineData" in part:
                return OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA
            else:
                raise ValueError(f"Unexpected part type: {part}")
        raise ValueError(f"Unexpected model turn event, no 'parts' key: {model_turn}")

    def map_generation_complete_event(
        self, delta_type: Optional[ALL_DELTA_TYPES]
    ) -> OpenAIRealtimeEventTypes:
        if delta_type == "text":
            return OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE
        elif delta_type == "audio":
            return OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE
        else:
            raise ValueError(f"Unexpected delta type: {delta_type}")

    def get_audio_mime_type(self, input_audio_format: str = "pcm16"):
        mime_types = {
            "pcm16": "audio/pcm;rate=24000",
            "g711_ulaw": "audio/pcmu",
            "g711_alaw": "audio/pcma",
        }

        return mime_types.get(input_audio_format, "application/octet-stream")

    def _manual_turn_detection_enabled(
        self, session_configuration_request: Optional[str]
    ) -> bool:
        if not session_configuration_request:
            return False
        try:
            setup = json.loads(session_configuration_request).get("setup", {})
            automatic_detection = setup.get("realtimeInputConfig", {}).get(
                "automaticActivityDetection", {}
            )
            return (
                isinstance(automatic_detection, dict)
                and automatic_detection.get("disabled") is True
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False

    def _handle_input_audio_buffer_commit_or_end(
        self, session_configuration_request: Optional[str]
    ) -> List[str]:
        """Map OpenAI buffer commit/end to Gemini Live turn-boundary signals."""
        if self._manual_turn_detection_enabled(session_configuration_request):
            realtime_input_dict: BidiGenerateContentRealtimeInput = {
                "activityEnd": True,
            }
            verbose_logger.debug(
                "Gemini Realtime: Sending activityEnd realtimeInput to backend"
            )
        else:
            realtime_input_dict = {"audioStreamEnd": True}
            verbose_logger.debug(
                "Gemini Realtime: Sending audioStreamEnd realtimeInput to backend"
            )
        return [json.dumps({"realtimeInput": realtime_input_dict})]

    def map_automatic_turn_detection(
        self, value: OpenAIRealtimeTurnDetection
    ) -> AutomaticActivityDetection:
        """Map OpenAI ``server_vad`` to Gemini ``automaticActivityDetection``.

        OpenAI ``semantic_vad`` has no Gemini Live equivalent — return an empty
        dict so callers omit ``realtimeInputConfig`` (mapping it with
        ``disabled: true`` breaks native-audio sessions).
        """
        if (
            isinstance(value, dict)
            and value.get("type") == "semantic_vad"
            and "create_response" not in value
        ):
            return AutomaticActivityDetection()

        automatic_activity_dection = AutomaticActivityDetection()
        if "create_response" in value and isinstance(value["create_response"], bool):
            automatic_activity_dection["disabled"] = not value["create_response"]
        elif isinstance(value, dict) and value.get("type") == "server_vad":
            # OpenAI server VAD enables activity detection by default.
            automatic_activity_dection["disabled"] = False
        else:
            automatic_activity_dection["disabled"] = True
        if "prefix_padding_ms" in value and isinstance(value["prefix_padding_ms"], int):
            automatic_activity_dection["prefixPaddingMs"] = value["prefix_padding_ms"]
        if "silence_duration_ms" in value and isinstance(
            value["silence_duration_ms"], int
        ):
            automatic_activity_dection["silenceDurationMs"] = value[
                "silence_duration_ms"
            ]
        return automatic_activity_dection

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "instructions",
            "temperature",
            "max_response_output_tokens",
            "modalities",
            "tools",
            "input_audio_transcription",
            "turn_detection",
            "voice",
        ]

    def map_openai_params(
        self, optional_params: dict, non_default_params: dict
    ) -> dict:
        if "generationConfig" not in optional_params:
            optional_params["generationConfig"] = {}
        for key, value in non_default_params.items():
            if key == "instructions":
                optional_params["systemInstruction"] = HttpxContentType(
                    role="user", parts=[{"text": value}]
                )
            elif key == "temperature":
                optional_params["generationConfig"]["temperature"] = value
            elif key == "max_response_output_tokens":
                optional_params["generationConfig"]["maxOutputTokens"] = value
            elif key == "modalities":
                optional_params["generationConfig"]["responseModalities"] = [
                    modality.upper() for modality in cast(List[str], value)
                ]
            elif key == "tools":
                from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
                    VertexGeminiConfig,
                )

                vertex_gemini_config = VertexGeminiConfig()
                # Tools should be at the top level of setup, not inside generationConfig
                optional_params["tools"] = vertex_gemini_config._map_function(
                    value=value, optional_params=optional_params
                )
            elif key == "input_audio_transcription" and value is not None:
                optional_params["inputAudioTranscription"] = {}
            elif key == "turn_detection":
                value_typed = cast(OpenAIRealtimeTurnDetection, value)
                if (
                    isinstance(value_typed, dict)
                    and value_typed.get("type") == "semantic_vad"
                    and "create_response" not in value_typed
                ):
                    # Pipecat/OpenAI GA semantic VAD — skip; Gemini uses its own VAD.
                    # Only skip when there is no create_response override so that
                    # a guardrail-injected create_response:false is not dropped.
                    continue
                transformed_audio_activity_config = self.map_automatic_turn_detection(
                    value_typed
                )
                if transformed_audio_activity_config:
                    optional_params["realtimeInputConfig"] = (
                        BidiGenerateContentRealtimeInputConfig(
                            automaticActivityDetection=transformed_audio_activity_config
                        )
                    )
            elif key == "voice":
                from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
                    VertexGeminiConfig,
                )

                vertex_gemini_config = VertexGeminiConfig()
                speech_config = vertex_gemini_config._map_audio_params({"voice": value})
                if speech_config:
                    optional_params["generationConfig"]["speechConfig"] = speech_config
        if len(optional_params["generationConfig"]) == 0:
            optional_params.pop("generationConfig")
        return optional_params

    @staticmethod
    def _extract_turn_detection(session: dict) -> Optional[dict]:
        """Extract turn_detection from a session.update payload.

        Handles both the flat beta shape (``session.turn_detection``) and the
        GA shape (``session.audio.input.turn_detection``).
        """
        if not isinstance(session, dict):
            return None
        td = session.get("turn_detection")
        if isinstance(td, dict):
            return td
        audio = session.get("audio")
        if isinstance(audio, dict):
            input_cfg = audio.get("input")
            if isinstance(input_cfg, dict):
                td = input_cfg.get("turn_detection")
                if isinstance(td, dict):
                    return td
        return None

    @staticmethod
    def _normalize_session_payload_for_mapping(session: dict) -> dict:
        """Normalize GA-remapped session fields back to their beta keys.

        ``map_openai_params`` only recognises the flat OpenAI-beta key names
        (``modalities``, ``input_audio_transcription``, ``turn_detection``).
        For GA clients the upstream shim renames these into the nested GA
        schema (``output_modalities``, ``audio.input.transcription``,
        ``audio.input.turn_detection``), which would otherwise be silently
        dropped here. Surface them back at the top level so the existing
        mapping logic picks them up without duplicating provider-specific
        knowledge of the GA schema in ``map_openai_params``.
        """
        if not isinstance(session, dict):
            return session

        normalized = dict(session)

        if "modalities" not in normalized and "output_modalities" in normalized:
            normalized["modalities"] = normalized["output_modalities"]

        audio = normalized.get("audio")
        if isinstance(audio, dict):
            input_cfg = audio.get("input")
            if isinstance(input_cfg, dict):
                if (
                    "input_audio_transcription" not in normalized
                    and "transcription" in input_cfg
                ):
                    normalized["input_audio_transcription"] = input_cfg["transcription"]
            output_cfg = audio.get("output")
            if isinstance(output_cfg, dict) and output_cfg.get("voice"):
                normalized["voice"] = output_cfg["voice"]

        extracted_turn_detection = GeminiRealtimeConfig._extract_turn_detection(
            normalized
        )
        if extracted_turn_detection is not None and not isinstance(
            normalized.get("turn_detection"), dict
        ):
            normalized["turn_detection"] = extracted_turn_detection

        return normalized

    @staticmethod
    def _finalize_gemini_live_setup(
        model: str, setup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Drop fields Gemini Live native-audio rejects on ``setup``."""
        if _GEMINI_NATIVE_AUDIO_MODEL_MARKER not in model.lower():
            return setup
        generation_config = setup.get("generationConfig")
        if isinstance(generation_config, dict):
            generation_config.pop("speechConfig", None)
        return setup

    def _handle_session_update(
        self,
        json_message: dict,
        model: str,
        session_configuration_request: Optional[str],
    ) -> List[str]:
        """
        Handle session.update by sending setup to Gemini.

        On the FIRST session.update (when session_configuration_request is None),
        the full setup with all configuration is sent. Gemini Live accepts setup
        as the first-and-only client message, so every later session.update is
        dropped rather than forwarded as a second setup (which Gemini rejects
        with a 1007, tearing the session down). To carry tools/instructions, send
        them on the first session.update before any conversation content.
        """
        session_payload = json_message.get("session") or {}
        # Normalize GA-remapped fields (``output_modalities``,
        # nested ``audio.input.transcription``,
        # ``audio.input.turn_detection``) back to their flat beta keys so
        # ``map_openai_params`` picks them up. Without this, GA clients'
        # explicit modality / transcription / turn-detection settings
        # would be silently dropped because ``map_openai_params`` only
        # recognises the flat OpenAI-beta key names.
        session_payload = self._normalize_session_payload_for_mapping(session_payload)
        new_overrides = self.map_openai_params(
            optional_params={}, non_default_params=session_payload
        )

        if session_configuration_request is None:
            generation_config = new_overrides.setdefault("generationConfig", {})
            generation_config.setdefault("responseModalities", ["AUDIO"])
            new_overrides.setdefault("inputAudioTranscription", {})
            new_overrides["model"] = f"models/{model}"
            verbose_logger.debug(
                "Gemini Realtime: Sending initial setup with tools to backend"
            )
            return [
                json.dumps(
                    {"setup": self._finalize_gemini_live_setup(model, new_overrides)}
                )
            ]

        # Gemini Live accepts exactly one ``setup`` message: the first and only
        # client message. A second ``setup`` closes the socket with
        # ``1007 Request contains an invalid argument``, so a session.update
        # after the initial setup must not be forwarded as a follow-up setup.
        # Every GA client (pipecat included) sends several session.updates while
        # configuring the session; forwarding a second one tears the session down
        # before the first turn, which surfaces to callers as silence after the
        # first response, reconnect/retry latency churn, and 1011 errors. Drop
        # it. The Vertex subclass already drops subsequent setups for this exact
        # reason; the constraint is identical on AI Studio.
        client_turn_detection = self._extract_turn_detection(session_payload)
        if (
            isinstance(client_turn_detection, dict)
            and client_turn_detection.get("create_response") is False
        ):
            verbose_logger.warning(
                "Gemini Realtime: Dropping subsequent session.update "
                "(turn_detection.create_response=False) — Gemini Live rejects a "
                "second setup message, so audio-transcription guardrails cannot "
                "suppress the model's auto-response mid-session."
            )
        else:
            verbose_logger.debug(
                "Gemini Realtime: Ignoring session.update (setup already sent)"
            )
        return []

    def _handle_conversation_item(self, json_message: dict) -> List[str]:
        """
        Handle conversation.item.create for user text or function call output.

        Converts OpenAI format to Gemini's clientContent (for user text) or
        toolResponse (for function outputs).
        """
        item = json_message.get("item", {})
        item_type = item.get("type")

        # Handle function call output (tool response)
        if item_type == "function_call_output":
            return self._handle_function_call_output(item)

        # Handle regular text content
        return self._handle_user_text_content(item)

    def _handle_function_call_output(self, item: dict) -> List[str]:
        """Transform function_call_output to Gemini toolResponse format."""
        call_id = item.get("call_id", "")
        output = item.get("output", "{}")

        verbose_logger.debug(
            f"Gemini Realtime: Transforming function_call_output for call_id={call_id}"
        )

        # Parse the output to get the result. Gemini's
        # functionResponses[].response field is a Struct, so it must be a
        # dict; wrap any non-dict (primitives, lists, invalid JSON) under a
        # `result` key.
        try:
            parsed_output = json.loads(output) if isinstance(output, str) else output
        except json.JSONDecodeError:
            parsed_output = output
        output_dict = (
            parsed_output
            if isinstance(parsed_output, dict)
            else {"result": parsed_output}
        )

        # Look up the function name from stored mapping. Keep the entry so a
        # client SDK that retries function_call_output (or sends it twice for
        # the same tool call) still produces a Gemini toolResponse with the
        # required ``name`` field; refresh the LRU position so an active
        # call_id stays warm across long sessions.
        function_name = self._tool_call_id_to_name.get(call_id)
        if function_name:
            self._tool_call_id_to_name.move_to_end(call_id)
        else:
            verbose_logger.warning(
                f"Gemini Realtime: Function name not found for call_id={call_id}. "
                "This may cause Gemini to reject the response."
            )

        # Build Gemini toolResponse format
        function_response = {
            "id": call_id,
            "response": output_dict,
        }
        if function_name:
            function_response["name"] = function_name

        tool_response_message = {
            "toolResponse": {"functionResponses": [function_response]}
        }

        return [json.dumps(tool_response_message)]

    def _handle_user_text_content(self, item: dict) -> List[str]:
        """Transform user text content to Gemini clientContent format."""
        content_list = item.get("content", [])
        text_parts = [
            c.get("text", "")
            for c in content_list
            if isinstance(c, dict) and c.get("type") == "input_text"
        ]
        text = " ".join(filter(None, text_parts))
        if not text:
            return []

        # Build clientContent message with turns (proper Gemini Live API format)
        client_content_message = {
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turnComplete": True,
            }
        }

        return [json.dumps(client_content_message)]

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: Optional[str] = None,
    ) -> List[str]:
        realtime_input_dict: BidiGenerateContentRealtimeInput = {}
        try:
            json_message = json.loads(message)
        except json.JSONDecodeError:
            if isinstance(message, bytes):
                message_str = message.decode("utf-8", errors="replace")
            else:
                message_str = str(message)
            raise ValueError(f"Invalid JSON message: {message_str}")

        messages: List[str] = []
        msg_type = json_message.get("type")

        ## HANDLE SESSION UPDATE — translate to Gemini setup ##
        if msg_type == "session.update":
            return self._handle_session_update(
                json_message, model, session_configuration_request
            )

        ## HANDLE response.create — Gemini responds automatically; nothing to forward ##
        if msg_type == "response.create":
            return []

        ## HANDLE conversation.item.create — extract user text or function call output ##
        if msg_type == "conversation.item.create":
            return self._handle_conversation_item(json_message)

        ## HANDLE INPUT AUDIO BUFFER - use realtimeInput for audio streaming ##
        if msg_type == "input_audio_buffer.append":
            realtime_input_dict["audio"] = HttpxBlobType(
                mimeType=self.get_audio_mime_type(), data=json_message["audio"]
            )

            realtime_input_dict = cast(
                BidiGenerateContentRealtimeInput,
                encode_unserializable_types(
                    cast(Dict[str, object], realtime_input_dict)
                ),
            )

            gemini_msg = json.dumps({"realtimeInput": realtime_input_dict})
            verbose_logger.debug(
                "Gemini Realtime: Sending audio realtimeInput to backend"
            )
            messages.append(gemini_msg)
            return messages

        if msg_type in ("input_audio_buffer.commit", "input_audio_buffer.end"):
            return self._handle_input_audio_buffer_commit_or_end(
                session_configuration_request
            )

        if msg_type == "input_audio_buffer.clear":
            # Local OpenAI buffer op — nothing to forward to Gemini Live.
            verbose_logger.debug(
                "Gemini Realtime: input_audio_buffer.clear is a local buffer op"
            )
            return []

        # Unknown/unsupported OpenAI event type — drop silently rather than
        # forwarding raw JSON as text input to the model.
        return []

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: Optional[str] = None,
    ) -> OpenAIRealtimeStreamSessionEvents:
        if session_configuration_request:
            session_configuration_request_dict: BidiGenerateContentSetup = json.loads(
                session_configuration_request
            ).get("setup", {})
        else:
            session_configuration_request_dict = {}

        _model = session_configuration_request_dict.get("model") or model
        generation_config = (
            session_configuration_request_dict.get("generationConfig", {}) or {}
        )
        gemini_modalities = generation_config.get("responseModalities", ["AUDIO"])
        _modalities = [
            modality.lower() for modality in cast(List[str], gemini_modalities)
        ]
        _system_instruction = session_configuration_request_dict.get(
            "systemInstruction"
        )
        session = OpenAIRealtimeStreamSession(
            id=logging_session_id,
            modalities=_modalities,
        )
        if _system_instruction is not None and isinstance(_system_instruction, str):
            session["instructions"] = _system_instruction
        if _model is not None and isinstance(_model, str):
            # Normalise to bare model name for OpenAI compatibility.
            # Vertex AI uses a full resource path:
            #   projects/{project}/locations/{location}/publishers/google/models/{model}
            # Google AI Studio uses:
            #   models/{model}
            if "/models/" in _model:
                session["model"] = _model.split("/models/")[-1]
            elif _model.startswith("models/"):
                session["model"] = _model[len("models/") :]
            else:
                session["model"] = _model

        return OpenAIRealtimeStreamSessionEvents(
            type="session.created",
            session=session,
            event_id=str(uuid.uuid4()),
        )

    def _is_new_content_delta(
        self,
        previous_messages: Optional[List[OpenAIRealtimeEvents]] = None,
    ) -> bool:
        if previous_messages is None or len(previous_messages) == 0:
            return True
        if "type" in previous_messages[-1] and previous_messages[-1]["type"].endswith(
            "delta"
        ):
            return False
        return True

    def return_new_content_delta_events(
        self,
        response_id: str,
        output_item_id: str,
        conversation_id: str,
        delta_type: ALL_DELTA_TYPES,
        session_configuration_request: Optional[str] = None,
    ) -> List[OpenAIRealtimeEvents]:
        session_configuration_request_dict: BidiGenerateContentSetup = {}
        if session_configuration_request is not None:
            try:
                session_configuration_request_dict = json.loads(
                    session_configuration_request
                ).get("setup", {})
            except json.JSONDecodeError:
                session_configuration_request_dict = {}
        generation_config = session_configuration_request_dict.get(
            "generationConfig", {}
        )
        gemini_modalities = generation_config.get("responseModalities", ["AUDIO"])
        _modalities = [
            modality.lower() for modality in cast(List[str], gemini_modalities)
        ]

        _temperature = generation_config.get("temperature")
        _max_output_tokens = generation_config.get("maxOutputTokens")

        response_items: List[OpenAIRealtimeEvents] = []

        ## - return response.created
        response_created = OpenAIRealtimeStreamResponseBaseObject(
            type="response.created",
            event_id="event_{}".format(uuid.uuid4()),
            response={
                "object": "realtime.response",
                "id": response_id,
                "status": "in_progress",
                "status_details": None,
                "output": [],
                "conversation_id": conversation_id,
                "modalities": _modalities,
                "temperature": _temperature,
                "max_output_tokens": _max_output_tokens,
            },
        )
        response_items.append(response_created)

        ## - return response.output_item.added
        response_output_item_added = OpenAIRealtimeStreamResponseOutputItemAdded(
            type="response.output_item.added",
            event_id="event_{}".format(uuid.uuid4()),
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
        response_items.append(response_output_item_added)
        ## - return conversation.item.added
        # Pipecat 1.3.x handles "conversation.item.added" (not ".created").
        # Sending ".created" raises "Unimplemented server event type" which
        # kills the receive task handler.
        response_items.append(
            cast(
                OpenAIRealtimeEvents,
                {
                    "type": "conversation.item.added",
                    "event_id": "event_{}".format(uuid.uuid4()),
                    "previous_item_id": None,
                    "item": {
                        "id": output_item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
        )
        ## - return response.content_part.added
        response_content_part_added = OpenAIRealtimeResponseContentPartAdded(
            type="response.content_part.added",
            content_index=0,
            output_index=0,
            event_id="event_{}".format(uuid.uuid4()),
            item_id=output_item_id,
            part=(
                {
                    "type": "text",
                    "text": "",
                }
                if delta_type == "text"
                else {
                    "type": "audio",
                    "transcript": "",
                }
            ),
            response_id=response_id,
        )
        response_items.append(response_content_part_added)
        return response_items

    def transform_content_delta_events(
        self,
        message: BidiGenerateContentServerContent,
        output_item_id: str,
        response_id: str,
        delta_type: ALL_DELTA_TYPES,
    ) -> OpenAIRealtimeResponseDelta:
        delta = ""
        try:
            if "modelTurn" in message and "parts" in message["modelTurn"]:
                for part in message["modelTurn"]["parts"]:
                    if "text" in part:
                        delta += part["text"]
                    elif "inlineData" in part:
                        delta += part["inlineData"].get("data", "")
        except Exception as e:
            raise ValueError(
                f"Error transforming content delta events: {e}, got message: {message}"
            )

        return OpenAIRealtimeResponseDelta(
            type=(
                "response.output_text.delta"
                if delta_type == "text"
                else "response.output_audio.delta"
            ),
            content_index=0,
            event_id="event_{}".format(uuid.uuid4()),
            item_id=output_item_id,
            output_index=0,
            response_id=response_id,
            delta=delta,
        )

    def transform_content_done_event(
        self,
        delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]],
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
        delta_type: ALL_DELTA_TYPES,
    ) -> Union[OpenAIRealtimeResponseTextDone, OpenAIRealtimeResponseAudioDone]:
        if delta_chunks:
            delta = "".join([delta_chunk["delta"] for delta_chunk in delta_chunks])
        else:
            delta = ""
        if current_output_item_id is None:
            current_output_item_id = "item_{}".format(uuid.uuid4())
        if current_response_id is None:
            current_response_id = "resp_{}".format(uuid.uuid4())
        if delta_type == "text":
            return OpenAIRealtimeResponseTextDone(
                type="response.output_text.done",
                content_index=0,
                event_id="event_{}".format(uuid.uuid4()),
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
                text=delta,
            )
        elif delta_type == "audio":
            return OpenAIRealtimeResponseAudioDone(
                type="response.output_audio.done",
                content_index=0,
                event_id="event_{}".format(uuid.uuid4()),
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
            )

    def return_additional_content_done_events(
        self,
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
        delta_done_event: Union[
            OpenAIRealtimeResponseTextDone, OpenAIRealtimeResponseAudioDone
        ],
        delta_type: ALL_DELTA_TYPES,
    ) -> List[OpenAIRealtimeEvents]:
        """
        - return response.content_part.done
        - return response.output_item.done
        """
        if current_output_item_id is None:
            current_output_item_id = "item_{}".format(uuid.uuid4())
        if current_response_id is None:
            current_response_id = "resp_{}".format(uuid.uuid4())
        returned_items: List[OpenAIRealtimeEvents] = []

        delta_done_event_text = cast(Optional[str], delta_done_event.get("text"))
        # response.content_part.done
        response_content_part_done = OpenAIRealtimeContentPartDone(
            type="response.content_part.done",
            content_index=0,
            event_id="event_{}".format(uuid.uuid4()),
            item_id=current_output_item_id,
            output_index=0,
            part=(
                {"type": "text", "text": delta_done_event_text}
                if delta_done_event_text and delta_type == "text"
                else {
                    "type": "audio",
                    "transcript": "",  # gemini doesn't return transcript for audio
                }
            ),
            response_id=current_response_id,
        )
        returned_items.append(response_content_part_done)
        # response.output_item.done
        response_output_item_done = OpenAIRealtimeOutputItemDone(
            type="response.output_item.done",
            event_id="event_{}".format(uuid.uuid4()),
            output_index=0,
            response_id=current_response_id,
            item={
                "id": current_output_item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    (
                        {"type": "text", "text": delta_done_event_text}
                        if delta_done_event_text and delta_type == "text"
                        else {
                            "type": "audio",
                            "transcript": "",
                        }
                    )
                ],
            },
        )
        returned_items.append(response_output_item_done)
        return returned_items

    def _consume_usage_metadata_for_response_done(self, frame: dict) -> Optional[dict]:
        """Return the ``usageMetadata`` to attribute to a ``response.done``.

        Gemini Live emits ``usageMetadata`` either alongside the closing
        frame (``serverContent.turnComplete`` / ``toolCall``) or as a
        standalone frame between turns. The standalone form would otherwise
        be discarded by the no-op branch in ``transform_realtime_response``
        and the consumed tokens silently dropped from spend/budget
        accounting. ``_pending_usage_metadata`` buffers any such standalone
        frames so the next emitted ``response.done`` carries the deferred
        token counts.

        Returns the in-frame ``usageMetadata`` if present (and clears the
        buffer since the in-frame counts are the authoritative attribution
        for this turn), otherwise returns the buffered counts. ``None`` is
        returned when neither is available so the caller can fall back to
        ``get_empty_usage()``.
        """
        # ``pop`` (rather than ``get``) so a single Gemini frame containing
        # multiple closing keys (e.g. both ``toolCall`` and
        # ``serverContent.turnComplete``) cannot attribute the same
        # ``usageMetadata`` to two ``response.done`` events and double-count
        # tokens in spend/budget accounting.
        in_frame = frame.pop("usageMetadata", None) if isinstance(frame, dict) else None
        if isinstance(in_frame, dict):
            self._pending_usage_metadata = None
            return in_frame
        buffered = self._pending_usage_metadata
        self._pending_usage_metadata = None
        return buffered

    def transform_tool_call_events(
        self,
        tool_call_message: dict,
        response_id: Optional[str] = None,
        output_item_id: Optional[str] = None,
    ) -> List[OpenAIRealtimeFunctionCallArgumentsDone]:
        """
        Transform Gemini toolCall message to OpenAI function call events.

        Converts Gemini's functionCalls format to OpenAI's response.function_call_arguments.done events.
        Also stores call_id → name mapping for later use in function_call_output responses.
        """
        function_calls = tool_call_message.get("functionCalls", [])
        resolved_response_id = response_id or f"resp_{uuid.uuid4()}"
        resolved_output_item_id = output_item_id or f"item_{uuid.uuid4()}"

        verbose_logger.debug(
            f"Gemini Realtime: Transforming {len(function_calls)} tool call(s) to OpenAI format"
        )

        events: List[OpenAIRealtimeFunctionCallArgumentsDone] = []
        for idx, fc in enumerate(function_calls):
            call_id = fc.get("id", "") or f"call_{uuid.uuid4().hex[:16]}"
            name = fc.get("name", "")

            # Store call_id → name mapping for round-trip. Use an LRU so
            # repeated function_call_output lookups (retries) still hit, while
            # sessions with many tool calls don't grow the dict unboundedly.
            if call_id and name:
                self._tool_call_id_to_name[call_id] = name
                self._tool_call_id_to_name.move_to_end(call_id)
                while len(self._tool_call_id_to_name) > self._TOOL_CALL_ID_TO_NAME_MAX:
                    self._tool_call_id_to_name.popitem(last=False)

            events.append(
                OpenAIRealtimeFunctionCallArgumentsDone(
                    type="response.function_call_arguments.done",
                    event_id=f"event_{uuid.uuid4()}",
                    response_id=resolved_response_id,
                    item_id=f"{resolved_output_item_id}_tool_{idx}",
                    output_index=idx,
                    call_id=call_id,
                    name=name,
                    arguments=json.dumps(fc.get("args", {})),
                )
            )

        return events

    @staticmethod
    def get_nested_value(obj: dict, path: str) -> Any:
        keys = path.split(".")
        current = obj
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def update_current_delta_chunks(
        self,
        transformed_message: Union[OpenAIRealtimeEvents, List[OpenAIRealtimeEvents]],
        current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]],
    ) -> Optional[List[OpenAIRealtimeResponseDelta]]:
        try:
            if isinstance(transformed_message, list):
                current_delta_chunks = []
                any_delta_chunk = False
                for event in transformed_message:
                    if event["type"] == "response.output_text.delta":
                        current_delta_chunks.append(
                            cast(OpenAIRealtimeResponseDelta, event)
                        )
                        any_delta_chunk = True
                if not any_delta_chunk:
                    current_delta_chunks = (
                        None  # reset current_delta_chunks if no delta chunks
                    )
            else:
                if (
                    transformed_message["type"] == "response.output_text.delta"
                ):  # ONLY ACCUMULATE TEXT DELTA CHUNKS - AUDIO WILL CAUSE SERVER MEMORY ISSUES
                    if current_delta_chunks is None:
                        current_delta_chunks = []
                    current_delta_chunks.append(
                        cast(OpenAIRealtimeResponseDelta, transformed_message)
                    )
                else:
                    current_delta_chunks = None
            return current_delta_chunks
        except Exception as e:
            raise ValueError(
                f"Error updating current delta chunks: {e}, got transformed_message: {transformed_message}"
            )

    def update_current_item_chunks(
        self,
        transformed_message: Union[OpenAIRealtimeEvents, List[OpenAIRealtimeEvents]],
        current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]],
    ) -> Optional[List[OpenAIRealtimeOutputItemDone]]:
        try:
            if isinstance(transformed_message, list):
                current_item_chunks = []
                any_item_chunk = False
                for event in transformed_message:
                    if event["type"] == "response.output_item.done":
                        current_item_chunks.append(
                            cast(OpenAIRealtimeOutputItemDone, event)
                        )
                        any_item_chunk = True
                if not any_item_chunk:
                    current_item_chunks = (
                        None  # reset current_item_chunks if no item chunks
                    )
            else:
                if transformed_message["type"] == "response.output_item.done":
                    if current_item_chunks is None:
                        current_item_chunks = []
                    current_item_chunks.append(
                        cast(OpenAIRealtimeOutputItemDone, transformed_message)
                    )
                else:
                    current_item_chunks = None
            return current_item_chunks
        except Exception as e:
            raise ValueError(
                f"Error updating current item chunks: {e}, got transformed_message: {transformed_message}"
            )

    def transform_response_done_event(
        self,
        message: BidiGenerateContentServerMessage,
        current_response_id: Optional[str],
        current_conversation_id: Optional[str],
        output_items: Optional[List[OpenAIRealtimeOutputItemDone]],
        session_configuration_request: Optional[str] = None,
    ) -> OpenAIRealtimeDoneEvent:
        if current_conversation_id is None:
            current_conversation_id = "conv_{}".format(uuid.uuid4())
        if current_response_id is None:
            current_response_id = "resp_{}".format(uuid.uuid4())

        if session_configuration_request:
            session_configuration_request_dict: BidiGenerateContentSetup = json.loads(
                session_configuration_request
            ).get("setup", {})
        else:
            session_configuration_request_dict = {}

        generation_config = session_configuration_request_dict.get(
            "generationConfig", {}
        )
        temperature = generation_config.get("temperature")
        max_output_tokens = generation_config.get("maxOutputTokens")
        gemini_modalities = generation_config.get("responseModalities", ["AUDIO"])
        _modalities = [
            modality.lower() for modality in cast(List[str], gemini_modalities)
        ]
        resolved_usage_metadata = self._consume_usage_metadata_for_response_done(
            cast(dict, message)
        )
        if resolved_usage_metadata is not None:
            _chat_completion_usage = VertexGeminiConfig._calculate_usage(
                completion_response=cast(
                    BidiGenerateContentServerMessage,
                    {**cast(dict, message), "usageMetadata": resolved_usage_metadata},
                ),
            )
        else:
            _chat_completion_usage = get_empty_usage()

        responses_api_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            _chat_completion_usage,
        )
        _usage_dict = responses_api_usage.model_dump()
        self._add_pipecat_usage_detail_aliases(_usage_dict)
        response_done_event = OpenAIRealtimeDoneEvent(
            type="response.done",
            event_id="event_{}".format(uuid.uuid4()),
            response=OpenAIRealtimeResponseDoneObject(
                object="realtime.response",
                id=current_response_id,
                status="completed",
                status_details=None,  # type: ignore[typeddict-item]
                output=(
                    [output_item["item"] for output_item in output_items]
                    if output_items
                    else []
                ),
                conversation_id=current_conversation_id,
                modalities=_modalities,
                usage=_usage_dict,
            ),
        )
        if temperature is not None:
            response_done_event["response"]["temperature"] = temperature
        if max_output_tokens is not None:
            response_done_event["response"]["max_output_tokens"] = cast(
                int, max_output_tokens
            )

        return response_done_event

    def handle_openai_modality_event(
        self,
        openai_event: OpenAIRealtimeEventTypes,
        json_message: dict,
        realtime_response_transform_input: RealtimeResponseTransformInput,
        delta_type: ALL_DELTA_TYPES,
    ) -> RealtimeModalityResponseTransformOutput:
        current_output_item_id = realtime_response_transform_input[
            "current_output_item_id"
        ]
        current_response_id = realtime_response_transform_input["current_response_id"]
        current_conversation_id = realtime_response_transform_input[
            "current_conversation_id"
        ]
        current_delta_chunks = realtime_response_transform_input["current_delta_chunks"]
        session_configuration_request = realtime_response_transform_input[
            "session_configuration_request"
        ]

        returned_message: List[OpenAIRealtimeEvents] = []
        if (
            openai_event == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA
            or openai_event == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA
        ):
            current_response_id = current_response_id or "resp_{}".format(uuid.uuid4())
            if not current_output_item_id:
                # send the list of standard 'new' content.delta events
                current_output_item_id = "item_{}".format(uuid.uuid4())
                current_conversation_id = current_conversation_id or "conv_{}".format(
                    uuid.uuid4()
                )
                returned_message = self.return_new_content_delta_events(
                    session_configuration_request=session_configuration_request,
                    response_id=current_response_id,
                    output_item_id=current_output_item_id,
                    conversation_id=current_conversation_id,
                    delta_type=delta_type,
                )

            # send the list of standard 'new' content.delta events
            transformed_message = self.transform_content_delta_events(
                BidiGenerateContentServerContent(**json_message["serverContent"]),
                current_output_item_id,
                current_response_id,
                delta_type=delta_type,
            )
            returned_message.append(transformed_message)
        elif (
            openai_event == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE
            or openai_event == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE
        ):
            transformed_content_done_event = self.transform_content_done_event(
                current_output_item_id=current_output_item_id,
                current_response_id=current_response_id,
                delta_chunks=current_delta_chunks,
                delta_type=delta_type,
            )
            returned_message = [transformed_content_done_event]

            # Use IDs from the done event — transform_content_done_event may have
            # generated UUID fallbacks when the originals were None.
            resolved_item_id = (
                transformed_content_done_event.get("item_id") or current_output_item_id
            )
            resolved_response_id = (
                transformed_content_done_event.get("response_id") or current_response_id
            )

            additional_items = self.return_additional_content_done_events(
                current_output_item_id=resolved_item_id,
                current_response_id=resolved_response_id,
                delta_done_event=transformed_content_done_event,
                delta_type=delta_type,
            )
            returned_message.extend(additional_items)

        return {
            "returned_message": returned_message,
            "current_output_item_id": current_output_item_id,
            "current_response_id": current_response_id,
            "current_conversation_id": current_conversation_id,
            "current_delta_chunks": current_delta_chunks,
            "current_delta_type": delta_type,
        }

    def map_openai_event(
        self,
        key: str,
        value: Any,
        current_delta_type: Optional[ALL_DELTA_TYPES],
    ) -> Union[OpenAIRealtimeEventTypes, ResponsesAPIStreamEvents]:
        if isinstance(value, dict):
            model_turn_event = value.get("modelTurn")
            generation_complete_event = value.get("generationComplete")
        else:
            model_turn_event = None
            generation_complete_event = None
        openai_event: Optional[
            Union[OpenAIRealtimeEventTypes, ResponsesAPIStreamEvents]
        ] = None
        if model_turn_event:  # check if model turn event
            openai_event = self.map_model_turn_event(model_turn_event)
        elif generation_complete_event:
            openai_event = self.map_generation_complete_event(
                delta_type=current_delta_type
            )
        else:
            # Check if this key or any nested key matches our mapping. Use a
            # distinct loop variable so we don't shadow ``openai_event`` and
            # leak the last dict value when no entry matches. Scope dotted-key
            # lookups to the current ``key``/``value`` pair — checking the
            # whole ``json_message`` would let a sibling key (e.g.
            # ``serverContent.turnComplete``) misclassify the event currently
            # being processed (e.g. ``toolCall``).
            for map_key, candidate_event in MAP_GEMINI_FIELD_TO_OPENAI_EVENT.items():
                if map_key == key:
                    openai_event = candidate_event
                    break
                if "." in map_key:
                    prefix, _, nested_path = map_key.partition(".")
                    if (
                        prefix == key
                        and isinstance(value, dict)
                        and GeminiRealtimeConfig.get_nested_value(value, nested_path)
                        is not None
                    ):
                        openai_event = candidate_event
                        break
        if openai_event is None:
            raise ValueError(f"Unknown openai event: {key}, value: {value}")
        return openai_event

    def transform_realtime_response(
        self,
        message: Union[str, bytes],
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        """
        Keep this state less - leave the state management (e.g. tracking current_output_item_id, current_response_id, current_conversation_id, current_delta_chunks) to the caller.
        """
        try:
            json_message = json.loads(message)
        except json.JSONDecodeError:
            if isinstance(message, bytes):
                message_str = message.decode("utf-8", errors="replace")
            else:
                message_str = str(message)
            raise ValueError(f"Invalid JSON message: {message_str}")

        verbose_logger.debug(
            "Realtime Response Transform: Gemini frame keys=%s",
            (
                sorted(json_message.keys())
                if isinstance(json_message, dict)
                else type(json_message).__name__
            ),
        )

        logging_session_id = logging_obj.litellm_trace_id

        current_output_item_id = realtime_response_transform_input[
            "current_output_item_id"
        ]
        current_response_id = realtime_response_transform_input["current_response_id"]
        current_conversation_id = realtime_response_transform_input[
            "current_conversation_id"
        ]
        current_delta_chunks = realtime_response_transform_input["current_delta_chunks"]
        session_configuration_request = realtime_response_transform_input[
            "session_configuration_request"
        ]
        current_item_chunks = realtime_response_transform_input["current_item_chunks"]
        current_delta_type: Optional[ALL_DELTA_TYPES] = (
            realtime_response_transform_input["current_delta_type"]
        )
        returned_message: List[OpenAIRealtimeEvents] = []

        # Handle transcription events that arrive independently from model
        # content.  Gemini sends inputTranscription / outputTranscription
        # inside serverContent, separately from modelTurn / turnComplete.
        server_content = json_message.get("serverContent")
        if isinstance(server_content, dict):
            input_tx = server_content.get("inputTranscription")
            if isinstance(input_tx, dict) and input_tx.get("text"):
                returned_message.append(
                    cast(
                        OpenAIRealtimeEvents,
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "event_id": "event_{}".format(uuid.uuid4()),
                            "transcript": input_tx["text"],
                            "item_id": "item_{}".format(uuid.uuid4()),
                            "content_index": 0,
                        },
                    )
                )

            output_tx = server_content.get("outputTranscription")
            if isinstance(output_tx, dict) and output_tx.get("text"):
                if current_response_id is None:
                    current_response_id = "resp_{}".format(uuid.uuid4())
                if current_output_item_id is None:
                    current_output_item_id = "item_{}".format(uuid.uuid4())
                    current_conversation_id = (
                        current_conversation_id or "conv_{}".format(uuid.uuid4())
                    )
                    returned_message.extend(
                        self.return_new_content_delta_events(
                            session_configuration_request=session_configuration_request,
                            response_id=current_response_id,
                            output_item_id=current_output_item_id,
                            conversation_id=current_conversation_id,
                            delta_type="audio",
                        )
                    )
                # Emit as the GA event name; _GA_TO_BETA_EVENT_TYPES translates
                # this back to response.audio_transcript.delta for beta clients.
                returned_message.append(
                    cast(
                        OpenAIRealtimeEvents,
                        {
                            "type": "response.output_audio_transcript.delta",
                            "event_id": "event_{}".format(uuid.uuid4()),
                            "transcript": output_tx["text"],
                            "item_id": current_output_item_id,
                            "content_index": 0,
                            "output_index": 0,
                            "response_id": current_response_id,
                            "delta": output_tx["text"],
                        },
                    )
                )

            # If serverContent only contained transcription(s) and no model
            # content, mark it as already handled so the main loop skips it
            # (map_openai_event would raise on an unknown serverContent
            # subkey). Fall through so sibling top-level keys such as
            # ``toolCall`` are still processed in the main loop.
            _model_content_keys = {
                "modelTurn",
                "turnComplete",
                "interrupted",
                "generationComplete",
            }
            server_content_handled = not any(
                k in server_content for k in _model_content_keys
            )
        else:
            server_content_handled = False

        tool_call_handled = False
        # Snapshot the items so handlers below can safely mutate
        # ``json_message`` (e.g. ``_consume_usage_metadata_for_response_done``
        # pops ``usageMetadata`` to prevent a single frame from attributing
        # the same token counts to two ``response.done`` events).
        for key, value in list(json_message.items()):
            # Skip sibling metadata keys (e.g. ``usageMetadata``) that can
            # accompany a primary payload like ``toolCall`` or ``serverContent``.
            # ``map_openai_event`` raises ValueError on unknown keys, which
            # would otherwise terminate the WebSocket session.
            if key not in _KNOWN_GEMINI_TOP_LEVEL_KEYS:
                continue
            # serverContent was a transcription-only payload already emitted
            # above; skip it here so map_openai_event doesn't raise on the
            # missing model-content subkeys.
            if key == "serverContent" and server_content_handled:
                continue
            # Check if this key or any nested key matches our mapping
            openai_event = self.map_openai_event(
                key=key,
                value=value,
                current_delta_type=current_delta_type,
            )

            if openai_event == OpenAIRealtimeEventTypes.SESSION_CREATED:
                transformed_message = self.transform_session_created_event(
                    model,
                    logging_session_id,
                    realtime_response_transform_input["session_configuration_request"],
                )
                returned_message.append(transformed_message)
            elif openai_event == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE:
                # Handle toolCall from Gemini. If the payload has no function
                # calls, emit nothing — an orphaned response.created/done pair
                # with no output items would confuse OpenAI-compatible clients.
                # Mark the key as intentionally consumed (mirroring
                # ``server_content_handled``) so any sibling keys in the same
                # frame are still processed by the rest of the loop and the
                # post-loop guard doesn't treat the no-op as fatal.
                if not value.get("functionCalls"):
                    tool_call_handled = True
                    continue

                if current_conversation_id is None:
                    current_conversation_id = f"conv_{uuid.uuid4()}"

                # Extract session-level response metadata once so both
                # response.created and response.done can include matching
                # modalities/temperature/max_output_tokens fields.
                session_setup: BidiGenerateContentSetup = {}
                if session_configuration_request is not None:
                    try:
                        session_setup = json.loads(session_configuration_request).get(
                            "setup", {}
                        )
                    except (json.JSONDecodeError, TypeError):
                        session_setup = {}
                tool_call_generation_config = (
                    session_setup.get("generationConfig", {}) or {}
                )
                tool_call_modalities = [
                    modality.lower()
                    for modality in cast(
                        List[str],
                        tool_call_generation_config.get(
                            "responseModalities", ["AUDIO"]
                        ),
                    )
                ]

                # Emit response.created preamble if this is the first event in the response
                if current_response_id is None:
                    current_response_id = f"resp_{uuid.uuid4()}"
                    current_output_item_id = f"item_{uuid.uuid4()}"

                    # Mirror the audio/text path: include modalities,
                    # temperature, and max_output_tokens on response.created so
                    # spec-compliant clients see consistent response metadata
                    # regardless of whether the response starts with content or
                    # a tool call.
                    returned_message.append(
                        {
                            "type": "response.created",
                            "event_id": f"event_{uuid.uuid4()}",
                            "response": {
                                "object": "realtime.response",
                                "id": current_response_id,
                                "status": "in_progress",
                                "status_details": None,
                                "output": [],
                                "conversation_id": current_conversation_id,
                                "modalities": tool_call_modalities,
                                "temperature": tool_call_generation_config.get(
                                    "temperature"
                                ),
                                "max_output_tokens": tool_call_generation_config.get(
                                    "maxOutputTokens"
                                ),
                            },
                        }
                    )

                tool_call_events = self.transform_tool_call_events(
                    value,
                    response_id=current_response_id,
                    output_item_id=current_output_item_id,
                )
                # Emit output_item.added and conversation.item.created for each function call
                for idx, tool_call in enumerate(tool_call_events):
                    item_id = tool_call["item_id"]
                    function_call_item: OpenAIRealtimeStreamResponseOutputItem = {
                        "id": item_id,
                        "object": "realtime.item",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": tool_call["call_id"],
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    }
                    # response.output_item.added
                    returned_message.append(
                        OpenAIRealtimeStreamResponseOutputItemAdded(
                            type="response.output_item.added",
                            event_id=f"event_{uuid.uuid4()}",
                            response_id=current_response_id,
                            output_index=idx,
                            item={
                                **function_call_item,
                                "status": "in_progress",
                                "arguments": "",
                            },
                        )
                    )
                    # conversation.item.added — Pipecat 1.3.x registers the
                    # call_id into _pending_function_calls inside
                    # _handle_evt_conversation_item_added, which is triggered
                    # by this event (NOT by response.output_item.added and NOT
                    # by the old conversation.item.created which Pipecat 1.3.x
                    # does not handle). Without this event the subsequent
                    # response.function_call_arguments.done finds an empty
                    # pending-calls dict and drops the tool invocation silently.
                    returned_message.append(
                        cast(
                            OpenAIRealtimeEvents,
                            {
                                "type": "conversation.item.added",
                                "event_id": f"event_{uuid.uuid4()}",
                                "previous_item_id": None,
                                "item": {
                                    **function_call_item,
                                    "status": "in_progress",
                                    "arguments": "",
                                },
                            },
                        )
                    )
                    # response.function_call_arguments.delta — Gemini delivers
                    # the full arguments string in a single toolCall frame
                    # rather than streaming partial chunks, so emit one delta
                    # carrying the complete payload before the matching
                    # ``.done`` event. Spec-compliant OpenAI Realtime SDK
                    # clients accumulate ``delta.delta`` and rely on at least
                    # one delta before ``.done``.
                    returned_message.append(
                        cast(
                            OpenAIRealtimeEvents,
                            {
                                "type": "response.function_call_arguments.delta",
                                "event_id": f"event_{uuid.uuid4()}",
                                "response_id": current_response_id,
                                "item_id": item_id,
                                "output_index": idx,
                                "call_id": tool_call["call_id"],
                                "delta": tool_call["arguments"],
                            },
                        )
                    )
                    # response.function_call_arguments.done
                    returned_message.append(tool_call)
                    # response.output_item.done — pass a fresh copy so
                    # downstream handlers that mutate the item dict (e.g. the
                    # beta-protocol translator) don't corrupt the references
                    # used by sibling events sharing the same function_call_item.
                    returned_message.append(
                        OpenAIRealtimeOutputItemDone(
                            type="response.output_item.done",
                            event_id=f"event_{uuid.uuid4()}",
                            response_id=current_response_id,
                            output_index=idx,
                            item={**function_call_item},
                        )
                    )

                # response.done - close the response so clients can submit tool
                # results. Mirror the non-tool-call RESPONSE_DONE path: if Gemini
                # delivered ``usageMetadata`` alongside this ``toolCall`` frame,
                # propagate the real token counts so spend/budget accounting
                # records the tokens consumed by the tool-call turn. Standalone
                # ``usageMetadata`` frames emitted in a separate WebSocket frame
                # are buffered on the instance so the next ``response.done``
                # picks them up (otherwise an authenticated client could drive
                # tool-call turns whose token usage is recorded as zero,
                # bypassing budgets). Falls back to an empty usage block when
                # neither is available (OpenAI-compatible clients expect
                # ``usage`` to always be present on response.done).
                resolved_tool_call_usage_metadata = (
                    self._consume_usage_metadata_for_response_done(json_message)
                )
                if resolved_tool_call_usage_metadata is not None:
                    _tool_call_chat_completion_usage = (
                        VertexGeminiConfig._calculate_usage(
                            completion_response=cast(
                                BidiGenerateContentServerMessage,
                                {
                                    **json_message,
                                    "usageMetadata": resolved_tool_call_usage_metadata,
                                },
                            ),
                        )
                    )
                else:
                    _tool_call_chat_completion_usage = get_empty_usage()
                tool_call_responses_api_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
                    _tool_call_chat_completion_usage,
                )
                _tool_usage_dict = tool_call_responses_api_usage.model_dump()
                self._add_pipecat_usage_detail_aliases(_tool_usage_dict)
                tool_call_done_event = OpenAIRealtimeDoneEvent(
                    type="response.done",
                    event_id=f"event_{uuid.uuid4()}",
                    response=OpenAIRealtimeResponseDoneObject(
                        id=current_response_id,
                        object="realtime.response",
                        status="completed",
                        status_details=None,  # type: ignore[typeddict-item]
                        output=[
                            {
                                "id": te["item_id"],
                                "object": "realtime.item",
                                "type": "function_call",
                                "status": "completed",
                                "call_id": te["call_id"],
                                "name": te["name"],
                                "arguments": te["arguments"],
                            }
                            for te in tool_call_events
                        ],
                        conversation_id=current_conversation_id,
                        modalities=tool_call_modalities,
                        usage=_tool_usage_dict,
                    ),
                )
                tool_call_temperature = tool_call_generation_config.get("temperature")
                if tool_call_temperature is not None:
                    tool_call_done_event["response"][
                        "temperature"
                    ] = tool_call_temperature
                tool_call_max_output_tokens = tool_call_generation_config.get(
                    "maxOutputTokens"
                )
                if tool_call_max_output_tokens is not None:
                    tool_call_done_event["response"]["max_output_tokens"] = cast(
                        int, tool_call_max_output_tokens
                    )
                returned_message.append(tool_call_done_event)
                # Reset IDs so the next model turn (after tool results) starts a
                # fresh response with its own response.created preamble.
                current_output_item_id = None
                current_response_id = None
            elif openai_event == OpenAIRealtimeEventTypes.RESPONSE_DONE:
                transformed_response_done_event = self.transform_response_done_event(
                    message=BidiGenerateContentServerMessage(**json_message),  # type: ignore
                    current_response_id=current_response_id,
                    current_conversation_id=current_conversation_id,
                    session_configuration_request=session_configuration_request,
                    output_items=None,
                )
                returned_message.append(transformed_response_done_event)
                # Reset IDs so a subsequent turn (e.g. a `toolCall` arriving in
                # a later WebSocket frame after `turnComplete`) starts a fresh
                # response with its own `response.created` preamble instead of
                # reusing the just-completed response ID.
                current_output_item_id = None
                current_response_id = None
            elif (
                openai_event == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA
                or openai_event == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE
                or openai_event == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA
                or openai_event == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE
            ):
                # Pass the locally-updated state (rather than the original
                # input snapshot) so that prior iterations of this loop —
                # e.g. a tool-call or response.done that just reset
                # current_response_id/current_output_item_id to None — are
                # honoured by the modality handler.
                _modality_input: RealtimeResponseTransformInput = {
                    **realtime_response_transform_input,
                    "current_output_item_id": current_output_item_id,
                    "current_response_id": current_response_id,
                    "current_conversation_id": current_conversation_id,
                    "current_delta_chunks": current_delta_chunks,
                    "current_item_chunks": current_item_chunks,
                    "current_delta_type": current_delta_type,
                    "session_configuration_request": session_configuration_request,
                }
                _returned_message = self.handle_openai_modality_event(
                    openai_event,
                    json_message,
                    _modality_input,
                    delta_type="text" if "text" in openai_event.value else "audio",
                )
                returned_message.extend(_returned_message["returned_message"])
                current_output_item_id = _returned_message["current_output_item_id"]
                current_response_id = _returned_message["current_response_id"]
                current_conversation_id = _returned_message["current_conversation_id"]
                current_delta_chunks = _returned_message["current_delta_chunks"]
                current_delta_type = _returned_message["current_delta_type"]
            else:
                raise ValueError(f"Unknown openai event: {openai_event}")
        if len(returned_message) == 0:
            # A frame whose only top-level keys are sibling metadata (e.g.
            # a standalone ``{"usageMetadata": {...}}`` emitted by Gemini
            # Live between turns) is not an error — there is just nothing
            # to forward to the OpenAI-shaped client. Returning the
            # unchanged state keeps the WebSocket alive; raising would
            # terminate the session for a benign no-op frame.
            # serverContent already consumed by the transcription handler is
            # a benign no-op for downstream — treat it like a metadata-only
            # key when deciding whether to raise.
            unhandled_known_keys = [
                key
                for key in json_message
                if key in _KNOWN_GEMINI_TOP_LEVEL_KEYS
                and not (key == "serverContent" and server_content_handled)
                and not (key == "toolCall" and tool_call_handled)
            ]
            # Buffer standalone usage metadata so the next response.done can
            # attribute the token counts. Without this, an authenticated
            # client driving turns whose usageMetadata is emitted in a
            # separate frame would have those tokens recorded as zero spend,
            # bypassing budget enforcement.
            standalone_usage_metadata = json_message.get("usageMetadata")
            if isinstance(standalone_usage_metadata, dict):
                self._pending_usage_metadata = standalone_usage_metadata
            if not unhandled_known_keys:
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
            if isinstance(message, bytes):
                message_str = message.decode("utf-8", errors="replace")
            else:
                message_str = str(message)
            raise ValueError(f"Unknown message type: {message_str}")

        current_delta_chunks = self.update_current_delta_chunks(
            transformed_message=returned_message,
            current_delta_chunks=current_delta_chunks,
        )
        current_item_chunks = self.update_current_item_chunks(
            transformed_message=returned_message,
            current_item_chunks=current_item_chunks,
        )

        for msg in returned_message:
            event_type = msg.get("type") if isinstance(msg, dict) else "unknown"
            verbose_logger.debug(
                "Realtime Response Transform: OpenAI event=%s", event_type
            )

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

    def requires_session_configuration(self) -> bool:
        # Default behavior is backwards-compatible: send setup on connect.
        # Opt-in to deferred setup for tool-injection flow via:
        #   litellm.gemini_live_defer_setup = True
        return not litellm.gemini_live_defer_setup

    def session_configuration_request(self, model: str) -> str:
        """

        ```
        {
            "model": string,
            "generationConfig": {
                "candidateCount": integer,
                "maxOutputTokens": integer,
                "temperature": number,
                "topP": number,
                "topK": integer,
                "presencePenalty": number,
                "frequencyPenalty": number,
                "responseModalities": [string],
                "speechConfig": object,
                "mediaResolution": object
            },
            "systemInstruction": string,
            "tools": [object]
        }
        ```
        """

        response_modalities: List[GeminiResponseModalities] = ["AUDIO"]
        output_audio_transcription = False
        # if "audio" in model: ## UNCOMMENT THIS WHEN AUDIO IS SUPPORTED
        #     output_audio_transcription = True

        setup_config: BidiGenerateContentSetup = {
            "model": f"models/{model}",
            "generationConfig": {"responseModalities": response_modalities},
            # Return input transcript so guardrails can inspect user speech.
            "inputAudioTranscription": {},
        }
        if output_audio_transcription:
            setup_config["outputAudioTranscription"] = {}
        return json.dumps(
            {
                "setup": setup_config,
            }
        )
