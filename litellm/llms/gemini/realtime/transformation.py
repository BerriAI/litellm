"""
This file contains the transformation logic for the Gemini realtime API.
"""

import json
from litellm._uuid import uuid
from typing import Any, Dict, List, Optional, Union, cast

from litellm import verbose_logger
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

MAP_GEMINI_FIELD_TO_OPENAI_EVENT: Dict[str, OpenAIRealtimeEventTypes] = {
    "setupComplete": OpenAIRealtimeEventTypes.SESSION_CREATED,
    "serverContent.generationComplete": OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE,
    "serverContent.turnComplete": OpenAIRealtimeEventTypes.RESPONSE_DONE,
    "serverContent.interrupted": OpenAIRealtimeEventTypes.RESPONSE_DONE,
}


class GeminiRealtimeConfig(BaseRealtimeConfig):
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
            "pcm16": "audio/pcm",
            "g711_ulaw": "audio/pcmu",
            "g711_alaw": "audio/pcma",
        }

        return mime_types.get(input_audio_format, "application/octet-stream")

    def map_automatic_turn_detection(
        self, value: OpenAIRealtimeTurnDetection
    ) -> AutomaticActivityDetection:
        automatic_activity_dection = AutomaticActivityDetection()
        if "create_response" in value and isinstance(value["create_response"], bool):
            automatic_activity_dection["disabled"] = not value["create_response"]
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
                vertex_gemini_config._map_function(value)
                optional_params["generationConfig"]["tools"] = (
                    vertex_gemini_config._map_function(value)
                )
            elif key == "input_audio_transcription" and value is not None:
                optional_params["inputAudioTranscription"] = {}
            elif key == "turn_detection":
                value_typed = cast(OpenAIRealtimeTurnDetection, value)
                transformed_audio_activity_config = self.map_automatic_turn_detection(
                    value_typed
                )
                if (
                    len(transformed_audio_activity_config) > 0
                ):  # if the config is not empty, add it to the optional params
                    optional_params["realtimeInputConfig"] = (
                        BidiGenerateContentRealtimeInputConfig(
                            automaticActivityDetection=transformed_audio_activity_config
                        )
                    )
        if len(optional_params["generationConfig"]) == 0:
            optional_params.pop("generationConfig")
        return optional_params

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

        ## HANDLE SESSION UPDATE ##
        messages: List[str] = []
        if "type" in json_message and json_message["type"] == "session.update":
            client_session_configuration_request = self.map_openai_params(
                optional_params={}, non_default_params=json_message["session"]
            )
            client_session_configuration_request["model"] = f"models/{model}"

            messages.append(
                json.dumps(
                    {
                        "setup": client_session_configuration_request,
                    }
                )
            )
        # elif session_configuration_request is None:
        #     default_session_configuration_request = self.session_configuration_request(model)
        #     messages.append(default_session_configuration_request)

        ## HANDLE INPUT AUDIO BUFFER ##
        if (
            "type" in json_message
            and json_message["type"] == "input_audio_buffer.append"
        ):
            realtime_input_dict["audio"] = HttpxBlobType(
                mimeType=self.get_audio_mime_type(), data=json_message["audio"]
            )
        else:
            realtime_input_dict["text"] = message

        if len(realtime_input_dict) != 1:
            raise ValueError(
                f"Only one argument can be set, got {len(realtime_input_dict)}:"
                f" {list(realtime_input_dict.keys())}"
            )

        realtime_input_dict = cast(
            BidiGenerateContentRealtimeInput,
            encode_unserializable_types(cast(Dict[str, object], realtime_input_dict)),
        )

        messages.append(json.dumps({"realtime_input": realtime_input_dict}))
        return messages

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
        gemini_modalities = generation_config.get("responseModalities", ["TEXT"])
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
            session["model"] = _model.strip(
                "models/"
            )  # keep it consistent with how openai returns the model name

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
        if session_configuration_request is None:
            raise ValueError(
                "session_configuration_request is required for Gemini API calls"
            )

        session_configuration_request_dict: BidiGenerateContentSetup = json.loads(
            session_configuration_request
        ).get("setup", {})
        generation_config = session_configuration_request_dict.get(
            "generationConfig", {}
        )
        gemini_modalities = generation_config.get("responseModalities", ["TEXT"])
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
                "output": [],
                "conversation_id": conversation_id,
                "modalities": _modalities,
                "temperature": _temperature,
                "max_output_tokens": _max_output_tokens,
            },
        )
        response_items.append(response_created)

        ## - return response.output_item.added ← adds ‘item_id’ same for all subsequent events
        response_output_item_added = OpenAIRealtimeStreamResponseOutputItemAdded(
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
        response_items.append(response_output_item_added)
        ## - return conversation.item.created
        conversation_item_created = OpenAIRealtimeConversationItemCreated(
            type="conversation.item.created",
            event_id="event_{}".format(uuid.uuid4()),
            item={
                "id": output_item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        )
        response_items.append(conversation_item_created)
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
                        delta += part["inlineData"]["data"]
        except Exception as e:
            raise ValueError(
                f"Error transforming content delta events: {e}, got message: {message}"
            )

        return OpenAIRealtimeResponseDelta(
            type=(
                "response.text.delta"
                if delta_type == "text"
                else "response.audio.delta"
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
        if current_output_item_id is None or current_response_id is None:
            raise ValueError(
                "current_output_item_id and current_response_id cannot be None for a 'done' event."
            )
        if delta_type == "text":
            return OpenAIRealtimeResponseTextDone(
                type="response.text.done",
                content_index=0,
                event_id="event_{}".format(uuid.uuid4()),
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
                text=delta,
            )
        elif delta_type == "audio":
            return OpenAIRealtimeResponseAudioDone(
                type="response.audio.done",
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
        if current_output_item_id is None or current_response_id is None:
            raise ValueError(
                "current_output_item_id and current_response_id cannot be None for a 'done' event."
            )
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
                    if event["type"] == "response.text.delta":
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
                    transformed_message["type"] == "response.text.delta"
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
        if current_conversation_id is None or current_response_id is None:
            raise ValueError(
                f"current_conversation_id and current_response_id must all be set for a 'done' event. Got=current_conversation_id: {current_conversation_id}, current_response_id: {current_response_id}"
            )

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
        max_output_tokens = generation_config.get("max_output_tokens")
        gemini_modalities = generation_config.get("responseModalities", ["TEXT"])
        _modalities = [
            modality.lower() for modality in cast(List[str], gemini_modalities)
        ]
        if "usageMetadata" in message:
            _chat_completion_usage = VertexGeminiConfig._calculate_usage(
                completion_response=message,
            )
        else:
            _chat_completion_usage = get_empty_usage()

        responses_api_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            _chat_completion_usage,
        )
        response_done_event = OpenAIRealtimeDoneEvent(
            type="response.done",
            event_id="event_{}".format(uuid.uuid4()),
            response=OpenAIRealtimeResponseDoneObject(
                object="realtime.response",
                id=current_response_id,
                status="completed",
                output=(
                    [output_item["item"] for output_item in output_items]
                    if output_items
                    else []
                ),
                conversation_id=current_conversation_id,
                modalities=_modalities,
                usage=responses_api_usage.model_dump(),
            ),
        )
        if temperature is not None:
            response_done_event["response"]["temperature"] = temperature
        if max_output_tokens is not None:
            response_done_event["response"]["max_output_tokens"] = max_output_tokens

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

            additional_items = self.return_additional_content_done_events(
                current_output_item_id=current_output_item_id,
                current_response_id=current_response_id,
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
        value: dict,
        current_delta_type: Optional[ALL_DELTA_TYPES],
        json_message: dict,
    ) -> OpenAIRealtimeEventTypes:
        model_turn_event = value.get("modelTurn")
        generation_complete_event = value.get("generationComplete")
        openai_event: Optional[OpenAIRealtimeEventTypes] = None
        if model_turn_event:  # check if model turn event
            openai_event = self.map_model_turn_event(model_turn_event)
        elif generation_complete_event:
            openai_event = self.map_generation_complete_event(
                delta_type=current_delta_type
            )
        else:
            # Check if this key or any nested key matches our mapping
            for map_key, openai_event in MAP_GEMINI_FIELD_TO_OPENAI_EVENT.items():
                if map_key == key or (
                    "." in map_key
                    and GeminiRealtimeConfig.get_nested_value(json_message, map_key)
                    is not None
                ):
                    openai_event = openai_event
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

        for key, value in json_message.items():
            # Check if this key or any nested key matches our mapping
            openai_event = self.map_openai_event(
                key=key,
                value=value,
                current_delta_type=current_delta_type,
                json_message=json_message,
            )

            if openai_event == OpenAIRealtimeEventTypes.SESSION_CREATED:
                transformed_message = self.transform_session_created_event(
                    model,
                    logging_session_id,
                    realtime_response_transform_input["session_configuration_request"],
                )
                session_configuration_request = json.dumps(transformed_message)
                returned_message.append(transformed_message)
            elif openai_event == OpenAIRealtimeEventTypes.RESPONSE_DONE:
                transformed_response_done_event = self.transform_response_done_event(
                    message=BidiGenerateContentServerMessage(**json_message),  # type: ignore
                    current_response_id=current_response_id,
                    current_conversation_id=current_conversation_id,
                    session_configuration_request=session_configuration_request,
                    output_items=None,
                )
                returned_message.append(transformed_response_done_event)
            elif (
                openai_event == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA
                or openai_event == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE
                or openai_event == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA
                or openai_event == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE
            ):
                _returned_message = self.handle_openai_modality_event(
                    openai_event,
                    json_message,
                    realtime_response_transform_input,
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
        return True

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
        }
        if output_audio_transcription:
            setup_config["outputAudioTranscription"] = {}
        return json.dumps(
            {
                "setup": setup_config,
            }
        )
