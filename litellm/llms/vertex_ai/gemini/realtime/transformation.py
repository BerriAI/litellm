"""
This file contains the transformation logic for the Vertex realtime API.
"""

import json, os
import uuid
from typing import Any, Dict, List, Optional, Union, cast

from litellm import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.llms.gemini.common_utils import encode_unserializable_types
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
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
    OpenAIRealtimeResponseTextDone,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeStreamSession,
    OpenAIRealtimeStreamSessionEvents,
    OpenAIRealtimeTurnDetection,
)
from litellm.types.llms.vertex_ai import (
    BlobType,
    GenerateContentResponseBody,
    PartType,
    ContentType,
    BidiGenerateContentClientContent,
    GenerationConfig,
    GeminiResponseModalities,
    HttpxBlobType,
    AutomaticActivityDetection,
    BidiGenerateContentRealtimeInput,
    BidiGenerateContentSetup,
    BidiGenerateContentServerContent,
    BidiGenerateContentServerMessage,
)
from litellm.types.realtime import (
    ALL_DELTA_TYPES,
    RealtimeModalityResponseTransformOutput,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)
from litellm.utils import get_empty_usage

# Same map as Gemini; server sends identical field names on Vertex Live
MAP_VERTEX_FIELD_TO_OPENAI_EVENT: Dict[str, OpenAIRealtimeEventTypes] = {
    "setupComplete": OpenAIRealtimeEventTypes.SESSION_CREATED,
    "serverContent.generationComplete": OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE,
    "serverContent.turnComplete": OpenAIRealtimeEventTypes.RESPONSE_DONE,
    "serverContent.interrupted": OpenAIRealtimeEventTypes.RESPONSE_DONE,
}


def _resolve_vertex_model_resource(
    model: str,
) -> str:
    """
    Returns a fully-qualified Vertex publisher model resource:
      projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}
    """
    VERTEX_PROJECT: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT", None)
    VERTEX_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if "/" in model:
        # e.g. "vertex_ai/gemini-live-..."
        model = model.split("/")[-1]

    if not VERTEX_PROJECT:
        raise ValueError("Vertex Live: GOOGLE_CLOUD_PROJECT not set and model is not a full resource name.")
    return f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}" f"/publishers/google/models/{model}"


class VertexLiveRealtimeConfig(BaseRealtimeConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vb = VertexBase()

    def validate_environment(self, headers: Dict, model: str, api_key: Optional[str] = None) -> Dict:
        if api_key:
            auth_header = api_key
            project_id = None
        else:
            auth_header, project_id = self._vb.get_access_token(
                credentials=None,
                project_id=None,
            )

        merged = self._vb.set_headers(auth_header, headers or {})
        merged.pop("Content-Type", None)
        return merged

    def get_complete_url(self, api_base: Optional[str], model: str, api_key: Optional[str] = None) -> str:
        """
        Example output:
        "BACKEND_WS_URL = "wss://aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent";
        """
        region = self._vb.get_vertex_region(vertex_region=None, model=model)
        http_base = self._vb.get_api_base(api_base=api_base, vertex_location=region)

        ws_base = http_base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"

    def map_model_turn_event(self, model_turn: ContentType) -> OpenAIRealtimeEventTypes:
        """
        Map the Vertex/Gemini model turn event to OpenAI Realtime events.

        Returns one of:
        - response.text.delta   when parts == [{"text": "..."}]
        - response.audio.delta  when parts == [{"inlineData": {"mimeType": "audio/pcm", "data": "<b64>"}}]

        Assumes `parts` is a single-element list.
        """
        parts = model_turn.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"Unexpected modelTurn: missing/invalid 'parts': {model_turn}")

        for raw_part in parts:
            if not isinstance(raw_part, dict):
                continue
            part = cast(PartType, raw_part)
            if part.get("thought") is True:
                continue
            if isinstance(part.get("text"), str):
                return OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA
            inline_data = cast(Optional[dict], part.get("inline_data") or part.get("inlineData"))
            if isinstance(inline_data, dict):
                mime_type = inline_data.get("mime_type") or inline_data.get("mimeType")
                if isinstance(mime_type, str) and mime_type.startswith("audio/"):
                    return OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA
        verbose_logger.warning(f"Realtime: Unhandled Vertex modelTurn parts: {parts}")
        return OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA

    def map_generation_complete_event(self, delta_type: Optional[ALL_DELTA_TYPES]) -> OpenAIRealtimeEventTypes:
        if delta_type == "text":
            return OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE
        elif delta_type == "audio":
            return OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE
        else:
            raise ValueError(f"Unexpected delta type: {delta_type}")

    def get_audio_mime_type(self, input_audio_format: str = "pcm16") -> str:
        return {
            "pcm16": "audio/pcm",
            "g711_ulaw": "audio/pcmu",
            "g711_alaw": "audio/pcma",
        }.get(input_audio_format, "application/octet-stream")

    def map_automatic_turn_detection(self, value: OpenAIRealtimeTurnDetection) -> AutomaticActivityDetection:
        automatic_activity_detection = AutomaticActivityDetection()
        create = value.get("create_response")
        if isinstance(create, bool):
            automatic_activity_detection["disabled"] = not create
        else:
            automatic_activity_detection["disabled"] = True
        prefix_padding_ms = value.get("prefix_padding_ms")
        if isinstance(prefix_padding_ms, int):
            automatic_activity_detection["prefix_padding_ms"] = prefix_padding_ms
        silence_duration_ms = value.get("silence_duration_ms")
        if isinstance(silence_duration_ms, int):
            automatic_activity_detection["silence_duration_ms"] = silence_duration_ms
        return automatic_activity_detection

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "instructions",
            "temperature",
            "max_output_tokens",
            "max_response_output_tokens",
            "modalities",
            "tools",
            "input_audio_transcription",
            "turn_detection",
        ]

    def map_openai_params(
        self,
        optional_params: Dict[str, Any],
        non_default_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "generation_config" not in optional_params:
            optional_params["generation_config"] = {}
        generation_config = optional_params.setdefault("generation_config", {})
        for key, value in non_default_params.items():
            if key == "instructions":
                optional_params["system_instruction"] = ContentType(role="user", parts=[{"text": value}])
            elif key == "temperature":
                generation_config["temperature"] = cast(float, value)
            elif key in ("max_output_tokens", "max_response_output_tokens"):
                generation_config["max_output_tokens"] = cast(int, value)
            elif key == "modalities":

                def _to_gemini_modalities(mods: List[str]) -> List[GeminiResponseModalities]:
                    out: List[GeminiResponseModalities] = []
                    for m in mods:
                        mu = (m or "").upper()
                        if mu in ("TEXT", "IMAGE", "AUDIO", "VIDEO"):
                            out.append(cast(GeminiResponseModalities, mu))
                    return out

                generation_config["responseModalities"] = _to_gemini_modalities(cast(List[str], value))
            elif key == "tools":
                try:
                    vertex_gemini_config = VertexGeminiConfig()
                    optional_params["tools"] = vertex_gemini_config._map_function(value)
                except Exception:
                    optional_params["tools"] = value
            elif key == "input_audio_transcription" and value is not None:
                optional_params["input_audio_transcription"] = {}
            elif key == "turn_detection":
                transformed = self.map_automatic_turn_detection(cast(OpenAIRealtimeTurnDetection, value))
                if transformed:
                    ric = optional_params.setdefault("realtime_input_config", {})
                    ric["automatic_activity_detection"] = transformed

        if len(optional_params.get("generation_config", {})) == 0:
            optional_params.pop("generation_config")
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

        messages: List[str] = []
        if "type" in json_message and json_message["type"] == "session.update":
            client_session_configuration_request = self.map_openai_params(
                optional_params={}, non_default_params=json_message["session"]
            )
            client_session_configuration_request["model"] = _resolve_vertex_model_resource(model)

            messages.append(
                json.dumps(
                    {
                        "setup": client_session_configuration_request,
                    }
                )
            )
        if "type" in json_message and json_message["type"] == "input_audio_buffer.append":
            realtime_input_dict["audio"] = BlobType(mime_type=self.get_audio_mime_type(), data=json_message["audio"])
        else:
            realtime_input_dict["text"] = message

        if len(realtime_input_dict) != 1:
            raise ValueError(
                f"Only one argument can be set, got {len(realtime_input_dict)}:" f" {list(realtime_input_dict.keys())}"
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
        generation_config = session_configuration_request_dict.get("generation_config", {}) or {}
        response_modalities = generation_config.get("responseModalities") or ["TEXT"]
        modalities = [str(m).lower() for m in response_modalities]
        system_instruction = session_configuration_request_dict.get("system_instruction")
        instructions = None
        if isinstance(system_instruction, dict):
            parts = system_instruction.get("parts") or []
            if parts and isinstance(parts[0], dict):
                instructions = parts[0].get("text")
        _model = session_configuration_request_dict.get("model") or model
        session = OpenAIRealtimeStreamSession(id=logging_session_id, modalities=modalities)
        if instructions:
            session["instructions"] = instructions
        if isinstance(_model, str):
            session["model"] = _model.split("models/", 1)[-1] if "models/" in _model else _model

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
        if "type" in previous_messages[-1] and previous_messages[-1]["type"].endswith("delta"):
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
        """
        Build initial OpenAI Realtime 'delta' scaffolding events for Vertex,
        using BidiGenerateContentSetup and ONLY snake_case generation_config.
        """
        if session_configuration_request is None:
            raise ValueError("session_configuration_request is required for Gemini API calls")
        session_configuration_request_dict: BidiGenerateContentSetup = json.loads(session_configuration_request).get(
            "setup", {}
        )
        generation_config = session_configuration_request_dict.get("generation_config", {})
        response_modalities = cast(List[str], generation_config.get("responseModalities") or ["TEXT"])
        modalities = [str(m).lower() for m in response_modalities]
        temperature = generation_config.get("temperature")
        max_output_tokens = generation_config.get("max_output_tokens")
        response_items: List[OpenAIRealtimeEvents] = []
        # response.created
        response_items.append(
            OpenAIRealtimeStreamResponseBaseObject(
                type="response.created",
                event_id=f"event_{uuid.uuid4()}",
                response={
                    "object": "realtime.response",
                    "id": response_id,
                    "status": "in_progress",
                    "output": [],
                    "conversation_id": conversation_id,
                    "modalities": modalities,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
        )
        # response.output_item.added
        response_items.append(
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
        response_items.append(
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
        response_items.append(
            OpenAIRealtimeResponseContentPartAdded(
                type="response.content_part.added",
                content_index=0,
                output_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=output_item_id,
                part=({"type": "text", "text": ""} if delta_type == "text" else {"type": "audio", "transcript": ""}),
                response_id=response_id,
            )
        )
        return response_items

    def transform_content_delta_events(
        self,
        message: BidiGenerateContentServerContent,
        output_item_id: str,
        response_id: str,
        delta_type: ALL_DELTA_TYPES,
    ) -> OpenAIRealtimeResponseDelta:
        model_turn = message.get("model_turn") or message.get("modelTurn")
        if not isinstance(model_turn, dict):
            return OpenAIRealtimeResponseDelta(
                type=("response.text.delta" if delta_type == "text" else "response.audio.delta"),
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=output_item_id,
                output_index=0,
                response_id=response_id,
                delta="",
            )
        content = cast(ContentType, model_turn)
        parts: List[PartType] = cast(List[PartType], content.get("parts") or [])
        chunks: List[str] = []
        if delta_type == "text":
            for part in parts:
                if part.get("thought") is True:
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        else:  # delta_type == "audio"
            for part in parts:
                if part.get("thought") is True:
                    continue
                inline = part.get("inline_data") or part.get("inlineData")
                if isinstance(inline, dict):
                    data_b64 = inline.get("data")
                    if isinstance(data_b64, str):
                        chunks.append(data_b64)
        return OpenAIRealtimeResponseDelta(
            type=("response.text.delta" if delta_type == "text" else "response.audio.delta"),
            content_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=output_item_id,
            output_index=0,
            response_id=response_id,
            delta="".join(chunks),
        )

    def transform_content_done_event(
        self,
        delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]],
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
        delta_type: ALL_DELTA_TYPES,
    ) -> Union[OpenAIRealtimeResponseTextDone, OpenAIRealtimeResponseAudioDone]:
        if current_output_item_id is None or current_response_id is None:
            raise ValueError("current_output_item_id and current_response_id cannot be None for a 'done' event.")
        delta = ""
        if delta_chunks:
            delta = "".join([d.get("delta", "") for d in delta_chunks])

        if delta_type == "text":
            return OpenAIRealtimeResponseTextDone(
                type="response.text.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
                text=delta,
            )
        else:
            return OpenAIRealtimeResponseAudioDone(
                type="response.audio.done",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
            )

    def return_additional_content_done_events(
        self,
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
        delta_done_event: Union[OpenAIRealtimeResponseTextDone, OpenAIRealtimeResponseAudioDone],
        delta_type: ALL_DELTA_TYPES,
    ) -> List[OpenAIRealtimeEvents]:
        if current_output_item_id is None or current_response_id is None:
            raise ValueError("current_output_item_id and current_response_id cannot be None for a 'done' event.")

        returned_items: List[OpenAIRealtimeEvents] = []
        delta_done_event_text = cast(Optional[str], delta_done_event.get("text"))
        safe_text: str = delta_done_event_text or ""

        response_content_part_done = OpenAIRealtimeContentPartDone(
            type="response.content_part.done",
            content_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            output_index=0,
            part=(
                {"type": "text", "text": safe_text}
                if delta_type == "text"
                else {"type": "audio", "transcript": ""}  # keep your audio shape
            ),
            response_id=current_response_id,
        )
        returned_items.append(response_content_part_done)

        response_output_item_done = OpenAIRealtimeOutputItemDone(
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
                "content": [
                    (
                        {"type": "text", "text": safe_text}
                        if delta_type == "text"
                        else {"type": "audio", "transcript": ""}
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
                current = current[key]  # <-- assign back
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
                        current_delta_chunks.append(cast(OpenAIRealtimeResponseDelta, event))
                        any_delta_chunk = True
                if not any_delta_chunk:
                    current_delta_chunks = None
            else:
                if transformed_message["type"] == "response.text.delta":
                    if current_delta_chunks is None:
                        current_delta_chunks = []
                    current_delta_chunks.append(cast(OpenAIRealtimeResponseDelta, transformed_message))
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
                new_items: List[OpenAIRealtimeOutputItemDone] = []
                for event in transformed_message:
                    if event["type"] == "response.output_item.done":
                        new_items.append(cast(OpenAIRealtimeOutputItemDone, event))
                return new_items or None
            else:
                if transformed_message["type"] == "response.output_item.done":
                    current_item_chunks = current_item_chunks or []
                    current_item_chunks.append(cast(OpenAIRealtimeOutputItemDone, transformed_message))
                    return current_item_chunks
                return None
        except Exception as e:
            raise ValueError(f"Error updating current item chunks: {e}, got transformed_message: {transformed_message}")

    def transform_response_done_event(
        self,
        message: BidiGenerateContentServerMessage,
        current_response_id: Optional[str],
        current_conversation_id: Optional[str],
        output_items: Optional[List[OpenAIRealtimeOutputItemDone]],
        session_configuration_request: Optional[str] = None,
    ) -> OpenAIRealtimeDoneEvent:
        if current_conversation_id is None or current_response_id is None:
            raise ValueError("current_conversation_id and current_response_id must be set for 'done'.")
        setup_dict: Dict[str, Any] = {}
        if session_configuration_request:
            try:
                setup_dict = json.loads(session_configuration_request).get("setup", {}) or {}
            except Exception:
                setup_dict = {}
        generation_config = cast(GenerationConfig, setup_dict.get("generation_config", {}) or {})
        temperature = generation_config.get("temperature")
        max_output_tokens = generation_config.get("max_output_tokens")
        response_modalities = generation_config.get("responseModalities") or ["TEXT"]
        if "usageMetadata" in message:
            usage_metadata = message.get("usageMetadata")
            calc_input = cast(GenerateContentResponseBody, {"usageMetadata": usage_metadata})
            chat_completion_usage = VertexGeminiConfig._calculate_usage(calc_input)
        else:
            chat_completion_usage = get_empty_usage()

        responses_api_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            chat_completion_usage,
        )
        response_done_event: OpenAIRealtimeDoneEvent = {
            "type": "response.done",
            "event_id": f"event_{uuid.uuid4()}",
            "response": {
                "object": "realtime.response",
                "id": current_response_id,
                "status": "completed",
                "output": ([oi["item"] for oi in output_items] if output_items else []),
                "conversation_id": current_conversation_id,
                "modalities": response_modalities,
                "usage": responses_api_usage.model_dump(),
            },
        }
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
        current_output_item_id = realtime_response_transform_input["current_output_item_id"]
        current_response_id = realtime_response_transform_input["current_response_id"]
        current_conversation_id = realtime_response_transform_input["current_conversation_id"]
        current_delta_chunks = realtime_response_transform_input["current_delta_chunks"]
        session_configuration_request = realtime_response_transform_input["session_configuration_request"]
        returned_message: List[OpenAIRealtimeEvents] = []
        if openai_event in (
            OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA,
            OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA,
        ):
            current_response_id = current_response_id or f"resp_{uuid.uuid4()}"
            if not current_output_item_id:
                current_output_item_id = f"item_{uuid.uuid4()}"
                current_conversation_id = current_conversation_id or f"conv_{uuid.uuid4()}"
                returned_message = self.return_new_content_delta_events(
                    session_configuration_request=session_configuration_request,
                    response_id=current_response_id,
                    output_item_id=current_output_item_id,
                    conversation_id=current_conversation_id,
                    delta_type=delta_type,
                )
            transformed_message = self.transform_content_delta_events(
                BidiGenerateContentServerContent(**json_message["serverContent"]),
                current_output_item_id,
                current_response_id,
                delta_type=delta_type,
            )
            returned_message.append(transformed_message)

        elif openai_event in (
            OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE,
            OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE,
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
        model_turn_event = cast(ContentType, value.get("modelTurn"))
        generation_complete_event = value.get("generationComplete")
        openai_event: Optional[OpenAIRealtimeEventTypes] = None
        if model_turn_event:
            openai_event = self.map_model_turn_event(model_turn_event)
        elif generation_complete_event:
            openai_event = self.map_generation_complete_event(delta_type=current_delta_type)
        else:
            for map_key, openai_event in MAP_VERTEX_FIELD_TO_OPENAI_EVENT.items():
                if map_key == key or (
                    "." in map_key and VertexLiveRealtimeConfig.get_nested_value(json_message, map_key) is not None
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
        try:
            json_message: Dict[str, Any] = json.loads(message)
        except json.JSONDecodeError:
            msg_str = message.decode("utf-8", errors="replace") if isinstance(message, bytes) else str(message)
            raise ValueError(f"Invalid JSON message: {msg_str}")

        logging_session_id = logging_obj.litellm_trace_id

        current_output_item_id = realtime_response_transform_input["current_output_item_id"]
        current_response_id = realtime_response_transform_input["current_response_id"]
        current_conversation_id = realtime_response_transform_input["current_conversation_id"]
        current_delta_chunks = realtime_response_transform_input["current_delta_chunks"]
        session_configuration_request = realtime_response_transform_input["session_configuration_request"]
        current_item_chunks = realtime_response_transform_input["current_item_chunks"]
        current_delta_type: Optional[ALL_DELTA_TYPES] = realtime_response_transform_input["current_delta_type"]

        returned_message: List[OpenAIRealtimeEvents] = []
        for key, value in json_message.items():
            openai_event = self.map_openai_event(
                key=key,
                value=value,
                current_delta_type=current_delta_type,
                json_message=json_message,
            )

            if openai_event == OpenAIRealtimeEventTypes.SESSION_CREATED:
                transformed_message = self.transform_session_created_event(
                    model=model,
                    logging_session_id=logging_session_id,
                    session_configuration_request=realtime_response_transform_input.get(
                        "session_configuration_request"
                    ),
                )
                session_configuration_request = json.dumps(transformed_message)
                returned_message.append(transformed_message)

            elif openai_event == OpenAIRealtimeEventTypes.RESPONSE_DONE:
                server_message = BidiGenerateContentServerMessage(**json_message)
                transformed_response_done_event = self.transform_response_done_event(
                    message=server_message,
                    current_response_id=current_response_id,
                    current_conversation_id=current_conversation_id,
                    session_configuration_request=session_configuration_request,
                    output_items=None,
                )
                returned_message.append(transformed_response_done_event)

            elif openai_event in (
                OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA,
                OpenAIRealtimeEventTypes.RESPONSE_TEXT_DONE,
                OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA,
                OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE,
            ):
                _returned = self.handle_openai_modality_event(
                    openai_event=openai_event,
                    json_message=json_message,
                    realtime_response_transform_input=realtime_response_transform_input,
                    delta_type="text" if "text" in openai_event.value else "audio",
                )
                returned_message.extend(_returned["returned_message"])
                current_output_item_id = _returned["current_output_item_id"]
                current_response_id = _returned["current_response_id"]
                current_conversation_id = _returned["current_conversation_id"]
                current_delta_chunks = _returned["current_delta_chunks"]
                current_delta_type = _returned["current_delta_type"]

            else:
                raise ValueError(f"Unknown openai event: {openai_event}")

        if not returned_message:
            msg_str = message.decode("utf-8", errors="replace") if isinstance(message, bytes) else str(message)
            raise ValueError(f"Unknown message type: {msg_str}")

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
        Build the Vertex 'setup' message. We resolve the model to a full resource
        here so config.yaml can keep friendly names (e.g., vertex_ai/<id>).
        """
        # Build full model resource for Vertex
        resolved_model = _resolve_vertex_model_resource(model)
        response_modalities: List[GeminiResponseModalities] = ["TEXT"]  # or ["AUDIO"] if you want audio out by default

        setup_config: BidiGenerateContentSetup = {
            "model": resolved_model,
            "generation_config": {"responseModalities": response_modalities},
        }
        return json.dumps(
            {
                "setup": setup_config,
            }
        )
