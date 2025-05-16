"""
This file contains the transformation logic for the Gemini realtime API.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Union, cast

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.gemini import (
    BidiGenerateContentServerContent,
    BidiGenerateContentServerMessage,
)
from litellm.types.llms.openai import (
    OpenAIRealtimeContentPartDone,
    OpenAIRealtimeConversationItemCreated,
    OpenAIRealtimeDoneEvent,
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseContentPartAdded,
    OpenAIRealtimeResponseDoneObject,
    OpenAIRealtimeResponseTextDelta,
    OpenAIRealtimeResponseTextDone,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeStreamSession,
    OpenAIRealtimeStreamSessionEvents,
)
from litellm.types.realtime import (
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)

from ..common_utils import encode_unserializable_types

MAP_GEMINI_FIELD_TO_OPENAI_EVENT = {
    "setupComplete": "session.created",
    "serverContent.modelTurn": "response.text.delta",
    "serverContent.generationComplete": "response.text.done",
    "serverContent.turnComplete": "response.done",
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
            api_key = os.environ.get("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("api_key is required for Gemini API calls")
        api_base = api_base.replace("https://", "wss://")
        api_base = api_base.replace("http://", "ws://")
        return f"{api_base}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"

    def transform_realtime_request(self, message: str) -> str:
        realtime_input_dict: Dict[str, Any] = {}
        realtime_input_dict["text"] = message

        if len(realtime_input_dict) != 1:
            raise ValueError(
                f"Only one argument can be set, got {len(realtime_input_dict)}:"
                f" {list(realtime_input_dict.keys())}"
            )

        realtime_input_dict = encode_unserializable_types(realtime_input_dict)

        return json.dumps({"realtime_input": realtime_input_dict})

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: Optional[str] = None,
    ) -> OpenAIRealtimeStreamSessionEvents:
        if session_configuration_request is None:
            raise ValueError(
                "session_configuration_request is required for Gemini API calls"
            )

        session_configuration_request_dict = json.loads(session_configuration_request)
        _model = session_configuration_request_dict.get("model") or model
        _modalities = session_configuration_request_dict.get(
            "generationConfig", {}
        ).get("responseModalities", ["TEXT"])
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
        session_configuration_request: Optional[str] = None,
    ) -> List[OpenAIRealtimeEvents]:
        if session_configuration_request is None:
            raise ValueError(
                "session_configuration_request is required for Gemini API calls"
            )

        session_configuration_request_dict = json.loads(session_configuration_request)
        _modalities = session_configuration_request_dict.get(
            "generationConfig", {}
        ).get("responseModalities", ["TEXT"])
        _temperature = session_configuration_request_dict.get(
            "generationConfig", {}
        ).get("temperature")
        _max_output_tokens = session_configuration_request_dict.get(
            "generationConfig", {}
        ).get("maxOutputTokens")

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
            part={
                "type": "text",
                "text": "",
            },
            response_id=response_id,
        )
        response_items.append(response_content_part_added)
        return response_items

    def transform_content_delta_events(
        self,
        message: BidiGenerateContentServerContent,
        output_item_id: str,
        response_id: str,
    ) -> OpenAIRealtimeResponseTextDelta:
        delta = ""
        try:
            if "modelTurn" in message and "parts" in message["modelTurn"]:
                for part in message["modelTurn"]["parts"]:
                    if "text" in part:
                        delta += part["text"]
        except Exception as e:
            raise ValueError(
                f"Error transforming content delta events: {e}, got message: {message}"
            )

        return OpenAIRealtimeResponseTextDelta(
            type="response.text.delta",
            content_index=0,
            event_id="event_{}".format(uuid.uuid4()),
            item_id=output_item_id,
            output_index=0,
            response_id=response_id,
            delta=delta,
        )

    def transform_content_done_event(
        self,
        delta_chunks: Optional[List[OpenAIRealtimeResponseTextDelta]],
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
    ) -> OpenAIRealtimeResponseTextDone:
        if delta_chunks:
            delta = "".join([delta_chunk["delta"] for delta_chunk in delta_chunks])
        else:
            delta = ""
        if current_output_item_id is None or current_response_id is None:
            raise ValueError(
                "current_output_item_id and current_response_id cannot be None for a 'done' event."
            )
        return OpenAIRealtimeResponseTextDone(
            type="response.text.done",
            content_index=0,
            event_id="event_{}".format(uuid.uuid4()),
            item_id=current_output_item_id,
            output_index=0,
            response_id=current_response_id,
            text=delta,
        )

    def return_additional_content_done_events(
        self,
        current_output_item_id: Optional[str],
        current_response_id: Optional[str],
        delta_done_event: OpenAIRealtimeResponseTextDone,
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
        # response.content_part.done
        response_content_part_done = OpenAIRealtimeContentPartDone(
            type="response.content_part.done",
            content_index=0,
            event_id="event_{}".format(uuid.uuid4()),
            item_id=current_output_item_id,
            output_index=0,
            part={
                "type": "text",
                "text": delta_done_event["text"],
            },
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
                    {
                        "type": "text",
                        "text": delta_done_event["text"],
                    }
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
        current_delta_chunks: Optional[List[OpenAIRealtimeResponseTextDelta]],
    ) -> Optional[List[OpenAIRealtimeResponseTextDelta]]:
        try:
            if isinstance(transformed_message, list):
                current_delta_chunks = []
                any_delta_chunk = False
                for event in transformed_message:
                    if event["type"] == "response.text.delta":
                        current_delta_chunks.append(
                            cast(OpenAIRealtimeResponseTextDelta, event)
                        )
                        any_delta_chunk = True
                if not any_delta_chunk:
                    current_delta_chunks = (
                        None  # reset current_delta_chunks if no delta chunks
                    )
            else:
                if transformed_message["type"] == "response.text.delta":
                    if current_delta_chunks is None:
                        current_delta_chunks = []
                    current_delta_chunks.append(
                        cast(OpenAIRealtimeResponseTextDelta, transformed_message)
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
        current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]],
        output_items: Optional[List[OpenAIRealtimeOutputItemDone]],
        session_configuration_request: Optional[str] = None,
    ) -> OpenAIRealtimeDoneEvent:
        if (
            current_conversation_id is None
            or current_response_id is None
            or current_item_chunks is None
        ):
            raise ValueError(
                "current_conversation_id and current_response_id and current_item_chunks cannot be None for a 'done' event."
            )
        if session_configuration_request is None:
            raise ValueError(
                "session_configuration_request is required for Gemini API calls"
            )

        session_configuration_request_dict = json.loads(session_configuration_request)
        temperature = session_configuration_request_dict.get(
            "generationConfig", {}
        ).get("temperature")
        max_output_tokens = session_configuration_request_dict.get(
            "generationConfig", {}
        ).get("maxOutputTokens")
        _modalities = session_configuration_request_dict.get(
            "generationConfig", {}
        ).get("responseModalities", ["TEXT"])
        _chat_completion_usage = VertexGeminiConfig()._calculate_usage(
            completion_response=message,
        )
        responses_api_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            _chat_completion_usage,
        )
        return OpenAIRealtimeDoneEvent(
            type="response.done",
            event_id="event_{}".format(uuid.uuid4()),
            response=OpenAIRealtimeResponseDoneObject(
                object="realtime.response",
                id=current_response_id,
                status="completed",
                output=[output_item["item"] for output_item in output_items]
                if output_items
                else [],
                conversation_id=current_conversation_id,
                modalities=_modalities,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                usage=responses_api_usage.model_dump(),
            ),
        )

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
        returned_message: Optional[
            Union[OpenAIRealtimeEvents, List[OpenAIRealtimeEvents]]
        ] = None
        for key, value in json_message.items():
            # Check if this key or any nested key matches our mapping
            for map_key, openai_event in MAP_GEMINI_FIELD_TO_OPENAI_EVENT.items():
                if map_key == key or (
                    "." in map_key
                    and GeminiRealtimeConfig.get_nested_value(json_message, map_key)
                    is not None
                ):
                    if openai_event == "session.created":
                        transformed_message = self.transform_session_created_event(
                            model,
                            logging_session_id,
                            realtime_response_transform_input[
                                "session_configuration_request"
                            ],
                        )
                        returned_message = transformed_message

                    elif openai_event == "response.text.delta":
                        # check if this is a new content.delta or a continuation of a previous content.delta
                        if not current_output_item_id:
                            # send the list of standard 'new' content.delta events
                            current_response_id = (
                                current_response_id or "resp_{}".format(uuid.uuid4())
                            )
                            current_output_item_id = "item_{}".format(uuid.uuid4())
                            current_conversation_id = (
                                current_conversation_id
                                or "conv_{}".format(uuid.uuid4())
                            )
                            response_items = self.return_new_content_delta_events(
                                session_configuration_request=session_configuration_request,
                                response_id=current_response_id,
                                output_item_id=current_output_item_id,
                                conversation_id=current_conversation_id,
                            )

                            transformed_message = self.transform_content_delta_events(
                                BidiGenerateContentServerContent(**json_message[key]),  # type: ignore
                                current_output_item_id,
                                current_response_id,
                            )
                            response_items.append(transformed_message)
                            returned_message = response_items
                        else:
                            current_response_id = (
                                current_response_id or "resp_{}".format(uuid.uuid4())
                            )
                            # send the list of standard 'new' content.delta events
                            transformed_message = self.transform_content_delta_events(
                                BidiGenerateContentServerContent(**json_message[key]),  # type: ignore
                                current_output_item_id,
                                current_response_id,
                            )
                            returned_message = transformed_message
                    elif openai_event == "response.text.done":
                        transformed_content_done_event = (
                            self.transform_content_done_event(
                                current_output_item_id=current_output_item_id,
                                current_response_id=current_response_id,
                                delta_chunks=current_delta_chunks,
                            )
                        )
                        returned_message = [transformed_content_done_event]

                        additional_items = self.return_additional_content_done_events(
                            current_output_item_id=current_output_item_id,
                            current_response_id=current_response_id,
                            delta_done_event=transformed_content_done_event,
                        )
                        returned_message.extend(additional_items)
                    elif openai_event == "response.done":
                        transformed_response_done_event = self.transform_response_done_event(
                            message=BidiGenerateContentServerMessage(**json_message),  # type: ignore
                            current_response_id=current_response_id,
                            current_conversation_id=current_conversation_id,
                            session_configuration_request=session_configuration_request,
                            output_items=current_item_chunks,
                        )
                        returned_message = transformed_response_done_event

        if returned_message is None:
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
        }

    def requires_session_configuration(self) -> bool:
        return True

    def session_configuration_request(self, model: str) -> Optional[str]:
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
        return json.dumps(
            {
                "setup": {
                    "model": f"models/{model}",
                    "generationConfig": {"responseModalities": ["TEXT"]},
                }
            }
        )
