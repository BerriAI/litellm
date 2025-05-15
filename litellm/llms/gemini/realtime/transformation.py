"""
This file contains the transformation logic for the Gemini realtime API.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Union

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.llms.gemini import (
    BidiGenerateContentServerContent,
    BidiGenerateContentSetupComplete,
)
from litellm.types.llms.openai import (
    OpenAIRealtimeConversationItemCreated,
    OpenAIRealtimeEvents,
    OpenAIRealtimeResponseContentPartAdded,
    OpenAIRealtimeResponseTextDelta,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeStreamSession,
    OpenAIRealtimeStreamSessionEvents,
)

from ..common_utils import encode_unserializable_types

MAP_GEMINI_FIELD_TO_OPENAI_EVENT = {
    "setupComplete": "session.created",
    "serverContent": "response.text.delta",
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
        message: BidiGenerateContentSetupComplete,
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
                "conversation_id": "conv_{}".format(uuid.uuid4()),
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
        if "parts" in message["modelTurn"]:
            for part in message["modelTurn"]["parts"]:
                if "text" in part:
                    delta += part["text"]

        return OpenAIRealtimeResponseTextDelta(
            type="response.text.delta",
            content_index=0,
            event_id="event_{}".format(uuid.uuid4()),
            item_id=output_item_id,
            output_index=0,
            response_id=response_id,
            delta=delta,
        )

    def transform_realtime_response(
        self,
        message: Union[str, bytes],
        model: str,
        logging_obj: LiteLLMLoggingObj,
        session_configuration_request: Optional[str] = None,
        previous_messages: Optional[List[OpenAIRealtimeEvents]] = None,
    ) -> Union[OpenAIRealtimeEvents, List[OpenAIRealtimeEvents]]:
        try:
            json_message = json.loads(message)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON message: {message}")

        logging_session_id = logging_obj.litellm_trace_id
        for key, value in json_message.items():
            if key in MAP_GEMINI_FIELD_TO_OPENAI_EVENT:
                openai_event = MAP_GEMINI_FIELD_TO_OPENAI_EVENT[key]
                if openai_event == "session.created":
                    transformed_message = self.transform_session_created_event(
                        BidiGenerateContentSetupComplete(**json_message),  # type: ignore
                        model,
                        logging_session_id,
                        session_configuration_request,
                    )
                    return transformed_message
                elif openai_event == "content.delta":
                    # check if this is a new content.delta or a continuation of a previous content.delta
                    if self._is_new_content_delta(previous_messages):
                        # send the list of standard 'new' content.delta events
                        response_id = "resp_{}".format(uuid.uuid4())
                        output_item_id = "item_{}".format(uuid.uuid4())
                        response_items = self.return_new_content_delta_events(
                            session_configuration_request=session_configuration_request,
                            response_id=response_id,
                            output_item_id=output_item_id,
                        )

                        transformed_message = self.transform_content_delta_events(
                            BidiGenerateContentServerContent(**json_message),  # type: ignore
                            output_item_id,
                            response_id,
                        )
                        response_items.append(transformed_message)
                        return response_items

        raise ValueError(f"Unknown message type: {message}")

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
