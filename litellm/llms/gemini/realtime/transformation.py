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
    OpenAIRealtimeEvents,
    OpenAIRealtimeStreamSession,
    OpenAIRealtimeStreamSessionEvents,
)

from ..common_utils import encode_unserializable_types

MAP_GEMINI_FIELD_TO_OPENAI_EVENT = {
    "setupComplete": "session.created",
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

    def transform_realtime_response(
        self,
        message: Union[str, bytes],
        model: str,
        logging_obj: LiteLLMLoggingObj,
        session_configuration_request: Optional[str] = None,
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
