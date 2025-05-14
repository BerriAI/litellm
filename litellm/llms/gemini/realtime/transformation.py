"""
This file contains the transformation logic for the Gemini realtime API.
"""

import json
import os
from typing import Any, Dict, Optional

from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig

from ..common_utils import encode_unserializable_types


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

    def transform_realtime_response(self, message: str) -> str:
        return message

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
