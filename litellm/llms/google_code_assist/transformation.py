import json
import uuid
import copy
import httpx
from typing import Any, List, Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import ModelResponse
from ..vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig


class GoogleCodeAssistError(BaseLLMException):
    """
    Exception raised for errors in the Google Code Assist API.
    """

    def __init__(self, status_code, message):
        super().__init__(status_code=status_code, message=message)


class GoogleCodeAssistConfig(VertexGeminiConfig):
    """
    Reference: https://cloud.google.com/gemini/docs/api/reference/rest/v1internal/projects.locations.codeAssist/generateContent

    The class `GoogleCodeAssistConfig` provides configuration for the Google Code Assist API.
    It inherits from `VertexGeminiConfig` to provide consistent parameter mapping.

    - `temperature` (float): This controls the degree of randomness in token selection.
    - `max_output_tokens` (integer): This sets the limitation for the maximum amount of token in the text output.
    - `top_p` (float): The tokens are selected from the most probable to the least probable until the sum of their probabilities equals the `top_p` value.
    - `top_k` (integer): The value of `top_k` determines how many of the most probable tokens are considered in the selection.
    - `stop_sequences` (List[str]): The set of character sequences that will stop output generation.
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list] = None,
    ) -> None:
        super().__init__(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )

    def get_supported_openai_params(self, model: str) -> List[str]:
        return super().get_supported_openai_params(model)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        messages: list,
    ) -> dict:
        return super().map_openai_params(
            non_default_params, optional_params, model, messages
        )

    def transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
    ) -> dict:
        """
        Transforms standard LiteLLM request to Code Assist API format.
        Matches gemini-cli 'toGenerateContentRequest'.
        """
        # 1. Map messages to Gemini format
        from ..vertex_ai.gemini.transformation import (
            _gemini_convert_messages_with_history,
            _transform_system_message,
        )

        # Create a copy to avoid mutating the original list
        messages_copy = copy.deepcopy(messages)

        # Separate system instruction
        system_instruction, filtered_messages = _transform_system_message(
            supports_system_message=True, messages=messages_copy
        )

        # Convert the rest of messages
        contents = _gemini_convert_messages_with_history(
            messages=filtered_messages, model=model
        )

        # 2. Build vertex-style nested request
        generation_config = {}
        # Handle parameter mapping
        base_params = self.map_openai_params(
            {}, optional_params.copy(), model, messages
        )

        for key in ["temperature", "topP", "topK", "maxOutputTokens", "stopSequences"]:
            if key in base_params:
                generation_config[key] = base_params.pop(key)

        # Support thinkingConfig
        if "thinkingConfig" in base_params:
            generation_config["thinkingConfig"] = base_params.pop("thinkingConfig")
        elif "include_thoughts" in base_params:
            generation_config["thinkingConfig"] = {
                "includeThoughts": base_params.pop("include_thoughts")
            }
        elif "thinkingConfig" in optional_params:
            generation_config["thinkingConfig"] = optional_params["thinkingConfig"]
        elif "include_thoughts" in optional_params:
            generation_config["thinkingConfig"] = {
                "includeThoughts": optional_params["include_thoughts"]
            }

        vertex_request = {
            "contents": contents,
            "session_id": litellm_params.get("session_id", str(uuid.uuid4())),
        }

        if system_instruction:
            vertex_request["systemInstruction"] = {
                "role": "system",
                "parts": system_instruction["parts"],
            }

        if generation_config:
            vertex_request["generationConfig"] = generation_config

        # 3. Wrap in Code Assist envelope (matches verified gemini-cli structure)
        user_prompt_id = f"litellm-{uuid.uuid4()}"[:13]
        model_name = model.split("/")[-1]

        ca_request = {
            "model": model_name,
            "user_prompt_id": user_prompt_id,
            "request": vertex_request,
        }

        # Add project ID if available
        if litellm_params.get("google_code_assist_project"):
            ca_request["project"] = litellm_params["google_code_assist_project"]

        return ca_request

    def transform_response(
        self,
        model: str,
        raw_response: Any,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
    ) -> ModelResponse:
        """
        Transforms Code Assist API response to standard LiteLLM format.
        """
        data = raw_response.json()
        # Code Assist wraps the response in a "response" key
        gemini_response = data.get("response", data)

        # Reuse base vertex transformation
        class MockResponse:
            def __init__(self, json_data):
                self._json = json_data
                self.status_code = 200
                self.text = json.dumps(json_data)
                self.headers = httpx.Headers({"content-type": "application/json"})

            def json(self):
                return self._json

        return super().transform_response(
            model=model,
            raw_response=MockResponse(gemini_response),
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )
