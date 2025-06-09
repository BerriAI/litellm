from typing import Dict, List, Optional, Any

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_generic_image_chunk_to_openai_image_obj,
    convert_to_anthropic_image_obj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import ContentType, PartType, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, Tools
from litellm.utils import supports_reasoning

from ...vertex_ai.gemini.transformation import _gemini_convert_messages_with_history
from ...vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig


class GoogleAIStudioGeminiConfig(VertexGeminiConfig):
    """
    Reference: https://ai.google.dev/api/rest/v1beta/GenerationConfig

    The class `GoogleAIStudioGeminiConfig` provides configuration for the Google AI Studio's Gemini API interface. Below are the parameters:

    - `temperature` (float): This controls the degree of randomness in token selection.

    - `max_output_tokens` (integer): This sets the limitation for the maximum amount of token in the text output. In this case, the default value is 256.

    - `top_p` (float): The tokens are selected from the most probable to the least probable until the sum of their probabilities equals the `top_p` value. Default is 0.95.

    - `top_k` (integer): The value of `top_k` determines how many of the most probable tokens are considered in the selection. For example, a `top_k` of 1 means the selected token is the most probable among all tokens. The default value is 40.

    - `response_mime_type` (str): The MIME type of the response. The default value is 'text/plain'. Other values - `application/json`.

    - `response_schema` (dict): Optional. Output response schema of the generated candidate text when response mime type can have schema. Schema can be objects, primitives or arrays and is a subset of OpenAPI schema. If set, a compatible response_mime_type must also be set. Compatible mimetypes: application/json: Schema for JSON response.

    - `candidate_count` (int): Number of generated responses to return.

    - `stop_sequences` (List[str]): The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    response_mime_type: Optional[str] = None
    response_schema: Optional[dict] = None
    candidate_count: Optional[int] = None
    stop_sequences: Optional[list] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[dict] = None,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[list] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def is_model_gemini_audio_model(self, model: str) -> bool:
        return "tts" in model

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = [
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "tools",
            "tool_choice",
            "functions",
            "response_format",
            "n",
            "stop",
            "logprobs",
            "frequency_penalty",
            "modalities",
            "parallel_tool_calls",
            "web_search_options",
        ]
        if supports_reasoning(model):
            supported_params.append("reasoning_effort")
            supported_params.append("thinking")
        if self.is_model_gemini_audio_model(model):
            supported_params.append("audio")
        return supported_params

    def _filter_gemini_unsupported_formats(self, schema: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Filter out format values that are not supported by Gemini API.
        
        According to the Gemini API error message, only 'enum' and 'date-time' formats
        are supported for STRING type. Other formats like 'email', 'uri', etc. should be removed.
        
        Args:
            schema: The schema dictionary to filter
            
        Returns:
            The filtered schema dictionary
        """
        if not isinstance(schema, dict):
            return schema

        # Recursively process the schema
        filtered_schema: Dict[Any, Any] = {}
        for key, value in schema.items():
            if key == "format" and isinstance(value, str):
                # Only keep 'enum' and 'date-time' formats for Gemini API
                if value in ["enum", "date-time"]:
                    filtered_schema[key] = value
                # Skip other format values like 'email', 'uri', etc.
            elif key == "properties" and isinstance(value, dict):
                # Recursively filter properties
                filtered_schema[key] = {
                    prop_key: self._filter_gemini_unsupported_formats(prop_value)
                    if isinstance(prop_value, dict)
                    else prop_value
                    for prop_key, prop_value in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                # Recursively filter array items
                filtered_schema[key] = self._filter_gemini_unsupported_formats(value)
            elif key == "anyOf" and isinstance(value, list):
                # Recursively filter anyOf items
                filtered_schema[key] = [
                    self._filter_gemini_unsupported_formats(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                filtered_schema[key] = value

        return filtered_schema

    def _map_function(self, value: List[dict]) -> List[Tools]:
        """
        Override the parent _map_function to apply Gemini-specific format filtering.
        
        This method calls the parent implementation and then filters out
        unsupported format values from the schema.
        """
        # First call the parent implementation to get the standard transformation
        tools = super()._map_function(value)

        # Then apply Gemini-specific format filtering
        for tool in tools:
            if "function_declarations" in tool:
                for func_declaration in tool["function_declarations"]:
                    if "parameters" in func_declaration:
                        parameters = func_declaration["parameters"]
                        if isinstance(parameters, dict):
                            func_declaration[
                                "parameters"
                            ] = self._filter_gemini_unsupported_formats(parameters)

        return tools

    def map_openai_params(
        self,
        non_default_params: Dict,
        optional_params: Dict,
        model: str,
        drop_params: bool,
    ) -> Dict:
        # Handle audio parameter for TTS models
        if self.is_model_gemini_audio_model(model):
            for param, value in non_default_params.items():
                if param == "audio" and isinstance(value, dict):
                    # Validate audio format - Gemini TTS only supports pcm16
                    audio_format = value.get("format")
                    if audio_format is not None and audio_format != "pcm16":
                        raise ValueError(
                            f"Unsupported audio format for Gemini TTS models: {audio_format}. "
                            f"Gemini TTS models only support 'pcm16' format as they return audio data in L16 PCM format. "
                            f"Please set audio format to 'pcm16'."
                        )

                    # Map OpenAI audio parameter to Gemini speech config
                    speech_config: SpeechConfig = {}

                    if "voice" in value:
                        prebuilt_voice_config: PrebuiltVoiceConfig = {
                            "voiceName": value["voice"]
                        }
                        voice_config: VoiceConfig = {
                            "prebuiltVoiceConfig": prebuilt_voice_config
                        }
                        speech_config["voiceConfig"] = voice_config

                    if speech_config:
                        optional_params["speechConfig"] = speech_config

                    # Ensure audio modality is set
                    if "responseModalities" not in optional_params:
                        optional_params["responseModalities"] = ["AUDIO"]
                    elif "AUDIO" not in optional_params["responseModalities"]:
                        optional_params["responseModalities"].append("AUDIO")

        if litellm.vertex_ai_safety_settings is not None:
            optional_params["safety_settings"] = litellm.vertex_ai_safety_settings
        return super().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
            drop_params=drop_params,
        )

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[ContentType]:
        """
        Google AI Studio Gemini does not support image urls in messages.
        """
        for message in messages:
            _message_content = message.get("content")
            if _message_content is not None and isinstance(_message_content, list):
                _parts: List[PartType] = []
                for element in _message_content:
                    if element.get("type") == "image_url":
                        img_element = element
                        _image_url: Optional[str] = None
                        format: Optional[str] = None
                        if isinstance(img_element.get("image_url"), dict):
                            _image_url = img_element["image_url"].get("url")  # type: ignore
                            format = img_element["image_url"].get("format")  # type: ignore
                        else:
                            _image_url = img_element.get("image_url")  # type: ignore
                        if _image_url and "https://" in _image_url:
                            image_obj = convert_to_anthropic_image_obj(
                                _image_url, format=format
                            )
                            img_element["image_url"] = (  # type: ignore
                                convert_generic_image_chunk_to_openai_image_obj(
                                    image_obj
                                )
                            )
        return _gemini_convert_messages_with_history(messages=messages)
