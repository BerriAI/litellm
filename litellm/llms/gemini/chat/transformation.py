from typing import List, Optional, cast

from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_generic_image_chunk_to_openai_image_obj,
    convert_to_anthropic_image_obj,
)
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    convert_url_to_base64,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionFileObject
from litellm.types.llms.vertex_ai import ContentType, PartType
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
            "presence_penalty",
            "modalities",
            "parallel_tool_calls",
            "web_search_options",
        ]
        if supports_reasoning(model, custom_llm_provider="gemini"):
            supported_params.append("reasoning_effort")
            supported_params.append("thinking")
        if self.is_model_gemini_audio_model(model):
            supported_params.append("audio")
        return supported_params

    def _transform_messages(
        self, messages: List[AllMessageValues], model: Optional[str] = None
    ) -> List[ContentType]:
        """
        Google AI Studio Gemini does not support HTTP/HTTPS URLs for files.
        Convert them to base64 data instead.
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
                        detail: Optional[str] = None
                        if isinstance(img_element.get("image_url"), dict):
                            _image_url = img_element["image_url"].get("url")  # type: ignore
                            format = img_element["image_url"].get("format")  # type: ignore
                            detail = img_element["image_url"].get("detail")  # type: ignore
                        else:
                            _image_url = img_element.get("image_url")  # type: ignore
                        if _image_url and "https://" in _image_url:
                            image_obj = convert_to_anthropic_image_obj(
                                _image_url, format=format
                            )
                            converted_image_url = convert_generic_image_chunk_to_openai_image_obj(
                                image_obj
                            )
                            if detail is not None:
                                img_element["image_url"] = {  # type: ignore
                                    "url": converted_image_url,
                                    "detail": detail
                                }
                            else:
                                img_element["image_url"] = converted_image_url  # type: ignore
                    elif element.get("type") == "file":
                        file_element = cast(ChatCompletionFileObject, element)
                        file_id = file_element["file"].get("file_id")
                        if file_id and ("http://" in file_id or "https://" in file_id):
                            # Convert HTTP/HTTPS file URL to base64 data
                            try:
                                base64_data = convert_url_to_base64(file_id)
                                file_element["file"]["file_data"] = base64_data  # type: ignore
                                file_element["file"].pop("file_id", None)  # type: ignore
                            except Exception:
                                # If conversion fails, leave as is and let the API handle it
                                pass
        return _gemini_convert_messages_with_history(messages=messages, model=model)
