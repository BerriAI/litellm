"""
GigaChat Chat Transformation

Transforms OpenAI-format requests to GigaChat format and back.
"""

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from ..authenticator import get_access_token
from ..file_handler import upload_file_sync

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

# GigaChat API endpoint
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"


def is_valid_json(value: str) -> bool:
    """Checks whether the value passed is a valid serialized JSON string"""
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    else:
        return True


class GigaChatError(BaseLLMException):
    """GigaChat API error."""

    pass


class GigaChatConfig(BaseConfig):
    """
    Configuration class for GigaChat API.

    GigaChat is Sber's (Russia's largest bank) LLM API.

    Supported parameters:
        temperature: Sampling temperature (0-2, default 0.87)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty factor
        profanity_check: Enable content filtering
        stream: Enable streaming
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    profanity_check: Optional[bool] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        profanity_check: Optional[bool] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        # Instance variables for current request context
        self._current_credentials: Optional[str] = None
        self._current_api_base: Optional[str] = None

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """Get complete API URL for chat completions."""
        base = api_base or get_secret_str("GIGACHAT_API_BASE") or GIGACHAT_BASE_URL
        return f"{base}/chat/completions"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Set up headers with OAuth token.
        """
        # Get access token
        credentials = (
            api_key
            or get_secret_str("GIGACHAT_CREDENTIALS")
            or get_secret_str("GIGACHAT_API_KEY")
        )
        access_token = get_access_token(credentials=credentials)

        # Store credentials for image uploads
        self._current_credentials = credentials
        self._current_api_base = api_base

        headers["Authorization"] = f"Bearer {access_token}"
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"

        return headers

    def get_supported_openai_params(self, model: str) -> List[str]:
        """Return list of supported OpenAI parameters."""
        return [
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "response_format",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """Map OpenAI parameters to GigaChat parameters."""
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            elif param == "temperature":
                # GigaChat: temperature 0 means use top_p=0 instead
                if value == 0:
                    optional_params["top_p"] = 0
                else:
                    optional_params["temperature"] = value
            elif param == "top_p":
                optional_params["top_p"] = value
            elif param in ("max_tokens", "max_completion_tokens"):
                optional_params["max_tokens"] = value
            elif param == "stop":
                # GigaChat doesn't support stop sequences
                pass
            elif param == "tools":
                # Convert tools to functions format
                optional_params["functions"] = self._convert_tools_to_functions(value)
            elif param == "tool_choice":
                # Map OpenAI tool_choice to GigaChat function_call
                mapped_choice = self._map_tool_choice(value)
                if mapped_choice is not None:
                    optional_params["function_call"] = mapped_choice
            elif param == "functions":
                optional_params["functions"] = value
            elif param == "function_call":
                optional_params["function_call"] = value
            elif param == "response_format":
                # Handle structured output via function calling
                if value.get("type") == "json_schema":
                    json_schema = value.get("json_schema", {})
                    schema_name = json_schema.get("name", "structured_output")
                    schema = json_schema.get("schema", {})

                    function_def = {
                        "name": schema_name,
                        "description": f"Output structured response: {schema_name}",
                        "parameters": schema,
                    }

                    if "functions" not in optional_params:
                        optional_params["functions"] = []
                    optional_params["functions"].append(function_def)
                    optional_params["function_call"] = {"name": schema_name}
                    optional_params["_structured_output"] = True

        return optional_params

    def _convert_tools_to_functions(self, tools: List[dict]) -> List[dict]:
        """Convert OpenAI tools format to GigaChat functions format."""
        functions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                functions.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )
        return functions

    def _map_tool_choice(
        self, tool_choice: Union[str, dict]
    ) -> Optional[Union[str, dict]]:
        """
        Map OpenAI tool_choice to GigaChat function_call format.

        OpenAI format:
        - "auto": Call zero, one, or multiple functions (default)
        - "required": Call one or more functions
        - "none": Don't call any functions
        - {"type": "function", "function": {"name": "get_weather"}}: Force specific function

        GigaChat format:
        - "none": Disable function calls
        - "auto": Automatic mode (default)
        - {"name": "get_weather"}: Force specific function

        Args:
            tool_choice: OpenAI tool_choice value

        Returns:
            GigaChat function_call value or None
        """
        if tool_choice == "none":
            return "none"
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            # GigaChat doesn't have a direct "required" equivalent
            # Use "auto" as the closest behavior
            return "auto"
        elif isinstance(tool_choice, dict):
            # OpenAI format: {"type": "function", "function": {"name": "func_name"}}
            # GigaChat format: {"name": "func_name"}
            if tool_choice.get("type") == "function":
                func_name = tool_choice.get("function", {}).get("name")
                if func_name:
                    return {"name": func_name}
        
        # Default to None (don't set function_call)
        return None

    def _upload_image(self, image_url: str) -> Optional[str]:
        """
        Upload image to GigaChat and return file_id.

        Args:
            image_url: URL or base64 data URL of the image

        Returns:
            file_id string or None if upload failed
        """
        try:
            return upload_file_sync(
                image_url=image_url,
                credentials=self._current_credentials,
                api_base=self._current_api_base,
            )
        except Exception as e:
            verbose_logger.error(f"Failed to upload image: {e}")
            return None

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Transform OpenAI request to GigaChat format."""
        # Transform messages
        giga_messages = self._transform_messages(messages)

        # Build request
        request_data = {
            "model": model.replace("gigachat/", ""),
            "messages": giga_messages,
        }

        # Add optional params
        for key in [
            "temperature",
            "top_p",
            "max_tokens",
            "stream",
            "repetition_penalty",
            "profanity_check",
        ]:
            if key in optional_params:
                request_data[key] = optional_params[key]

        # Add functions if present
        if "functions" in optional_params:
            request_data["functions"] = optional_params["functions"]
        if "function_call" in optional_params:
            request_data["function_call"] = optional_params["function_call"]

        return request_data

    def _transform_messages(self, messages: List[AllMessageValues]) -> List[dict]:
        """Transform OpenAI messages to GigaChat format."""
        transformed = []

        for i, msg in enumerate(messages):
            message = dict(msg)

            # Remove unsupported fields
            message.pop("name", None)

            # Transform roles
            role = message.get("role", "user")
            if role == "developer":
                message["role"] = "system"
            elif role == "system" and i > 0:
                # GigaChat only allows system message as first message
                message["role"] = "user"
            elif role == "tool":
                message["role"] = "function"
                content = message.get("content", "")
                if not isinstance(content, str) or not is_valid_json(content):
                    message["content"] = json.dumps(content, ensure_ascii=False)

            # Handle None content
            if message.get("content") is None:
                message["content"] = ""

            # Handle list content (multimodal) - extract text and images
            content = message.get("content")
            if isinstance(content, list):
                texts = []
                attachments = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            texts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            # Extract image URL and upload to GigaChat
                            image_url = part.get("image_url", {})
                            if isinstance(image_url, str):
                                url = image_url
                            else:
                                url = image_url.get("url", "")
                            if url:
                                file_id = self._upload_image(url)
                                if file_id:
                                    attachments.append(file_id)
                message["content"] = "\n".join(texts) if texts else ""
                if attachments:
                    message["attachments"] = attachments

            # Transform tool_calls to function_call
            tool_calls = message.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                func = tool_call.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                message["function_call"] = {
                    "name": func.get("name", ""),
                    "arguments": args,
                }
                message.pop("tool_calls", None)

            transformed.append(message)

        return transformed

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """Transform GigaChat response to OpenAI format."""
        try:
            response_json = raw_response.json()
        except Exception:
            raise GigaChatError(
                status_code=raw_response.status_code,
                message=f"Invalid JSON response: {raw_response.text}",
            )

        is_structured_output = optional_params.get("_structured_output", False)

        choices = []
        for choice in response_json.get("choices", []):
            message_data = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Transform function_call to tool_calls or content
            if finish_reason == "function_call" and message_data.get("function_call"):
                func_call = message_data["function_call"]
                args = func_call.get("arguments", {})

                if is_structured_output:
                    # Convert to content for structured output
                    if isinstance(args, dict):
                        content = json.dumps(args, ensure_ascii=False)
                    else:
                        content = str(args)
                    message_data["content"] = content
                    message_data.pop("function_call", None)
                    message_data.pop("functions_state_id", None)
                    finish_reason = "stop"
                else:
                    # Convert to tool_calls format
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    message_data["tool_calls"] = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": func_call.get("name", ""),
                                "arguments": args,
                            },
                        }
                    ]
                    message_data.pop("function_call", None)
                    finish_reason = "tool_calls"

            # Clean up GigaChat-specific fields
            message_data.pop("functions_state_id", None)

            choices.append(
                Choices(
                    index=choice.get("index", 0),
                    message=Message(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content"),
                        tool_calls=message_data.get("tool_calls"),
                    ),
                    finish_reason=finish_reason,
                )
            )

        # Build usage
        usage_data = response_json.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        model_response.id = response_json.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
        model_response.created = response_json.get("created", int(time.time()))
        model_response.model = model
        model_response.choices = choices  # type: ignore
        setattr(model_response, "usage", usage)

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        """Return GigaChat error class."""
        return GigaChatError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        """Return streaming response iterator."""
        from .streaming import GigaChatModelResponseIterator

        return GigaChatModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
