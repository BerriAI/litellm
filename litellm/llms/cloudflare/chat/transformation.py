import json
import time
from typing import AsyncIterator, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _audio_or_image_in_message_content,
)
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    convert_url_to_base64,
)
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import (
    BaseConfig,
    BaseLLMException,
    LiteLLMLoggingObj,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
    ModelResponse,
    Usage,
)


class CloudflareError(BaseLLMException):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.cloudflare.com")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            request=self.request,
            response=self.response,
        )  # Call the base class constructor with the parameters it needs


class CloudflareChatConfig(BaseConfig):
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[dict] = None
    guided_json: Optional[dict] = None
    raw: Optional[bool] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        guided_json: Optional[dict] = None,
        raw: Optional[bool] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

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
        dynamic_api_key = api_key or get_secret_str("CLOUDFLARE_API_KEY")
        if dynamic_api_key is None:
            raise ValueError(
                "Missing CloudflareError API Key - A call is being made to cloudflare but no key is set either in the environment variables or via params"
            )
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "Bearer " + dynamic_api_key,
        }
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            account_id = get_secret_str("CLOUDFLARE_ACCOUNT_ID")
            api_base = (
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
            )
        return api_base + model

    def get_supported_openai_params(self, model: str) -> List[str]:
        # Base parameters supported by all Cloudflare models
        base_params = [
            "stream",
            "max_tokens",
        ]

        # Advanced parameters supported by newer Llama models
        if model in [
            "@cf/meta/llama-4-scout-17b-16e-instruct",
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        ]:
            advanced_params = [
                "temperature",
                "top_p",
                "top_k",
                "seed",
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "tools",
                "tool_choice",
            ]

            # Llama 4 Scout has additional parameters
            if model == "@cf/meta/llama-4-scout-17b-16e-instruct":
                advanced_params.extend(
                    [
                        "response_format",
                        "guided_json",
                        "raw",
                    ]
                )

            return base_params + advanced_params
        else:
            # For older models like llama-2-7b-chat-int8, only support basic parameters
            return base_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                # Apply model-specific validation for advanced Llama models
                if model in [
                    "@cf/meta/llama-4-scout-17b-16e-instruct",
                    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                ]:
                    if param == "temperature":
                        if value < 0 or value > 5:
                            if litellm.drop_params is True or drop_params is True:
                                continue
                            else:
                                raise ValueError(
                                    "Temperature must be between 0 and 5 for Cloudflare models"
                                )
                    elif param == "top_k":
                        if value < 1 or value > 50:
                            if litellm.drop_params is True or drop_params is True:
                                continue
                            else:
                                raise ValueError(
                                    "top_k must be between 1 and 50 for Cloudflare models"
                                )
                    elif param == "top_p":
                        if value < 0 or value > 2:
                            if litellm.drop_params is True or drop_params is True:
                                continue
                            else:
                                raise ValueError(
                                    "top_p must be between 0 and 2 for Cloudflare models"
                                )
                    elif param == "seed":
                        if value < 1 or value > 9999999999:
                            if litellm.drop_params is True or drop_params is True:
                                continue
                            else:
                                raise ValueError(
                                    "seed must be between 1 and 9999999999 for Cloudflare models"
                                )
                    elif param == "repetition_penalty":
                        if value < 0 or value > 2:
                            if litellm.drop_params is True or drop_params is True:
                                continue
                            else:
                                raise ValueError(
                                    "repetition_penalty must be between 0 and 2 for Cloudflare models"
                                )
                    elif param == "frequency_penalty":
                        if value < 0 or value > 2:
                            if litellm.drop_params is True or drop_params is True:
                                continue
                            else:
                                raise ValueError(
                                    "frequency_penalty must be between 0 and 2 for Cloudflare models"
                                )
                    elif param == "presence_penalty":
                        if value < 0 or value > 2:
                            if litellm.drop_params is True or drop_params is True:
                                continue
                            else:
                                raise ValueError(
                                    "presence_penalty must be between 0 and 2 for Cloudflare models"
                                )

                optional_params[param] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        config = litellm.CloudflareChatConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        if any(_audio_or_image_in_message_content(message) for message in messages):
            messages = self._validate_multimodal_messages(messages)

        data = {
            "messages": messages,
            **optional_params,
        }
        return data

    def _validate_multimodal_messages(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        Validate and convert image inputs for Cloudflare Workers AI requirements.

        Cloudflare Workers AI only accepts images in data URI format.
        This method will automatically convert HTTP URLs to data URIs.
        """
        processed_messages: List[AllMessageValues] = []

        for message in messages:
            message_content = message.get("content")
            if isinstance(message_content, list):
                processed_content = []
                for content_item in message_content:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "image_url"
                    ):
                        image_url_dict = content_item.get("image_url")
                        if isinstance(image_url_dict, dict):
                            image_url = image_url_dict.get("url", "")
                            if not image_url.startswith("data:"):
                                image_url = convert_url_to_base64(image_url)
                                image_url_dict["url"] = image_url
                        processed_content.append(content_item)
                    else:
                        processed_content.append(content_item)

                processed_message = message.copy()
                processed_message["content"] = processed_content  # type: ignore
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)

        return processed_messages

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
        encoding: str,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        completion_response = raw_response.json()

        if not completion_response.get("success", True):
            errors = completion_response.get("errors", [])
            error_message = "; ".join(
                [f"Code {err.get('code')}: {err.get('message')}" for err in errors]
            )
            raise CloudflareError(
                status_code=raw_response.status_code,
                message=error_message,
            )

        if (
            "tool_calls" in completion_response["result"]
            and completion_response["result"]["tool_calls"]
        ):
            # Transform raw tool_calls from Cloudflare to proper ChatCompletionMessageToolCall objects
            tool_calls = []
            for tool_call_dict in completion_response["result"]["tool_calls"]:
                from litellm.types.utils import ChatCompletionMessageToolCall, Function

                function = Function(
                    name=tool_call_dict.get("function", {}).get("name"),
                    arguments=tool_call_dict.get("function", {}).get("arguments", ""),
                )

                tool_call = ChatCompletionMessageToolCall(
                    id=tool_call_dict.get("id"),
                    type=tool_call_dict.get("type", "function"),
                    function=function,
                )
                tool_calls.append(tool_call)

            choice = model_response.choices[0]
            if hasattr(choice, "message"):
                choice.message.tool_calls = tool_calls  # type: ignore
                choice.message.content = None  # type: ignore
            elif hasattr(choice, "delta"):
                choice.delta["tool_calls"] = tool_calls  # type: ignore
                choice.delta["content"] = None  # type: ignore
        else:
            choice = model_response.choices[0]
            if hasattr(choice, "message"):
                choice.message.content = completion_response["result"]["response"]  # type: ignore
            elif hasattr(choice, "delta"):
                choice.delta["content"] = completion_response["result"]["response"]  # type: ignore

        if "usage" in completion_response["result"]:
            usage_data = completion_response["result"]["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
        else:
            prompt_tokens = litellm.utils.get_token_count(
                messages=messages, model=model
            )
            content = ""
            choice = model_response.choices[0]
            # Handle both Choices (non-streaming) and StreamingChoices (streaming)
            if hasattr(choice, "message") and hasattr(choice.message, "content"):  # type: ignore
                content = choice.message.content or ""  # type: ignore
            elif hasattr(choice, "delta"):
                content = choice.delta.get("content", "") or ""  # type: ignore
            completion_tokens = len(encoding.encode(content))
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        model_response.created = int(time.time())
        model_response.model = "cloudflare/" + model
        setattr(model_response, "usage", usage)
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CloudflareError(
            status_code=status_code,
            message=error_message,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return CloudflareChatResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class CloudflareChatResponseIterator(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            index = int(chunk.get("index", 0))

            if "response" in chunk:
                text = chunk["response"]

            if "tool_calls" in chunk and chunk["tool_calls"]:
                tool_call = chunk["tool_calls"][0]
                from litellm.types.llms.openai import (
                    ChatCompletionToolCallFunctionChunk,
                )

                tool_use = ChatCompletionToolCallChunk(
                    id=tool_call.get("id"),
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=tool_call.get("function", {}).get("name"),
                        arguments=tool_call.get("function", {}).get("arguments", ""),
                    ),
                    index=index,
                )

            returned_chunk = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
                provider_specific_fields=provider_specific_fields,
            )

            return returned_chunk

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")
