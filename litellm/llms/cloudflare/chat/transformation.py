import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segments
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


def _extract_cloudflare_chat_content(result: Dict[str, Any]) -> str:
    # Legacy Cloudflare fields take precedence, and "" is a valid response.
    if "response" in result and result["response"] is not None:
        if isinstance(result["response"], str):
            return result["response"]
        raise CloudflareError(
            status_code=500,
            message=f"Unable to parse Cloudflare chat response. Invalid response field: {result}",
        )

    if "response_text" in result and result["response_text"] is not None:
        if isinstance(result["response_text"], str):
            return result["response_text"]
        raise CloudflareError(
            status_code=500,
            message=f"Unable to parse Cloudflare chat response. Invalid response_text field: {result}",
        )

    if "choices" in result:
        choices = result["choices"]
        if (
            isinstance(choices, list)
            and len(choices) > 0
            and isinstance(choices[0], dict)
        ):
            message = choices[0].get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                content = message["content"]
                if isinstance(content, str):
                    return content

    raise CloudflareError(
        status_code=500,
        message=f"Unable to parse Cloudflare chat response. Response result: {result}",
    )


class CloudflareChatConfig(BaseConfig):
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
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
        if api_key is None:
            raise ValueError(
                "Missing CloudflareError API Key - A call is being made to cloudflare but no key is set either in the environment variables or via params"
            )
        headers = {
            "accept": "application/json",
            "content-type": "apbplication/json",
            "Authorization": "Bearer " + api_key,
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
        encoded_model = encode_url_path_segments(model, field_name="model")
        return api_base + encoded_model

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "stream",
            "max_tokens",
        ]

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

        data = {
            "messages": messages,
            **optional_params,
        }
        return data

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

        result = completion_response["result"]
        model_response.choices[0].message.content = _extract_cloudflare_chat_content(  # type: ignore
            result=result
        )

        prompt_tokens = litellm.utils.get_token_count(messages=messages, model=model)
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )

        model_response.created = int(time.time())
        model_response.model = "cloudflare/" + model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
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
        text = ""
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        is_finished = False
        finish_reason = ""
        usage: Optional[ChatCompletionUsageBlock] = None
        provider_specific_fields = None

        index = int(chunk.get("index", 0))

        if "response" in chunk and chunk["response"] is not None:
            text = chunk["response"]
        elif "response_text" in chunk and chunk["response_text"] is not None:
            text = chunk["response_text"]
        elif "choices" in chunk and isinstance(chunk["choices"], list):
            choices = chunk["choices"]
            if len(choices) > 0 and isinstance(choices[0], dict):
                choice = choices[0]
                index = int(choice.get("index", index))
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("content") is not None:
                    text = delta["content"]
                if choice.get("finish_reason") is not None:
                    is_finished = True
                    finish_reason = choice["finish_reason"]

        if not is_finished and chunk.get("finish_reason") is not None:
            is_finished = True
            finish_reason = chunk["finish_reason"]

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
