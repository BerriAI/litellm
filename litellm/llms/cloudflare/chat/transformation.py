import json
import time
from typing import AsyncIterator, Iterator, List, Optional, Union

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
        result = completion_response.get("result", {})

        # Reasoning-capable Workers AI models (e.g. @cf/google/gemma-4-26b-a4b-it)
        # return an OpenAI-style choices block and leave the flat "response" field
        # empty, putting chain-of-thought in choices[0].message.reasoning. Read the
        # choices block first, then fall back to legacy "response"/"response_text"
        # (newer models like Nemotron use "response_text").
        message = {}
        choices = result.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}

        content = message.get("content")
        if not content:
            content = result.get("response")
        if not content:
            content = result.get("response_text", "")
        model_response.choices[0].message.content = content  # type: ignore

        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            model_response.choices[0].message.reasoning_content = reasoning  # type: ignore

        model_response.created = int(time.time())
        model_response.model = "cloudflare/" + model

        # Prefer Cloudflare's own usage; estimate any field it omits so we never
        # silently report zero tokens. Reasoning models always return usage; the
        # legacy text shape may not.
        usage_block = result.get("usage") or {}
        prompt_tokens = usage_block.get("prompt_tokens")
        if not prompt_tokens:
            prompt_tokens = litellm.utils.get_token_count(messages=messages, model=model)
        completion_tokens = usage_block.get("completion_tokens")
        if not completion_tokens:
            completion_tokens = len(encoding.encode(content or ""))
        total_tokens = usage_block.get("total_tokens") or (
            prompt_tokens + completion_tokens
        )

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
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
        try:
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
            else:
                # OpenAI-style stream (reasoning + newer models): text and
                # chain-of-thought arrive on choices[0].delta.
                choices = chunk.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    index = int(choices[0].get("index", index))
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    reasoning = delta.get("reasoning") or delta.get(
                        "reasoning_content"
                    )
                    if reasoning:
                        provider_specific_fields = {"reasoning_content": reasoning}
                    finish_reason = choices[0].get("finish_reason") or ""
                    is_finished = bool(finish_reason)

            usage_block = chunk.get("usage")
            if usage_block:
                prompt_tokens = usage_block.get("prompt_tokens", 0)
                completion_tokens = usage_block.get("completion_tokens", 0)
                usage = ChatCompletionUsageBlock(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=usage_block.get("total_tokens")
                    or (prompt_tokens + completion_tokens),
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
