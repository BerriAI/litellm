"""
Translation logic for anthropic's `/v1/complete` endpoint

Litellm provider slug: `anthropic_text/<model_name>`
"""

import json
import time
from typing import AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    custom_prompt,
    prompt_factory,
)
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import (
    BaseConfig,
    BaseLLMException,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
    ModelResponse,
    Usage,
)


class AnthropicTextError(BaseLLMException):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.anthropic.com/v1/complete"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            message=self.message,
            status_code=self.status_code,
            request=self.request,
            response=self.response,
        )  # Call the base class constructor with the parameters it needs


class AnthropicTextConfig(BaseConfig):
    """
    Reference: https://docs.anthropic.com/claude/reference/complete_post

    to pass metadata to anthropic, it's {"user_id": "any-relevant-information"}
    """

    max_tokens_to_sample: Optional[int] = (
        litellm.max_tokens
    )  # anthropic requires a default
    stop_sequences: Optional[list] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    metadata: Optional[dict] = None

    def __init__(
        self,
        max_tokens_to_sample: Optional[int] = 256,  # anthropic requires a default
        stop_sequences: Optional[list] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    # makes headers for API call
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            raise ValueError(
                "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params"
            )
        _headers = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key,
        }
        headers.update(_headers)
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        prompt = self._get_anthropic_text_prompt_from_messages(
            messages=messages, model=model
        )
        ## Load Config
        config = litellm.AnthropicTextConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        data = {
            "model": model,
            "prompt": prompt,
            **optional_params,
        }

        return data

    def get_supported_openai_params(self, model: str):
        """
        Anthropic /complete API Ref: https://docs.anthropic.com/en/api/complete
        """
        return [
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "temperature",
            "top_p",
            "extra_headers",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Follows the same logic as the AnthropicConfig.map_openai_params method (which is the Anthropic /messages API)

        Note: the only difference is in the get supported openai params method between the AnthropicConfig and AnthropicTextConfig
        API Ref: https://docs.anthropic.com/en/api/complete
        """
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens_to_sample"] = value
            if param == "max_completion_tokens":
                optional_params["max_tokens_to_sample"] = value
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "stop" and (isinstance(value, str) or isinstance(value, list)):
                _value = litellm.AnthropicConfig()._map_stop_sequences(value)
                if _value is not None:
                    optional_params["stop_sequences"] = _value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "user":
                optional_params["metadata"] = {"user_id": value}

        return optional_params

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
        try:
            completion_response = raw_response.json()
        except Exception:
            raise AnthropicTextError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        prompt = self._get_anthropic_text_prompt_from_messages(
            messages=messages, model=model
        )
        if "error" in completion_response:
            raise AnthropicTextError(
                message=str(completion_response["error"]),
                status_code=raw_response.status_code,
            )
        else:
            if len(completion_response["completion"]) > 0:
                model_response.choices[0].message.content = completion_response[  # type: ignore
                    "completion"
                ]
            model_response.choices[0].finish_reason = completion_response["stop_reason"]

        ## CALCULATING USAGE
        prompt_tokens = len(
            encoding.encode(prompt)
        )  ##[TODO] use the anthropic tokenizer here
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )  ##[TODO] use the anthropic tokenizer here

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        setattr(model_response, "usage", usage)
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, httpx.Headers]
    ) -> BaseLLMException:
        return AnthropicTextError(
            status_code=status_code,
            message=error_message,
        )

    @staticmethod
    def _is_anthropic_text_model(model: str) -> bool:
        return model == "claude-2" or model == "claude-instant-1"

    def _get_anthropic_text_prompt_from_messages(
        self, messages: List[AllMessageValues], model: str
    ) -> str:
        custom_prompt_dict = litellm.custom_prompt_dict
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"],
                initial_prompt_value=model_prompt_details["initial_prompt_value"],
                final_prompt_value=model_prompt_details["final_prompt_value"],
                messages=messages,
            )
        else:
            prompt = prompt_factory(
                model=model, messages=messages, custom_llm_provider="anthropic"
            )

        return str(prompt)

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return AnthropicTextCompletionResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class AnthropicTextCompletionResponseIterator(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None
            index = int(chunk.get("index", 0))
            _chunk_text = chunk.get("completion", None)
            if _chunk_text is not None and isinstance(_chunk_text, str):
                text = _chunk_text
            finish_reason = chunk.get("stop_reason", None)
            if finish_reason is not None:
                is_finished = True
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
