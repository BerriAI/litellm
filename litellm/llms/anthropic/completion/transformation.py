"""
Translation logic for anthropic's `/v1/complete` endpoint

Litellm provider slug: `anthropic_text/<model_name>`
"""

import json
import os
import time
import types
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

import httpx
import requests

import litellm
from litellm.llms.base_llm.transformation import (
    BaseConfig,
    BaseLLMException,
    LiteLLMLoggingObj,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.prompt_templates.factory import custom_prompt, prompt_factory
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage


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
        return litellm.AnthropicConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

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

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        "Not required"
        raise NotImplementedError
