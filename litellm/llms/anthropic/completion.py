"""
Translation logic for anthropic's `/v1/complete` endpoint
"""

import json
import os
import time
import types
from enum import Enum
from typing import Callable, Optional

import httpx
import requests

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from ..base import BaseLLM
from ..prompt_templates.factory import custom_prompt, prompt_factory


class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "


class AnthropicError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.anthropic.com/v1/complete"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class AnthropicTextConfig:
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

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }


# makes headers for API call
def validate_environment(api_key, user_headers):
    if api_key is None:
        raise ValueError(
            "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params"
        )
    headers = {
        "accept": "application/json",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    if user_headers is not None and isinstance(user_headers, dict):
        headers = {**headers, **user_headers}
    return headers


class AnthropicTextCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def _process_response(
        self, model_response: ModelResponse, response, encoding, prompt: str, model: str
    ):
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except Exception:
            raise AnthropicError(
                message=response.text, status_code=response.status_code
            )
        if "error" in completion_response:
            raise AnthropicError(
                message=str(completion_response["error"]),
                status_code=response.status_code,
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

    async def async_completion(
        self,
        model: str,
        model_response: ModelResponse,
        api_base: str,
        logging_obj,
        encoding,
        headers: dict,
        data: dict,
        client=None,
    ):
        if client is None:
            client = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

        response = await client.post(api_base, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise AnthropicError(
                status_code=response.status_code, message=response.text
            )

        ## LOGGING
        logging_obj.post_call(
            input=data["prompt"],
            api_key=headers.get("x-api-key"),
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )

        response = self._process_response(
            model_response=model_response,
            response=response,
            encoding=encoding,
            prompt=data["prompt"],
            model=model,
        )
        return response

    async def async_streaming(
        self,
        model: str,
        api_base: str,
        logging_obj,
        headers: dict,
        data: Optional[dict],
        client=None,
    ):
        if client is None:
            client = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

        response = await client.post(api_base, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise AnthropicError(
                status_code=response.status_code, message=response.text
            )

        completion_stream = response.aiter_lines()

        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="anthropic_text",
            logging_obj=logging_obj,
        )
        return streamwrapper

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        acompletion: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client=None,
    ):
        headers = validate_environment(api_key, headers)
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

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        ## COMPLETION CALL
        if "stream" in optional_params and optional_params["stream"] is True:
            if acompletion is True:
                return self.async_streaming(
                    model=model,
                    api_base=api_base,
                    logging_obj=logging_obj,
                    headers=headers,
                    data=data,
                    client=None,
                )

            if client is None:
                client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

            response = client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
                # stream=optional_params["stream"],
            )

            if response.status_code != 200:
                raise AnthropicError(
                    status_code=response.status_code, message=response.text
                )
            completion_stream = response.iter_lines()
            stream_response = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="anthropic_text",
                logging_obj=logging_obj,
            )
            return stream_response
        elif acompletion is True:
            return self.async_completion(
                model=model,
                model_response=model_response,
                api_base=api_base,
                logging_obj=logging_obj,
                encoding=encoding,
                headers=headers,
                data=data,
                client=client,
            )
        else:
            if client is None:
                client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))
            response = client.post(api_base, headers=headers, data=json.dumps(data))
            if response.status_code != 200:
                raise AnthropicError(
                    status_code=response.status_code, message=response.text
                )

            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )
            print_verbose(f"raw model_response: {response.text}")

            response = self._process_response(
                model_response=model_response,
                response=response,
                encoding=encoding,
                prompt=data["prompt"],
                model=model,
            )
            return response

    def embedding(self):
        # logic for parsing in - calling - parsing out model embedding calls
        pass
