# What is this?
## Handler file for databricks API https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
import copy
import json
import os
import time
import types
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.databricks.client import (
    DatabricksModelServingClientWrapper,
    get_databricks_model_serving_client_wrapper,
)
from litellm.llms.databricks.exceptions import DatabricksError
from litellm.types.utils import (
    CustomStreamingDecoder,
    GenericStreamingChunk,
    ProviderField,
)
from litellm.utils import CustomStreamWrapper, EmbeddingResponse, ModelResponse, Usage

from ..base import BaseLLM
from ..prompt_templates.factory import custom_prompt, prompt_factory


class DatabricksConfig:
    """
    Reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
    """

    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None
    n: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        stop: Optional[Union[List[str], str]] = None,
        n: Optional[int] = None,
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

    def get_required_params(self) -> List[ProviderField]:
        """For a given provider, return it's required fields with a description"""
        return [
            ProviderField(
                field_name="api_key",
                field_type="string",
                field_description="Your Databricks API Key.",
                field_value="dapi...",
            ),
            ProviderField(
                field_name="api_base",
                field_type="string",
                field_description="Your Databricks API Base.",
                field_value="https://adb-..",
            ),
        ]

    def get_supported_openai_params(self):
        return ["stream", "stop", "temperature", "top_p", "max_tokens", "n"]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "n":
                optional_params["n"] = value
            if param == "stream" and value == True:
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stop":
                optional_params["stop"] = value
        return optional_params


class DatabricksEmbeddingConfig:
    """
    Reference: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-models/api-reference#--embedding-task
    """

    instruction: Optional[str] = (
        None  # An optional instruction to pass to the embedding model. BGE Authors recommend 'Represent this sentence for searching relevant passages:' for retrieval queries
    )

    def __init__(self, instruction: Optional[str] = None) -> None:
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

    def get_supported_openai_params(
        self,
    ):  # no optional openai embedding params supported
        return []

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        return optional_params


class DatabricksChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def _validate_environment(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        endpoint_type: Literal["chat_completions", "embeddings"],
        custom_endpoint: Optional[bool],
        headers: Optional[dict],
    ) -> Tuple[str, dict]:
        if api_key is None and headers is None:
            raise DatabricksError(
                status_code=400,
                message="Missing API Key - A call is being made to LLM Provider but no key is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
            )

        if api_base is None:
            raise DatabricksError(
                status_code=400,
                message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
            )

        if headers is None:
            headers = {
                "Authorization": "Bearer {}".format(api_key),
                "Content-Type": "application/json",
            }
        else:
            if api_key is not None:
                headers.update({"Authorization": "Bearer {}".format(api_key)})

        if endpoint_type == "chat_completions" and custom_endpoint is not True:
            api_base = "{}/chat/completions".format(api_base)
        elif endpoint_type == "embeddings" and custom_endpoint is not True:
            api_base = "{}/embeddings".format(api_base)
        return api_base, headers

    def completion(
        self,
        model: str,
        messages: list,
        api_base: Optional[str],
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        custom_endpoint: Optional[bool] = None,
        streaming_decoder: Optional[
            CustomStreamingDecoder
        ] = None,  # if openai-compatible api needs custom stream decoder - e.g. sagemaker
    ):
        def emit_log_event(log_fn: Callable, response: Optional[str] = None):
            response_kwargs = {"original_response": response} if response else {}
            log_fn(
                input=messages,
                api_key=api_key,
                additional_args={
                    "complete_input_dict": {
                        "model": model,
                        "messages": messages,
                        **optional_params,
                    },
                    "api_base": api_base,
                    "headers": headers,
                },
                **response_kwargs,
            )

        custom_endpoint = custom_endpoint or optional_params.pop(
            "custom_endpoint", None
        )
        databricks_client = get_databricks_model_serving_client_wrapper(
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
            support_async=acompletion,
            api_key=api_key,
            api_base=api_base,
            http_handler=client,
            timeout=timeout,
            custom_endpoint=custom_endpoint,
            headers=headers,
            streaming_decoder=streaming_decoder,
        )

        for k, v in litellm.DatabricksConfig().get_config().items():
            optional_params.setdefault(k, v)
        stream: bool = optional_params.get("stream", False)
        optional_params["stream"] = stream

        emit_log_event(log_fn=logging_obj.pre_call)

        def format_response(response: Union[ModelResponse, CustomStreamWrapper]):
            base_model: Optional[str] = optional_params.pop("base_model", None)
            if response.model is not None:
                response.model = custom_llm_provider + "/" + response.model
            if base_model is not None and response.model is not None:
                response._hidden_params["model"] = base_model

            return response

        if acompletion is True:

            async def get_async_completion():
                response = await databricks_client.acompletion(
                    endpoint_name=model,
                    messages=messages,
                    optional_params=optional_params,
                    stream=stream,
                )
                emit_log_event(log_fn=logging_obj.post_call, response=str(response))
                return format_response(response)

            return get_async_completion()
        else:
            response = databricks_client.completion(
                endpoint_name=model,
                messages=messages,
                optional_params=optional_params,
                stream=stream,
            )
            emit_log_event(log_fn=logging_obj.post_call, response=str(response))
            return format_response(response)

    async def aembedding(
        self,
        input: list,
        data: dict,
        model_response: ModelResponse,
        timeout: float,
        api_key: str,
        api_base: str,
        logging_obj,
        headers: dict,
        client=None,
    ) -> EmbeddingResponse:
        response = None
        try:
            if client is None or isinstance(client, AsyncHTTPHandler):
                self.async_client = AsyncHTTPHandler(timeout=timeout)  # type: ignore
            else:
                self.async_client = client

            try:
                response = await self.async_client.post(
                    api_base,
                    headers=headers,
                    data=json.dumps(data),
                )  # type: ignore

                response.raise_for_status()

                response_json = response.json()
            except httpx.HTTPStatusError as e:
                raise DatabricksError(
                    status_code=e.response.status_code,
                    message=response.text if response else str(e),
                )
            except httpx.TimeoutException as e:
                raise DatabricksError(
                    status_code=408, message="Timeout error occurred."
                )
            except Exception as e:
                raise DatabricksError(status_code=500, message=str(e))

            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response_json,
            )
            return EmbeddingResponse(**response_json)
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                original_response=str(e),
            )
            raise e

    def embedding(
        self,
        model: str,
        input: list,
        timeout: float,
        logging_obj,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        model_response: Optional[litellm.utils.EmbeddingResponse] = None,
        client=None,
        aembedding=None,
        headers: Optional[dict] = None,
    ) -> EmbeddingResponse:
        api_base, headers = self._validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="embeddings",
            custom_endpoint=False,
            headers=headers,
        )
        model = model
        data = {"model": model, "input": input, **optional_params}

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data, "api_base": api_base},
        )

        if aembedding == True:
            return self.aembedding(data=data, input=input, logging_obj=logging_obj, model_response=model_response, api_base=api_base, api_key=api_key, timeout=timeout, client=client, headers=headers)  # type: ignore
        if client is None or isinstance(client, AsyncHTTPHandler):
            self.client = HTTPHandler(timeout=timeout)  # type: ignore
        else:
            self.client = client

        ## EMBEDDING CALL
        try:
            response = self.client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
            )  # type: ignore

            response.raise_for_status()  # type: ignore

            response_json = response.json()  # type: ignore
        except httpx.HTTPStatusError as e:
            raise DatabricksError(
                status_code=e.response.status_code,
                message=response.text if response else str(e),
            )
        except httpx.TimeoutException as e:
            raise DatabricksError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise DatabricksError(status_code=500, message=str(e))

        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response_json,
        )

        return litellm.EmbeddingResponse(**response_json)
