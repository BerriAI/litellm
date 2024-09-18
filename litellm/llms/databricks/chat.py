# What is this?
## Handler file for databricks API https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
import copy
import json
import os
import time
import types
from enum import Enum
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.databricks.exceptions import DatabricksError
from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.types.llms.openai import (
    ChatCompletionDeltaChunk,
    ChatCompletionResponseMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
)
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
        return [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
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


async def make_call(
    client: AsyncHTTPHandler,
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    streaming_decoder: Optional[CustomStreamingDecoder] = None,
):
    response = await client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200:
        raise DatabricksError(status_code=response.status_code, message=response.text)

    if streaming_decoder is not None:
        completion_stream: Any = streaming_decoder.aiter_bytes(
            response.aiter_bytes(chunk_size=1024)
        )
    else:
        completion_stream = ModelResponseIterator(
            streaming_response=response.aiter_lines(), sync_stream=False
        )
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,  # Pass the completion stream for logging
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    streaming_decoder: Optional[CustomStreamingDecoder] = None,
):
    if client is None:
        client = HTTPHandler()  # Create a new client if none provided

    response = client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200:
        raise DatabricksError(status_code=response.status_code, message=response.read())

    if streaming_decoder is not None:
        completion_stream = streaming_decoder.iter_bytes(
            response.iter_bytes(chunk_size=1024)
        )
    else:
        completion_stream = ModelResponseIterator(
            streaming_response=response.iter_lines(), sync_stream=True
        )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class DatabricksChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    # makes headers for API call

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

    async def acompletion_stream_function(
        self,
        model: str,
        messages: list,
        custom_llm_provider: str,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        stream,
        data: dict,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
    ) -> CustomStreamWrapper:

        data["stream"] = True
        streamwrapper = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                api_base=api_base,
                headers=headers,
                data=json.dumps(data),
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                streaming_decoder=streaming_decoder,
            ),
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streamwrapper

    async def acompletion_function(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        custom_llm_provider: str,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        stream,
        data: dict,
        base_model: Optional[str],
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> ModelResponse:
        if timeout is None:
            timeout = httpx.Timeout(timeout=600.0, connect=5.0)

        self.async_handler = AsyncHTTPHandler(timeout=timeout)

        try:
            response = await self.async_handler.post(
                api_base, headers=headers, data=json.dumps(data)
            )
            response.raise_for_status()

            response_json = response.json()
        except httpx.HTTPStatusError as e:
            raise DatabricksError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except httpx.TimeoutException:
            raise DatabricksError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise DatabricksError(status_code=500, message=str(e))

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": data},
        )
        response = ModelResponse(**response_json)

        response.model = custom_llm_provider + "/" + (response.model or "")

        if base_model is not None:
            response._hidden_params["model"] = base_model
        return response

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj,
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
        custom_endpoint = custom_endpoint or optional_params.pop(
            "custom_endpoint", None
        )
        base_model: Optional[str] = optional_params.pop("base_model", None)
        api_base, headers = self._validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="chat_completions",
            custom_endpoint=custom_endpoint,
            headers=headers,
        )
        ## Load Config
        config = litellm.DatabricksConfig().get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        stream: bool = optional_params.get("stream", None) or False
        optional_params["stream"] = stream

        data = {
            "model": model,
            "messages": messages,
            **optional_params,
        }

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )
        if acompletion is True:
            if client is not None and isinstance(client, HTTPHandler):
                client = None
            if (
                stream is not None and stream is True
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
                print_verbose("makes async anthropic streaming POST request")
                data["stream"] = stream
                return self.acompletion_stream_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    client=client,
                    custom_llm_provider=custom_llm_provider,
                    streaming_decoder=streaming_decoder,
                )
            else:
                return self.acompletion_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    custom_llm_provider=custom_llm_provider,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    timeout=timeout,
                    base_model=base_model,
                )
        else:
            if client is None or not isinstance(client, HTTPHandler):
                client = HTTPHandler(timeout=timeout)  # type: ignore
            ## COMPLETION CALL
            if stream is True:
                return CustomStreamWrapper(
                    completion_stream=None,
                    make_call=partial(
                        make_sync_call,
                        client=None,
                        api_base=api_base,
                        headers=headers,  # type: ignore
                        data=json.dumps(data),
                        model=model,
                        messages=messages,
                        logging_obj=logging_obj,
                        streaming_decoder=streaming_decoder,
                    ),
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    logging_obj=logging_obj,
                )
            else:
                try:
                    response = client.post(
                        api_base, headers=headers, data=json.dumps(data)
                    )
                    response.raise_for_status()

                    response_json = response.json()
                except httpx.HTTPStatusError as e:
                    raise DatabricksError(
                        status_code=e.response.status_code, message=e.response.text
                    )
                except httpx.TimeoutException as e:
                    raise DatabricksError(
                        status_code=408, message="Timeout error occurred."
                    )
                except Exception as e:
                    raise DatabricksError(status_code=500, message=str(e))

        response = ModelResponse(**response_json)

        response.model = custom_llm_provider + "/" + (response.model or "")

        if base_model is not None:
            response._hidden_params["model"] = base_model

        return response

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
                message=e.response.text,
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
