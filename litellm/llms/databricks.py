# What is this?
## Handler file for databricks API https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
from functools import partial
import os, types
import json
from enum import Enum
import requests, copy  # type: ignore
import time
from typing import Callable, Optional, List, Union, Tuple, Literal
from litellm.utils import (
    ModelResponse,
    Usage,
    CustomStreamWrapper,
    EmbeddingResponse,
)
from litellm.litellm_core_utils.core_helpers import map_finish_reason
import litellm
from .prompt_templates.factory import prompt_factory, custom_prompt
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from .base import BaseLLM
import httpx  # type: ignore
from litellm.types.llms.databricks import GenericStreamingChunk
from litellm.types.utils import ProviderField


class DatabricksError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://docs.databricks.com/")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


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

    def _chunk_parser(self, chunk_data: str) -> GenericStreamingChunk:
        try:
            text = ""
            is_finished = False
            finish_reason = None
            logprobs = None
            usage = None
            original_chunk = None  # this is used for function/tool calling
            chunk_data = chunk_data.replace("data:", "")
            chunk_data = chunk_data.strip()
            if len(chunk_data) == 0 or chunk_data == "[DONE]":
                return {
                    "text": "",
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            chunk_data_dict = json.loads(chunk_data)
            str_line = litellm.ModelResponse(**chunk_data_dict, stream=True)

            if len(str_line.choices) > 0:
                if (
                    str_line.choices[0].delta is not None  # type: ignore
                    and str_line.choices[0].delta.content is not None  # type: ignore
                ):
                    text = str_line.choices[0].delta.content  # type: ignore
                else:  # function/tool calling chunk - when content is None. in this case we just return the original chunk from openai
                    original_chunk = str_line
                if str_line.choices[0].finish_reason:
                    is_finished = True
                    finish_reason = str_line.choices[0].finish_reason
                    if finish_reason == "content_filter":
                        if hasattr(str_line.choices[0], "content_filter_result"):
                            error_message = json.dumps(
                                str_line.choices[0].content_filter_result  # type: ignore
                            )
                        else:
                            error_message = "Azure Response={}".format(
                                str(dict(str_line))
                            )
                        raise litellm.AzureOpenAIError(
                            status_code=400, message=error_message
                        )

                # checking for logprobs
                if (
                    hasattr(str_line.choices[0], "logprobs")
                    and str_line.choices[0].logprobs is not None
                ):
                    logprobs = str_line.choices[0].logprobs
                else:
                    logprobs = None

            usage = getattr(str_line, "usage", None)

            return GenericStreamingChunk(
                text=text,
                is_finished=is_finished,
                finish_reason=finish_reason,
                logprobs=logprobs,
                original_chunk=original_chunk,
                usage=usage,
            )
        except Exception as e:
            raise e


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
):
    response = await client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200:
        raise DatabricksError(status_code=response.status_code, message=response.text)

    completion_stream = response.aiter_lines()
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,  # Pass the completion stream for logging
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
    ) -> Tuple[str, dict]:
        if api_key is None:
            raise DatabricksError(
                status_code=400,
                message="Missing Databricks API Key - A call is being made to Databricks but no key is set either in the environment variables (DATABRICKS_API_KEY) or via params",
            )

        if api_base is None:
            raise DatabricksError(
                status_code=400,
                message="Missing Databricks API Base - A call is being made to Databricks but no api base is set either in the environment variables (DATABRICKS_API_BASE) or via params",
            )

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        }

        if endpoint_type == "chat_completions":
            api_base = "{}/chat/completions".format(api_base)
        elif endpoint_type == "embeddings":
            api_base = "{}/embeddings".format(api_base)
        return api_base, headers

    def process_response(
        self,
        model: str,
        response: Union[requests.Response, httpx.Response],
        model_response: ModelResponse,
        stream: bool,
        logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except:
            raise DatabricksError(
                message=response.text, status_code=response.status_code
            )
        if "error" in completion_response:
            raise DatabricksError(
                message=str(completion_response["error"]),
                status_code=response.status_code,
            )
        else:
            text_content = ""
            tool_calls = []
            for content in completion_response["content"]:
                if content["type"] == "text":
                    text_content += content["text"]
                ## TOOL CALLING
                elif content["type"] == "tool_use":
                    tool_calls.append(
                        {
                            "id": content["id"],
                            "type": "function",
                            "function": {
                                "name": content["name"],
                                "arguments": json.dumps(content["input"]),
                            },
                        }
                    )

            _message = litellm.Message(
                tool_calls=tool_calls,
                content=text_content or None,
            )
            model_response.choices[0].message = _message  # type: ignore
            model_response._hidden_params["original_response"] = completion_response[
                "content"
            ]  # allow user to access raw anthropic tool calling response

            model_response.choices[0].finish_reason = map_finish_reason(
                completion_response["stop_reason"]
            )

        ## CALCULATING USAGE
        prompt_tokens = completion_response["usage"]["input_tokens"]
        completion_tokens = completion_response["usage"]["output_tokens"]
        total_tokens = prompt_tokens + completion_tokens

        model_response["created"] = int(time.time())
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        setattr(model_response, "usage", usage)  # type: ignore
        return model_response

    async def acompletion_stream_function(
        self,
        model: str,
        messages: list,
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
            ),
            model=model,
            custom_llm_provider="databricks",
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
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        stream,
        data: dict,
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
                message=response.text if response else str(e),
            )
        except httpx.TimeoutException as e:
            raise DatabricksError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise DatabricksError(status_code=500, message=str(e))

        return ModelResponse(**response_json)

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ):
        api_base, headers = self._validate_environment(
            api_base=api_base, api_key=api_key, endpoint_type="chat_completions"
        )
        ## Load Config
        config = litellm.DatabricksConfig().get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        stream = optional_params.pop("stream", None)

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
        if acompletion == True:
            if client is not None and isinstance(client, HTTPHandler):
                client = None
            if (
                stream is not None and stream == True
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
                )
            else:
                return self.acompletion_function(
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
                    timeout=timeout,
                )
        else:
            if client is None or isinstance(client, AsyncHTTPHandler):
                self.client = HTTPHandler(timeout=timeout)  # type: ignore
            else:
                self.client = client
            ## COMPLETION CALL
            if (
                stream is not None and stream == True
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
                print_verbose("makes dbrx streaming POST request")
                data["stream"] = stream
                try:
                    response = self.client.post(
                        api_base, headers=headers, data=json.dumps(data), stream=stream
                    )
                    response.raise_for_status()
                    completion_stream = response.iter_lines()
                except httpx.HTTPStatusError as e:
                    raise DatabricksError(
                        status_code=e.response.status_code, message=response.text
                    )
                except httpx.TimeoutException as e:
                    raise DatabricksError(
                        status_code=408, message="Timeout error occurred."
                    )
                except Exception as e:
                    raise DatabricksError(status_code=408, message=str(e))

                streaming_response = CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="databricks",
                    logging_obj=logging_obj,
                )
                return streaming_response

            else:
                try:
                    response = self.client.post(
                        api_base, headers=headers, data=json.dumps(data)
                    )
                    response.raise_for_status()

                    response_json = response.json()
                except httpx.HTTPStatusError as e:
                    raise DatabricksError(
                        status_code=e.response.status_code, message=response.text
                    )
                except httpx.TimeoutException as e:
                    raise DatabricksError(
                        status_code=408, message="Timeout error occurred."
                    )
                except Exception as e:
                    raise DatabricksError(status_code=500, message=str(e))

        return ModelResponse(**response_json)

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
    ) -> EmbeddingResponse:
        api_base, headers = self._validate_environment(
            api_base=api_base, api_key=api_key, endpoint_type="embeddings"
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
