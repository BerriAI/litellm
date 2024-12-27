"""
OpenAI-like chat completion handler

For handling OpenAI-like chat completions, like IBM WatsonX, etc.
"""

import json
from typing import Any, Callable, Optional, Union

import httpx

import litellm
from litellm import LlmProviders
from litellm.llms.bedrock.chat.invoke_handler import MockResponseIterator
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.utils import CustomStreamingDecoder, ModelResponse
from litellm.utils import CustomStreamWrapper, ProviderConfigManager

from ..common_utils import OpenAILikeBase, OpenAILikeError
from .transformation import OpenAILikeChatConfig


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    streaming_decoder: Optional[CustomStreamingDecoder] = None,
    fake_stream: bool = False,
):
    if client is None:
        client = litellm.module_level_aclient

    response = await client.post(
        api_base, headers=headers, data=data, stream=not fake_stream
    )

    if streaming_decoder is not None:
        completion_stream: Any = streaming_decoder.aiter_bytes(
            response.aiter_bytes(chunk_size=1024)
        )
    elif fake_stream:
        model_response = ModelResponse(**response.json())
        completion_stream = MockResponseIterator(model_response=model_response)
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
    fake_stream: bool = False,
):
    if client is None:
        client = litellm.module_level_client  # Create a new client if none provided

    response = client.post(api_base, headers=headers, data=data, stream=not fake_stream)

    if response.status_code != 200:
        raise OpenAILikeError(status_code=response.status_code, message=response.read())

    if streaming_decoder is not None:
        completion_stream = streaming_decoder.iter_bytes(
            response.iter_bytes(chunk_size=1024)
        )
    elif fake_stream:
        model_response = ModelResponse(**response.json())
        completion_stream = MockResponseIterator(model_response=model_response)
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


class OpenAILikeChatHandler(OpenAILikeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        fake_stream: bool = False,
    ) -> CustomStreamWrapper:
        data["stream"] = True
        completion_stream = await make_call(
            client=client,
            api_base=api_base,
            headers=headers,
            data=json.dumps(data),
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            streaming_decoder=streaming_decoder,
        )
        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
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
        client: Optional[AsyncHTTPHandler],
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
        json_mode: bool = False,
    ) -> ModelResponse:
        if timeout is None:
            timeout = httpx.Timeout(timeout=600.0, connect=5.0)

        if client is None:
            client = litellm.module_level_aclient

        try:
            response = await client.post(
                api_base, headers=headers, data=json.dumps(data), timeout=timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise OpenAILikeError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except httpx.TimeoutException:
            raise OpenAILikeError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise OpenAILikeError(status_code=500, message=str(e))

        return OpenAILikeChatConfig._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            encoding=encoding,
            json_mode=json_mode,
            custom_llm_provider=custom_llm_provider,
            base_model=base_model,
        )

    def completion(
        self,
        *,
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
        fake_stream: bool = False,
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

        stream: bool = optional_params.pop("stream", None) or False
        extra_body = optional_params.pop("extra_body", {})
        json_mode = optional_params.pop("json_mode", None)
        optional_params.pop("max_retries", None)
        if not fake_stream:
            optional_params["stream"] = stream

        if messages is not None and custom_llm_provider is not None:
            provider_config = ProviderConfigManager.get_provider_chat_config(
                model=model, provider=LlmProviders(custom_llm_provider)
            )
            if isinstance(provider_config, OpenAIGPTConfig) or isinstance(
                provider_config, OpenAIConfig
            ):
                messages = provider_config._transform_messages(
                    messages=messages, model=model
                )

        data = {
            "model": model,
            "messages": messages,
            **optional_params,
            **extra_body,
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
            if client is None or not isinstance(client, AsyncHTTPHandler):
                client = None
            if (
                stream is True
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
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
                    fake_stream=fake_stream,
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
                    client=client,
                )
        else:
            ## COMPLETION CALL
            if stream is True:
                completion_stream = make_sync_call(
                    client=(
                        client
                        if client is not None and isinstance(client, HTTPHandler)
                        else None
                    ),
                    api_base=api_base,
                    headers=headers,
                    data=json.dumps(data),
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                    streaming_decoder=streaming_decoder,
                    fake_stream=fake_stream,
                )
                # completion_stream.__iter__()
                return CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    logging_obj=logging_obj,
                )
            else:
                if client is None or not isinstance(client, HTTPHandler):
                    client = HTTPHandler(timeout=timeout)  # type: ignore
                try:
                    response = client.post(
                        url=api_base, headers=headers, data=json.dumps(data)
                    )
                    response.raise_for_status()

                except httpx.HTTPStatusError as e:
                    raise OpenAILikeError(
                        status_code=e.response.status_code,
                        message=e.response.text,
                    )
                except httpx.TimeoutException:
                    raise OpenAILikeError(
                        status_code=408, message="Timeout error occurred."
                    )
                except Exception as e:
                    raise OpenAILikeError(status_code=500, message=str(e))
        return OpenAILikeChatConfig._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            encoding=encoding,
            json_mode=json_mode,
            custom_llm_provider=custom_llm_provider,
            base_model=base_model,
        )
