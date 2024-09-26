# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you ! We ❤️ you! - Krrish & Ishaan

import asyncio
import contextvars
import datetime
import inspect
import json
import os
import random
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Type, Union

import dotenv
import httpx
import openai
import tiktoken
from pydantic import BaseModel
from typing_extensions import overload

import litellm
from litellm import (  # type: ignore
    Logging,
    client,
    exception_type,
    get_litellm_params,
    get_optional_params,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.secret_managers.main import get_secret_str
from litellm.utils import (
    CustomStreamWrapper,
    Usage,
    async_mock_completion_streaming_obj,
    completion_with_fallbacks,
    convert_to_model_response_object,
    create_pretrained_tokenizer,
    create_tokenizer,
    get_api_key,
    get_llm_provider,
    get_optional_params_embeddings,
    get_optional_params_image_gen,
    get_optional_params_transcription,
    get_secret,
    mock_completion_streaming_obj,
    read_config_args,
    supports_httpx_timeout,
    token_counter,
)

from ._logging import verbose_logger
from .caching import disable_cache, enable_cache, update_cache
from .llms import (
    aleph_alpha,
    baseten,
    clarifai,
    cloudflare,
    maritalk,
    nlp_cloud,
    ollama,
    ollama_chat,
    oobabooga,
    openrouter,
    palm,
    petals,
    replicate,
    vllm,
)
from .llms.AI21 import completion as ai21
from .llms.anthropic.chat import AnthropicChatCompletion
from .llms.anthropic.completion import AnthropicTextCompletion
from .llms.azure_ai.chat.handler import AzureAIChatCompletion
from .llms.azure_text import AzureTextCompletion
from .llms.AzureOpenAI.audio_transcriptions import AzureAudioTranscription
from .llms.AzureOpenAI.azure import AzureChatCompletion, _check_dynamic_azure_params
from .llms.bedrock import image_generation as bedrock_image_generation  # type: ignore
from .llms.bedrock.chat import BedrockConverseLLM, BedrockLLM
from .llms.bedrock.embed.embedding import BedrockEmbedding
from .llms.cohere import chat as cohere_chat
from .llms.cohere import completion as cohere_completion  # type: ignore
from .llms.cohere import embed as cohere_embed
from .llms.custom_llm import CustomLLM, custom_chat_llm_router
from .llms.databricks.chat import DatabricksChatCompletion
from .llms.huggingface_restapi import Huggingface
from .llms.OpenAI.audio_transcriptions import OpenAIAudioTranscription
from .llms.OpenAI.chat.o1_handler import OpenAIO1ChatCompletion
from .llms.OpenAI.openai import OpenAIChatCompletion, OpenAITextCompletion
from .llms.predibase import PredibaseChatCompletion
from .llms.prompt_templates.factory import (
    custom_prompt,
    function_call_prompt,
    map_system_message_pt,
    prompt_factory,
    stringify_json_tool_call_content,
)
from .llms.sagemaker.sagemaker import SagemakerLLM
from .llms.text_completion_codestral import CodestralTextCompletion
from .llms.triton import TritonChatCompletion
from .llms.vertex_ai_and_google_ai_studio import (
    vertex_ai_anthropic,
    vertex_ai_non_gemini,
)
from .llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
    VertexLLM,
)
from .llms.vertex_ai_and_google_ai_studio.gemini_embeddings.batch_embed_content_handler import (
    GoogleBatchEmbeddings,
)
from .llms.vertex_ai_and_google_ai_studio.image_generation.image_generation_handler import (
    VertexImageGeneration,
)
from .llms.vertex_ai_and_google_ai_studio.multimodal_embeddings.embedding_handler import (
    VertexMultimodalEmbedding,
)
from .llms.vertex_ai_and_google_ai_studio.text_to_speech.text_to_speech_handler import (
    VertexTextToSpeechAPI,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_ai_partner_models.main import (
    VertexAIPartnerModels,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_embeddings import (
    embedding_handler as vertex_ai_embedding_handler,
)
from .llms.watsonx import IBMWatsonXAI
from .types.llms.openai import HttpxBinaryResponseContent
from .types.utils import (
    AdapterCompletionStreamWrapper,
    ChatCompletionMessageToolCall,
    FileTypes,
    HiddenParams,
    all_litellm_params,
)

encoding = tiktoken.get_encoding("cl100k_base")
from litellm.utils import (
    Choices,
    CustomStreamWrapper,
    EmbeddingResponse,
    ImageResponse,
    Message,
    ModelResponse,
    TextChoices,
    TextCompletionResponse,
    TextCompletionStreamWrapper,
    TranscriptionResponse,
    get_secret,
    read_config_args,
)

####### ENVIRONMENT VARIABLES ###################
openai_chat_completions = OpenAIChatCompletion()
openai_text_completions = OpenAITextCompletion()
openai_o1_chat_completions = OpenAIO1ChatCompletion()
openai_audio_transcriptions = OpenAIAudioTranscription()
databricks_chat_completions = DatabricksChatCompletion()
azure_ai_chat_completions = AzureAIChatCompletion()
anthropic_chat_completions = AnthropicChatCompletion()
anthropic_text_completions = AnthropicTextCompletion()
azure_chat_completions = AzureChatCompletion()
azure_text_completions = AzureTextCompletion()
azure_audio_transcriptions = AzureAudioTranscription()
huggingface = Huggingface()
predibase_chat_completions = PredibaseChatCompletion()
codestral_text_completions = CodestralTextCompletion()
triton_chat_completions = TritonChatCompletion()
bedrock_chat_completion = BedrockLLM()
bedrock_converse_chat_completion = BedrockConverseLLM()
bedrock_embedding = BedrockEmbedding()
vertex_chat_completion = VertexLLM()
vertex_multimodal_embedding = VertexMultimodalEmbedding()
vertex_image_generation = VertexImageGeneration()
google_batch_embeddings = GoogleBatchEmbeddings()
vertex_partner_models_chat_completion = VertexAIPartnerModels()
vertex_text_to_speech = VertexTextToSpeechAPI()
watsonxai = IBMWatsonXAI()
sagemaker_llm = SagemakerLLM()
####### COMPLETION ENDPOINTS ################


class LiteLLM:
    def __init__(
        self,
        *,
        api_key=None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 600,
        max_retries: Optional[int] = litellm.num_retries,
        default_headers: Optional[Mapping[str, str]] = None,
    ):
        self.params = locals()
        self.chat = Chat(self.params, router_obj=None)


class Chat:
    def __init__(self, params, router_obj: Optional[Any]):
        self.params = params
        if self.params.get("acompletion", False) == True:
            self.params.pop("acompletion")
            self.completions: Union[AsyncCompletions, Completions] = AsyncCompletions(
                self.params, router_obj=router_obj
            )
        else:
            self.completions = Completions(self.params, router_obj=router_obj)


class Completions:
    def __init__(self, params, router_obj: Optional[Any]):
        self.params = params
        self.router_obj = router_obj

    def create(self, messages, model=None, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
        model = model or self.params.get("model")
        if self.router_obj is not None:
            response = self.router_obj.completion(
                model=model, messages=messages, **self.params
            )
        else:
            response = completion(model=model, messages=messages, **self.params)
        return response


class AsyncCompletions:
    def __init__(self, params, router_obj: Optional[Any]):
        self.params = params
        self.router_obj = router_obj

    async def create(self, messages, model=None, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
        model = model or self.params.get("model")
        if self.router_obj is not None:
            response = await self.router_obj.acompletion(
                model=model, messages=messages, **self.params
            )
        else:
            response = await acompletion(model=model, messages=messages, **self.params)
        return response


@client
async def acompletion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: List = [],
    functions: Optional[List] = None,
    function_call: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop=None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    # openai v1.0+ new params
    response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[str] = None,
    parallel_tool_calls: Optional[bool] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    deployment_id=None,
    # set api_base, api_version, api_key
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
    extra_headers: Optional[dict] = None,
    # Optional liteLLM function params
    **kwargs,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Asynchronously executes a litellm.completion() call for any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly)

    Parameters:
        model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
        messages (List): A list of message objects representing the conversation context (default is an empty list).

        OPTIONAL PARAMS
        functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
        function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
        temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
        top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
        n (int, optional): The number of completions to generate (default is 1).
        stream (bool, optional): If True, return a streaming response (default is False).
        stream_options (dict, optional): A dictionary containing options for the streaming response. Only use this if stream is True.
        stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
        max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
        max_completion_tokens (integer, optional): An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
        presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
        frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
        logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
        user (str, optional):  A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
        metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc.
        api_base (str, optional): Base URL for the API (default is None).
        api_version (str, optional): API version (default is None).
        api_key (str, optional): API key (default is None).
        model_list (list, optional): List of api base, version, keys
        timeout (float, optional): The maximum execution time in seconds for the completion request.

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Notes:
        - This function is an asynchronous version of the `completion` function.
        - The `completion` function is called using `run_in_executor` to execute synchronously in the event loop.
        - If `stream` is True, the function returns an async generator that yields completion lines.
    """
    loop = asyncio.get_event_loop()
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    # Adjusted to use explicit arguments instead of *args and **kwargs
    completion_kwargs = {
        "model": model,
        "messages": messages,
        "functions": functions,
        "function_call": function_call,
        "timeout": timeout,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stream_options": stream_options,
        "stop": stop,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_completion_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "user": user,
        "response_format": response_format,
        "seed": seed,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "deployment_id": deployment_id,
        "base_url": base_url,
        "api_version": api_version,
        "api_key": api_key,
        "model_list": model_list,
        "extra_headers": extra_headers,
        "acompletion": True,  # assuming this is a required parameter
    }
    if custom_llm_provider is None:
        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=completion_kwargs.get("base_url", None)
        )
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(completion, **completion_kwargs, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        if (
            custom_llm_provider == "openai"
            or custom_llm_provider == "azure"
            or custom_llm_provider == "azure_text"
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "anyscale"
            or custom_llm_provider == "mistral"
            or custom_llm_provider == "openrouter"
            or custom_llm_provider == "deepinfra"
            or custom_llm_provider == "perplexity"
            or custom_llm_provider == "groq"
            or custom_llm_provider == "nvidia_nim"
            or custom_llm_provider == "cerebras"
            or custom_llm_provider == "sambanova"
            or custom_llm_provider == "ai21_chat"
            or custom_llm_provider == "volcengine"
            or custom_llm_provider == "codestral"
            or custom_llm_provider == "text-completion-codestral"
            or custom_llm_provider == "deepseek"
            or custom_llm_provider == "text-completion-openai"
            or custom_llm_provider == "huggingface"
            or custom_llm_provider == "ollama"
            or custom_llm_provider == "ollama_chat"
            or custom_llm_provider == "replicate"
            or custom_llm_provider == "vertex_ai"
            or custom_llm_provider == "vertex_ai_beta"
            or custom_llm_provider == "gemini"
            or custom_llm_provider == "sagemaker"
            or custom_llm_provider == "sagemaker_chat"
            or custom_llm_provider == "anthropic"
            or custom_llm_provider == "predibase"
            or custom_llm_provider == "bedrock"
            or custom_llm_provider == "databricks"
            or custom_llm_provider == "triton"
            or custom_llm_provider == "clarifai"
            or custom_llm_provider == "watsonx"
            or custom_llm_provider in litellm.openai_compatible_providers
            or custom_llm_provider in litellm._custom_providers
        ):  # currently implemented aiohttp calls for just azure, openai, hf, ollama, vertex ai soon all.
            init_response = await loop.run_in_executor(None, func_with_context)
            if isinstance(init_response, dict) or isinstance(
                init_response, ModelResponse
            ):  ## CACHING SCENARIO
                if isinstance(init_response, dict):
                    response = ModelResponse(**init_response)
                response = init_response
            elif asyncio.iscoroutine(init_response):
                response = await init_response
            else:
                response = init_response  # type: ignore

            if (
                custom_llm_provider == "text-completion-openai"
                or custom_llm_provider == "text-completion-codestral"
            ) and isinstance(response, TextCompletionResponse):
                response = litellm.OpenAITextCompletionConfig().convert_to_chat_model_response_object(
                    response_object=response,
                    model_response_object=litellm.ModelResponse(),
                )
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)  # type: ignore
        if isinstance(response, CustomStreamWrapper):
            response.set_logging_event_loop(
                loop=loop
            )  # sets the logging event loop if the user does sync streaming (e.g. on proxy for sagemaker calls)
        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )


async def _async_streaming(response, model, custom_llm_provider, args):
    try:
        print_verbose(f"received response in _async_streaming: {response}")
        if asyncio.iscoroutine(response):
            response = await response
        async for line in response:
            print_verbose(f"line in async streaming: {line}")
            yield line
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
        )


def mock_completion(
    model: str,
    messages: List,
    stream: Optional[bool] = False,
    n: Optional[int] = None,
    mock_response: Union[str, Exception, dict] = "This is a mock request",
    mock_tool_calls: Optional[List] = None,
    logging=None,
    custom_llm_provider=None,
    **kwargs,
):
    """
    Generate a mock completion response for testing or debugging purposes.

    This is a helper function that simulates the response structure of the OpenAI completion API.

    Parameters:
        model (str): The name of the language model for which the mock response is generated.
        messages (List): A list of message objects representing the conversation context.
        stream (bool, optional): If True, returns a mock streaming response (default is False).
        mock_response (str, optional): The content of the mock response (default is "This is a mock request").
        **kwargs: Additional keyword arguments that can be used but are not required.

    Returns:
        litellm.ModelResponse: A ModelResponse simulating a completion response with the specified model, messages, and mock response.

    Raises:
        Exception: If an error occurs during the generation of the mock completion response.

    Note:
        - This function is intended for testing or debugging purposes to generate mock completion responses.
        - If 'stream' is True, it returns a response that mimics the behavior of a streaming completion.
    """
    try:
        ## LOGGING
        if logging is not None:
            logging.pre_call(
                input=messages,
                api_key="mock-key",
            )
        if isinstance(mock_response, Exception):
            if isinstance(mock_response, openai.APIError):
                raise mock_response
            raise litellm.MockException(
                status_code=getattr(mock_response, "status_code", 500),  # type: ignore
                message=getattr(mock_response, "text", str(mock_response)),
                llm_provider=getattr(mock_response, "llm_provider", custom_llm_provider or "openai"),  # type: ignore
                model=model,  # type: ignore
                request=httpx.Request(method="POST", url="https://api.openai.com/v1/"),
            )
        elif (
            isinstance(mock_response, str) and mock_response == "litellm.RateLimitError"
        ):
            raise litellm.RateLimitError(
                message="this is a mock rate limit error",
                llm_provider=getattr(mock_response, "llm_provider", custom_llm_provider or "openai"),  # type: ignore
                model=model,
            )
        elif (
            isinstance(mock_response, str)
            and mock_response == "litellm.InternalServerError"
        ):
            raise litellm.InternalServerError(
                message="this is a mock internal server error",
                llm_provider=getattr(mock_response, "llm_provider", custom_llm_provider or "openai"),  # type: ignore
                model=model,
            )
        elif isinstance(mock_response, str) and mock_response.startswith(
            "Exception: content_filter_policy"
        ):
            raise litellm.MockException(
                status_code=400,
                message=mock_response,
                llm_provider="azure",
                model=model,  # type: ignore
                request=httpx.Request(method="POST", url="https://api.openai.com/v1/"),
            )
        elif isinstance(mock_response, str) and mock_response.startswith(
            "Exception: mock_streaming_error"
        ):
            mock_response = litellm.MockException(
                message="This is a mock error raised mid-stream",
                llm_provider="anthropic",
                model=model,
                status_code=529,
            )
        time_delay = kwargs.get("mock_delay", None)
        if time_delay is not None:
            time.sleep(time_delay)

        if isinstance(mock_response, dict):
            return ModelResponse(**mock_response)

        model_response = ModelResponse(stream=stream)
        if stream is True:
            # don't try to access stream object,
            if kwargs.get("acompletion", False) is True:
                return CustomStreamWrapper(
                    completion_stream=async_mock_completion_streaming_obj(
                        model_response, mock_response=mock_response, model=model, n=n
                    ),
                    model=model,
                    custom_llm_provider="openai",
                    logging_obj=logging,
                )
            return CustomStreamWrapper(
                completion_stream=mock_completion_streaming_obj(
                    model_response, mock_response=mock_response, model=model, n=n
                ),
                model=model,
                custom_llm_provider="openai",
                logging_obj=logging,
            )
        if isinstance(mock_response, litellm.MockException):
            raise mock_response
        if n is None:
            model_response.choices[0].message.content = mock_response  # type: ignore
        else:
            _all_choices = []
            for i in range(n):
                _choice = litellm.utils.Choices(
                    index=i,
                    message=litellm.utils.Message(
                        content=mock_response, role="assistant"
                    ),
                )
                _all_choices.append(_choice)
            model_response.choices = _all_choices  # type: ignore
        model_response.created = int(time.time())
        model_response.model = model

        if mock_tool_calls:
            model_response.choices[0].message.tool_calls = [  # type: ignore
                ChatCompletionMessageToolCall(**tool_call)
                for tool_call in mock_tool_calls
            ]

        setattr(
            model_response,
            "usage",
            Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        try:
            _, custom_llm_provider, _, _ = litellm.utils.get_llm_provider(model=model)
            model_response._hidden_params["custom_llm_provider"] = custom_llm_provider
        except Exception:
            # dont let setting a hidden param block a mock_respose
            pass

        if logging is not None:
            logging.post_call(
                input=messages,
                api_key="my-secret-key",
                original_response="my-original-response",
            )
        return model_response

    except Exception as e:
        if isinstance(e, openai.APIError):
            raise e
        raise Exception("Mock completion response failed")


@client
def completion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: List = [],
    timeout: Optional[Union[float, str, httpx.Timeout]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop=None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    # openai v1.0+ new params
    response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    deployment_id=None,
    extra_headers: Optional[dict] = None,
    # soon to be deprecated params by OpenAI
    functions: Optional[List] = None,
    function_call: Optional[str] = None,
    # set api_base, api_version, api_key
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    **kwargs,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Perform a completion() using any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly)
    Parameters:
        model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
        messages (List): A list of message objects representing the conversation context (default is an empty list).

        OPTIONAL PARAMS
        functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
        function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
        temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
        top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
        n (int, optional): The number of completions to generate (default is 1).
        stream (bool, optional): If True, return a streaming response (default is False).
        stream_options (dict, optional): A dictionary containing options for the streaming response. Only set this when you set stream: true.
        stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
        max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
        max_completion_tokens (integer, optional): An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
        presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
        frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
        logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
        user (str, optional):  A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
        logprobs (bool, optional): Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message
        top_logprobs (int, optional): An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.
        metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc.
        api_base (str, optional): Base URL for the API (default is None).
        api_version (str, optional): API version (default is None).
        api_key (str, optional): API key (default is None).
        model_list (list, optional): List of api base, version, keys
        extra_headers (dict, optional): Additional headers to include in the request.

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
        max_retries (int, optional): The number of retries to attempt (default is 0).
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Note:
        - This function is used to perform completions() using the specified language model.
        - It supports various optional parameters for customizing the completion behavior.
        - If 'mock_response' is provided, a mock completion response is returned for testing or debugging.
    """
    ######### unpacking kwargs #####################
    args = locals()
    api_base = kwargs.get("api_base", None)
    mock_response = kwargs.get("mock_response", None)
    mock_tool_calls = kwargs.get("mock_tool_calls", None)
    force_timeout = kwargs.get("force_timeout", 600)  ## deprecated
    logger_fn = kwargs.get("logger_fn", None)
    verbose = kwargs.get("verbose", False)
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    litellm_logging_obj = kwargs.get("litellm_logging_obj", None)
    id = kwargs.get("id", None)
    metadata = kwargs.get("metadata", None)
    model_info = kwargs.get("model_info", None)
    proxy_server_request = kwargs.get("proxy_server_request", None)
    fallbacks = kwargs.get("fallbacks", None)
    headers = kwargs.get("headers", None) or extra_headers
    num_retries = kwargs.get(
        "num_retries", None
    )  ## alt. param for 'max_retries'. Use this to pass retries w/ instructor.
    max_retries = kwargs.get("max_retries", None)
    cooldown_time = kwargs.get("cooldown_time", None)
    context_window_fallback_dict = kwargs.get("context_window_fallback_dict", None)
    organization = kwargs.get("organization", None)
    ### CUSTOM MODEL COST ###
    input_cost_per_token = kwargs.get("input_cost_per_token", None)
    output_cost_per_token = kwargs.get("output_cost_per_token", None)
    input_cost_per_second = kwargs.get("input_cost_per_second", None)
    output_cost_per_second = kwargs.get("output_cost_per_second", None)
    ### CUSTOM PROMPT TEMPLATE ###
    initial_prompt_value = kwargs.get("initial_prompt_value", None)
    roles = kwargs.get("roles", None)
    final_prompt_value = kwargs.get("final_prompt_value", None)
    bos_token = kwargs.get("bos_token", None)
    eos_token = kwargs.get("eos_token", None)
    preset_cache_key = kwargs.get("preset_cache_key", None)
    hf_model_name = kwargs.get("hf_model_name", None)
    supports_system_message = kwargs.get("supports_system_message", None)
    base_model = kwargs.get("base_model", None)
    ### TEXT COMPLETION CALLS ###
    text_completion = kwargs.get("text_completion", False)
    atext_completion = kwargs.get("atext_completion", False)
    ### ASYNC CALLS ###
    acompletion = kwargs.get("acompletion", False)
    client = kwargs.get("client", None)
    ### Admin Controls ###
    no_log = kwargs.get("no-log", False)
    ### COPY MESSAGES ### - related issue https://github.com/BerriAI/litellm/discussions/4489
    messages = deepcopy(messages)
    ######## end of unpacking kwargs ###########
    openai_params = [
        "functions",
        "function_call",
        "temperature",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stream_options",
        "stop",
        "max_completion_tokens",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "request_timeout",
        "api_base",
        "api_version",
        "api_key",
        "deployment_id",
        "organization",
        "base_url",
        "default_headers",
        "timeout",
        "response_format",
        "seed",
        "tools",
        "tool_choice",
        "max_retries",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "extra_headers",
    ]

    default_params = openai_params + all_litellm_params
    litellm_params = {}  # used to prevent unbound var errors
    non_default_params = {
        k: v for k, v in kwargs.items() if k not in default_params
    }  # model-specific params - pass them straight to the model/provider

    try:
        if base_url is not None:
            api_base = base_url
        if num_retries is not None:
            max_retries = num_retries
        logging = litellm_logging_obj
        fallbacks = fallbacks or litellm.model_fallbacks
        if fallbacks is not None:
            return completion_with_fallbacks(**args)
        if model_list is not None:
            deployments = [
                m["litellm_params"] for m in model_list if m["model_name"] == model
            ]
            return batch_completion_models(deployments=deployments, **args)
        if litellm.model_alias_map and model in litellm.model_alias_map:
            model = litellm.model_alias_map[
                model
            ]  # update the model to the actual value if an alias has been passed in
        model_response = ModelResponse()
        setattr(model_response, "usage", litellm.Usage())
        if (
            kwargs.get("azure", False) == True
        ):  # don't remove flag check, to remain backwards compatible for repos like Codium
            custom_llm_provider = "azure"
        if deployment_id != None:  # azure llms
            model = deployment_id
            custom_llm_provider = "azure"
        model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
        )
        if model_response is not None and hasattr(model_response, "_hidden_params"):
            model_response._hidden_params["custom_llm_provider"] = custom_llm_provider
            model_response._hidden_params["region_name"] = kwargs.get(
                "aws_region_name", None
            )  # support region-based pricing for bedrock

        ### TIMEOUT LOGIC ###
        timeout = timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default
        if isinstance(timeout, httpx.Timeout) and not supports_httpx_timeout(
            custom_llm_provider
        ):
            timeout = timeout.read or 600  # default 10 min timeout
        elif not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore

        ### REGISTER CUSTOM MODEL PRICING -- IF GIVEN ###
        if input_cost_per_token is not None and output_cost_per_token is not None:
            litellm.register_model(
                {
                    f"{custom_llm_provider}/{model}": {
                        "input_cost_per_token": input_cost_per_token,
                        "output_cost_per_token": output_cost_per_token,
                        "litellm_provider": custom_llm_provider,
                    },
                    model: {
                        "input_cost_per_token": input_cost_per_token,
                        "output_cost_per_token": output_cost_per_token,
                        "litellm_provider": custom_llm_provider,
                    },
                }
            )
        elif (
            input_cost_per_second is not None
        ):  # time based pricing just needs cost in place
            output_cost_per_second = output_cost_per_second
            litellm.register_model(
                {
                    f"{custom_llm_provider}/{model}": {
                        "input_cost_per_second": input_cost_per_second,
                        "output_cost_per_second": output_cost_per_second,
                        "litellm_provider": custom_llm_provider,
                    },
                    model: {
                        "input_cost_per_second": input_cost_per_second,
                        "output_cost_per_second": output_cost_per_second,
                        "litellm_provider": custom_llm_provider,
                    },
                }
            )
        ### BUILD CUSTOM PROMPT TEMPLATE -- IF GIVEN ###
        custom_prompt_dict = {}  # type: ignore
        if (
            initial_prompt_value
            or roles
            or final_prompt_value
            or bos_token
            or eos_token
        ):
            custom_prompt_dict = {model: {}}
            if initial_prompt_value:
                custom_prompt_dict[model]["initial_prompt_value"] = initial_prompt_value
            if roles:
                custom_prompt_dict[model]["roles"] = roles
            if final_prompt_value:
                custom_prompt_dict[model]["final_prompt_value"] = final_prompt_value
            if bos_token:
                custom_prompt_dict[model]["bos_token"] = bos_token
            if eos_token:
                custom_prompt_dict[model]["eos_token"] = eos_token

        if (
            supports_system_message is not None
            and isinstance(supports_system_message, bool)
            and supports_system_message is False
        ):
            messages = map_system_message_pt(messages=messages)
        model_api_key = get_api_key(
            llm_provider=custom_llm_provider, dynamic_api_key=api_key
        )  # get the api key from the environment if required for the model

        if dynamic_api_key is not None:
            api_key = dynamic_api_key
        # check if user passed in any of the OpenAI optional params
        optional_params = get_optional_params(
            functions=functions,
            function_call=function_call,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            # params to identify the model
            model=model,
            custom_llm_provider=custom_llm_provider,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            max_retries=max_retries,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_headers=extra_headers,
            api_version=api_version,
            parallel_tool_calls=parallel_tool_calls,
            **non_default_params,
        )

        if litellm.add_function_to_prompt and optional_params.get(
            "functions_unsupported_model", None
        ):  # if user opts to add it to prompt, when API doesn't support function calling
            functions_unsupported_model = optional_params.pop(
                "functions_unsupported_model"
            )
            messages = function_call_prompt(
                messages=messages, functions=functions_unsupported_model
            )

        # For logging - save the values of the litellm-specific params passed in
        litellm_params = get_litellm_params(
            acompletion=acompletion,
            api_key=api_key,
            force_timeout=force_timeout,
            logger_fn=logger_fn,
            verbose=verbose,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            litellm_call_id=kwargs.get("litellm_call_id", None),
            model_alias_map=litellm.model_alias_map,
            completion_call_id=id,
            metadata=metadata,
            model_info=model_info,
            proxy_server_request=proxy_server_request,
            preset_cache_key=preset_cache_key,
            no_log=no_log,
            input_cost_per_second=input_cost_per_second,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_second=output_cost_per_second,
            output_cost_per_token=output_cost_per_token,
            cooldown_time=cooldown_time,
            text_completion=kwargs.get("text_completion"),
            azure_ad_token_provider=kwargs.get("azure_ad_token_provider"),
            user_continue_message=kwargs.get("user_continue_message"),
            base_model=base_model,
        )
        logging.update_environment_variables(
            model=model,
            user=user,
            optional_params=optional_params,
            litellm_params=litellm_params,
            custom_llm_provider=custom_llm_provider,
        )
        if mock_response or mock_tool_calls:
            return mock_completion(
                model,
                messages,
                stream=stream,
                n=n,
                mock_response=mock_response,
                mock_tool_calls=mock_tool_calls,
                logging=logging,
                acompletion=acompletion,
                mock_delay=kwargs.get("mock_delay", None),
                custom_llm_provider=custom_llm_provider,
            )

        if custom_llm_provider == "azure":
            # azure configs
            ## check dynamic params ##
            dynamic_params = False
            if client is not None and (
                isinstance(client, openai.AzureOpenAI)
                or isinstance(client, openai.AsyncAzureOpenAI)
            ):
                dynamic_params = _check_dynamic_azure_params(
                    azure_client_params={"api_version": api_version},
                    azure_client=client,
                )

            api_type = get_secret("AZURE_API_TYPE") or "azure"

            api_base = api_base or litellm.api_base or get_secret("AZURE_API_BASE")

            api_version = (
                api_version
                or litellm.api_version
                or get_secret("AZURE_API_VERSION")
                or litellm.AZURE_DEFAULT_API_VERSION
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )

            azure_ad_token = optional_params.get("extra_body", {}).pop(
                "azure_ad_token", None
            ) or get_secret("AZURE_AD_TOKEN")

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.AzureOpenAIConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## COMPLETION CALL
            response = azure_chat_completions.completion(
                model=model,
                messages=messages,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                api_type=api_type,
                dynamic_params=dynamic_params,
                azure_ad_token=azure_ad_token,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                logging_obj=logging,
                acompletion=acompletion,
                timeout=timeout,  # type: ignore
                client=client,  # pass AsyncAzureOpenAI, AzureOpenAI client
            )

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={
                        "headers": headers,
                        "api_version": api_version,
                        "api_base": api_base,
                    },
                )
        elif custom_llm_provider == "azure_text":
            # azure configs
            api_type = get_secret("AZURE_API_TYPE") or "azure"

            api_base = api_base or litellm.api_base or get_secret("AZURE_API_BASE")

            api_version = (
                api_version or litellm.api_version or get_secret("AZURE_API_VERSION")
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )

            azure_ad_token = optional_params.get("extra_body", {}).pop(
                "azure_ad_token", None
            ) or get_secret("AZURE_AD_TOKEN")

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.AzureOpenAIConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## COMPLETION CALL
            response = azure_text_completions.completion(
                model=model,
                messages=messages,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                api_type=api_type,
                azure_ad_token=azure_ad_token,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                logging_obj=logging,
                acompletion=acompletion,
                timeout=timeout,
                client=client,  # pass AsyncAzureOpenAI, AzureOpenAI client
            )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={
                        "headers": headers,
                        "api_version": api_version,
                        "api_base": api_base,
                    },
                )
        elif custom_llm_provider == "azure_ai":
            api_base = (
                api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
                or litellm.api_base
                or get_secret("AZURE_AI_API_BASE")
            )
            # set API KEY
            api_key = (
                api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
                or litellm.openai_key
                or get_secret("AZURE_AI_API_KEY")
            )

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.AzureAIStudioConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## FOR COHERE
            if "command-r" in model:  # make sure tool call in messages are str
                messages = stringify_json_tool_call_content(messages=messages)

            ## COMPLETION CALL
            try:
                response = azure_ai_chat_completions.completion(
                    model=model,
                    messages=messages,
                    headers=headers,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    api_key=api_key,
                    api_base=api_base,
                    acompletion=acompletion,
                    logging_obj=logging,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,  # type: ignore
                    custom_prompt_dict=custom_prompt_dict,
                    client=client,  # pass AsyncOpenAI, OpenAI client
                    organization=organization,
                    custom_llm_provider=custom_llm_provider,
                    drop_params=non_default_params.get("drop_params"),
                )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"headers": headers},
                )
                raise e

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={"headers": headers},
                )
        elif (
            custom_llm_provider == "text-completion-openai"
            or "ft:babbage-002" in model
            or "ft:davinci-002" in model  # support for finetuned completion models
            or custom_llm_provider
            in litellm.openai_text_completion_compatible_providers
            and kwargs.get("text_completion") is True
        ):
            openai.api_type = "openai"

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )

            openai.api_version = None
            # set API KEY

            api_key = (
                api_key
                or litellm.api_key
                or litellm.openai_key
                or get_secret("OPENAI_API_KEY")
            )

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.OpenAITextCompletionConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > openai_text_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v
            if litellm.organization:
                openai.organization = litellm.organization

            if (
                len(messages) > 0
                and "content" in messages[0]
                and type(messages[0]["content"]) == list
            ):
                # text-davinci-003 can accept a string or array, if it's an array, assume the array is set in messages[0]['content']
                # https://platform.openai.com/docs/api-reference/completions/create
                prompt = messages[0]["content"]
            else:
                prompt = " ".join([message["content"] for message in messages])  # type: ignore

            ## COMPLETION CALL
            _response = openai_text_completions.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                api_key=api_key,
                api_base=api_base,
                acompletion=acompletion,
                client=client,  # pass AsyncOpenAI, OpenAI client
                logging_obj=logging,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,  # type: ignore
            )

            if (
                optional_params.get("stream", False) == False
                and acompletion == False
                and text_completion == False
            ):
                # convert to chat completion response
                _response = litellm.OpenAITextCompletionConfig().convert_to_chat_model_response_object(
                    response_object=_response, model_response_object=model_response
                )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=_response,
                    additional_args={"headers": headers},
                )
            response = _response

        elif (
            model in litellm.open_ai_chat_completion_models
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "deepinfra"
            or custom_llm_provider == "perplexity"
            or custom_llm_provider == "groq"
            or custom_llm_provider == "nvidia_nim"
            or custom_llm_provider == "cerebras"
            or custom_llm_provider == "sambanova"
            or custom_llm_provider == "ai21_chat"
            or custom_llm_provider == "volcengine"
            or custom_llm_provider == "codestral"
            or custom_llm_provider == "deepseek"
            or custom_llm_provider == "anyscale"
            or custom_llm_provider == "mistral"
            or custom_llm_provider == "openai"
            or custom_llm_provider == "together_ai"
            or custom_llm_provider in litellm.openai_compatible_providers
            or "ft:gpt-3.5-turbo" in model  # finetune gpt-3.5-turbo
        ):  # allow user to make an openai call with a custom base
            # note: if a user sets a custom base - we should ensure this works
            # allow for the setting of dynamic and stateful api-bases
            api_base = (
                api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
                or litellm.api_base
                or get_secret("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            openai.organization = (
                organization
                or litellm.organization
                or get_secret("OPENAI_ORGANIZATION")
                or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = (
                api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
                or litellm.openai_key
                or get_secret("OPENAI_API_KEY")
            )

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.OpenAIConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## COMPLETION CALL
            try:
                if litellm.OpenAIO1Config().is_model_o1_reasoning_model(model=model):
                    response = openai_o1_chat_completions.completion(
                        model=model,
                        messages=messages,
                        headers=headers,
                        model_response=model_response,
                        print_verbose=print_verbose,
                        api_key=api_key,
                        api_base=api_base,
                        acompletion=acompletion,
                        logging_obj=logging,
                        optional_params=optional_params,
                        litellm_params=litellm_params,
                        logger_fn=logger_fn,
                        timeout=timeout,  # type: ignore
                        custom_prompt_dict=custom_prompt_dict,
                        client=client,  # pass AsyncOpenAI, OpenAI client
                        organization=organization,
                        custom_llm_provider=custom_llm_provider,
                    )
                else:
                    response = openai_chat_completions.completion(
                        model=model,
                        messages=messages,
                        headers=headers,
                        model_response=model_response,
                        print_verbose=print_verbose,
                        api_key=api_key,
                        api_base=api_base,
                        acompletion=acompletion,
                        logging_obj=logging,
                        optional_params=optional_params,
                        litellm_params=litellm_params,
                        logger_fn=logger_fn,
                        timeout=timeout,  # type: ignore
                        custom_prompt_dict=custom_prompt_dict,
                        client=client,  # pass AsyncOpenAI, OpenAI client
                        organization=organization,
                        custom_llm_provider=custom_llm_provider,
                    )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"headers": headers},
                )
                raise e

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={"headers": headers},
                )
        elif (
            "replicate" in model
            or custom_llm_provider == "replicate"
            or model in litellm.replicate_models
        ):
            # Setting the relevant API KEY for replicate, replicate defaults to using os.environ.get("REPLICATE_API_TOKEN")
            replicate_key = None
            replicate_key = (
                api_key
                or litellm.replicate_key
                or litellm.api_key
                or get_secret("REPLICATE_API_KEY")
                or get_secret("REPLICATE_API_TOKEN")
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("REPLICATE_API_BASE")
                or "https://api.replicate.com/v1"
            )

            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict

            model_response = replicate.completion(  # type: ignore
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,  # for calculating input/output tokens
                api_key=replicate_key,
                logging_obj=logging,
                custom_prompt_dict=custom_prompt_dict,
                acompletion=acompletion,
            )

            if optional_params.get("stream", False) == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=replicate_key,
                    original_response=model_response,
                )

            response = model_response
        elif (
            "clarifai" in model
            or custom_llm_provider == "clarifai"
            or model in litellm.clarifai_models
        ):
            clarifai_key = None
            clarifai_key = (
                api_key
                or litellm.clarifai_key
                or litellm.api_key
                or get_secret("CLARIFAI_API_KEY")
                or get_secret("CLARIFAI_API_TOKEN")
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("CLARIFAI_API_BASE")
                or "https://api.clarifai.com/v2"
            )

            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
            model_response = clarifai.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                acompletion=acompletion,
                logger_fn=logger_fn,
                encoding=encoding,  # for calculating input/output tokens
                api_key=clarifai_key,
                logging_obj=logging,
                custom_prompt_dict=custom_prompt_dict,
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=model_response,
                )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=clarifai_key,
                    original_response=model_response,
                )
            response = model_response

        elif custom_llm_provider == "anthropic":
            api_key = (
                api_key
                or litellm.anthropic_key
                or litellm.api_key
                or os.environ.get("ANTHROPIC_API_KEY")
            )
            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict

            if (model == "claude-2") or (model == "claude-instant-1"):
                # call anthropic /completion, only use this route for claude-2, claude-instant-1
                api_base = (
                    api_base
                    or litellm.api_base
                    or get_secret("ANTHROPIC_API_BASE")
                    or get_secret("ANTHROPIC_BASE_URL")
                    or "https://api.anthropic.com/v1/complete"
                )

                if api_base is not None and not api_base.endswith("/v1/complete"):
                    api_base += "/v1/complete"

                response = anthropic_text_completions.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    acompletion=acompletion,
                    custom_prompt_dict=litellm.custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,  # for calculating input/output tokens
                    api_key=api_key,
                    logging_obj=logging,
                    headers=headers,
                )
            else:
                # call /messages
                # default route for all anthropic models
                api_base = (
                    api_base
                    or litellm.api_base
                    or get_secret("ANTHROPIC_API_BASE")
                    or get_secret("ANTHROPIC_BASE_URL")
                    or "https://api.anthropic.com/v1/messages"
                )

                if api_base is not None and not api_base.endswith("/v1/messages"):
                    api_base += "/v1/messages"

                response = anthropic_chat_completions.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    acompletion=acompletion,
                    custom_prompt_dict=litellm.custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,  # for calculating input/output tokens
                    api_key=api_key,
                    logging_obj=logging,
                    headers=headers,
                    timeout=timeout,
                    client=client,
                )
            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                )
            response = response
        elif custom_llm_provider == "nlp_cloud":
            nlp_cloud_key = (
                api_key
                or litellm.nlp_cloud_key
                or get_secret("NLP_CLOUD_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("NLP_CLOUD_API_BASE")
                or "https://api.nlpcloud.io/v1/gpu/"
            )

            response = nlp_cloud.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=nlp_cloud_key,
                logging_obj=logging,
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    response,
                    model,
                    custom_llm_provider="nlp_cloud",
                    logging_obj=logging,
                )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                )

            response = response
        elif custom_llm_provider == "aleph_alpha":
            aleph_alpha_key = (
                api_key
                or litellm.aleph_alpha_key
                or get_secret("ALEPH_ALPHA_API_KEY")
                or get_secret("ALEPHALPHA_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("ALEPH_ALPHA_API_BASE")
                or "https://api.aleph-alpha.com/complete"
            )

            model_response = aleph_alpha.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                default_max_tokens_to_sample=litellm.max_tokens,
                api_key=aleph_alpha_key,
                logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="aleph_alpha",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "cohere":
            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("COHERE_API_BASE")
                or "https://api.cohere.ai/v1/generate"
            )

            headers = headers or litellm.headers or {}
            if headers is None:
                headers = {}

            if extra_headers is not None:
                headers.update(extra_headers)

            model_response = cohere_completion.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                headers=headers,
                api_key=cohere_key,
                logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="cohere",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "cohere_chat":
            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("COHERE_API_BASE")
                or "https://api.cohere.ai/v1/chat"
            )

            headers = headers or litellm.headers or {}
            if headers is None:
                headers = {}

            if extra_headers is not None:
                headers.update(extra_headers)

            model_response = cohere_chat.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=cohere_key,
                logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="cohere_chat",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "maritalk":
            maritalk_key = (
                api_key
                or litellm.maritalk_key
                or get_secret("MARITALK_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("MARITALK_API_BASE")
                or "https://chat.maritaca.ai/api/chat/inference"
            )

            model_response = maritalk.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=maritalk_key,
                logging_obj=logging,
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="maritalk",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "huggingface":
            custom_llm_provider = "huggingface"
            huggingface_key = (
                api_key
                or litellm.huggingface_key
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_API_KEY")
                or litellm.api_key
            )
            hf_headers = headers or litellm.headers

            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
            model_response = huggingface.completion(
                model=model,
                messages=messages,
                api_base=api_base,  # type: ignore
                headers=hf_headers,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=huggingface_key,
                acompletion=acompletion,
                logging_obj=logging,
                custom_prompt_dict=custom_prompt_dict,
                timeout=timeout,  # type: ignore
            )
            if (
                "stream" in optional_params
                and optional_params["stream"] == True
                and acompletion is False
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="huggingface",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "oobabooga":
            custom_llm_provider = "oobabooga"
            model_response = oobabooga.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                api_base=api_base,  # type: ignore
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=None,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="oobabooga",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "databricks":
            api_base = (
                api_base  # for databricks we check in get_llm_provider and pass in the api base from there
                or litellm.api_base
                or os.getenv("DATABRICKS_API_BASE")
            )

            # set API KEY
            api_key = (
                api_key
                or litellm.api_key  # for databricks we check in get_llm_provider and pass in the api key from there
                or litellm.databricks_key
                or get_secret("DATABRICKS_API_KEY")
            )

            headers = headers or litellm.headers

            ## COMPLETION CALL
            try:
                response = databricks_chat_completions.completion(
                    model=model,
                    messages=messages,
                    headers=headers,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    api_key=api_key,
                    api_base=api_base,
                    acompletion=acompletion,
                    logging_obj=logging,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,  # type: ignore
                    custom_prompt_dict=custom_prompt_dict,
                    client=client,  # pass AsyncOpenAI, OpenAI client
                    encoding=encoding,
                    custom_llm_provider="databricks",
                )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"headers": headers},
                )
                raise e

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={"headers": headers},
                )
        elif custom_llm_provider == "openrouter":
            api_base = api_base or litellm.api_base or "https://openrouter.ai/api/v1"

            api_key = (
                api_key
                or litellm.api_key
                or litellm.openrouter_key
                or get_secret("OPENROUTER_API_KEY")
                or get_secret("OR_API_KEY")
            )

            openrouter_site_url = get_secret("OR_SITE_URL") or "https://litellm.ai"
            openrouter_app_name = get_secret("OR_APP_NAME") or "liteLLM"

            openrouter_headers = {
                "HTTP-Referer": openrouter_site_url,
                "X-Title": openrouter_app_name,
            }

            _headers = headers or litellm.headers
            if _headers:
                openrouter_headers.update(_headers)

            headers = openrouter_headers

            ## Load Config
            config = openrouter.OpenrouterConfig.get_config()
            for k, v in config.items():
                if k == "extra_body":
                    # we use openai 'extra_body' to pass openrouter specific params - transforms, route, models
                    if "extra_body" in optional_params:
                        optional_params[k].update(v)
                    else:
                        optional_params[k] = v
                elif k not in optional_params:
                    optional_params[k] = v

            data = {"model": model, "messages": messages, **optional_params}

            ## COMPLETION CALL
            response = openai_chat_completions.completion(
                model=model,
                messages=messages,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                logging_obj=logging,
                acompletion=acompletion,
                timeout=timeout,  # type: ignore
                custom_llm_provider="openrouter",
            )
            ## LOGGING
            logging.post_call(
                input=messages, api_key=openai.api_key, original_response=response
            )
        elif (
            custom_llm_provider == "together_ai"
            or ("togethercomputer" in model)
            or (model in litellm.together_ai_models)
        ):
            """
            Deprecated. We now do together ai calls via the openai client - https://docs.together.ai/docs/openai-api-compatibility
            """
            pass
        elif custom_llm_provider == "palm":
            palm_api_key = api_key or get_secret("PALM_API_KEY") or litellm.api_key

            # palm does not support streaming as yet :(
            model_response = palm.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=palm_api_key,
                logging_obj=logging,
            )
            # fake palm streaming
            if "stream" in optional_params and optional_params["stream"] == True:
                # fake streaming for palm
                resp_string = model_response["choices"][0]["message"]["content"]
                response = CustomStreamWrapper(
                    resp_string, model, custom_llm_provider="palm", logging_obj=logging
                )
                return response
            response = model_response
        elif custom_llm_provider == "vertex_ai_beta" or custom_llm_provider == "gemini":
            vertex_ai_project = (
                optional_params.pop("vertex_project", None)
                or optional_params.pop("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret("VERTEXAI_PROJECT")
            )
            vertex_ai_location = (
                optional_params.pop("vertex_location", None)
                or optional_params.pop("vertex_ai_location", None)
                or litellm.vertex_location
                or get_secret("VERTEXAI_LOCATION")
            )
            vertex_credentials = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret("VERTEXAI_CREDENTIALS")
            )

            gemini_api_key = (
                api_key
                or get_secret("GEMINI_API_KEY")
                or get_secret("PALM_API_KEY")  # older palm api key should also work
                or litellm.api_key
            )

            new_params = deepcopy(optional_params)
            response = vertex_chat_completion.completion(  # type: ignore
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=new_params,
                litellm_params=litellm_params,  # type: ignore
                logger_fn=logger_fn,
                encoding=encoding,
                vertex_location=vertex_ai_location,
                vertex_project=vertex_ai_project,
                vertex_credentials=vertex_credentials,
                gemini_api_key=gemini_api_key,
                logging_obj=logging,
                acompletion=acompletion,
                timeout=timeout,
                custom_llm_provider=custom_llm_provider,
                client=client,
                api_base=api_base,
                extra_headers=extra_headers,
            )

        elif custom_llm_provider == "vertex_ai":
            vertex_ai_project = (
                optional_params.pop("vertex_project", None)
                or optional_params.pop("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret("VERTEXAI_PROJECT")
            )
            vertex_ai_location = (
                optional_params.pop("vertex_location", None)
                or optional_params.pop("vertex_ai_location", None)
                or litellm.vertex_location
                or get_secret("VERTEXAI_LOCATION")
            )
            vertex_credentials = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret("VERTEXAI_CREDENTIALS")
            )

            new_params = deepcopy(optional_params)
            if "claude-3" in model:
                model_response = vertex_ai_anthropic.completion(
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=new_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,
                    vertex_location=vertex_ai_location,
                    vertex_project=vertex_ai_project,
                    vertex_credentials=vertex_credentials,
                    logging_obj=logging,
                    acompletion=acompletion,
                    headers=headers,
                    custom_prompt_dict=custom_prompt_dict,
                    timeout=timeout,
                    client=client,
                )
            elif (
                model.startswith("meta/")
                or model.startswith("mistral")
                or model.startswith("codestral")
                or model.startswith("jamba")
            ):
                model_response = vertex_partner_models_chat_completion.completion(
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=new_params,
                    litellm_params=litellm_params,  # type: ignore
                    logger_fn=logger_fn,
                    encoding=encoding,
                    vertex_location=vertex_ai_location,
                    vertex_project=vertex_ai_project,
                    vertex_credentials=vertex_credentials,
                    logging_obj=logging,
                    acompletion=acompletion,
                    headers=headers,
                    custom_prompt_dict=custom_prompt_dict,
                    timeout=timeout,
                    client=client,
                )
            elif "gemini" in model or (
                litellm_params.get("base_model") is not None
                and "gemini" in litellm_params["base_model"]
            ):
                model_response = vertex_chat_completion.completion(  # type: ignore
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=new_params,
                    litellm_params=litellm_params,  # type: ignore
                    logger_fn=logger_fn,
                    encoding=encoding,
                    vertex_location=vertex_ai_location,
                    vertex_project=vertex_ai_project,
                    vertex_credentials=vertex_credentials,
                    gemini_api_key=None,
                    logging_obj=logging,
                    acompletion=acompletion,
                    timeout=timeout,
                    custom_llm_provider=custom_llm_provider,
                    client=client,
                    api_base=api_base,
                    extra_headers=extra_headers,
                )
            else:
                model_response = vertex_ai_non_gemini.completion(
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=new_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,
                    vertex_location=vertex_ai_location,
                    vertex_project=vertex_ai_project,
                    vertex_credentials=vertex_credentials,
                    logging_obj=logging,
                    acompletion=acompletion,
                )

                if (
                    "stream" in optional_params
                    and optional_params["stream"] is True
                    and acompletion is False
                ):
                    response = CustomStreamWrapper(
                        model_response,
                        model,
                        custom_llm_provider="vertex_ai",
                        logging_obj=logging,
                    )
                    return response
            response = model_response
        elif custom_llm_provider == "predibase":
            tenant_id = (
                optional_params.pop("tenant_id", None)
                or optional_params.pop("predibase_tenant_id", None)
                or litellm.predibase_tenant_id
                or get_secret("PREDIBASE_TENANT_ID")
            )

            api_base = (
                api_base
                or optional_params.pop("api_base", None)
                or optional_params.pop("base_url", None)
                or litellm.api_base
                or get_secret("PREDIBASE_API_BASE")
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.predibase_key
                or get_secret("PREDIBASE_API_KEY")
            )

            _model_response = predibase_chat_completions.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
                acompletion=acompletion,
                api_base=api_base,
                custom_prompt_dict=custom_prompt_dict,
                api_key=api_key,
                tenant_id=tenant_id,
                timeout=timeout,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] is True
                and acompletion is False
            ):
                return _model_response
            response = _model_response
        elif custom_llm_provider == "text-completion-codestral":

            api_base = (
                api_base
                or optional_params.pop("api_base", None)
                or optional_params.pop("base_url", None)
                or litellm.api_base
                or "https://codestral.mistral.ai/v1/fim/completions"
            )

            api_key = api_key or litellm.api_key or get_secret("CODESTRAL_API_KEY")

            text_completion_model_response = litellm.TextCompletionResponse(
                stream=stream
            )

            _model_response = codestral_text_completions.completion(  # type: ignore
                model=model,
                messages=messages,
                model_response=text_completion_model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
                acompletion=acompletion,
                api_base=api_base,
                custom_prompt_dict=custom_prompt_dict,
                api_key=api_key,
                timeout=timeout,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] is True
                and acompletion is False
            ):
                return _model_response
            response = _model_response
        elif custom_llm_provider == "ai21":
            custom_llm_provider = "ai21"
            ai21_key = (
                api_key
                or litellm.ai21_key
                or os.environ.get("AI21_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("AI21_API_BASE")
                or "https://api.ai21.com/studio/v1/"
            )

            model_response = ai21.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=ai21_key,
                logging_obj=logging,
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="ai21",
                    logging_obj=logging,
                )
                return response

            ## RESPONSE OBJECT
            response = model_response
        elif (
            custom_llm_provider == "sagemaker"
            or custom_llm_provider == "sagemaker_chat"
        ):
            # boto3 reads keys from .env
            model_response = sagemaker_llm.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                custom_prompt_dict=custom_prompt_dict,
                hf_model_name=hf_model_name,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
                acompletion=acompletion,
                use_messages_api=(
                    True if custom_llm_provider == "sagemaker_chat" else False
                ),
            )
            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=None,
                    original_response=model_response,
                )

            ## RESPONSE OBJECT
            response = model_response
        elif custom_llm_provider == "bedrock":
            # boto3 reads keys from .env
            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict

            if "aws_bedrock_client" in optional_params:
                verbose_logger.warning(
                    "'aws_bedrock_client' is a deprecated param. Please move to another auth method - https://docs.litellm.ai/docs/providers/bedrock#boto3---authentication."
                )
                # Extract credentials for legacy boto3 client and pass thru to httpx
                aws_bedrock_client = optional_params.pop("aws_bedrock_client")
                creds = aws_bedrock_client._get_credentials().get_frozen_credentials()

                if creds.access_key:
                    optional_params["aws_access_key_id"] = creds.access_key
                if creds.secret_key:
                    optional_params["aws_secret_access_key"] = creds.secret_key
                if creds.token:
                    optional_params["aws_session_token"] = creds.token
                if (
                    "aws_region_name" not in optional_params
                    or optional_params["aws_region_name"] is None
                ):
                    optional_params["aws_region_name"] = (
                        aws_bedrock_client.meta.region_name
                    )

            if model in litellm.BEDROCK_CONVERSE_MODELS:

                response = bedrock_converse_chat_completion.completion(
                    model=model,
                    messages=messages,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,  # type: ignore
                    logger_fn=logger_fn,
                    encoding=encoding,
                    logging_obj=logging,
                    extra_headers=extra_headers,
                    timeout=timeout,
                    acompletion=acompletion,
                    client=client,
                    api_base=api_base,
                )
            else:
                response = bedrock_chat_completion.completion(
                    model=model,
                    messages=messages,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,
                    logging_obj=logging,
                    extra_headers=extra_headers,
                    timeout=timeout,
                    acompletion=acompletion,
                    client=client,
                    api_base=api_base,
                )

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=None,
                    original_response=response,
                )

            ## RESPONSE OBJECT
            response = response
        elif custom_llm_provider == "watsonx":
            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
            response = watsonxai.completion(
                model=model,
                messages=messages,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,  # type: ignore
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
                timeout=timeout,  # type: ignore
                acompletion=acompletion,
            )
            if (
                "stream" in optional_params
                and optional_params["stream"] == True
                and not isinstance(response, CustomStreamWrapper)
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    iter(response),
                    model,
                    custom_llm_provider="watsonx",
                    logging_obj=logging,
                )

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=None,
                    original_response=response,
                )
            ## RESPONSE OBJECT
            response = response
        elif custom_llm_provider == "vllm":
            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
            model_response = vllm.completion(
                model=model,
                messages=messages,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
            )

            if (
                "stream" in optional_params and optional_params["stream"] == True
            ):  ## [BETA]
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="vllm",
                    logging_obj=logging,
                )
                return response

            ## RESPONSE OBJECT
            response = model_response
        elif custom_llm_provider == "ollama":
            api_base = (
                litellm.api_base
                or api_base
                or get_secret("OLLAMA_API_BASE")
                or "http://localhost:11434"
            )
            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
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
                    model=model,
                    messages=messages,
                    custom_llm_provider=custom_llm_provider,
                )
                if isinstance(prompt, dict):
                    # for multimode models - ollama/llava prompt_factory returns a dict {
                    #     "prompt": prompt,
                    #     "images": images
                    # }
                    prompt, images = prompt["prompt"], prompt["images"]
                    optional_params["images"] = images

            ## LOGGING
            generator = ollama.get_ollama_response(
                api_base=api_base,
                model=model,
                prompt=prompt,
                optional_params=optional_params,
                logging_obj=logging,
                acompletion=acompletion,
                model_response=model_response,
                encoding=encoding,
            )
            if acompletion is True or optional_params.get("stream", False) == True:
                return generator

            response = generator
        elif custom_llm_provider == "ollama_chat":
            api_base = (
                litellm.api_base
                or api_base
                or get_secret("OLLAMA_API_BASE")
                or "http://localhost:11434"
            )

            api_key = (
                api_key
                or litellm.ollama_key
                or os.environ.get("OLLAMA_API_KEY")
                or litellm.api_key
            )
            ## LOGGING
            generator = ollama_chat.get_ollama_response(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                optional_params=optional_params,
                logging_obj=logging,
                acompletion=acompletion,
                model_response=model_response,
                encoding=encoding,
            )
            if acompletion is True or optional_params.get("stream", False) is True:
                return generator

            response = generator

        elif custom_llm_provider == "triton":
            api_base = litellm.api_base or api_base
            model_response = triton_chat_completions.completion(
                api_base=api_base,
                timeout=timeout,  # type: ignore
                model=model,
                messages=messages,
                model_response=model_response,
                optional_params=optional_params,
                logging_obj=logging,
                stream=stream,
                acompletion=acompletion,
            )

            ## RESPONSE OBJECT
            response = model_response
            return response

        elif custom_llm_provider == "cloudflare":
            api_key = (
                api_key
                or litellm.cloudflare_api_key
                or litellm.api_key
                or get_secret("CLOUDFLARE_API_KEY")
            )
            account_id = get_secret("CLOUDFLARE_ACCOUNT_ID")
            api_base = (
                api_base
                or litellm.api_base
                or get_secret("CLOUDFLARE_API_BASE")
                or f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
            )

            custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
            response = cloudflare.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                custom_prompt_dict=litellm.custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,  # for calculating input/output tokens
                api_key=api_key,
                logging_obj=logging,
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    response,
                    model,
                    custom_llm_provider="cloudflare",
                    logging_obj=logging,
                )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                )
            response = response
        elif (
            custom_llm_provider == "baseten"
            or litellm.api_base == "https://app.baseten.co"
        ):
            custom_llm_provider = "baseten"
            baseten_key = (
                api_key
                or litellm.baseten_key
                or os.environ.get("BASETEN_API_KEY")
                or litellm.api_key
            )

            model_response = baseten.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=baseten_key,
                logging_obj=logging,
            )
            if inspect.isgenerator(model_response) or (
                "stream" in optional_params and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="baseten",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "petals" or model in litellm.petals_models:
            api_base = api_base or litellm.api_base

            custom_llm_provider = "petals"
            stream = optional_params.pop("stream", False)
            model_response = petals.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
            )
            if stream == True:  ## [BETA]
                # Fake streaming for petals
                resp_string = model_response["choices"][0]["message"]["content"]
                response = CustomStreamWrapper(
                    resp_string,
                    model,
                    custom_llm_provider="petals",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "custom":
            import requests

            url = litellm.api_base or api_base or ""
            if url == None or url == "":
                raise ValueError(
                    "api_base not set. Set api_base or litellm.api_base for custom endpoints"
                )

            """
            assume input to custom LLM api bases follow this format:
            resp = requests.post(
                api_base,
                json={
                    'model': 'meta-llama/Llama-2-13b-hf', # model name
                    'params': {
                        'prompt': ["The capital of France is P"],
                        'max_tokens': 32,
                        'temperature': 0.7,
                        'top_p': 1.0,
                        'top_k': 40,
                    }
                }
            )

            """
            prompt = " ".join([message["content"] for message in messages])  # type: ignore
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "params": {
                        "prompt": [prompt],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": kwargs.get("top_k", 40),
                    },
                },
                verify=litellm.ssl_verify,
            )
            response_json = resp.json()
            """
            assume all responses from custom api_bases of this format:
            {
                'data': [
                    {
                        'prompt': 'The capital of France is P',
                        'output': ['The capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France'],
                        'params': {'temperature': 0.7, 'top_k': 40, 'top_p': 1}}],
                        'message': 'ok'
                    }
                ]
            }
            """
            string_response = response_json["data"][0]["output"][0]
            ## RESPONSE OBJECT
            model_response.choices[0].message.content = string_response  # type: ignore
            model_response.created = int(time.time())
            model_response.model = model
            response = model_response
        elif (
            custom_llm_provider in litellm._custom_providers
        ):  # Assume custom LLM provider
            # Get the Custom Handler
            custom_handler: Optional[CustomLLM] = None
            for item in litellm.custom_provider_map:
                if item["provider"] == custom_llm_provider:
                    custom_handler = item["custom_handler"]

            if custom_handler is None:
                raise ValueError(
                    f"Unable to map your input to a model. Check your input - {args}"
                )

            ## ROUTE LLM CALL ##
            handler_fn = custom_chat_llm_router(
                async_fn=acompletion, stream=stream, custom_llm=custom_handler
            )

            headers = headers or litellm.headers

            ## CALL FUNCTION
            response = handler_fn(
                model=model,
                messages=messages,
                headers=headers,
                model_response=model_response,
                print_verbose=print_verbose,
                api_key=api_key,
                api_base=api_base,
                acompletion=acompletion,
                logging_obj=logging,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,  # type: ignore
                custom_prompt_dict=custom_prompt_dict,
                client=client,  # pass AsyncOpenAI, OpenAI client
                encoding=encoding,
            )
            if stream is True:
                return CustomStreamWrapper(
                    completion_stream=response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    logging_obj=logging,
                )

        else:
            raise ValueError(
                f"Unable to map your input to a model. Check your input - {args}"
            )
        return response
    except Exception as e:
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


def completion_with_retries(*args, **kwargs):
    """
    Executes a litellm.completion() with 3 retries
    """
    try:
        import tenacity
    except Exception as e:
        raise Exception(
            f"tenacity import failed please run `pip install tenacity`. Error{e}"
        )

    num_retries = kwargs.pop("num_retries", 3)
    retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = kwargs.pop("retry_strategy", "constant_retry")  # type: ignore
    original_function = kwargs.pop("original_function", completion)
    if retry_strategy == "constant_retry":
        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(num_retries), reraise=True
        )
    elif retry_strategy == "exponential_backoff_retry":
        retryer = tenacity.Retrying(
            wait=tenacity.wait_exponential(multiplier=1, max=10),
            stop=tenacity.stop_after_attempt(num_retries),
            reraise=True,
        )
    return retryer(original_function, *args, **kwargs)


async def acompletion_with_retries(*args, **kwargs):
    """
    [DEPRECATED]. Use 'acompletion' or router.acompletion instead!
    Executes a litellm.completion() with 3 retries
    """
    try:
        import tenacity
    except Exception as e:
        raise Exception(
            f"tenacity import failed please run `pip install tenacity`. Error{e}"
        )

    num_retries = kwargs.pop("num_retries", 3)
    retry_strategy = kwargs.pop("retry_strategy", "constant_retry")
    original_function = kwargs.pop("original_function", completion)
    if retry_strategy == "constant_retry":
        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(num_retries), reraise=True
        )
    elif retry_strategy == "exponential_backoff_retry":
        retryer = tenacity.Retrying(
            wait=tenacity.wait_exponential(multiplier=1, max=10),
            stop=tenacity.stop_after_attempt(num_retries),
            reraise=True,
        )
    return await retryer(original_function, *args, **kwargs)


def batch_completion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: List = [],
    functions: Optional[List] = None,
    function_call: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stop=None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    deployment_id=None,
    request_timeout: Optional[int] = None,
    timeout: Optional[int] = 600,
    # Optional liteLLM function params
    **kwargs,
):
    """
    Batch litellm.completion function for a given model.

    Args:
        model (str): The model to use for generating completions.
        messages (List, optional): List of messages to use as input for generating completions. Defaults to [].
        functions (List, optional): List of functions to use as input for generating completions. Defaults to [].
        function_call (str, optional): The function call to use as input for generating completions. Defaults to "".
        temperature (float, optional): The temperature parameter for generating completions. Defaults to None.
        top_p (float, optional): The top-p parameter for generating completions. Defaults to None.
        n (int, optional): The number of completions to generate. Defaults to None.
        stream (bool, optional): Whether to stream completions or not. Defaults to None.
        stop (optional): The stop parameter for generating completions. Defaults to None.
        max_tokens (float, optional): The maximum number of tokens to generate. Defaults to None.
        presence_penalty (float, optional): The presence penalty for generating completions. Defaults to None.
        frequency_penalty (float, optional): The frequency penalty for generating completions. Defaults to None.
        logit_bias (dict, optional): The logit bias for generating completions. Defaults to {}.
        user (str, optional): The user string for generating completions. Defaults to "".
        deployment_id (optional): The deployment ID for generating completions. Defaults to None.
        request_timeout (int, optional): The request timeout for generating completions. Defaults to None.

    Returns:
        list: A list of completion results.
    """
    args = locals()

    batch_messages = messages
    completions = []
    model = model
    custom_llm_provider = None
    if model.split("/", 1)[0] in litellm.provider_list:
        custom_llm_provider = model.split("/", 1)[0]
        model = model.split("/", 1)[1]
    if custom_llm_provider == "vllm":
        optional_params = get_optional_params(
            functions=functions,
            function_call=function_call,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            # params to identify the model
            model=model,
            custom_llm_provider=custom_llm_provider,
        )
        results = vllm.batch_completions(
            model=model,
            messages=batch_messages,
            custom_prompt_dict=litellm.custom_prompt_dict,
            optional_params=optional_params,
        )
    # all non VLLM models for batch completion models
    else:

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        with ThreadPoolExecutor(max_workers=100) as executor:
            for sub_batch in chunks(batch_messages, 100):
                for message_list in sub_batch:
                    kwargs_modified = args.copy()
                    kwargs_modified["messages"] = message_list
                    original_kwargs = {}
                    if "kwargs" in kwargs_modified:
                        original_kwargs = kwargs_modified.pop("kwargs")
                    future = executor.submit(
                        completion, **kwargs_modified, **original_kwargs
                    )
                    completions.append(future)

        # Retrieve the results from the futures
        # results = [future.result() for future in completions]
        # return exceptions if any
        results = []
        for future in completions:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)

    return results


# send one request to multiple models
# return as soon as one of the llms responds
def batch_completion_models(*args, **kwargs):
    """
    Send a request to multiple language models concurrently and return the response
    as soon as one of the models responds.

    Args:
        *args: Variable-length positional arguments passed to the completion function.
        **kwargs: Additional keyword arguments:
            - models (str or list of str): The language models to send requests to.
            - Other keyword arguments to be passed to the completion function.

    Returns:
        str or None: The response from one of the language models, or None if no response is received.

    Note:
        This function utilizes a ThreadPoolExecutor to parallelize requests to multiple models.
        It sends requests concurrently and returns the response from the first model that responds.
    """
    import concurrent

    if "model" in kwargs:
        kwargs.pop("model")
    if "models" in kwargs:
        models = kwargs["models"]
        kwargs.pop("models")
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
            for model in models:
                futures[model] = executor.submit(
                    completion, *args, model=model, **kwargs
                )

            for model, future in sorted(
                futures.items(), key=lambda x: models.index(x[0])
            ):
                if future.result() is not None:
                    return future.result()
    elif "deployments" in kwargs:
        deployments = kwargs["deployments"]
        kwargs.pop("deployments")
        kwargs.pop("model_list")
        nested_kwargs = kwargs.pop("kwargs", {})
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(deployments)
        ) as executor:
            for deployment in deployments:
                for key in kwargs.keys():
                    if (
                        key not in deployment
                    ):  # don't override deployment values e.g. model name, api base, etc.
                        deployment[key] = kwargs[key]
                kwargs = {**deployment, **nested_kwargs}
                futures[deployment["model"]] = executor.submit(completion, **kwargs)

            while futures:
                # wait for the first returned future
                print_verbose("\n\n waiting for next result\n\n")
                done, _ = concurrent.futures.wait(
                    futures.values(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                print_verbose(f"done list\n{done}")
                for future in done:
                    try:
                        result = future.result()
                        return result
                    except Exception as e:
                        # if model 1 fails, continue with response from model 2, model3
                        print_verbose(
                            f"\n\ngot an exception, ignoring, removing from futures"
                        )
                        print_verbose(futures)
                        new_futures = {}
                        for key, value in futures.items():
                            if future == value:
                                print_verbose(f"removing key{key}")
                                continue
                            else:
                                new_futures[key] = value
                        futures = new_futures
                        print_verbose(f"new futures{futures}")
                        continue

                print_verbose("\n\ndone looping through futures\n\n")
                print_verbose(futures)

    return None  # If no response is received from any model


def batch_completion_models_all_responses(*args, **kwargs):
    """
    Send a request to multiple language models concurrently and return a list of responses
    from all models that respond.

    Args:
        *args: Variable-length positional arguments passed to the completion function.
        **kwargs: Additional keyword arguments:
            - models (str or list of str): The language models to send requests to.
            - Other keyword arguments to be passed to the completion function.

    Returns:
        list: A list of responses from the language models that responded.

    Note:
        This function utilizes a ThreadPoolExecutor to parallelize requests to multiple models.
        It sends requests concurrently and collects responses from all models that respond.
    """
    import concurrent.futures

    # ANSI escape codes for colored output
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    if "model" in kwargs:
        kwargs.pop("model")
    if "models" in kwargs:
        models = kwargs["models"]
        kwargs.pop("models")

    responses = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        for idx, model in enumerate(models):
            future = executor.submit(completion, *args, model=model, **kwargs)
            if future.result() is not None:
                responses.append(future.result())

    return responses


### EMBEDDING ENDPOINTS ####################
@client
async def aembedding(*args, **kwargs) -> EmbeddingResponse:
    """
    Asynchronously calls the `embedding` function with the given arguments and keyword arguments.

    Parameters:
    - `args` (tuple): Positional arguments to be passed to the `embedding` function.
    - `kwargs` (dict): Keyword arguments to be passed to the `embedding` function.

    Returns:
    - `response` (Any): The response returned by the `embedding` function.
    """
    loop = asyncio.get_event_loop()
    model = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Embedding ###
    kwargs["aembedding"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(embedding, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=kwargs.get("api_base", None)
        )

        if (
            custom_llm_provider == "openai"
            or custom_llm_provider == "azure"
            or custom_llm_provider == "xinference"
            or custom_llm_provider == "voyage"
            or custom_llm_provider == "mistral"
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "triton"
            or custom_llm_provider == "anyscale"
            or custom_llm_provider == "openrouter"
            or custom_llm_provider == "deepinfra"
            or custom_llm_provider == "perplexity"
            or custom_llm_provider == "groq"
            or custom_llm_provider == "nvidia_nim"
            or custom_llm_provider == "cerebras"
            or custom_llm_provider == "sambanova"
            or custom_llm_provider == "ai21_chat"
            or custom_llm_provider == "volcengine"
            or custom_llm_provider == "deepseek"
            or custom_llm_provider == "fireworks_ai"
            or custom_llm_provider == "ollama"
            or custom_llm_provider == "vertex_ai"
            or custom_llm_provider == "gemini"
            or custom_llm_provider == "databricks"
            or custom_llm_provider == "watsonx"
            or custom_llm_provider == "cohere"
            or custom_llm_provider == "huggingface"
            or custom_llm_provider == "bedrock"
        ):  # currently implemented aiohttp calls for just azure and openai, soon all.
            # Await normally
            init_response = await loop.run_in_executor(None, func_with_context)
            if isinstance(init_response, dict):
                response = EmbeddingResponse(**init_response)
            elif isinstance(init_response, EmbeddingResponse):  ## CACHING SCENARIO
                response = init_response
            elif asyncio.iscoroutine(init_response):
                response = await init_response
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)
        if response is not None and hasattr(response, "_hidden_params"):
            response._hidden_params["custom_llm_provider"] = custom_llm_provider
        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def embedding(
    model,
    input=[],
    # Optional params
    dimensions: Optional[int] = None,
    timeout=600,  # default to 10 minutes
    # set api_base, api_version, api_key
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    api_type: Optional[str] = None,
    caching: bool = False,
    user: Optional[str] = None,
    custom_llm_provider=None,
    litellm_call_id=None,
    litellm_logging_obj=None,
    logger_fn=None,
    **kwargs,
) -> EmbeddingResponse:
    """
    Embedding function that calls an API to generate embeddings for the given input.

    Parameters:
    - model: The embedding model to use.
    - input: The input for which embeddings are to be generated.
    - dimensions: The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.
    - timeout: The timeout value for the API call, default 10 mins
    - litellm_call_id: The call ID for litellm logging.
    - litellm_logging_obj: The litellm logging object.
    - logger_fn: The logger function.
    - api_base: Optional. The base URL for the API.
    - api_version: Optional. The version of the API.
    - api_key: Optional. The API key to use.
    - api_type: Optional. The type of the API.
    - caching: A boolean indicating whether to enable caching.
    - custom_llm_provider: The custom llm provider.

    Returns:
    - response: The response received from the API call.

    Raises:
    - exception_type: If an exception occurs during the API call.
    """
    azure = kwargs.get("azure", None)
    client = kwargs.pop("client", None)
    rpm = kwargs.pop("rpm", None)
    tpm = kwargs.pop("tpm", None)
    cooldown_time = kwargs.get("cooldown_time", None)
    max_parallel_requests = kwargs.pop("max_parallel_requests", None)
    model_info = kwargs.get("model_info", None)
    metadata = kwargs.get("metadata", None)
    encoding_format = kwargs.get("encoding_format", None)
    proxy_server_request = kwargs.get("proxy_server_request", None)
    aembedding = kwargs.get("aembedding", None)
    extra_headers = kwargs.get("extra_headers", None)
    ### CUSTOM MODEL COST ###
    input_cost_per_token = kwargs.get("input_cost_per_token", None)
    output_cost_per_token = kwargs.get("output_cost_per_token", None)
    input_cost_per_second = kwargs.get("input_cost_per_second", None)
    output_cost_per_second = kwargs.get("output_cost_per_second", None)
    openai_params = [
        "user",
        "dimensions",
        "request_timeout",
        "api_base",
        "api_version",
        "api_key",
        "deployment_id",
        "organization",
        "base_url",
        "default_headers",
        "timeout",
        "max_retries",
        "encoding_format",
    ]
    litellm_params = [
        "metadata",
        "aembedding",
        "caching",
        "mock_response",
        "api_key",
        "api_version",
        "api_base",
        "force_timeout",
        "logger_fn",
        "verbose",
        "custom_llm_provider",
        "litellm_logging_obj",
        "litellm_call_id",
        "use_client",
        "id",
        "fallbacks",
        "azure",
        "headers",
        "model_list",
        "num_retries",
        "context_window_fallback_dict",
        "retry_policy",
        "roles",
        "final_prompt_value",
        "bos_token",
        "eos_token",
        "request_timeout",
        "complete_response",
        "self",
        "client",
        "rpm",
        "tpm",
        "max_parallel_requests",
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_second",
        "output_cost_per_second",
        "hf_model_name",
        "proxy_server_request",
        "model_info",
        "preset_cache_key",
        "caching_groups",
        "ttl",
        "cache",
        "no-log",
        "region_name",
        "allowed_model_region",
        "model_config",
        "cooldown_time",
        "tags",
        "azure_ad_token_provider",
        "tenant_id",
        "client_id",
        "client_secret",
        "extra_headers",
    ]
    default_params = openai_params + litellm_params
    non_default_params = {
        k: v for k, v in kwargs.items() if k not in default_params
    }  # model-specific params - pass them straight to the model/provider

    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )
    optional_params = get_optional_params_embeddings(
        model=model,
        user=user,
        dimensions=dimensions,
        encoding_format=encoding_format,
        custom_llm_provider=custom_llm_provider,
        **non_default_params,
    )
    ### REGISTER CUSTOM MODEL PRICING -- IF GIVEN ###
    if input_cost_per_token is not None and output_cost_per_token is not None:
        litellm.register_model(
            {
                model: {
                    "input_cost_per_token": input_cost_per_token,
                    "output_cost_per_token": output_cost_per_token,
                    "litellm_provider": custom_llm_provider,
                }
            }
        )
    if input_cost_per_second is not None:  # time based pricing just needs cost in place
        output_cost_per_second = output_cost_per_second or 0.0
        litellm.register_model(
            {
                model: {
                    "input_cost_per_second": input_cost_per_second,
                    "output_cost_per_second": output_cost_per_second,
                    "litellm_provider": custom_llm_provider,
                }
            }
        )
    try:
        response = None
        logging: Logging = litellm_logging_obj  # type: ignore
        logging.update_environment_variables(
            model=model,
            user=user,
            optional_params=optional_params,
            litellm_params={
                "timeout": timeout,
                "azure": azure,
                "litellm_call_id": litellm_call_id,
                "logger_fn": logger_fn,
                "proxy_server_request": proxy_server_request,
                "model_info": model_info,
                "metadata": metadata,
                "aembedding": aembedding,
                "preset_cache_key": None,
                "stream_response": {},
                "cooldown_time": cooldown_time,
            },
        )
        if azure is True or custom_llm_provider == "azure":
            # azure configs
            api_type = get_secret_str("AZURE_API_TYPE") or "azure"

            api_base = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")

            api_version = (
                api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
                or litellm.AZURE_DEFAULT_API_VERSION
            )

            azure_ad_token = optional_params.pop(
                "azure_ad_token", None
            ) or get_secret_str("AZURE_AD_TOKEN")

            api_key = (
                api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_API_KEY")
            )

            if api_base is None:
                raise ValueError(
                    f"No API Base provided for Azure OpenAI LLM provider. Set 'AZURE_API_BASE' in .env"
                )

            ## EMBEDDING CALL
            response = azure_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif (
            model in litellm.open_ai_embedding_models or custom_llm_provider == "openai"
        ):
            api_base = (
                api_base
                or litellm.api_base
                or get_secret_str("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            openai.organization = (
                litellm.organization
                or get_secret_str("OPENAI_ORGANIZATION")
                or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = (
                api_key
                or litellm.api_key
                or litellm.openai_key
                or get_secret_str("OPENAI_API_KEY")
            )
            api_type = "openai"
            api_version = None

            ## EMBEDDING CALL
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "databricks":
            api_base = (
                api_base or litellm.api_base or get_secret("DATABRICKS_API_BASE")
            )  # type: ignore

            # set API KEY
            api_key = (
                api_key
                or litellm.api_key
                or litellm.databricks_key
                or get_secret("DATABRICKS_API_KEY")
            )  # type: ignore

            ## EMBEDDING CALL
            response = databricks_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "cohere" or custom_llm_provider == "cohere_chat":
            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )

            if extra_headers is not None and isinstance(extra_headers, dict):
                headers = extra_headers
            else:
                headers = {}
            response = cohere_embed.embedding(
                model=model,
                input=input,
                optional_params=optional_params,
                encoding=encoding,
                api_key=cohere_key,  # type: ignore
                headers=headers,
                logging_obj=logging,
                model_response=EmbeddingResponse(),
                aembedding=aembedding,
                timeout=timeout,
                client=client,
            )
        elif custom_llm_provider == "huggingface":
            api_key = (
                api_key
                or litellm.huggingface_key
                or get_secret("HUGGINGFACE_API_KEY")
                or litellm.api_key
            )  # type: ignore
            response = huggingface.embedding(
                model=model,
                input=input,
                encoding=encoding,  # type: ignore
                api_key=api_key,
                api_base=api_base,
                logging_obj=logging,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "bedrock":
            if isinstance(input, str):
                transformed_input = [input]
            else:
                transformed_input = input
            response = bedrock_embedding.embeddings(
                model=model,
                input=transformed_input,
                encoding=encoding,
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                client=client,
                timeout=timeout,
                aembedding=aembedding,
                litellm_params={},
                api_base=api_base,
                print_verbose=print_verbose,
                extra_headers=extra_headers,
            )
        elif custom_llm_provider == "triton":
            if api_base is None:
                raise ValueError(
                    "api_base is required for triton. Please pass `api_base`"
                )
            response = triton_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "gemini":

            gemini_api_key = (
                api_key or get_secret_str("GEMINI_API_KEY") or litellm.api_key
            )

            response = google_batch_embeddings.batch_embeddings(  # type: ignore
                model=model,
                input=input,
                encoding=encoding,
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                vertex_project=None,
                vertex_location=None,
                vertex_credentials=None,
                aembedding=aembedding,
                print_verbose=print_verbose,
                custom_llm_provider="gemini",
                api_key=gemini_api_key,
            )

        elif custom_llm_provider == "vertex_ai":
            vertex_ai_project = (
                optional_params.pop("vertex_project", None)
                or optional_params.pop("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret("VERTEXAI_PROJECT")
                or get_secret("VERTEX_PROJECT")
            )
            vertex_ai_location = (
                optional_params.pop("vertex_location", None)
                or optional_params.pop("vertex_ai_location", None)
                or litellm.vertex_location
                or get_secret("VERTEXAI_LOCATION")
                or get_secret("VERTEX_LOCATION")
            )
            vertex_credentials = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret("VERTEXAI_CREDENTIALS")
                or get_secret("VERTEX_CREDENTIALS")
            )

            if (
                "image" in optional_params
                or "video" in optional_params
                or model
                in vertex_multimodal_embedding.SUPPORTED_MULTIMODAL_EMBEDDING_MODELS
            ):
                # multimodal embedding is supported on vertex httpx
                response = vertex_multimodal_embedding.multimodal_embedding(
                    model=model,
                    input=input,
                    encoding=encoding,
                    logging_obj=logging,
                    optional_params=optional_params,
                    model_response=EmbeddingResponse(),
                    vertex_project=vertex_ai_project,
                    vertex_location=vertex_ai_location,
                    vertex_credentials=vertex_credentials,
                    aembedding=aembedding,
                    print_verbose=print_verbose,
                    custom_llm_provider="vertex_ai",
                )
            else:
                response = vertex_ai_embedding_handler.embedding(
                    model=model,
                    input=input,
                    encoding=encoding,
                    logging_obj=logging,
                    optional_params=optional_params,
                    model_response=EmbeddingResponse(),
                    vertex_project=vertex_ai_project,
                    vertex_location=vertex_ai_location,
                    vertex_credentials=vertex_credentials,
                    aembedding=aembedding,
                    print_verbose=print_verbose,
                )
        elif custom_llm_provider == "oobabooga":
            response = oobabooga.embedding(
                model=model,
                input=input,
                encoding=encoding,
                api_base=api_base,
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
            )
        elif custom_llm_provider == "ollama":
            api_base = (
                litellm.api_base
                or api_base
                or get_secret("OLLAMA_API_BASE")
                or "http://localhost:11434"
            )  # type: ignore
            if isinstance(input, str):
                input = [input]
            if not all(isinstance(item, str) for item in input):
                raise litellm.BadRequestError(
                    message=f"Invalid input for ollama embeddings. input={input}",
                    model=model,  # type: ignore
                    llm_provider="ollama",  # type: ignore
                )
            ollama_embeddings_fn = (
                ollama.ollama_aembeddings
                if aembedding is True
                else ollama.ollama_embeddings
            )
            response = ollama_embeddings_fn(  # type: ignore
                api_base=api_base,
                model=model,
                prompts=input,
                encoding=encoding,
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
            )
        elif custom_llm_provider == "sagemaker":
            response = sagemaker_llm.embedding(
                model=model,
                input=input,
                encoding=encoding,
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                print_verbose=print_verbose,
            )
        elif custom_llm_provider == "mistral":
            api_key = api_key or litellm.api_key or get_secret_str("MISTRAL_API_KEY")
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "fireworks_ai":
            api_key = (
                api_key or litellm.api_key or get_secret_str("FIREWORKS_AI_API_KEY")
            )
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "voyage":
            api_key = api_key or litellm.api_key or get_secret_str("VOYAGE_API_KEY")
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "xinference":
            api_key = (
                api_key
                or litellm.api_key
                or get_secret("XINFERENCE_API_KEY")
                or "stub-xinference-key"
            )  # xinference does not need an api key, pass a stub key if user did not set one
            api_base = (
                api_base
                or litellm.api_base
                or get_secret("XINFERENCE_API_BASE")
                or "http://127.0.0.1:9997/v1"
            )
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "watsonx":
            response = watsonxai.embedding(
                model=model,
                input=input,
                encoding=encoding,
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                aembedding=aembedding,
            )
        else:
            args = locals()
            raise ValueError(f"No valid embedding model args passed in - {args}")
        if response is not None and hasattr(response, "_hidden_params"):
            response._hidden_params["custom_llm_provider"] = custom_llm_provider
        return response
    except Exception as e:
        ## LOGGING
        logging.post_call(
            input=input,
            api_key=api_key,
            original_response=str(e),
        )
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            original_exception=e,
            custom_llm_provider=custom_llm_provider,
            extra_kwargs=kwargs,
        )


###### Text Completion ################
@client
async def atext_completion(
    *args, **kwargs
) -> Union[TextCompletionResponse, TextCompletionStreamWrapper]:
    """
    Implemented to handle async streaming for the text completion endpoint
    """
    loop = asyncio.get_event_loop()
    model = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO COMPLETION ###
    kwargs["acompletion"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(text_completion, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=kwargs.get("api_base", None)
        )

        if (
            custom_llm_provider == "openai"
            or custom_llm_provider == "azure"
            or custom_llm_provider == "azure_text"
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "anyscale"
            or custom_llm_provider == "mistral"
            or custom_llm_provider == "openrouter"
            or custom_llm_provider == "deepinfra"
            or custom_llm_provider == "perplexity"
            or custom_llm_provider == "groq"
            or custom_llm_provider == "nvidia_nim"
            or custom_llm_provider == "cerebras"
            or custom_llm_provider == "sambanova"
            or custom_llm_provider == "ai21_chat"
            or custom_llm_provider == "volcengine"
            or custom_llm_provider == "text-completion-codestral"
            or custom_llm_provider == "deepseek"
            or custom_llm_provider == "fireworks_ai"
            or custom_llm_provider == "text-completion-openai"
            or custom_llm_provider == "huggingface"
            or custom_llm_provider == "ollama"
            or custom_llm_provider == "vertex_ai"
            or custom_llm_provider in litellm.openai_compatible_providers
        ):  # currently implemented aiohttp calls for just azure and openai, soon all.
            # Await normally
            response = await loop.run_in_executor(None, func_with_context)
            if asyncio.iscoroutine(response):
                response = await response
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)
        if kwargs.get("stream", False) is True:  # return an async generator
            return TextCompletionStreamWrapper(
                completion_stream=_async_streaming(
                    response=response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    args=args,
                ),
                model=model,
                custom_llm_provider=custom_llm_provider,
            )
        else:
            transformed_logprobs = None
            # only supported for TGI models
            try:
                raw_response = response._hidden_params.get("original_response", None)
                transformed_logprobs = litellm.utils.transform_logprobs(raw_response)
            except Exception as e:
                print_verbose(f"LiteLLM non blocking exception: {e}")

            ## TRANSLATE CHAT TO TEXT FORMAT ##
            if isinstance(response, TextCompletionResponse):
                return response
            elif asyncio.iscoroutine(response):
                response = await response

            text_completion_response = TextCompletionResponse()
            text_completion_response["id"] = response.get("id", None)
            text_completion_response["object"] = "text_completion"
            text_completion_response["created"] = response.get("created", None)
            text_completion_response["model"] = response.get("model", None)
            text_choices = TextChoices()
            text_choices["text"] = response["choices"][0]["message"]["content"]
            text_choices["index"] = response["choices"][0]["index"]
            text_choices["logprobs"] = transformed_logprobs
            text_choices["finish_reason"] = response["choices"][0]["finish_reason"]
            text_completion_response["choices"] = [text_choices]
            text_completion_response["usage"] = response.get("usage", None)
            text_completion_response._hidden_params = HiddenParams(
                **response._hidden_params
            )
            return text_completion_response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def text_completion(
    prompt: Union[
        str, List[Union[str, List[Union[str, List[int]]]]]
    ],  # Required: The prompt(s) to generate completions for.
    model: Optional[str] = None,  # Optional: either `model` or `engine` can be set
    best_of: Optional[
        int
    ] = None,  # Optional: Generates best_of completions server-side.
    echo: Optional[
        bool
    ] = None,  # Optional: Echo back the prompt in addition to the completion.
    frequency_penalty: Optional[
        float
    ] = None,  # Optional: Penalize new tokens based on their existing frequency.
    logit_bias: Optional[
        Dict[int, int]
    ] = None,  # Optional: Modify the likelihood of specified tokens.
    logprobs: Optional[
        int
    ] = None,  # Optional: Include the log probabilities on the most likely tokens.
    max_tokens: Optional[
        int
    ] = None,  # Optional: The maximum number of tokens to generate in the completion.
    n: Optional[
        int
    ] = None,  # Optional: How many completions to generate for each prompt.
    presence_penalty: Optional[
        float
    ] = None,  # Optional: Penalize new tokens based on whether they appear in the text so far.
    stop: Optional[
        Union[str, List[str]]
    ] = None,  # Optional: Sequences where the API will stop generating further tokens.
    stream: Optional[bool] = None,  # Optional: Whether to stream back partial progress.
    stream_options: Optional[dict] = None,
    suffix: Optional[
        str
    ] = None,  # Optional: The suffix that comes after a completion of inserted text.
    temperature: Optional[float] = None,  # Optional: Sampling temperature to use.
    top_p: Optional[float] = None,  # Optional: Nucleus sampling parameter.
    user: Optional[
        str
    ] = None,  # Optional: A unique identifier representing your end-user.
    # set api_base, api_version, api_key
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    custom_llm_provider: Optional[str] = None,
    *args,
    **kwargs,
):
    global print_verbose
    import copy

    """
    Generate text completions using the OpenAI API.

    Args:
        model (str): ID of the model to use.
        prompt (Union[str, List[Union[str, List[Union[str, List[int]]]]]): The prompt(s) to generate completions for.
        best_of (Optional[int], optional): Generates best_of completions server-side. Defaults to 1.
        echo (Optional[bool], optional): Echo back the prompt in addition to the completion. Defaults to False.
        frequency_penalty (Optional[float], optional): Penalize new tokens based on their existing frequency. Defaults to 0.
        logit_bias (Optional[Dict[int, int]], optional): Modify the likelihood of specified tokens. Defaults to None.
        logprobs (Optional[int], optional): Include the log probabilities on the most likely tokens. Defaults to None.
        max_tokens (Optional[int], optional): The maximum number of tokens to generate in the completion. Defaults to 16.
        n (Optional[int], optional): How many completions to generate for each prompt. Defaults to 1.
        presence_penalty (Optional[float], optional): Penalize new tokens based on whether they appear in the text so far. Defaults to 0.
        stop (Optional[Union[str, List[str]]], optional): Sequences where the API will stop generating further tokens. Defaults to None.
        stream (Optional[bool], optional): Whether to stream back partial progress. Defaults to False.
        suffix (Optional[str], optional): The suffix that comes after a completion of inserted text. Defaults to None.
        temperature (Optional[float], optional): Sampling temperature to use. Defaults to 1.
        top_p (Optional[float], optional): Nucleus sampling parameter. Defaults to 1.
        user (Optional[str], optional): A unique identifier representing your end-user.
    Returns:
        TextCompletionResponse: A response object containing the generated completion and associated metadata.

    Example:
        Your example of how to use this function goes here.
    """
    if "engine" in kwargs:
        if model == None:
            # only use engine when model not passed
            model = kwargs["engine"]
        kwargs.pop("engine")

    text_completion_response = TextCompletionResponse()

    optional_params: Dict[str, Any] = {}
    # default values for all optional params are none, litellm only passes them to the llm when they are set to non None values
    if best_of is not None:
        optional_params["best_of"] = best_of
    if echo is not None:
        optional_params["echo"] = echo
    if frequency_penalty is not None:
        optional_params["frequency_penalty"] = frequency_penalty
    if logit_bias is not None:
        optional_params["logit_bias"] = logit_bias
    if logprobs is not None:
        optional_params["logprobs"] = logprobs
    if max_tokens is not None:
        optional_params["max_tokens"] = max_tokens
    if n is not None:
        optional_params["n"] = n
    if presence_penalty is not None:
        optional_params["presence_penalty"] = presence_penalty
    if stop is not None:
        optional_params["stop"] = stop
    if stream is not None:
        optional_params["stream"] = stream
    if stream_options is not None:
        optional_params["stream_options"] = stream_options
    if suffix is not None:
        optional_params["suffix"] = suffix
    if temperature is not None:
        optional_params["temperature"] = temperature
    if top_p is not None:
        optional_params["top_p"] = top_p
    if user is not None:
        optional_params["user"] = user
    if api_base is not None:
        optional_params["api_base"] = api_base
    if api_version is not None:
        optional_params["api_version"] = api_version
    if api_key is not None:
        optional_params["api_key"] = api_key
    if custom_llm_provider is not None:
        optional_params["custom_llm_provider"] = custom_llm_provider

    # get custom_llm_provider
    _model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider, api_base=api_base)  # type: ignore

    if custom_llm_provider == "huggingface":
        # if echo == True, for TGI llms we need to set top_n_tokens to 3
        if echo == True:
            # for tgi llms
            if "top_n_tokens" not in kwargs:
                kwargs["top_n_tokens"] = 3

        # processing prompt - users can pass raw tokens to OpenAI Completion()
        if type(prompt) == list:
            import concurrent.futures

            tokenizer = tiktoken.encoding_for_model("text-davinci-003")
            ## if it's a 2d list - each element in the list is a text_completion() request
            if len(prompt) > 0 and type(prompt[0]) == list:
                responses = [None for x in prompt]  # init responses

                def process_prompt(i, individual_prompt):
                    decoded_prompt = tokenizer.decode(individual_prompt)
                    all_params = {**kwargs, **optional_params}
                    response = text_completion(
                        model=model,
                        prompt=decoded_prompt,
                        num_retries=3,  # ensure this does not fail for the batch
                        *args,
                        **all_params,
                    )

                    text_completion_response["id"] = response.get("id", None)
                    text_completion_response["object"] = "text_completion"
                    text_completion_response["created"] = response.get("created", None)
                    text_completion_response["model"] = response.get("model", None)
                    return response["choices"][0]

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_prompt, i, individual_prompt)
                        for i, individual_prompt in enumerate(prompt)
                    ]
                    for i, future in enumerate(
                        concurrent.futures.as_completed(futures)
                    ):
                        responses[i] = future.result()
                    text_completion_response.choices = responses  # type: ignore

                return text_completion_response
    # else:
    # check if non default values passed in for best_of, echo, logprobs, suffix
    # these are the params supported by Completion() but not ChatCompletion

    # default case, non OpenAI requests go through here
    # handle prompt formatting if prompt is a string vs. list of strings
    messages = []
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], str):
        for p in prompt:
            message = {"role": "user", "content": p}
            messages.append(message)
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif (
        (
            custom_llm_provider == "openai"
            or custom_llm_provider == "azure"
            or custom_llm_provider == "azure_text"
            or custom_llm_provider == "text-completion-codestral"
            or custom_llm_provider == "text-completion-openai"
        )
        and isinstance(prompt, list)
        and len(prompt) > 0
        and isinstance(prompt[0], list)
    ):
        verbose_logger.warning(
            msg="List of lists being passed. If this is for tokens, then it might not work across all models."
        )
        messages = [{"role": "user", "content": prompt}]  # type: ignore
    else:
        raise Exception(
            f"Unmapped prompt format. Your prompt is neither a list of strings nor a string. prompt={prompt}. File an issue - https://github.com/BerriAI/litellm/issues"
        )

    kwargs.pop("prompt", None)

    if _model is not None and (
        custom_llm_provider == "openai"
    ):  # for openai compatible endpoints - e.g. vllm, call the native /v1/completions endpoint for text completion calls
        if _model not in litellm.open_ai_chat_completion_models:
            model = "text-completion-openai/" + _model
            optional_params.pop("custom_llm_provider", None)

    kwargs["text_completion"] = True
    response = completion(
        model=model,
        messages=messages,
        *args,
        **kwargs,
        **optional_params,
    )
    if kwargs.get("acompletion", False) is True:
        return response
    if stream is True or kwargs.get("stream", False) is True:
        response = TextCompletionStreamWrapper(
            completion_stream=response,
            model=model,
            stream_options=stream_options,
            custom_llm_provider=custom_llm_provider,
        )
        return response
    transformed_logprobs = None
    # only supported for TGI models
    try:
        raw_response = response._hidden_params.get("original_response", None)
        transformed_logprobs = litellm.utils.transform_logprobs(raw_response)
    except Exception as e:
        print_verbose(f"LiteLLM non blocking exception: {e}")

    if isinstance(response, TextCompletionResponse):
        return response

    text_completion_response["id"] = response.get("id", None)
    text_completion_response["object"] = "text_completion"
    text_completion_response["created"] = response.get("created", None)
    text_completion_response["model"] = response.get("model", None)
    text_choices = TextChoices()
    text_choices["text"] = response["choices"][0]["message"]["content"]
    text_choices["index"] = response["choices"][0]["index"]
    text_choices["logprobs"] = transformed_logprobs
    text_choices["finish_reason"] = response["choices"][0]["finish_reason"]
    text_completion_response["choices"] = [text_choices]
    text_completion_response["usage"] = response.get("usage", None)
    text_completion_response._hidden_params = HiddenParams(**response._hidden_params)

    return text_completion_response


###### Adapter Completion ################


async def aadapter_completion(
    *, adapter_id: str, **kwargs
) -> Optional[Union[BaseModel, AdapterCompletionStreamWrapper]]:
    """
    Implemented to handle async calls for adapter_completion()
    """
    try:
        translation_obj: Optional[CustomLogger] = None
        for item in litellm.adapters:
            if item["id"] == adapter_id:
                translation_obj = item["adapter"]

        if translation_obj is None:
            raise ValueError(
                "No matching adapter given. Received 'adapter_id'={}, litellm.adapters={}".format(
                    adapter_id, litellm.adapters
                )
            )

        new_kwargs = translation_obj.translate_completion_input_params(kwargs=kwargs)

        response: Union[ModelResponse, CustomStreamWrapper] = await acompletion(**new_kwargs)  # type: ignore
        translated_response: Optional[
            Union[BaseModel, AdapterCompletionStreamWrapper]
        ] = None
        if isinstance(response, ModelResponse):
            translated_response = translation_obj.translate_completion_output_params(
                response=response
            )
        if isinstance(response, CustomStreamWrapper):
            translated_response = (
                translation_obj.translate_completion_output_params_streaming(
                    completion_stream=response
                )
            )

        return translated_response
    except Exception as e:
        raise e


def adapter_completion(
    *, adapter_id: str, **kwargs
) -> Optional[Union[BaseModel, AdapterCompletionStreamWrapper]]:
    translation_obj: Optional[CustomLogger] = None
    for item in litellm.adapters:
        if item["id"] == adapter_id:
            translation_obj = item["adapter"]

    if translation_obj is None:
        raise ValueError(
            "No matching adapter given. Received 'adapter_id'={}, litellm.adapters={}".format(
                adapter_id, litellm.adapters
            )
        )

    new_kwargs = translation_obj.translate_completion_input_params(kwargs=kwargs)

    response: Union[ModelResponse, CustomStreamWrapper] = completion(**new_kwargs)  # type: ignore
    translated_response: Optional[Union[BaseModel, AdapterCompletionStreamWrapper]] = (
        None
    )
    if isinstance(response, ModelResponse):
        translated_response = translation_obj.translate_completion_output_params(
            response=response
        )
    elif isinstance(response, CustomStreamWrapper) or inspect.isgenerator(response):
        translated_response = (
            translation_obj.translate_completion_output_params_streaming(
                completion_stream=response
            )
        )

    return translated_response


##### Moderation #######################


def moderation(
    input: str, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs
):
    # only supports open ai for now
    api_key = (
        api_key or litellm.api_key or litellm.openai_key or get_secret("OPENAI_API_KEY")
    )

    openai_client = kwargs.get("client", None)
    if openai_client is None:
        openai_client = openai.OpenAI(
            api_key=api_key,
        )

    response = openai_client.moderations.create(input=input, model=model)
    return response


@client
async def amoderation(
    input: str, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs
):
    # only supports open ai for now
    api_key = (
        api_key or litellm.api_key or litellm.openai_key or get_secret("OPENAI_API_KEY")
    )
    openai_client = kwargs.get("client", None)
    if openai_client is None:

        # call helper to get OpenAI client
        # _get_openai_client maintains in-memory caching logic for OpenAI clients
        openai_client = openai_chat_completions._get_openai_client(
            is_async=True,
            api_key=api_key,
        )
    response = await openai_client.moderations.create(input=input, model=model)
    return response


##### Image Generation #######################
@client
async def aimage_generation(*args, **kwargs) -> ImageResponse:
    """
    Asynchronously calls the `image_generation` function with the given arguments and keyword arguments.

    Parameters:
    - `args` (tuple): Positional arguments to be passed to the `image_generation` function.
    - `kwargs` (dict): Keyword arguments to be passed to the `image_generation` function.

    Returns:
    - `response` (Any): The response returned by the `image_generation` function.
    """
    loop = asyncio.get_event_loop()
    model = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Image Generation ###
    kwargs["aimg_generation"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(image_generation, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=kwargs.get("api_base", None)
        )

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if isinstance(init_response, dict) or isinstance(
            init_response, ImageResponse
        ):  ## CACHING SCENARIO
            if isinstance(init_response, dict):
                init_response = ImageResponse(**init_response)
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)
        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def image_generation(
    prompt: str,
    model: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[str] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    litellm_logging_obj=None,
    custom_llm_provider=None,
    **kwargs,
) -> ImageResponse:
    """
    Maps the https://api.openai.com/v1/images/generations endpoint.

    Currently supports just Azure + OpenAI.
    """
    try:
        aimg_generation = kwargs.get("aimg_generation", False)
        litellm_call_id = kwargs.get("litellm_call_id", None)
        logger_fn = kwargs.get("logger_fn", None)
        proxy_server_request = kwargs.get("proxy_server_request", None)
        model_info = kwargs.get("model_info", None)
        metadata = kwargs.get("metadata", {})
        client = kwargs.get("client", None)

        model_response = litellm.utils.ImageResponse()
        if model is not None or custom_llm_provider is not None:
            model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider, api_base=api_base)  # type: ignore
        else:
            model = "dall-e-2"
            custom_llm_provider = "openai"  # default to dall-e-2 on openai
        model_response._hidden_params["model"] = model
        openai_params = [
            "user",
            "request_timeout",
            "api_base",
            "api_version",
            "api_key",
            "deployment_id",
            "organization",
            "base_url",
            "default_headers",
            "timeout",
            "max_retries",
            "n",
            "quality",
            "size",
            "style",
        ]
        litellm_params = [
            "metadata",
            "aimg_generation",
            "caching",
            "mock_response",
            "api_key",
            "api_version",
            "api_base",
            "force_timeout",
            "logger_fn",
            "verbose",
            "custom_llm_provider",
            "litellm_logging_obj",
            "litellm_call_id",
            "use_client",
            "id",
            "fallbacks",
            "azure",
            "headers",
            "model_list",
            "num_retries",
            "context_window_fallback_dict",
            "retry_policy",
            "roles",
            "final_prompt_value",
            "bos_token",
            "eos_token",
            "request_timeout",
            "complete_response",
            "self",
            "client",
            "rpm",
            "tpm",
            "max_parallel_requests",
            "input_cost_per_token",
            "output_cost_per_token",
            "hf_model_name",
            "proxy_server_request",
            "model_info",
            "preset_cache_key",
            "caching_groups",
            "ttl",
            "cache",
            "region_name",
            "allowed_model_region",
            "model_config",
        ]
        default_params = openai_params + litellm_params
        non_default_params = {
            k: v for k, v in kwargs.items() if k not in default_params
        }  # model-specific params - pass them straight to the model/provider
        optional_params = get_optional_params_image_gen(
            n=n,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
            user=user,
            custom_llm_provider=custom_llm_provider,
            **non_default_params,
        )
        logging: Logging = litellm_logging_obj
        logging.update_environment_variables(
            model=model,
            user=user,
            optional_params=optional_params,
            litellm_params={
                "timeout": timeout,
                "azure": False,
                "litellm_call_id": litellm_call_id,
                "logger_fn": logger_fn,
                "proxy_server_request": proxy_server_request,
                "model_info": model_info,
                "metadata": metadata,
                "preset_cache_key": None,
                "stream_response": {},
            },
            custom_llm_provider=custom_llm_provider,
        )

        if custom_llm_provider == "azure":
            # azure configs
            api_type = get_secret("AZURE_API_TYPE") or "azure"

            api_base = api_base or litellm.api_base or get_secret("AZURE_API_BASE")

            api_version = (
                api_version or litellm.api_version or get_secret("AZURE_API_VERSION")
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )

            azure_ad_token = optional_params.pop("azure_ad_token", None) or get_secret(
                "AZURE_AD_TOKEN"
            )

            model_response = azure_chat_completions.image_generation(
                model=model,
                prompt=prompt,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                api_version=api_version,
                aimg_generation=aimg_generation,
                client=client,
            )
        elif custom_llm_provider == "openai":
            model_response = openai_chat_completions.image_generation(
                model=model,
                prompt=prompt,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                aimg_generation=aimg_generation,
                client=client,
            )
        elif custom_llm_provider == "bedrock":
            if model is None:
                raise Exception("Model needs to be set for bedrock")
            model_response = bedrock_image_generation.image_generation(
                model=model,
                prompt=prompt,
                timeout=timeout,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                aimg_generation=aimg_generation,
            )
        elif custom_llm_provider == "vertex_ai":
            vertex_ai_project = (
                optional_params.pop("vertex_project", None)
                or optional_params.pop("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret("VERTEXAI_PROJECT")
            )
            vertex_ai_location = (
                optional_params.pop("vertex_location", None)
                or optional_params.pop("vertex_ai_location", None)
                or litellm.vertex_location
                or get_secret("VERTEXAI_LOCATION")
            )
            vertex_credentials = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret("VERTEXAI_CREDENTIALS")
            )
            model_response = vertex_image_generation.image_generation(
                model=model,
                prompt=prompt,
                timeout=timeout,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                vertex_project=vertex_ai_project,
                vertex_location=vertex_ai_location,
                vertex_credentials=vertex_credentials,
                aimg_generation=aimg_generation,
            )

        return model_response
    except Exception as e:
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=locals(),
            extra_kwargs=kwargs,
        )


##### Transcription #######################


@client
async def atranscription(*args, **kwargs) -> TranscriptionResponse:
    """
    Calls openai + azure whisper endpoints.

    Allows router to load balance between them
    """
    loop = asyncio.get_event_loop()
    model = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Image Generation ###
    kwargs["atranscription"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(transcription, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=kwargs.get("api_base", None)
        )

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if isinstance(init_response, dict):
            response = TranscriptionResponse(**init_response)
        elif isinstance(init_response, TranscriptionResponse):  ## CACHING SCENARIO
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)
        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def transcription(
    model: str,
    file: FileTypes,
    ## OPTIONAL OPENAI PARAMS ##
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: Optional[
        Literal["json", "text", "srt", "verbose_json", "vtt"]
    ] = None,
    temperature: Optional[int] = None,  # openai defaults this to 0
    ## LITELLM PARAMS ##
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    max_retries: Optional[int] = None,
    litellm_logging_obj: Optional[LiteLLMLoggingObj] = None,
    custom_llm_provider=None,
    **kwargs,
) -> TranscriptionResponse:
    """
    Calls openai + azure whisper endpoints.

    Allows router to load balance between them
    """
    atranscription = kwargs.get("atranscription", False)
    litellm_call_id = kwargs.get("litellm_call_id", None)
    logger_fn = kwargs.get("logger_fn", None)
    proxy_server_request = kwargs.get("proxy_server_request", None)
    model_info = kwargs.get("model_info", None)
    metadata = kwargs.get("metadata", {})
    tags = kwargs.pop("tags", [])

    drop_params = kwargs.get("drop_params", None)
    client: Optional[
        Union[
            openai.AsyncOpenAI,
            openai.OpenAI,
            openai.AzureOpenAI,
            openai.AsyncAzureOpenAI,
        ]
    ] = kwargs.pop("client", None)

    if litellm_logging_obj:
        litellm_logging_obj.model_call_details["client"] = str(client)

    if max_retries is None:
        max_retries = openai.DEFAULT_MAX_RETRIES

    model_response = litellm.utils.TranscriptionResponse()

    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider, api_base=api_base)  # type: ignore

    if dynamic_api_key is not None:
        api_key = dynamic_api_key

    optional_params = get_optional_params_transcription(
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        custom_llm_provider=custom_llm_provider,
        drop_params=drop_params,
    )
    # optional_params = {
    #     "language": language,
    #     "prompt": prompt,
    #     "response_format": response_format,
    #     "temperature": None,  # openai defaults this to 0
    # }

    if custom_llm_provider == "azure":
        # azure configs
        api_base = api_base or litellm.api_base or get_secret("AZURE_API_BASE")

        api_version = (
            api_version or litellm.api_version or get_secret("AZURE_API_VERSION")
        )

        azure_ad_token = kwargs.pop("azure_ad_token", None) or get_secret(
            "AZURE_AD_TOKEN"
        )

        api_key = (
            api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret("AZURE_API_KEY")
        )  # type: ignore

        response = azure_audio_transcriptions.audio_transcriptions(
            model=model,
            audio_file=file,
            optional_params=optional_params,
            model_response=model_response,
            atranscription=atranscription,
            client=client,
            timeout=timeout,
            logging_obj=litellm_logging_obj,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            max_retries=max_retries,
        )
    elif custom_llm_provider == "openai" or custom_llm_provider == "groq":
        api_base = (
            api_base
            or litellm.api_base
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )  # type: ignore
        openai.organization = (
            litellm.organization
            or get_secret("OPENAI_ORGANIZATION")
            or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
        )
        # set API KEY
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret("OPENAI_API_KEY")
        )  # type: ignore
        response = openai_audio_transcriptions.audio_transcriptions(
            model=model,
            audio_file=file,
            optional_params=optional_params,
            model_response=model_response,
            atranscription=atranscription,
            client=client,
            timeout=timeout,
            logging_obj=litellm_logging_obj,
            max_retries=max_retries,
            api_base=api_base,
            api_key=api_key,
        )
    return response


@client
async def aspeech(*args, **kwargs) -> HttpxBinaryResponseContent:
    """
    Calls openai tts endpoints.
    """
    loop = asyncio.get_event_loop()
    model = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Image Generation ###
    kwargs["aspeech"] = True
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(speech, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=kwargs.get("api_base", None)
        )

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)
        return response  # type: ignore
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def speech(
    model: str,
    input: str,
    voice: Optional[Union[str, dict]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    max_retries: Optional[int] = None,
    metadata: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    response_format: Optional[str] = None,
    speed: Optional[int] = None,
    client=None,
    headers: Optional[dict] = None,
    custom_llm_provider: Optional[str] = None,
    aspeech: Optional[bool] = None,
    **kwargs,
) -> HttpxBinaryResponseContent:

    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider, api_base=api_base)  # type: ignore
    tags = kwargs.pop("tags", [])

    optional_params = {}
    if response_format is not None:
        optional_params["response_format"] = response_format
    if speed is not None:
        optional_params["speed"] = speed  # type: ignore

    if timeout is None:
        timeout = litellm.request_timeout

    if max_retries is None:
        max_retries = litellm.num_retries or openai.DEFAULT_MAX_RETRIES

    logging_obj = kwargs.get("litellm_logging_obj", None)
    response: Optional[HttpxBinaryResponseContent] = None
    if custom_llm_provider == "openai":
        if voice is None or not (isinstance(voice, str)):
            raise litellm.BadRequestError(
                message="'voice' is required to be passed as a string for OpenAI TTS",
                model=model,
                llm_provider=custom_llm_provider,
            )
        api_base = (
            api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
            or litellm.api_base
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )  # type: ignore
        # set API KEY
        api_key = (
            api_key
            or litellm.api_key  # for deepinfra/perplexity/anyscale we check in get_llm_provider and pass in the api key from there
            or litellm.openai_key
            or get_secret("OPENAI_API_KEY")
        )  # type: ignore

        organization = (
            organization
            or litellm.organization
            or get_secret("OPENAI_ORGANIZATION")
            or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
        )  # type: ignore

        project = (
            project
            or litellm.project
            or get_secret("OPENAI_PROJECT")
            or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
        )  # type: ignore

        headers = headers or litellm.headers

        response = openai_chat_completions.audio_speech(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            organization=organization,
            project=project,
            max_retries=max_retries,
            timeout=timeout,
            client=client,  # pass AsyncOpenAI, OpenAI client
            aspeech=aspeech,
        )
    elif custom_llm_provider == "azure":
        # azure configs
        if voice is None or not (isinstance(voice, str)):
            raise litellm.BadRequestError(
                message="'voice' is required to be passed as a string for Azure TTS",
                model=model,
                llm_provider=custom_llm_provider,
            )
        api_base = api_base or litellm.api_base or get_secret("AZURE_API_BASE")  # type: ignore

        api_version = (
            api_version or litellm.api_version or get_secret("AZURE_API_VERSION")
        )  # type: ignore

        api_key = (
            api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret("AZURE_OPENAI_API_KEY")
            or get_secret("AZURE_API_KEY")
        )  # type: ignore

        azure_ad_token: Optional[str] = optional_params.get("extra_body", {}).pop(  # type: ignore
            "azure_ad_token", None
        ) or get_secret(
            "AZURE_AD_TOKEN"
        )

        headers = headers or litellm.headers

        response = azure_chat_completions.audio_speech(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            organization=organization,
            max_retries=max_retries,
            timeout=timeout,
            client=client,  # pass AsyncOpenAI, OpenAI client
            aspeech=aspeech,
        )
    elif custom_llm_provider == "vertex_ai" or custom_llm_provider == "vertex_ai_beta":
        from litellm.types.router import GenericLiteLLMParams

        generic_optional_params = GenericLiteLLMParams(**kwargs)

        api_base = generic_optional_params.api_base or ""
        vertex_ai_project = (
            generic_optional_params.vertex_project
            or litellm.vertex_project
            or get_secret("VERTEXAI_PROJECT")
        )
        vertex_ai_location = (
            generic_optional_params.vertex_location
            or litellm.vertex_location
            or get_secret("VERTEXAI_LOCATION")
        )
        vertex_credentials = generic_optional_params.vertex_credentials or get_secret(
            "VERTEXAI_CREDENTIALS"
        )

        if voice is not None and not isinstance(voice, dict):
            raise litellm.BadRequestError(
                message=f"'voice' is required to be passed as a dict for Vertex AI TTS, passed in voice={voice}",
                model=model,
                llm_provider=custom_llm_provider,
            )
        response = vertex_text_to_speech.audio_speech(
            _is_async=aspeech,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_ai_project,
            vertex_location=vertex_ai_location,
            timeout=timeout,
            api_base=api_base,
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            kwargs=kwargs,
            logging_obj=logging_obj,
        )

    if response is None:
        raise Exception(
            "Unable to map the custom llm provider={} to a known provider={}.".format(
                custom_llm_provider, litellm.provider_list
            )
        )
    return response


##### Health Endpoints #######################


async def ahealth_check(
    model_params: dict,
    mode: Optional[
        Literal[
            "completion", "embedding", "image_generation", "chat", "batch", "rerank"
        ]
    ] = None,
    prompt: Optional[str] = None,
    input: Optional[List] = None,
    default_timeout: float = 6000,
):
    """
    Support health checks for different providers. Return remaining rate limit, etc.

    For azure/openai -> completion.with_raw_response
    For rest -> litellm.acompletion()
    """
    passed_in_mode: Optional[str] = None
    try:

        model: Optional[str] = model_params.get("model", None)

        if model is None:
            raise Exception("model not set")

        if model in litellm.model_cost and mode is None:
            mode = litellm.model_cost[model].get("mode")

        model, custom_llm_provider, _, _ = get_llm_provider(model=model)

        if model in litellm.model_cost and mode is None:
            mode = litellm.model_cost[model].get("mode")

        mode = mode
        passed_in_mode = mode
        if mode is None:
            mode = "chat"  # default to chat completion calls

        if custom_llm_provider == "azure":
            api_key = (
                model_params.get("api_key")
                or get_secret("AZURE_API_KEY")
                or get_secret("AZURE_OPENAI_API_KEY")
            )

            api_base = (
                model_params.get("api_base")
                or get_secret("AZURE_API_BASE")
                or get_secret("AZURE_OPENAI_API_BASE")
            )

            api_version = (
                model_params.get("api_version")
                or get_secret("AZURE_API_VERSION")
                or get_secret("AZURE_OPENAI_API_VERSION")
            )

            timeout = (
                model_params.get("timeout")
                or litellm.request_timeout
                or default_timeout
            )

            response = await azure_chat_completions.ahealth_check(
                model=model,
                messages=model_params.get(
                    "messages", None
                ),  # Replace with your actual messages list
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                timeout=timeout,
                mode=mode,
                prompt=prompt,
                input=input,
            )
        elif (
            custom_llm_provider == "openai"
            or custom_llm_provider == "text-completion-openai"
        ):
            api_key = model_params.get("api_key") or get_secret("OPENAI_API_KEY")
            organization = model_params.get("organization")

            timeout = (
                model_params.get("timeout")
                or litellm.request_timeout
                or default_timeout
            )

            api_base = model_params.get("api_base") or get_secret("OPENAI_API_BASE")

            if custom_llm_provider == "text-completion-openai":
                mode = "completion"

            response = await openai_chat_completions.ahealth_check(
                model=model,
                messages=model_params.get(
                    "messages", None
                ),  # Replace with your actual messages list
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                mode=mode,
                prompt=prompt,
                input=input,
                organization=organization,
            )
        else:
            model_params["cache"] = {
                "no-cache": True
            }  # don't used cached responses for making health check calls
            if mode == "embedding":
                model_params.pop("messages", None)
                model_params["input"] = input
                await litellm.aembedding(**model_params)
                response = {}
            elif mode == "image_generation":
                model_params.pop("messages", None)
                model_params["prompt"] = prompt
                await litellm.aimage_generation(**model_params)
                response = {}
            elif mode == "rerank":
                model_params.pop("messages", None)
                model_params["query"] = prompt
                model_params["documents"] = ["my sample text"]
                await litellm.arerank(**model_params)
                response = {}
            elif "*" in model:
                from litellm.litellm_core_utils.llm_request_utils import (
                    pick_cheapest_chat_model_from_llm_provider,
                )

                # this is a wildcard model, we need to pick a random model from the provider
                cheapest_model = pick_cheapest_chat_model_from_llm_provider(
                    custom_llm_provider=custom_llm_provider
                )
                model_params["model"] = cheapest_model
                await acompletion(**model_params)
                response = {}  # args like remaining ratelimit etc.
            else:  # default to completion calls
                await acompletion(**model_params)
                response = {}  # args like remaining ratelimit etc.
        return response
    except Exception as e:
        stack_trace = traceback.format_exc()
        if isinstance(stack_trace, str):
            stack_trace = stack_trace[:1000]

        if passed_in_mode is None:
            return {
                "error": f"error:{str(e)}. Missing `mode`. Set the `mode` for the model - https://docs.litellm.ai/docs/proxy/health#embedding-models  \nstacktrace: {stack_trace}"
            }

        error_to_return = (
            str(e)
            + "\nHave you set 'mode' - https://docs.litellm.ai/docs/proxy/health#embedding-models"
            + "\nstack trace: "
            + stack_trace
        )
        return {"error": error_to_return}


####### HELPER FUNCTIONS ################
## Set verbose to true -> ```litellm.set_verbose = True```
def print_verbose(print_statement):
    try:
        verbose_logger.debug(print_statement)
        if litellm.set_verbose:
            print(print_statement)  # noqa
    except:
        pass


def config_completion(**kwargs):
    if litellm.config_path != None:
        config_args = read_config_args(litellm.config_path)
        # overwrite any args passed in with config args
        return completion(**kwargs, **config_args)
    else:
        raise ValueError(
            "No config path set, please set a config path using `litellm.config_path = 'path/to/config.json'`"
        )


def stream_chunk_builder_text_completion(chunks: list, messages: Optional[List] = None):
    id = chunks[0]["id"]
    object = chunks[0]["object"]
    created = chunks[0]["created"]
    model = chunks[0]["model"]
    system_fingerprint = chunks[0].get("system_fingerprint", None)
    finish_reason = chunks[-1]["choices"][0]["finish_reason"]
    logprobs = chunks[-1]["choices"][0]["logprobs"]

    response = {
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "choices": [
            {
                "text": None,
                "index": 0,
                "logprobs": logprobs,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }
    content_list = []
    for chunk in chunks:
        choices = chunk["choices"]
        for choice in choices:
            if (
                choice is not None
                and hasattr(choice, "text")
                and choice.get("text") is not None
            ):
                _choice = choice.get("text")
                content_list.append(_choice)

    # Combine the "content" strings into a single string || combine the 'function' strings into a single string
    combined_content = "".join(content_list)

    # Update the "content" field within the response dictionary
    response["choices"][0]["text"] = combined_content

    if len(combined_content) > 0:
        completion_output = combined_content
    else:
        completion_output = ""
    # # Update usage information if needed
    try:
        response["usage"]["prompt_tokens"] = token_counter(
            model=model, messages=messages
        )
    except:  # don't allow this failing to block a complete streaming response from being returned
        print_verbose(f"token_counter failed, assuming prompt tokens is 0")
        response["usage"]["prompt_tokens"] = 0
    response["usage"]["completion_tokens"] = token_counter(
        model=model,
        text=combined_content,
        count_response_tokens=True,  # count_response_tokens is a Flag to tell token counter this is a response, No need to add extra tokens we do for input messages
    )
    response["usage"]["total_tokens"] = (
        response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"]
    )
    return response


def stream_chunk_builder(
    chunks: list, messages: Optional[list] = None, start_time=None, end_time=None
) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
    try:
        model_response = litellm.ModelResponse()
        ### BASE-CASE ###
        if len(chunks) == 0:
            return None
        ### SORT CHUNKS BASED ON CREATED ORDER ##
        print_verbose("Goes into checking if chunk has hiddden created at param")
        if chunks[0]._hidden_params.get("created_at", None):
            print_verbose("Chunks have a created at hidden param")
            # Sort chunks based on created_at in ascending order
            chunks = sorted(
                chunks, key=lambda x: x._hidden_params.get("created_at", float("inf"))
            )
            print_verbose("Chunks sorted")

        # set hidden params from chunk to model_response
        if model_response is not None and hasattr(model_response, "_hidden_params"):
            model_response._hidden_params = chunks[0].get("_hidden_params", {})
        id = chunks[0]["id"]
        object = chunks[0]["object"]
        created = chunks[0]["created"]
        model = chunks[0]["model"]
        system_fingerprint = chunks[0].get("system_fingerprint", None)

        if isinstance(
            chunks[0]["choices"][0], litellm.utils.TextChoices
        ):  # route to the text completion logic
            return stream_chunk_builder_text_completion(
                chunks=chunks, messages=messages
            )
        role = chunks[0]["choices"][0]["delta"]["role"]
        finish_reason = chunks[-1]["choices"][0]["finish_reason"]

        # Initialize the response dictionary
        response = {
            "id": id,
            "object": object,
            "created": created,
            "model": model,
            "system_fingerprint": system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": role, "content": ""},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Modify as needed
                "completion_tokens": 0,  # Modify as needed
                "total_tokens": 0,  # Modify as needed
            },
        }

        # Extract the "content" strings from the nested dictionaries within "choices"
        content_list = []
        combined_content = ""
        combined_arguments = ""

        tool_call_chunks = [
            chunk
            for chunk in chunks
            if "tool_calls" in chunk["choices"][0]["delta"]
            and chunk["choices"][0]["delta"]["tool_calls"] is not None
        ]

        if len(tool_call_chunks) > 0:
            argument_list: List = []
            delta = tool_call_chunks[0]["choices"][0]["delta"]
            message = response["choices"][0]["message"]
            message["tool_calls"] = []
            id = None
            name = None
            type = None
            tool_calls_list = []
            prev_index = None
            prev_name = None
            prev_id = None
            curr_id = None
            curr_index = 0
            for chunk in tool_call_chunks:
                choices = chunk["choices"]
                for choice in choices:
                    delta = choice.get("delta", {})
                    tool_calls = delta.get("tool_calls", "")
                    # Check if a tool call is present
                    if tool_calls and tool_calls[0].function is not None:
                        if tool_calls[0].id:
                            id = tool_calls[0].id
                            curr_id = id
                            if prev_id is None:
                                prev_id = curr_id
                        if tool_calls[0].index:
                            curr_index = tool_calls[0].index
                        if tool_calls[0].function.arguments:
                            # Now, tool_calls is expected to be a dictionary
                            arguments = tool_calls[0].function.arguments
                            argument_list.append(arguments)
                        if tool_calls[0].function.name:
                            name = tool_calls[0].function.name
                        if tool_calls[0].type:
                            type = tool_calls[0].type
                if prev_index is None:
                    prev_index = curr_index
                if prev_name is None:
                    prev_name = name
                if curr_index != prev_index:  # new tool call
                    combined_arguments = "".join(argument_list)
                    tool_calls_list.append(
                        {
                            "id": prev_id,
                            "function": {
                                "arguments": combined_arguments,
                                "name": prev_name,
                            },
                            "type": type,
                        }
                    )
                    argument_list = []  # reset
                    prev_index = curr_index
                    prev_id = curr_id
                    prev_name = name

            combined_arguments = (
                "".join(argument_list) or "{}"
            )  # base case, return empty dict

            tool_calls_list.append(
                {
                    "id": id,
                    "function": {"arguments": combined_arguments, "name": name},
                    "type": type,
                }
            )
            response["choices"][0]["message"]["content"] = None
            response["choices"][0]["message"]["tool_calls"] = tool_calls_list

        function_call_chunks = [
            chunk
            for chunk in chunks
            if "function_call" in chunk["choices"][0]["delta"]
            and chunk["choices"][0]["delta"]["function_call"] is not None
        ]

        if len(function_call_chunks) > 0:
            argument_list = []
            delta = function_call_chunks[0]["choices"][0]["delta"]
            function_call = delta.get("function_call", "")
            function_call_name = function_call.name

            message = response["choices"][0]["message"]
            message["function_call"] = {}
            message["function_call"]["name"] = function_call_name

            for chunk in function_call_chunks:
                choices = chunk["choices"]
                for choice in choices:
                    delta = choice.get("delta", {})
                    function_call = delta.get("function_call", "")

                    # Check if a function call is present
                    if function_call:
                        # Now, function_call is expected to be a dictionary
                        arguments = function_call.arguments
                        argument_list.append(arguments)

            combined_arguments = "".join(argument_list)
            response["choices"][0]["message"]["content"] = None
            response["choices"][0]["message"]["function_call"][
                "arguments"
            ] = combined_arguments

        content_chunks = [
            chunk
            for chunk in chunks
            if "content" in chunk["choices"][0]["delta"]
            and chunk["choices"][0]["delta"]["content"] is not None
        ]

        if len(content_chunks) > 0:
            for chunk in chunks:
                choices = chunk["choices"]
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content is None:
                        continue  # openai v1.0.0 sets content = None for chunks
                    content_list.append(content)

            # Combine the "content" strings into a single string || combine the 'function' strings into a single string
            combined_content = "".join(content_list)

            # Update the "content" field within the response dictionary
            response["choices"][0]["message"]["content"] = combined_content

        completion_output = ""
        if len(combined_content) > 0:
            completion_output += combined_content
        if len(combined_arguments) > 0:
            completion_output += combined_arguments

        # # Update usage information if needed
        prompt_tokens = 0
        completion_tokens = 0
        ## anthropic prompt caching information ##
        cache_creation_input_tokens: Optional[int] = None
        cache_read_input_tokens: Optional[int] = None
        for chunk in chunks:
            usage_chunk: Optional[Usage] = None
            if "usage" in chunk:
                usage_chunk = chunk.usage
            elif hasattr(chunk, "_hidden_params") and "usage" in chunk._hidden_params:
                usage_chunk = chunk._hidden_params["usage"]
            if usage_chunk is not None:
                if "prompt_tokens" in usage_chunk:
                    prompt_tokens = usage_chunk.get("prompt_tokens", 0) or 0
                if "completion_tokens" in usage_chunk:
                    completion_tokens = usage_chunk.get("completion_tokens", 0) or 0
                if "cache_creation_input_tokens" in usage_chunk:
                    cache_creation_input_tokens = usage_chunk.get(
                        "cache_creation_input_tokens"
                    )
                if "cache_read_input_tokens" in usage_chunk:
                    cache_read_input_tokens = usage_chunk.get("cache_read_input_tokens")

        try:
            response["usage"]["prompt_tokens"] = prompt_tokens or token_counter(
                model=model, messages=messages
            )
        except (
            Exception
        ):  # don't allow this failing to block a complete streaming response from being returned
            print_verbose("token_counter failed, assuming prompt tokens is 0")
            response["usage"]["prompt_tokens"] = 0
        response["usage"]["completion_tokens"] = completion_tokens or token_counter(
            model=model,
            text=completion_output,
            count_response_tokens=True,  # count_response_tokens is a Flag to tell token counter this is a response, No need to add extra tokens we do for input messages
        )
        response["usage"]["total_tokens"] = (
            response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"]
        )

        if cache_creation_input_tokens is not None:
            response["usage"][
                "cache_creation_input_tokens"
            ] = cache_creation_input_tokens
        if cache_read_input_tokens is not None:
            response["usage"]["cache_read_input_tokens"] = cache_read_input_tokens

        return convert_to_model_response_object(
            response_object=response,
            model_response_object=model_response,
            start_time=start_time,
            end_time=end_time,
        )  # type: ignore
    except Exception as e:
        verbose_logger.exception(
            "litellm.main.py::stream_chunk_builder() - Exception occurred - {}".format(
                str(e)
            )
        )
        raise litellm.APIError(
            status_code=500,
            message="Error building chunks for logging/streaming usage calculation",
            llm_provider="",
            model="",
        )
