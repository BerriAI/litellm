import hashlib
import time
import types
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
    cast,
)
from urllib.parse import urlparse

import httpx
import openai
from openai import AsyncOpenAI, OpenAI
from openai.types.beta.assistant_deleted import AssistantDeleted
from openai.types.file_deleted import FileDeleted
from pydantic import BaseModel
from typing_extensions import overload

import litellm
from litellm import LlmProviders
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.bedrock.chat.invoke_handler import MockResponseIterator
from litellm.llms.custom_httpx.http_handler import _DEFAULT_TTL_FOR_HTTPX_CLIENTS
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
)
from litellm.utils import (
    CustomStreamWrapper,
    ProviderConfigManager,
    convert_to_model_response_object,
)

from ...types.llms.openai import *
from ..base import BaseLLM
from .chat.o_series_transformation import OpenAIOSeriesConfig
from .common_utils import OpenAIError, drop_params_from_unprocessable_entity_error

openaiOSeriesConfig = OpenAIOSeriesConfig()


class MistralEmbeddingConfig:
    """
    Reference: https://docs.mistral.ai/api/#operation/createEmbedding
    """

    def __init__(
        self,
    ) -> None:
        locals_ = locals().copy()
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

    def get_supported_openai_params(self):
        return [
            "encoding_format",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "encoding_format":
                optional_params["encoding_format"] = value
        return optional_params


class OpenAIConfig(BaseConfig):
    """
    Reference: https://platform.openai.com/docs/api-reference/chat/create

    The class `OpenAIConfig` provides configuration for the OpenAI's Chat API interface. Below are the parameters:

    - `frequency_penalty` (number or null): Defaults to 0. Allows a value between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, thereby minimizing repetition.

    - `function_call` (string or object): This optional parameter controls how the model calls functions.

    - `functions` (array): An optional parameter. It is a list of functions for which the model may generate JSON inputs.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion. OpenAI has now deprecated in favor of max_completion_tokens, and is not compatible with o1 series models.

    - `max_completion_tokens` (integer or null): An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.

    - `n` (integer or null): This optional parameter helps to set how many chat completion choices to generate for each input message.

    - `presence_penalty` (number or null): Defaults to 0. It penalizes new tokens based on if they appear in the text so far, hence increasing the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """

    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        This function returns the list
        of supported openai parameters for a given OpenAI Model

        - If O1 model, returns O1 supported params
        - If gpt-audio model, returns gpt-audio supported params
        - Else, returns gpt supported params

        Args:
            model (str): OpenAI model

        Returns:
            list: List of supported openai parameters
        """
        if openaiOSeriesConfig.is_model_o_series_model(model=model):
            return openaiOSeriesConfig.get_supported_openai_params(model=model)
        elif litellm.openAIGPTAudioConfig.is_model_gpt_audio_model(model=model):
            return litellm.openAIGPTAudioConfig.get_supported_openai_params(model=model)
        else:
            return litellm.openAIGPTConfig.get_supported_openai_params(model=model)

    def _map_openai_params(
        self, non_default_params: dict, optional_params: dict, model: str
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str
    ) -> List[AllMessageValues]:
        return messages

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """ """
        if openaiOSeriesConfig.is_model_o_series_model(model=model):
            return openaiOSeriesConfig.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=drop_params,
            )
        elif litellm.openAIGPTAudioConfig.is_model_gpt_audio_model(model=model):
            return litellm.openAIGPTAudioConfig.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=drop_params,
            )

        return litellm.openAIGPTConfig.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        messages = self._transform_messages(messages=messages, model=model)
        return {"model": model, "messages": messages, **optional_params}

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
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:

        logging_obj.post_call(original_response=raw_response.text)
        logging_obj.model_call_details["response_headers"] = raw_response.headers
        final_response_obj = cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=raw_response.json(),
                model_response_object=model_response,
                hidden_params={"headers": raw_response.headers},
                _response_headers=dict(raw_response.headers),
            ),
        )

        return final_response_obj

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return {
            "Authorization": f"Bearer {api_key}",
            **headers,
        }

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return OpenAIChatCompletionResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class OpenAIChatCompletionResponseIterator(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        {'choices': [{'delta': {'content': '', 'role': 'assistant'}, 'finish_reason': None, 'index': 0, 'logprobs': None}], 'created': 1735763082, 'id': 'a83a2b0fbfaf4aab9c2c93cb8ba346d7', 'model': 'mistral-large', 'object': 'chat.completion.chunk'}
        """
        try:
            return ModelResponseStream(**chunk)
        except Exception as e:
            raise e


class OpenAIChatCompletion(BaseLLM):

    def __init__(self) -> None:
        super().__init__()

    def _get_openai_client(
        self,
        is_async: bool,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = httpx.Timeout(None),
        max_retries: Optional[int] = 2,
        organization: Optional[str] = None,
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ):
        args = locals()
        if client is None:
            if not isinstance(max_retries, int):
                raise OpenAIError(
                    status_code=422,
                    message="max retries must be an int. Passed in value: {}".format(
                        max_retries
                    ),
                )
            # Creating a new OpenAI Client
            # check in memory cache before creating a new one
            # Convert the API key to bytes
            hashed_api_key = None
            if api_key is not None:
                hash_object = hashlib.sha256(api_key.encode())
                # Hexadecimal representation of the hash
                hashed_api_key = hash_object.hexdigest()

            _cache_key = f"hashed_api_key={hashed_api_key},api_base={api_base},timeout={timeout},max_retries={max_retries},organization={organization},is_async={is_async}"

            _cached_client = litellm.in_memory_llm_clients_cache.get_cache(_cache_key)
            if _cached_client:
                return _cached_client
            if is_async:
                _new_client: Union[OpenAI, AsyncOpenAI] = AsyncOpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=litellm.aclient_session,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                )
            else:

                _new_client = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=litellm.client_session,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                )

            ## SAVE CACHE KEY
            litellm.in_memory_llm_clients_cache.set_cache(
                key=_cache_key,
                value=_new_client,
                ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
            )
            return _new_client

        else:
            return client

    @track_llm_api_timing()
    async def make_openai_chat_completion_request(
        self,
        openai_aclient: AsyncOpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
    ) -> Tuple[dict, BaseModel]:
        """
        Helper to:
        - call chat.completions.create.with_raw_response when litellm.return_response_headers is True
        - call chat.completions.create by default
        """
        start_time = time.time()
        try:
            raw_response = (
                await openai_aclient.chat.completions.with_raw_response.create(
                    **data, timeout=timeout
                )
            )
            end_time = time.time()

            if hasattr(raw_response, "headers"):
                headers = dict(raw_response.headers)
            else:
                headers = {}
            response = raw_response.parse()
            return headers, response
        except openai.APITimeoutError as e:
            end_time = time.time()
            time_delta = round(end_time - start_time, 2)
            e.message += f" - timeout value={timeout}, time taken={time_delta} seconds"
            raise e
        except Exception as e:
            raise e

    @track_llm_api_timing()
    def make_sync_openai_chat_completion_request(
        self,
        openai_client: OpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
    ) -> Tuple[dict, BaseModel]:
        """
        Helper to:
        - call chat.completions.create.with_raw_response when litellm.return_response_headers is True
        - call chat.completions.create by default
        """
        raw_response = None
        try:
            raw_response = openai_client.chat.completions.with_raw_response.create(
                **data, timeout=timeout
            )

            if hasattr(raw_response, "headers"):
                headers = dict(raw_response.headers)
            else:
                headers = {}
            response = raw_response.parse()
            return headers, response
        except Exception as e:
            if raw_response is not None:
                raise Exception(
                    "error - {}, Received response - {}, Type of response - {}".format(
                        e, raw_response, type(raw_response)
                    )
                )
            else:
                raise e

    def mock_streaming(
        self,
        response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        model: str,
        stream_options: Optional[dict] = None,
    ) -> CustomStreamWrapper:
        completion_stream = MockResponseIterator(model_response=response)
        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="openai",
            logging_obj=logging_obj,
            stream_options=stream_options,
        )

        return streaming_response

    def completion(  # type: ignore # noqa: PLR0915
        self,
        model_response: ModelResponse,
        timeout: Union[float, httpx.Timeout],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: Any,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        print_verbose: Optional[Callable] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        dynamic_params: Optional[bool] = None,
        azure_ad_token: Optional[str] = None,
        acompletion: bool = False,
        logger_fn=None,
        headers: Optional[dict] = None,
        custom_prompt_dict: dict = {},
        client=None,
        organization: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        drop_params: Optional[bool] = None,
    ):

        super().completion()
        try:
            fake_stream: bool = False
            inference_params = optional_params.copy()
            stream_options: Optional[dict] = inference_params.pop(
                "stream_options", None
            )
            stream: Optional[bool] = inference_params.pop("stream", False)
            provider_config: Optional[BaseConfig] = None

            if custom_llm_provider is not None and model is not None:
                provider_config = ProviderConfigManager.get_provider_chat_config(
                    model=model, provider=LlmProviders(custom_llm_provider)
                )

            if provider_config:
                fake_stream = provider_config.should_fake_stream(
                    model=model, custom_llm_provider=custom_llm_provider, stream=stream
                )

            if headers:
                inference_params["extra_headers"] = headers
            if model is None or messages is None:
                raise OpenAIError(status_code=422, message="Missing model or messages")

            if not isinstance(timeout, float) and not isinstance(
                timeout, httpx.Timeout
            ):
                raise OpenAIError(
                    status_code=422,
                    message="Timeout needs to be a float or httpx.Timeout",
                )

            if custom_llm_provider is not None and custom_llm_provider != "openai":
                model_response.model = f"{custom_llm_provider}/{model}"

            for _ in range(
                2
            ):  # if call fails due to alternating messages, retry with reformatted message

                if provider_config is not None:
                    data = provider_config.transform_request(
                        model=model,
                        messages=messages,
                        optional_params=inference_params,
                        litellm_params=litellm_params,
                        headers=headers or {},
                    )
                else:
                    data = OpenAIConfig().transform_request(
                        model=model,
                        messages=messages,
                        optional_params=inference_params,
                        litellm_params=litellm_params,
                        headers=headers or {},
                    )
                try:
                    max_retries = data.pop("max_retries", 2)
                    if acompletion is True:
                        if stream is True and fake_stream is False:
                            return self.async_streaming(
                                logging_obj=logging_obj,
                                headers=headers,
                                data=data,
                                model=model,
                                api_base=api_base,
                                api_key=api_key,
                                api_version=api_version,
                                timeout=timeout,
                                client=client,
                                max_retries=max_retries,
                                organization=organization,
                                drop_params=drop_params,
                                stream_options=stream_options,
                            )
                        else:
                            return self.acompletion(
                                data=data,
                                headers=headers,
                                model=model,
                                logging_obj=logging_obj,
                                model_response=model_response,
                                api_base=api_base,
                                api_key=api_key,
                                api_version=api_version,
                                timeout=timeout,
                                client=client,
                                max_retries=max_retries,
                                organization=organization,
                                drop_params=drop_params,
                                fake_stream=fake_stream,
                            )
                    elif stream is True and fake_stream is False:
                        return self.streaming(
                            logging_obj=logging_obj,
                            headers=headers,
                            data=data,
                            model=model,
                            api_base=api_base,
                            api_key=api_key,
                            api_version=api_version,
                            timeout=timeout,
                            client=client,
                            max_retries=max_retries,
                            organization=organization,
                            stream_options=stream_options,
                        )
                    else:
                        if not isinstance(max_retries, int):
                            raise OpenAIError(
                                status_code=422, message="max retries must be an int"
                            )
                        openai_client: OpenAI = self._get_openai_client(  # type: ignore
                            is_async=False,
                            api_key=api_key,
                            api_base=api_base,
                            api_version=api_version,
                            timeout=timeout,
                            max_retries=max_retries,
                            organization=organization,
                            client=client,
                        )

                        ## LOGGING
                        logging_obj.pre_call(
                            input=messages,
                            api_key=openai_client.api_key,
                            additional_args={
                                "headers": headers,
                                "api_base": openai_client._base_url._uri_reference,
                                "acompletion": acompletion,
                                "complete_input_dict": data,
                            },
                        )

                        headers, response = (
                            self.make_sync_openai_chat_completion_request(
                                openai_client=openai_client,
                                data=data,
                                timeout=timeout,
                                logging_obj=logging_obj,
                            )
                        )

                        logging_obj.model_call_details["response_headers"] = headers
                        stringified_response = response.model_dump()
                        logging_obj.post_call(
                            input=messages,
                            api_key=api_key,
                            original_response=stringified_response,
                            additional_args={"complete_input_dict": data},
                        )

                        final_response_obj = convert_to_model_response_object(
                            response_object=stringified_response,
                            model_response_object=model_response,
                            _response_headers=headers,
                        )
                        if fake_stream is True:
                            return self.mock_streaming(
                                response=cast(ModelResponse, final_response_obj),
                                logging_obj=logging_obj,
                                model=model,
                                stream_options=stream_options,
                            )

                        return final_response_obj
                except openai.UnprocessableEntityError as e:
                    ## check if body contains unprocessable params - related issue https://github.com/BerriAI/litellm/issues/4800
                    if litellm.drop_params is True or drop_params is True:
                        inference_params = drop_params_from_unprocessable_entity_error(
                            e, inference_params
                        )
                    else:
                        raise e
                    # e.message
                except Exception as e:
                    if print_verbose is not None:
                        print_verbose(f"openai.py: Received openai error - {str(e)}")
                    if (
                        "Conversation roles must alternate user/assistant" in str(e)
                        or "user and assistant roles should be alternating" in str(e)
                    ) and messages is not None:
                        if print_verbose is not None:
                            print_verbose("openai.py: REFORMATS THE MESSAGE!")
                        # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, add a blank 'user' or 'assistant' message to ensure compatibility
                        new_messages = []
                        for i in range(len(messages) - 1):  # type: ignore
                            new_messages.append(messages[i])
                            if messages[i]["role"] == messages[i + 1]["role"]:
                                if messages[i]["role"] == "user":
                                    new_messages.append(
                                        {"role": "assistant", "content": ""}
                                    )
                                else:
                                    new_messages.append({"role": "user", "content": ""})
                        new_messages.append(messages[-1])
                        messages = new_messages
                    elif (
                        "Last message must have role `user`" in str(e)
                    ) and messages is not None:
                        new_messages = messages
                        new_messages.append({"role": "user", "content": ""})
                        messages = new_messages
                    elif "unknown field: parameter index is not a valid field" in str(
                        e
                    ):
                        litellm.remove_index_from_tool_calls(messages=messages)
                    else:
                        raise e
        except OpenAIError as e:
            raise e
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(
                status_code=status_code, message=error_text, headers=error_headers
            )

    async def acompletion(
        self,
        data: dict,
        model: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        client=None,
        max_retries=None,
        headers=None,
        drop_params: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        fake_stream: bool = False,
    ):
        response = None
        for _ in range(
            2
        ):  # if call fails due to alternating messages, retry with reformatted message

            try:
                openai_aclient: AsyncOpenAI = self._get_openai_client(  # type: ignore
                    is_async=True,
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                )

                ## LOGGING
                logging_obj.pre_call(
                    input=data["messages"],
                    api_key=openai_aclient.api_key,
                    additional_args={
                        "headers": {
                            "Authorization": f"Bearer {openai_aclient.api_key}"
                        },
                        "api_base": openai_aclient._base_url._uri_reference,
                        "acompletion": True,
                        "complete_input_dict": data,
                    },
                )

                headers, response = await self.make_openai_chat_completion_request(
                    openai_aclient=openai_aclient,
                    data=data,
                    timeout=timeout,
                    logging_obj=logging_obj,
                )
                stringified_response = response.model_dump()

                logging_obj.post_call(
                    input=data["messages"],
                    api_key=api_key,
                    original_response=stringified_response,
                    additional_args={"complete_input_dict": data},
                )
                logging_obj.model_call_details["response_headers"] = headers
                final_response_obj = convert_to_model_response_object(
                    response_object=stringified_response,
                    model_response_object=model_response,
                    hidden_params={"headers": headers},
                    _response_headers=headers,
                )

                if fake_stream is True:
                    return self.mock_streaming(
                        response=cast(ModelResponse, final_response_obj),
                        logging_obj=logging_obj,
                        model=model,
                        stream_options=stream_options,
                    )

                return final_response_obj
            except openai.UnprocessableEntityError as e:
                ## check if body contains unprocessable params - related issue https://github.com/BerriAI/litellm/issues/4800
                if litellm.drop_params is True or drop_params is True:
                    data = drop_params_from_unprocessable_entity_error(e, data)
                else:
                    raise e
                # e.message
            except Exception as e:
                exception_response = getattr(e, "response", None)
                status_code = getattr(e, "status_code", 500)
                error_headers = getattr(e, "headers", None)
                if error_headers is None and exception_response:
                    error_headers = getattr(exception_response, "headers", None)
                message = getattr(e, "message", str(e))

                raise OpenAIError(
                    status_code=status_code, message=message, headers=error_headers
                )

    def streaming(
        self,
        logging_obj,
        timeout: Union[float, httpx.Timeout],
        data: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        client=None,
        max_retries=None,
        headers=None,
        stream_options: Optional[dict] = None,
    ):
        data["stream"] = True
        data.update(
            self.get_stream_options(stream_options=stream_options, api_base=api_base)
        )

        openai_client: OpenAI = self._get_openai_client(  # type: ignore
            is_async=False,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )
        ## LOGGING
        logging_obj.pre_call(
            input=data["messages"],
            api_key=api_key,
            additional_args={
                "headers": {"Authorization": f"Bearer {openai_client.api_key}"},
                "api_base": openai_client._base_url._uri_reference,
                "acompletion": False,
                "complete_input_dict": data,
            },
        )
        headers, response = self.make_sync_openai_chat_completion_request(
            openai_client=openai_client,
            data=data,
            timeout=timeout,
            logging_obj=logging_obj,
        )

        logging_obj.model_call_details["response_headers"] = headers
        streamwrapper = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="openai",
            logging_obj=logging_obj,
            stream_options=data.get("stream_options", None),
            _response_headers=headers,
        )
        return streamwrapper

    async def async_streaming(
        self,
        timeout: Union[float, httpx.Timeout],
        data: dict,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        client=None,
        max_retries=None,
        headers=None,
        drop_params: Optional[bool] = None,
        stream_options: Optional[dict] = None,
    ):
        response = None
        data["stream"] = True
        data.update(
            self.get_stream_options(stream_options=stream_options, api_base=api_base)
        )
        for _ in range(2):
            try:
                openai_aclient: AsyncOpenAI = self._get_openai_client(  # type: ignore
                    is_async=True,
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                )
                ## LOGGING
                logging_obj.pre_call(
                    input=data["messages"],
                    api_key=api_key,
                    additional_args={
                        "headers": headers,
                        "api_base": api_base,
                        "acompletion": True,
                        "complete_input_dict": data,
                    },
                )

                headers, response = await self.make_openai_chat_completion_request(
                    openai_aclient=openai_aclient,
                    data=data,
                    timeout=timeout,
                    logging_obj=logging_obj,
                )
                logging_obj.model_call_details["response_headers"] = headers
                streamwrapper = CustomStreamWrapper(
                    completion_stream=response,
                    model=model,
                    custom_llm_provider="openai",
                    logging_obj=logging_obj,
                    stream_options=data.get("stream_options", None),
                    _response_headers=headers,
                )
                return streamwrapper
            except openai.UnprocessableEntityError as e:
                ## check if body contains unprocessable params - related issue https://github.com/BerriAI/litellm/issues/4800
                if litellm.drop_params is True or drop_params is True:
                    data = drop_params_from_unprocessable_entity_error(e, data)
                else:
                    raise e
            except (
                Exception
            ) as e:  # need to exception handle here. async exceptions don't get caught in sync functions.

                if isinstance(e, OpenAIError):
                    raise e

                error_headers = getattr(e, "headers", None)
                status_code = getattr(e, "status_code", 500)
                error_response = getattr(e, "response", None)
                if error_headers is None and error_response:
                    error_headers = getattr(error_response, "headers", None)
                if response is not None and hasattr(response, "text"):
                    raise OpenAIError(
                        status_code=status_code,
                        message=f"{str(e)}\n\nOriginal Response: {response.text}",  # type: ignore
                        headers=error_headers,
                    )
                else:
                    if type(e).__name__ == "ReadTimeout":
                        raise OpenAIError(
                            status_code=408,
                            message=f"{type(e).__name__}",
                            headers=error_headers,
                        )
                    elif hasattr(e, "status_code"):
                        raise OpenAIError(
                            status_code=getattr(e, "status_code", 500),
                            message=str(e),
                            headers=error_headers,
                        )
                    else:
                        raise OpenAIError(
                            status_code=500, message=f"{str(e)}", headers=error_headers
                        )

    def get_stream_options(
        self, stream_options: Optional[dict], api_base: Optional[str]
    ) -> dict:
        """
        Pass `stream_options` to the data dict for OpenAI requests
        """
        if stream_options is not None:
            return {"stream_options": stream_options}
        else:
            # by default litellm will include usage for openai endpoints
            if api_base is None or urlparse(api_base).hostname == "api.openai.com":
                return {"stream_options": {"include_usage": True}}
        return {}

    # Embedding
    @track_llm_api_timing()
    async def make_openai_embedding_request(
        self,
        openai_aclient: AsyncOpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
    ):
        """
        Helper to:
        - call embeddings.create.with_raw_response when litellm.return_response_headers is True
        - call embeddings.create by default
        """
        try:
            raw_response = await openai_aclient.embeddings.with_raw_response.create(
                **data, timeout=timeout
            )  # type: ignore
            headers = dict(raw_response.headers)
            response = raw_response.parse()
            return headers, response
        except Exception as e:
            raise e

    @track_llm_api_timing()
    def make_sync_openai_embedding_request(
        self,
        openai_client: OpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
    ):
        """
        Helper to:
        - call embeddings.create.with_raw_response when litellm.return_response_headers is True
        - call embeddings.create by default
        """
        try:
            raw_response = openai_client.embeddings.with_raw_response.create(
                **data, timeout=timeout
            )  # type: ignore

            headers = dict(raw_response.headers)
            response = raw_response.parse()
            return headers, response
        except Exception as e:
            raise e

    async def aembedding(
        self,
        input: list,
        data: dict,
        model_response: EmbeddingResponse,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        max_retries=None,
    ):
        try:
            openai_aclient: AsyncOpenAI = self._get_openai_client(  # type: ignore
                is_async=True,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )
            headers, response = await self.make_openai_embedding_request(
                openai_aclient=openai_aclient,
                data=data,
                timeout=timeout,
                logging_obj=logging_obj,
            )
            logging_obj.model_call_details["response_headers"] = headers
            stringified_response = response.model_dump()
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=stringified_response,
            )
            returned_response: EmbeddingResponse = convert_to_model_response_object(
                response_object=stringified_response,
                model_response_object=model_response,
                response_type="embedding",
                _response_headers=headers,
            )  # type: ignore
            return returned_response
        except OpenAIError as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            raise e
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(
                status_code=status_code, message=error_text, headers=error_headers
            )

    def embedding(  # type: ignore
        self,
        model: str,
        input: list,
        timeout: float,
        logging_obj,
        model_response: EmbeddingResponse,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
        aembedding=None,
        max_retries: Optional[int] = None,
    ) -> EmbeddingResponse:
        super().embedding()
        try:
            model = model
            data = {"model": model, "input": input, **optional_params}
            max_retries = max_retries or litellm.DEFAULT_MAX_RETRIES
            if not isinstance(max_retries, int):
                raise OpenAIError(status_code=422, message="max retries must be an int")
            ## LOGGING
            logging_obj.pre_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data, "api_base": api_base},
            )

            if aembedding is True:
                return self.aembedding(  # type: ignore
                    data=data,
                    input=input,
                    logging_obj=logging_obj,
                    model_response=model_response,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    client=client,
                    max_retries=max_retries,
                )

            openai_client: OpenAI = self._get_openai_client(  # type: ignore
                is_async=False,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )

            ## embedding CALL
            headers: Optional[Dict] = None
            headers, sync_embedding_response = self.make_sync_openai_embedding_request(
                openai_client=openai_client,
                data=data,
                timeout=timeout,
                logging_obj=logging_obj,
            )  # type: ignore

            ## LOGGING
            logging_obj.model_call_details["response_headers"] = headers
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=sync_embedding_response,
            )
            response: EmbeddingResponse = convert_to_model_response_object(
                response_object=sync_embedding_response.model_dump(),
                model_response_object=model_response,
                _response_headers=headers,
                response_type="embedding",
            )  # type: ignore
            return response
        except OpenAIError as e:
            raise e
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(
                status_code=status_code, message=error_text, headers=error_headers
            )

    async def aimage_generation(
        self,
        prompt: str,
        data: dict,
        model_response: ModelResponse,
        timeout: float,
        logging_obj: Any,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
        max_retries=None,
    ):
        response = None
        try:

            openai_aclient = self._get_openai_client(
                is_async=True,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )

            response = await openai_aclient.images.generate(**data, timeout=timeout)  # type: ignore
            stringified_response = response.model_dump()
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=stringified_response,
            )
            return convert_to_model_response_object(response_object=stringified_response, model_response_object=model_response, response_type="image_generation")  # type: ignore
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                original_response=str(e),
            )
            raise e

    def image_generation(
        self,
        model: Optional[str],
        prompt: str,
        timeout: float,
        optional_params: dict,
        logging_obj: Any,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_response: Optional[ImageResponse] = None,
        client=None,
        aimg_generation=None,
    ) -> ImageResponse:
        data = {}
        try:
            model = model
            data = {"model": model, "prompt": prompt, **optional_params}
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int):
                raise OpenAIError(status_code=422, message="max retries must be an int")

            if aimg_generation is True:
                return self.aimage_generation(data=data, prompt=prompt, logging_obj=logging_obj, model_response=model_response, api_base=api_base, api_key=api_key, timeout=timeout, client=client, max_retries=max_retries)  # type: ignore

            openai_client: OpenAI = self._get_openai_client(  # type: ignore
                is_async=False,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=openai_client.api_key,
                additional_args={
                    "headers": {"Authorization": f"Bearer {openai_client.api_key}"},
                    "api_base": openai_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )

            ## COMPLETION CALL
            _response = openai_client.images.generate(**data, timeout=timeout)  # type: ignore

            response = _response.model_dump()
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response,
            )
            return convert_to_model_response_object(response_object=response, model_response_object=model_response, response_type="image_generation")  # type: ignore
        except OpenAIError as e:

            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            raise e
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            if hasattr(e, "status_code"):
                raise OpenAIError(
                    status_code=getattr(e, "status_code", 500), message=str(e)
                )
            else:
                raise OpenAIError(status_code=500, message=str(e))

    def audio_speech(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        organization: Optional[str],
        project: Optional[str],
        max_retries: int,
        timeout: Union[float, httpx.Timeout],
        aspeech: Optional[bool] = None,
        client=None,
    ) -> HttpxBinaryResponseContent:

        if aspeech is not None and aspeech is True:
            return self.async_audio_speech(
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
                client=client,
            )  # type: ignore

        openai_client = self._get_openai_client(
            is_async=False,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

        response = cast(OpenAI, openai_client).audio.speech.create(
            model=model,
            voice=voice,  # type: ignore
            input=input,
            **optional_params,
        )
        return HttpxBinaryResponseContent(response=response.response)

    async def async_audio_speech(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        organization: Optional[str],
        project: Optional[str],
        max_retries: int,
        timeout: Union[float, httpx.Timeout],
        client=None,
    ) -> HttpxBinaryResponseContent:

        openai_client = cast(
            AsyncOpenAI,
            self._get_openai_client(
                is_async=True,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            ),
        )

        response = await openai_client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore
            input=input,
            **optional_params,
        )

        return HttpxBinaryResponseContent(response=response.response)


class OpenAIFilesAPI(BaseLLM):
    """
    OpenAI methods to support for batches
    - create_file()
    - retrieve_file()
    - list_files()
    - delete_file()
    - file_content()
    - update_file()
    """

    def __init__(self) -> None:
        super().__init__()

    def get_openai_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
        _is_async: bool = False,
    ) -> Optional[Union[OpenAI, AsyncOpenAI]]:
        received_args = locals()
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = None
        if client is None:
            data = {}
            for k, v in received_args.items():
                if k == "self" or k == "client" or k == "_is_async":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            if _is_async is True:
                openai_client = AsyncOpenAI(**data)
            else:
                openai_client = OpenAI(**data)  # type: ignore
        else:
            openai_client = client

        return openai_client

    async def acreate_file(
        self,
        create_file_data: CreateFileRequest,
        openai_client: AsyncOpenAI,
    ) -> FileObject:
        response = await openai_client.files.create(**create_file_data)
        return response

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: str,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ) -> Union[FileObject, Coroutine[Any, Any, FileObject]]:
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acreate_file(  # type: ignore
                create_file_data=create_file_data, openai_client=openai_client
            )
        response = openai_client.files.create(**create_file_data)
        return response

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        openai_client: AsyncOpenAI,
    ) -> HttpxBinaryResponseContent:
        response = await openai_client.files.content(**file_content_request)
        return HttpxBinaryResponseContent(response=response.response)

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: str,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ) -> Union[
        HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]
    ]:
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.afile_content(  # type: ignore
                file_content_request=file_content_request,
                openai_client=openai_client,
            )
        response = cast(OpenAI, openai_client).files.content(**file_content_request)

        return HttpxBinaryResponseContent(response=response.response)

    async def aretrieve_file(
        self,
        file_id: str,
        openai_client: AsyncOpenAI,
    ) -> FileObject:
        response = await openai_client.files.retrieve(file_id=file_id)
        return response

    def retrieve_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: str,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ):
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.aretrieve_file(  # type: ignore
                file_id=file_id,
                openai_client=openai_client,
            )
        response = openai_client.files.retrieve(file_id=file_id)

        return response

    async def adelete_file(
        self,
        file_id: str,
        openai_client: AsyncOpenAI,
    ) -> FileDeleted:
        response = await openai_client.files.delete(file_id=file_id)
        return response

    def delete_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: str,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ):
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.adelete_file(  # type: ignore
                file_id=file_id,
                openai_client=openai_client,
            )
        response = openai_client.files.delete(file_id=file_id)

        return response

    async def alist_files(
        self,
        openai_client: AsyncOpenAI,
        purpose: Optional[str] = None,
    ):
        if isinstance(purpose, str):
            response = await openai_client.files.list(purpose=purpose)
        else:
            response = await openai_client.files.list()
        return response

    def list_files(
        self,
        _is_async: bool,
        api_base: str,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        purpose: Optional[str] = None,
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ):
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.alist_files(  # type: ignore
                purpose=purpose,
                openai_client=openai_client,
            )

        if isinstance(purpose, str):
            response = openai_client.files.list(purpose=purpose)
        else:
            response = openai_client.files.list()

        return response


class OpenAIBatchesAPI(BaseLLM):
    """
    OpenAI methods to support for batches
    - create_batch()
    - retrieve_batch()
    - cancel_batch()
    - list_batch()
    """

    def __init__(self) -> None:
        super().__init__()

    def get_openai_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
        _is_async: bool = False,
    ) -> Optional[Union[OpenAI, AsyncOpenAI]]:
        received_args = locals()
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = None
        if client is None:
            data = {}
            for k, v in received_args.items():
                if k == "self" or k == "client" or k == "_is_async":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            if _is_async is True:
                openai_client = AsyncOpenAI(**data)
            else:
                openai_client = OpenAI(**data)  # type: ignore
        else:
            openai_client = client

        return openai_client

    async def acreate_batch(
        self,
        create_batch_data: CreateBatchRequest,
        openai_client: AsyncOpenAI,
    ) -> Batch:
        response = await openai_client.batches.create(**create_batch_data)
        return response

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ) -> Union[Batch, Coroutine[Any, Any, Batch]]:
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acreate_batch(  # type: ignore
                create_batch_data=create_batch_data, openai_client=openai_client
            )
        response = openai_client.batches.create(**create_batch_data)
        return response

    async def aretrieve_batch(
        self,
        retrieve_batch_data: RetrieveBatchRequest,
        openai_client: AsyncOpenAI,
    ) -> Batch:
        verbose_logger.debug("retrieving batch, args= %s", retrieve_batch_data)
        response = await openai_client.batches.retrieve(**retrieve_batch_data)
        return response

    def retrieve_batch(
        self,
        _is_async: bool,
        retrieve_batch_data: RetrieveBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[OpenAI] = None,
    ):
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.aretrieve_batch(  # type: ignore
                retrieve_batch_data=retrieve_batch_data, openai_client=openai_client
            )
        response = openai_client.batches.retrieve(**retrieve_batch_data)
        return response

    async def acancel_batch(
        self,
        cancel_batch_data: CancelBatchRequest,
        openai_client: AsyncOpenAI,
    ) -> Batch:
        verbose_logger.debug("async cancelling batch, args= %s", cancel_batch_data)
        response = await openai_client.batches.cancel(**cancel_batch_data)
        return response

    def cancel_batch(
        self,
        _is_async: bool,
        cancel_batch_data: CancelBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[OpenAI] = None,
    ):
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acancel_batch(  # type: ignore
                cancel_batch_data=cancel_batch_data, openai_client=openai_client
            )

        response = openai_client.batches.cancel(**cancel_batch_data)
        return response

    async def alist_batches(
        self,
        openai_client: AsyncOpenAI,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        verbose_logger.debug("listing batches, after= %s, limit= %s", after, limit)
        response = await openai_client.batches.list(after=after, limit=limit)  # type: ignore
        return response

    def list_batches(
        self,
        _is_async: bool,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        after: Optional[str] = None,
        limit: Optional[int] = None,
        client: Optional[OpenAI] = None,
    ):
        openai_client: Optional[Union[OpenAI, AsyncOpenAI]] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.alist_batches(  # type: ignore
                openai_client=openai_client, after=after, limit=limit
            )
        response = openai_client.batches.list(after=after, limit=limit)  # type: ignore
        return response


class OpenAIAssistantsAPI(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def get_openai_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[OpenAI] = None,
    ) -> OpenAI:
        received_args = locals()
        if client is None:
            data = {}
            for k, v in received_args.items():
                if k == "self" or k == "client":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            openai_client = OpenAI(**data)  # type: ignore
        else:
            openai_client = client

        return openai_client

    def async_get_openai_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI] = None,
    ) -> AsyncOpenAI:
        received_args = locals()
        if client is None:
            data = {}
            for k, v in received_args.items():
                if k == "self" or k == "client":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            openai_client = AsyncOpenAI(**data)  # type: ignore
        else:
            openai_client = client

        return openai_client

    ### ASSISTANTS ###

    async def async_get_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
        order: Optional[str] = "desc",
        limit: Optional[int] = 20,
        before: Optional[str] = None,
        after: Optional[str] = None,
    ) -> AsyncCursorPage[Assistant]:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )
        request_params = {
            "order": order,
            "limit": limit,
        }
        if before:
            request_params["before"] = before
        if after:
            request_params["after"] = after

        response = await openai_client.beta.assistants.list(**request_params)  # type: ignore

        return response

    # fmt: off

    @overload
    def get_assistants(
        self, 
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
        aget_assistants: Literal[True], 
    ) -> Coroutine[None, None, AsyncCursorPage[Assistant]]:
        ...

    @overload
    def get_assistants(
        self, 
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[OpenAI],
        aget_assistants: Optional[Literal[False]], 
    ) -> SyncCursorPage[Assistant]: 
        ...

    # fmt: on

    def get_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client=None,
        aget_assistants=None,
        order: Optional[str] = "desc",
        limit: Optional[int] = 20,
        before: Optional[str] = None,
        after: Optional[str] = None,
    ):
        if aget_assistants is not None and aget_assistants is True:
            return self.async_get_assistants(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        request_params = {
            "order": order,
            "limit": limit,
        }

        if before:
            request_params["before"] = before
        if after:
            request_params["after"] = after

        response = openai_client.beta.assistants.list(**request_params)  # type: ignore

        return response

    # Create Assistant
    async def async_create_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
        create_assistant_data: dict,
    ) -> Assistant:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = await openai_client.beta.assistants.create(**create_assistant_data)

        return response

    def create_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        create_assistant_data: dict,
        client=None,
        async_create_assistants=None,
    ):
        if async_create_assistants is not None and async_create_assistants is True:
            return self.async_create_assistants(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
                create_assistant_data=create_assistant_data,
            )
        openai_client = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = openai_client.beta.assistants.create(**create_assistant_data)
        return response

    # Delete Assistant
    async def async_delete_assistant(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
        assistant_id: str,
    ) -> AssistantDeleted:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = await openai_client.beta.assistants.delete(assistant_id=assistant_id)

        return response

    def delete_assistant(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        assistant_id: str,
        client=None,
        async_delete_assistants=None,
    ):
        if async_delete_assistants is not None and async_delete_assistants is True:
            return self.async_delete_assistant(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
                assistant_id=assistant_id,
            )
        openai_client = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = openai_client.beta.assistants.delete(assistant_id=assistant_id)
        return response

    ### MESSAGES ###

    async def a_add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI] = None,
    ) -> OpenAIMessage:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        thread_message: OpenAIMessage = await openai_client.beta.threads.messages.create(  # type: ignore
            thread_id, **message_data  # type: ignore
        )

        response_obj: Optional[OpenAIMessage] = None
        if getattr(thread_message, "status", None) is None:
            thread_message.status = "completed"
            response_obj = OpenAIMessage(**thread_message.dict())
        else:
            response_obj = OpenAIMessage(**thread_message.dict())
        return response_obj

    # fmt: off

    @overload
    def add_message(
        self, 
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
        a_add_message: Literal[True], 
    ) -> Coroutine[None, None, OpenAIMessage]:
        ...

    @overload
    def add_message(
        self, 
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[OpenAI],
        a_add_message: Optional[Literal[False]], 
    ) -> OpenAIMessage: 
        ...

    # fmt: on

    def add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client=None,
        a_add_message: Optional[bool] = None,
    ):
        if a_add_message is not None and a_add_message is True:
            return self.a_add_message(
                thread_id=thread_id,
                message_data=message_data,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        thread_message: OpenAIMessage = openai_client.beta.threads.messages.create(  # type: ignore
            thread_id, **message_data  # type: ignore
        )

        response_obj: Optional[OpenAIMessage] = None
        if getattr(thread_message, "status", None) is None:
            thread_message.status = "completed"
            response_obj = OpenAIMessage(**thread_message.dict())
        else:
            response_obj = OpenAIMessage(**thread_message.dict())
        return response_obj

    async def async_get_messages(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI] = None,
    ) -> AsyncCursorPage[OpenAIMessage]:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = await openai_client.beta.threads.messages.list(thread_id=thread_id)

        return response

    # fmt: off

    @overload
    def get_messages(
        self, 
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
        aget_messages: Literal[True], 
    ) -> Coroutine[None, None, AsyncCursorPage[OpenAIMessage]]:
        ...

    @overload
    def get_messages(
        self, 
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[OpenAI],
        aget_messages: Optional[Literal[False]], 
    ) -> SyncCursorPage[OpenAIMessage]: 
        ...

    # fmt: on

    def get_messages(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client=None,
        aget_messages=None,
    ):
        if aget_messages is not None and aget_messages is True:
            return self.async_get_messages(
                thread_id=thread_id,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = openai_client.beta.threads.messages.list(thread_id=thread_id)

        return response

    ### THREADS ###

    async def async_create_thread(
        self,
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
    ) -> Thread:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        data = {}
        if messages is not None:
            data["messages"] = messages  # type: ignore
        if metadata is not None:
            data["metadata"] = metadata  # type: ignore

        message_thread = await openai_client.beta.threads.create(**data)  # type: ignore

        return Thread(**message_thread.dict())

    # fmt: off

    @overload
    def create_thread(
        self, 
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client: Optional[AsyncOpenAI],
        acreate_thread: Literal[True], 
    ) -> Coroutine[None, None, Thread]:
        ...

    @overload
    def create_thread(
        self, 
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client: Optional[OpenAI],
        acreate_thread: Optional[Literal[False]], 
    ) -> Thread: 
        ...

    # fmt: on

    def create_thread(
        self,
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client=None,
        acreate_thread=None,
    ):
        """
        Here's an example:
        ```
        from litellm.llms.openai.openai import OpenAIAssistantsAPI, MessageData

        # create thread
        message: MessageData = {"role": "user", "content": "Hey, how's it going?"}
        openai_api.create_thread(messages=[message])
        ```
        """
        if acreate_thread is not None and acreate_thread is True:
            return self.async_create_thread(
                metadata=metadata,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
                messages=messages,
            )
        openai_client = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        data = {}
        if messages is not None:
            data["messages"] = messages  # type: ignore
        if metadata is not None:
            data["metadata"] = metadata  # type: ignore

        message_thread = openai_client.beta.threads.create(**data)  # type: ignore

        return Thread(**message_thread.dict())

    async def async_get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
    ) -> Thread:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = await openai_client.beta.threads.retrieve(thread_id=thread_id)

        return Thread(**response.dict())

    # fmt: off

    @overload
    def get_thread(
        self, 
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
        aget_thread: Literal[True], 
    ) -> Coroutine[None, None, Thread]:
        ...

    @overload
    def get_thread(
        self, 
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[OpenAI],
        aget_thread: Optional[Literal[False]], 
    ) -> Thread: 
        ...

    # fmt: on

    def get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client=None,
        aget_thread=None,
    ):
        if aget_thread is not None and aget_thread is True:
            return self.async_get_thread(
                thread_id=thread_id,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = openai_client.beta.threads.retrieve(thread_id=thread_id)

        return Thread(**response.dict())

    def delete_thread(self):
        pass

    ### RUNS ###

    async def arun_thread(
        self,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[object],
        model: Optional[str],
        stream: Optional[bool],
        tools: Optional[Iterable[AssistantToolParam]],
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[AsyncOpenAI],
    ) -> Run:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = await openai_client.beta.threads.runs.create_and_poll(  # type: ignore
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            tools=tools,
        )

        return response

    def async_run_thread_stream(
        self,
        client: AsyncOpenAI,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[object],
        model: Optional[str],
        tools: Optional[Iterable[AssistantToolParam]],
        event_handler: Optional[AssistantEventHandler],
    ) -> AsyncAssistantStreamManager[AsyncAssistantEventHandler]:
        data = {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "additional_instructions": additional_instructions,
            "instructions": instructions,
            "metadata": metadata,
            "model": model,
            "tools": tools,
        }
        if event_handler is not None:
            data["event_handler"] = event_handler
        return client.beta.threads.runs.stream(**data)  # type: ignore

    def run_thread_stream(
        self,
        client: OpenAI,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[object],
        model: Optional[str],
        tools: Optional[Iterable[AssistantToolParam]],
        event_handler: Optional[AssistantEventHandler],
    ) -> AssistantStreamManager[AssistantEventHandler]:
        data = {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "additional_instructions": additional_instructions,
            "instructions": instructions,
            "metadata": metadata,
            "model": model,
            "tools": tools,
        }
        if event_handler is not None:
            data["event_handler"] = event_handler
        return client.beta.threads.runs.stream(**data)  # type: ignore

    # fmt: off

    @overload
    def run_thread(
        self, 
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[object],
        model: Optional[str],
        stream: Optional[bool],
        tools: Optional[Iterable[AssistantToolParam]],
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client,
        arun_thread: Literal[True], 
        event_handler: Optional[AssistantEventHandler],
    ) -> Coroutine[None, None, Run]:
        ...

    @overload
    def run_thread(
        self, 
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[object],
        model: Optional[str],
        stream: Optional[bool],
        tools: Optional[Iterable[AssistantToolParam]],
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client,
        arun_thread: Optional[Literal[False]], 
        event_handler: Optional[AssistantEventHandler],
    ) -> Run: 
        ...

    # fmt: on

    def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[object],
        model: Optional[str],
        stream: Optional[bool],
        tools: Optional[Iterable[AssistantToolParam]],
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client=None,
        arun_thread=None,
        event_handler: Optional[AssistantEventHandler] = None,
    ):
        if arun_thread is not None and arun_thread is True:
            if stream is not None and stream is True:
                _client = self.async_get_openai_client(
                    api_key=api_key,
                    api_base=api_base,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                )
                return self.async_run_thread_stream(
                    client=_client,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    additional_instructions=additional_instructions,
                    instructions=instructions,
                    metadata=metadata,
                    model=model,
                    tools=tools,
                    event_handler=event_handler,
                )
            return self.arun_thread(
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=additional_instructions,
                instructions=instructions,
                metadata=metadata,
                model=model,
                stream=stream,
                tools=tools,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        if stream is not None and stream is True:
            return self.run_thread_stream(
                client=openai_client,
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=additional_instructions,
                instructions=instructions,
                metadata=metadata,
                model=model,
                tools=tools,
                event_handler=event_handler,
            )

        response = openai_client.beta.threads.runs.create_and_poll(  # type: ignore
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            tools=tools,
        )

        return response
