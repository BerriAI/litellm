import hashlib
import json
import os
import time
import traceback
import types
from typing import Any, Callable, Coroutine, Iterable, Literal, Optional, Union, cast

import httpx
import openai
from openai import AsyncOpenAI, OpenAI
from openai.types.beta.assistant_deleted import AssistantDeleted
from openai.types.file_deleted import FileDeleted
from pydantic import BaseModel
from typing_extensions import overload, override

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ProviderField
from litellm.utils import (
    Choices,
    CustomStreamWrapper,
    Message,
    ModelResponse,
    TextCompletionResponse,
    Usage,
    convert_to_model_response_object,
)

from ...types.llms.openai import *
from ..base import BaseLLM
from ..prompt_templates.common_utils import convert_content_list_to_str
from ..prompt_templates.factory import custom_prompt, prompt_factory
from .common_utils import drop_params_from_unprocessable_entity_error
from .completion.utils import is_tokens_or_list_of_tokens


class OpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        if response:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


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


class DeepInfraConfig:
    """
    Reference: https://deepinfra.com/docs/advanced/openai_api

    The class `DeepInfra` provides configuration for the DeepInfra's Chat Completions API interface. Below are the parameters:
    """

    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
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
            "stream",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "stop",
            "temperature",
            "top_p",
            "response_format",
            "tools",
            "tool_choice",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params()
        for param, value in non_default_params.items():
            if (
                param == "temperature"
                and value == 0
                and model == "mistralai/Mistral-7B-Instruct-v0.1"
            ):  # this model does no support temperature == 0
                value = 0.0001  # close to 0
            if param == "tool_choice":
                if (
                    value != "auto" and value != "none"
                ):  # https://deepinfra.com/docs/advanced/function_calling
                    ## UNSUPPORTED TOOL CHOICE VALUE
                    if litellm.drop_params is True or drop_params is True:
                        value = None
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message="Deepinfra doesn't support tool_choice={}. To drop unsupported openai params from the call, set `litellm.drop_params = True`".format(
                                value
                            ),
                            status_code=400,
                        )
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value
        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # deepinfra is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = (
            api_base
            or get_secret_str("DEEPINFRA_API_BASE")
            or "https://api.deepinfra.com/v1/openai"
        )
        dynamic_api_key = api_key or get_secret_str("DEEPINFRA_API_KEY")
        return api_base, dynamic_api_key


class OpenAIConfig:
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

    def get_supported_openai_params(self, model: str) -> list:
        """
        This function returns the list of supported openai parameters for a given OpenAI Model

        - If O1 model, returns O1 supported params
        - If gpt-audio model, returns gpt-audio supported params
        - Else, returns gpt supported params

        Args:
            model (str): OpenAI model

        Returns:
            list: List of supported openai parameters
        """
        if litellm.openAIO1Config.is_model_o1_reasoning_model(model=model):
            return litellm.openAIO1Config.get_supported_openai_params(model=model)
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

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """ """
        if litellm.openAIO1Config.is_model_o1_reasoning_model(model=model):
            return litellm.openAIO1Config.map_openai_params(
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


class OpenAITextCompletionConfig:
    """
    Reference: https://platform.openai.com/docs/api-reference/completions/create

    The class `OpenAITextCompletionConfig` provides configuration for the OpenAI's text completion API interface. Below are the parameters:

    - `best_of` (integer or null): This optional parameter generates server-side completions and returns the one with the highest log probability per token.

    - `echo` (boolean or null): This optional parameter will echo back the prompt in addition to the completion.

    - `frequency_penalty` (number or null): Defaults to 0. It is a numbers from -2.0 to 2.0, where positive values decrease the model's likelihood to repeat the same line.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `logprobs` (integer or null): This optional parameter includes the log probabilities on the most likely tokens as well as the chosen tokens.

    - `max_tokens` (integer or null): This optional parameter sets the maximum number of tokens to generate in the completion.

    - `n` (integer or null): This optional parameter sets how many completions to generate for each prompt.

    - `presence_penalty` (number or null): Defaults to 0 and can be between -2.0 and 2.0. Positive values increase the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `suffix` (string or null): Defines the suffix that comes after a completion of inserted text.

    - `temperature` (number or null): This optional parameter defines the sampling temperature to use.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """

    best_of: Optional[int] = None
    echo: Optional[bool] = None
    frequency_penalty: Optional[int] = None
    logit_bias: Optional[dict] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    def __init__(
        self,
        best_of: Optional[int] = None,
        echo: Optional[bool] = None,
        frequency_penalty: Optional[int] = None,
        logit_bias: Optional[dict] = None,
        logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
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

    def _transform_prompt(
        self,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
    ) -> AllPromptValues:
        if len(messages) == 1:  # base case
            message_content = messages[0].get("content")
            if (
                message_content
                and isinstance(message_content, list)
                and is_tokens_or_list_of_tokens(message_content)
            ):
                openai_prompt: AllPromptValues = cast(AllPromptValues, message_content)
            else:
                openai_prompt = ""
                content = convert_content_list_to_str(
                    cast(AllMessageValues, messages[0])
                )
                openai_prompt += content
        else:
            prompt_str_list: List[str] = []
            for m in messages:
                try:  # expect list of int/list of list of int to be a 1 message array only.
                    content = convert_content_list_to_str(cast(AllMessageValues, m))
                    prompt_str_list.append(content)
                except Exception as e:
                    raise e
            openai_prompt = prompt_str_list
        return openai_prompt

    def convert_to_chat_model_response_object(
        self,
        response_object: Optional[TextCompletionResponse] = None,
        model_response_object: Optional[ModelResponse] = None,
    ):
        try:
            ## RESPONSE OBJECT
            if response_object is None or model_response_object is None:
                raise ValueError("Error in response object format")
            choice_list = []
            for idx, choice in enumerate(response_object["choices"]):
                message = Message(
                    content=choice["text"],
                    role="assistant",
                )
                choice = Choices(
                    finish_reason=choice["finish_reason"], index=idx, message=message
                )
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object:
                setattr(model_response_object, "usage", response_object["usage"])

            if "id" in response_object:
                model_response_object.id = response_object["id"]

            if "model" in response_object:
                model_response_object.model = response_object["model"]

            model_response_object._hidden_params["original_response"] = (
                response_object  # track original response, if users make a litellm.text_completion() request, we can return the original response
            )
            return model_response_object
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

            if _cache_key in litellm.in_memory_llm_clients_cache:
                return litellm.in_memory_llm_clients_cache[_cache_key]
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
            litellm.in_memory_llm_clients_cache[_cache_key] = _new_client
            return _new_client

        else:
            return client

    async def make_openai_chat_completion_request(
        self,
        openai_aclient: AsyncOpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
    ) -> Tuple[dict, BaseModel]:
        """
        Helper to:
        - call chat.completions.create.with_raw_response when litellm.return_response_headers is True
        - call chat.completions.create by default
        """
        try:
            raw_response = (
                await openai_aclient.chat.completions.with_raw_response.create(
                    **data, timeout=timeout
                )
            )

            if hasattr(raw_response, "headers"):
                headers = dict(raw_response.headers)
            else:
                headers = {}
            response = raw_response.parse()
            return headers, response
        except Exception as e:
            raise e

    def make_sync_openai_chat_completion_request(
        self,
        openai_client: OpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
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

    def completion(  # type: ignore # noqa: PLR0915
        self,
        model_response: ModelResponse,
        timeout: Union[float, httpx.Timeout],
        optional_params: dict,
        logging_obj: Any,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        print_verbose: Optional[Callable] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        acompletion: bool = False,
        litellm_params=None,
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
            if headers:
                optional_params["extra_headers"] = headers
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
                # process all OpenAI compatible provider logic here
                if custom_llm_provider == "mistral":
                    # check if message content passed in as list, and not string
                    messages = prompt_factory(  # type: ignore
                        model=model,
                        messages=messages,
                        custom_llm_provider=custom_llm_provider,
                    )
                if custom_llm_provider == "perplexity" and messages is not None:
                    # check if messages.name is passed + supported, if not supported remove
                    messages = prompt_factory(  # type: ignore
                        model=model,
                        messages=messages,
                        custom_llm_provider=custom_llm_provider,
                    )
            if (
                litellm.openAIO1Config.is_model_o1_reasoning_model(model=model)
                and messages is not None
            ):
                messages = litellm.openAIO1Config.o1_prompt_factory(
                    messages=messages,
                )

            for _ in range(
                2
            ):  # if call fails due to alternating messages, retry with reformatted message
                data = {"model": model, "messages": messages, **optional_params}

                try:
                    max_retries = data.pop("max_retries", 2)
                    if acompletion is True:
                        if optional_params.get("stream", False):
                            return self.async_streaming(
                                logging_obj=logging_obj,
                                headers=headers,
                                data=data,
                                model=model,
                                api_base=api_base,
                                api_key=api_key,
                                timeout=timeout,
                                client=client,
                                max_retries=max_retries,
                                organization=organization,
                                drop_params=drop_params,
                            )
                        else:
                            return self.acompletion(
                                data=data,
                                headers=headers,
                                logging_obj=logging_obj,
                                model_response=model_response,
                                api_base=api_base,
                                api_key=api_key,
                                timeout=timeout,
                                client=client,
                                max_retries=max_retries,
                                organization=organization,
                                drop_params=drop_params,
                            )
                    elif optional_params.get("stream", False):
                        return self.streaming(
                            logging_obj=logging_obj,
                            headers=headers,
                            data=data,
                            model=model,
                            api_base=api_base,
                            api_key=api_key,
                            timeout=timeout,
                            client=client,
                            max_retries=max_retries,
                            organization=organization,
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
                        return convert_to_model_response_object(
                            response_object=stringified_response,
                            model_response_object=model_response,
                            _response_headers=headers,
                        )
                except openai.UnprocessableEntityError as e:
                    ## check if body contains unprocessable params - related issue https://github.com/BerriAI/litellm/issues/4800
                    if litellm.drop_params is True or drop_params is True:
                        optional_params = drop_params_from_unprocessable_entity_error(
                            e, optional_params
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
                    elif (
                        "unknown field: parameter index is not a valid field" in str(e)
                    ) and "tools" in data:
                        litellm.remove_index_from_tool_calls(
                            tool_calls=data["tools"], messages=messages
                        )
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
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        organization: Optional[str] = None,
        client=None,
        max_retries=None,
        headers=None,
        drop_params: Optional[bool] = None,
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
                    openai_aclient=openai_aclient, data=data, timeout=timeout
                )
                stringified_response = response.model_dump()
                logging_obj.post_call(
                    input=data["messages"],
                    api_key=api_key,
                    original_response=stringified_response,
                    additional_args={"complete_input_dict": data},
                )
                logging_obj.model_call_details["response_headers"] = headers
                return convert_to_model_response_object(
                    response_object=stringified_response,
                    model_response_object=model_response,
                    hidden_params={"headers": headers},
                    _response_headers=headers,
                )
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

                raise OpenAIError(
                    status_code=status_code, message=str(e), headers=error_headers
                )

    def streaming(
        self,
        logging_obj,
        timeout: Union[float, httpx.Timeout],
        data: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        organization: Optional[str] = None,
        client=None,
        max_retries=None,
        headers=None,
    ):
        openai_client: OpenAI = self._get_openai_client(  # type: ignore
            is_async=False,
            api_key=api_key,
            api_base=api_base,
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
        organization: Optional[str] = None,
        client=None,
        max_retries=None,
        headers=None,
        drop_params: Optional[bool] = None,
    ):
        response = None
        for _ in range(2):
            try:
                openai_aclient: AsyncOpenAI = self._get_openai_client(  # type: ignore
                    is_async=True,
                    api_key=api_key,
                    api_base=api_base,
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
                    openai_aclient=openai_aclient, data=data, timeout=timeout
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

    # Embedding
    async def make_openai_embedding_request(
        self,
        openai_aclient: AsyncOpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
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

    def make_sync_openai_embedding_request(
        self,
        openai_client: OpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
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
        model_response: litellm.utils.EmbeddingResponse,
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
                openai_aclient=openai_aclient, data=data, timeout=timeout
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
            returned_response: (
                litellm.EmbeddingResponse
            ) = convert_to_model_response_object(
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
        model_response: litellm.utils.EmbeddingResponse,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
        aembedding=None,
    ) -> litellm.EmbeddingResponse:
        super().embedding()
        try:
            model = model
            data = {"model": model, "input": input, **optional_params}
            max_retries = data.pop("max_retries", 2)
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
                openai_client=openai_client, data=data, timeout=timeout
            )  # type: ignore

            ## LOGGING
            logging_obj.model_call_details["response_headers"] = headers
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=sync_embedding_response,
            )
            response: litellm.EmbeddingResponse = convert_to_model_response_object(
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
                input=input,
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
        model_response: Optional[litellm.utils.ImageResponse] = None,
        client=None,
        aimg_generation=None,
    ) -> litellm.ImageResponse:
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

        response = openai_client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore
            input=input,
            **optional_params,
        )
        return response  # type: ignore

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

        openai_client = self._get_openai_client(
            is_async=True,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

        response = await openai_client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore
            input=input,
            **optional_params,
        )

        return response

    async def ahealth_check(
        self,
        model: Optional[str],
        api_key: Optional[str],
        timeout: float,
        mode: str,
        messages: Optional[list] = None,
        input: Optional[list] = None,
        prompt: Optional[str] = None,
        organization: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            organization=organization,
            base_url=api_base,
        )
        if model is None and mode != "image_generation":
            raise Exception("model is not set")

        completion = None

        if mode == "completion":
            completion = await client.completions.with_raw_response.create(
                model=model,  # type: ignore
                prompt=prompt,  # type: ignore
            )
        elif mode == "chat":
            if messages is None:
                raise Exception("messages is not set")
            completion = await client.chat.completions.with_raw_response.create(
                model=model,  # type: ignore
                messages=messages,  # type: ignore
            )
        elif mode == "embedding":
            if input is None:
                raise Exception("input is not set")
            completion = await client.embeddings.with_raw_response.create(
                model=model,  # type: ignore
                input=input,  # type: ignore
            )
        elif mode == "image_generation":
            if prompt is None:
                raise Exception("prompt is not set")
            completion = await client.images.with_raw_response.generate(
                model=model,  # type: ignore
                prompt=prompt,  # type: ignore
            )
        elif mode == "audio_transcription":
            # Get the current directory of the file being run
            pwd = os.path.dirname(os.path.realpath(__file__))
            file_path = os.path.join(
                pwd, "../../../tests/gettysburg.wav"
            )  # proxy address
            audio_file = open(file_path, "rb")
            completion = await client.audio.transcriptions.with_raw_response.create(
                file=audio_file,
                model=model,  # type: ignore
                prompt=prompt,  # type: ignore
            )
        elif mode == "audio_speech":
            # Get the current directory of the file being run
            completion = await client.audio.speech.with_raw_response.create(
                model=model,  # type: ignore
                input=prompt,  # type: ignore
                voice="alloy",
            )
        else:
            raise ValueError("mode not set, passed in mode: " + mode)
        response = {}

        if completion is None or not hasattr(completion, "headers"):
            raise Exception("invalid completion response")

        if (
            completion.headers.get("x-ratelimit-remaining-requests", None) is not None
        ):  # not provided for dall-e requests
            response["x-ratelimit-remaining-requests"] = completion.headers[
                "x-ratelimit-remaining-requests"
            ]

        if completion.headers.get("x-ratelimit-remaining-tokens", None) is not None:
            response["x-ratelimit-remaining-tokens"] = completion.headers[
                "x-ratelimit-remaining-tokens"
            ]
        return response


class OpenAITextCompletion(BaseLLM):
    openai_text_completion_global_config = OpenAITextCompletionConfig()

    def __init__(self) -> None:
        super().__init__()

    def validate_environment(self, api_key):
        headers = {
            "content-type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def completion(
        self,
        model_response: ModelResponse,
        api_key: str,
        model: str,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        print_verbose: Optional[Callable] = None,
        api_base: Optional[str] = None,
        acompletion: bool = False,
        litellm_params=None,
        logger_fn=None,
        client=None,
        organization: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        try:
            if headers is None:
                headers = self.validate_environment(api_key=api_key)
            if model is None or messages is None:
                raise OpenAIError(status_code=422, message="Missing model or messages")

            # don't send max retries to the api, if set

            prompt = self.openai_text_completion_global_config._transform_prompt(
                messages
            )

            data = {"model": model, "prompt": prompt, **optional_params}
            max_retries = data.pop("max_retries", 2)
            ## LOGGING
            logging_obj.pre_call(
                input=messages,
                api_key=api_key,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                    "complete_input_dict": data,
                },
            )
            if acompletion is True:
                if optional_params.get("stream", False):
                    return self.async_streaming(
                        logging_obj=logging_obj,
                        api_base=api_base,
                        api_key=api_key,
                        data=data,
                        headers=headers,
                        model_response=model_response,
                        model=model,
                        timeout=timeout,
                        max_retries=max_retries,
                        client=client,
                        organization=organization,
                    )
                else:
                    return self.acompletion(api_base=api_base, data=data, headers=headers, model_response=model_response, prompt=prompt, api_key=api_key, logging_obj=logging_obj, model=model, timeout=timeout, max_retries=max_retries, organization=organization, client=client)  # type: ignore
            elif optional_params.get("stream", False):
                return self.streaming(
                    logging_obj=logging_obj,
                    api_base=api_base,
                    api_key=api_key,
                    data=data,
                    headers=headers,
                    model_response=model_response,
                    model=model,
                    timeout=timeout,
                    max_retries=max_retries,  # type: ignore
                    client=client,
                    organization=organization,
                )
            else:
                if client is None:
                    openai_client = OpenAI(
                        api_key=api_key,
                        base_url=api_base,
                        http_client=litellm.client_session,
                        timeout=timeout,
                        max_retries=max_retries,  # type: ignore
                        organization=organization,
                    )
                else:
                    openai_client = client

                raw_response = openai_client.completions.with_raw_response.create(**data)  # type: ignore
                response = raw_response.parse()
                response_json = response.model_dump()

                ## LOGGING
                logging_obj.post_call(
                    input=prompt,
                    api_key=api_key,
                    original_response=response_json,
                    additional_args={
                        "headers": headers,
                        "api_base": api_base,
                    },
                )

                ## RESPONSE OBJECT
                return TextCompletionResponse(**response_json)
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
        logging_obj,
        api_base: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        prompt: str,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        organization: Optional[str] = None,
        client=None,
    ):
        try:
            if client is None:
                openai_aclient = AsyncOpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=litellm.aclient_session,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                )
            else:
                openai_aclient = client

            raw_response = await openai_aclient.completions.with_raw_response.create(
                **data
            )
            response = raw_response.parse()
            response_json = response.model_dump()

            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                original_response=response,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                },
            )
            ## RESPONSE OBJECT
            response_obj = TextCompletionResponse(**response_json)
            response_obj._hidden_params.original_response = json.dumps(response_json)
            return response_obj
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

    def streaming(
        self,
        logging_obj,
        api_key: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        model: str,
        timeout: float,
        api_base: Optional[str] = None,
        max_retries=None,
        client=None,
        organization=None,
    ):

        if client is None:
            openai_client = OpenAI(
                api_key=api_key,
                base_url=api_base,
                http_client=litellm.client_session,
                timeout=timeout,
                max_retries=max_retries,  # type: ignore
                organization=organization,
            )
        else:
            openai_client = client

        try:
            raw_response = openai_client.completions.with_raw_response.create(**data)
            response = raw_response.parse()
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
        streamwrapper = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="text-completion-openai",
            logging_obj=logging_obj,
            stream_options=data.get("stream_options", None),
        )

        try:
            for chunk in streamwrapper:
                yield chunk
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

    async def async_streaming(
        self,
        logging_obj,
        api_key: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        model: str,
        timeout: float,
        max_retries: int,
        api_base: Optional[str] = None,
        client=None,
        organization=None,
    ):
        if client is None:
            openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                http_client=litellm.aclient_session,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
            )
        else:
            openai_client = client

        raw_response = await openai_client.completions.with_raw_response.create(**data)
        response = raw_response.parse()
        streamwrapper = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="text-completion-openai",
            logging_obj=logging_obj,
            stream_options=data.get("stream_options", None),
        )

        try:
            async for transformed_chunk in streamwrapper:
                yield transformed_chunk
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
        return response

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
        response = openai_client.files.content(**file_content_request)

        return response

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
    ) -> AsyncCursorPage[Assistant]:
        openai_client = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response = await openai_client.beta.assistants.list()

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

        response = openai_client.beta.assistants.list()

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
        from litellm.llms.OpenAI.openai import OpenAIAssistantsAPI, MessageData

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
