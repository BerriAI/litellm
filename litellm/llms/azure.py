from typing import Optional, Union, Any
import types, requests
from .base import BaseLLM
from litellm.utils import (
    ModelResponse,
    Choices,
    Message,
    CustomStreamWrapper,
    convert_to_model_response_object,
    TranscriptionResponse,
)
from typing import Callable, Optional, BinaryIO
from litellm import OpenAIConfig
import litellm, json
import httpx
from .custom_httpx.azure_dall_e_2 import CustomHTTPTransport, AsyncCustomHTTPTransport
from openai import AzureOpenAI, AsyncAzureOpenAI
import uuid


class AzureOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ):
        self.status_code = status_code
        self.message = message
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


class AzureOpenAIConfig(OpenAIConfig):
    """
    Reference: https://platform.openai.com/docs/api-reference/chat/create

    The class `AzureOpenAIConfig` provides configuration for the OpenAI's Chat API interface, for use with Azure. It inherits from `OpenAIConfig`. Below are the parameters::

    - `frequency_penalty` (number or null): Defaults to 0. Allows a value between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, thereby minimizing repetition.

    - `function_call` (string or object): This optional parameter controls how the model calls functions.

    - `functions` (array): An optional parameter. It is a list of functions for which the model may generate JSON inputs.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion.

    - `n` (integer or null): This optional parameter helps to set how many chat completion choices to generate for each input message.

    - `presence_penalty` (number or null): Defaults to 0. It penalizes new tokens based on if they appear in the text so far, hence increasing the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """

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
    ) -> None:
        super().__init__(
            frequency_penalty,
            function_call,
            functions,
            logit_bias,
            max_tokens,
            n,
            presence_penalty,
            stop,
            temperature,
            top_p,
        )


def select_azure_base_url_or_endpoint(azure_client_params: dict):
    # azure_client_params = {
    #     "api_version": api_version,
    #     "azure_endpoint": api_base,
    #     "azure_deployment": model,
    #     "http_client": litellm.client_session,
    #     "max_retries": max_retries,
    #     "timeout": timeout,
    # }
    azure_endpoint = azure_client_params.get("azure_endpoint", None)
    if azure_endpoint is not None:
        # see : https://github.com/openai/openai-python/blob/3d61ed42aba652b547029095a7eb269ad4e1e957/src/openai/lib/azure.py#L192
        if "/openai/deployments" in azure_endpoint:
            # this is base_url, not an azure_endpoint
            azure_client_params["base_url"] = azure_endpoint
            azure_client_params.pop("azure_endpoint")

    return azure_client_params


class AzureChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def validate_environment(self, api_key, azure_ad_token):
        headers = {
            "content-type": "application/json",
        }
        if api_key is not None:
            headers["api-key"] = api_key
        elif azure_ad_token is not None:
            headers["Authorization"] = f"Bearer {azure_ad_token}"
        return headers

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        api_key: str,
        api_base: str,
        api_version: str,
        api_type: str,
        azure_ad_token: str,
        print_verbose: Callable,
        timeout,
        logging_obj,
        optional_params,
        litellm_params,
        logger_fn,
        acompletion: bool = False,
        headers: Optional[dict] = None,
        client=None,
    ):
        super().completion()
        exception_mapping_worked = False
        try:
            if model is None or messages is None:
                raise AzureOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )

            max_retries = optional_params.pop("max_retries", 2)

            ### CHECK IF CLOUDFLARE AI GATEWAY ###
            ### if so - set the model as part of the base url
            if "gateway.ai.cloudflare.com" in api_base:
                ## build base url - assume api base includes resource name
                if client is None:
                    if not api_base.endswith("/"):
                        api_base += "/"
                    api_base += f"{model}"

                    azure_client_params = {
                        "api_version": api_version,
                        "base_url": f"{api_base}",
                        "http_client": litellm.client_session,
                        "max_retries": max_retries,
                        "timeout": timeout,
                    }
                    if api_key is not None:
                        azure_client_params["api_key"] = api_key
                    elif azure_ad_token is not None:
                        azure_client_params["azure_ad_token"] = azure_ad_token

                    if acompletion is True:
                        client = AsyncAzureOpenAI(**azure_client_params)
                    else:
                        client = AzureOpenAI(**azure_client_params)

                data = {"model": None, "messages": messages, **optional_params}
            else:
                data = {
                    "model": model,  # type: ignore
                    "messages": messages,
                    **optional_params,
                }

            if acompletion is True:
                if optional_params.get("stream", False):
                    return self.async_streaming(
                        logging_obj=logging_obj,
                        api_base=api_base,
                        data=data,
                        model=model,
                        api_key=api_key,
                        api_version=api_version,
                        azure_ad_token=azure_ad_token,
                        timeout=timeout,
                        client=client,
                    )
                else:
                    return self.acompletion(
                        api_base=api_base,
                        data=data,
                        model_response=model_response,
                        api_key=api_key,
                        api_version=api_version,
                        model=model,
                        azure_ad_token=azure_ad_token,
                        timeout=timeout,
                        client=client,
                        logging_obj=logging_obj,
                    )
            elif "stream" in optional_params and optional_params["stream"] == True:
                return self.streaming(
                    logging_obj=logging_obj,
                    api_base=api_base,
                    data=data,
                    model=model,
                    api_key=api_key,
                    api_version=api_version,
                    azure_ad_token=azure_ad_token,
                    timeout=timeout,
                    client=client,
                )
            else:
                ## LOGGING
                logging_obj.pre_call(
                    input=messages,
                    api_key=api_key,
                    additional_args={
                        "headers": {
                            "api_key": api_key,
                            "azure_ad_token": azure_ad_token,
                        },
                        "api_version": api_version,
                        "api_base": api_base,
                        "complete_input_dict": data,
                    },
                )
                if not isinstance(max_retries, int):
                    raise AzureOpenAIError(
                        status_code=422, message="max retries must be an int"
                    )
                # init AzureOpenAI Client
                azure_client_params = {
                    "api_version": api_version,
                    "azure_endpoint": api_base,
                    "azure_deployment": model,
                    "http_client": litellm.client_session,
                    "max_retries": max_retries,
                    "timeout": timeout,
                }
                azure_client_params = select_azure_base_url_or_endpoint(
                    azure_client_params=azure_client_params
                )
                if api_key is not None:
                    azure_client_params["api_key"] = api_key
                elif azure_ad_token is not None:
                    azure_client_params["azure_ad_token"] = azure_ad_token
                if client is None:
                    azure_client = AzureOpenAI(**azure_client_params)
                else:
                    azure_client = client
                    if api_version is not None and isinstance(
                        azure_client._custom_query, dict
                    ):
                        # set api_version to version passed by user
                        azure_client._custom_query.setdefault(
                            "api-version", api_version
                        )

                response = azure_client.chat.completions.create(**data, timeout=timeout)  # type: ignore
                stringified_response = response.model_dump()
                ## LOGGING
                logging_obj.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=stringified_response,
                    additional_args={
                        "headers": headers,
                        "api_version": api_version,
                        "api_base": api_base,
                    },
                )
                return convert_to_model_response_object(
                    response_object=stringified_response,
                    model_response_object=model_response,
                )
        except AzureOpenAIError as e:
            exception_mapping_worked = True
            raise e
        except Exception as e:
            if hasattr(e, "status_code"):
                raise AzureOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise AzureOpenAIError(status_code=500, message=str(e))

    async def acompletion(
        self,
        api_key: str,
        api_version: str,
        model: str,
        api_base: str,
        data: dict,
        timeout: Any,
        model_response: ModelResponse,
        azure_ad_token: Optional[str] = None,
        client=None,  # this is the AsyncAzureOpenAI
        logging_obj=None,
    ):
        response = None
        try:
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int):
                raise AzureOpenAIError(
                    status_code=422, message="max retries must be an int"
                )

            # init AzureOpenAI Client
            azure_client_params = {
                "api_version": api_version,
                "azure_endpoint": api_base,
                "azure_deployment": model,
                "http_client": litellm.client_session,
                "max_retries": max_retries,
                "timeout": timeout,
            }
            azure_client_params = select_azure_base_url_or_endpoint(
                azure_client_params=azure_client_params
            )
            if api_key is not None:
                azure_client_params["api_key"] = api_key
            elif azure_ad_token is not None:
                azure_client_params["azure_ad_token"] = azure_ad_token

            # setting Azure client
            if client is None:
                azure_client = AsyncAzureOpenAI(**azure_client_params)
            else:
                azure_client = client
                if api_version is not None and isinstance(
                    azure_client._custom_query, dict
                ):
                    # set api_version to version passed by user
                    azure_client._custom_query.setdefault("api-version", api_version)
            ## LOGGING
            logging_obj.pre_call(
                input=data["messages"],
                api_key=azure_client.api_key,
                additional_args={
                    "headers": {"Authorization": f"Bearer {azure_client.api_key}"},
                    "api_base": azure_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )
            response = await azure_client.chat.completions.create(
                **data, timeout=timeout
            )
            return convert_to_model_response_object(
                response_object=response.model_dump(),
                model_response_object=model_response,
            )
        except AzureOpenAIError as e:
            exception_mapping_worked = True
            raise e
        except Exception as e:
            if hasattr(e, "status_code"):
                raise e
            else:
                raise AzureOpenAIError(status_code=500, message=str(e))

    def streaming(
        self,
        logging_obj,
        api_base: str,
        api_key: str,
        api_version: str,
        data: dict,
        model: str,
        timeout: Any,
        azure_ad_token: Optional[str] = None,
        client=None,
    ):
        max_retries = data.pop("max_retries", 2)
        if not isinstance(max_retries, int):
            raise AzureOpenAIError(
                status_code=422, message="max retries must be an int"
            )
        # init AzureOpenAI Client
        azure_client_params = {
            "api_version": api_version,
            "azure_endpoint": api_base,
            "azure_deployment": model,
            "http_client": litellm.client_session,
            "max_retries": max_retries,
            "timeout": timeout,
        }
        azure_client_params = select_azure_base_url_or_endpoint(
            azure_client_params=azure_client_params
        )
        if api_key is not None:
            azure_client_params["api_key"] = api_key
        elif azure_ad_token is not None:
            azure_client_params["azure_ad_token"] = azure_ad_token
        if client is None:
            azure_client = AzureOpenAI(**azure_client_params)
        else:
            azure_client = client
            if api_version is not None and isinstance(azure_client._custom_query, dict):
                # set api_version to version passed by user
                azure_client._custom_query.setdefault("api-version", api_version)
        ## LOGGING
        logging_obj.pre_call(
            input=data["messages"],
            api_key=azure_client.api_key,
            additional_args={
                "headers": {"Authorization": f"Bearer {azure_client.api_key}"},
                "api_base": azure_client._base_url._uri_reference,
                "acompletion": True,
                "complete_input_dict": data,
            },
        )
        response = azure_client.chat.completions.create(**data, timeout=timeout)
        streamwrapper = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="azure",
            logging_obj=logging_obj,
        )
        return streamwrapper

    async def async_streaming(
        self,
        logging_obj,
        api_base: str,
        api_key: str,
        api_version: str,
        data: dict,
        model: str,
        timeout: Any,
        azure_ad_token: Optional[str] = None,
        client=None,
    ):
        try:
            # init AzureOpenAI Client
            azure_client_params = {
                "api_version": api_version,
                "azure_endpoint": api_base,
                "azure_deployment": model,
                "http_client": litellm.client_session,
                "max_retries": data.pop("max_retries", 2),
                "timeout": timeout,
            }
            azure_client_params = select_azure_base_url_or_endpoint(
                azure_client_params=azure_client_params
            )
            if api_key is not None:
                azure_client_params["api_key"] = api_key
            elif azure_ad_token is not None:
                azure_client_params["azure_ad_token"] = azure_ad_token
            if client is None:
                azure_client = AsyncAzureOpenAI(**azure_client_params)
            else:
                azure_client = client
                if api_version is not None and isinstance(
                    azure_client._custom_query, dict
                ):
                    # set api_version to version passed by user
                    azure_client._custom_query.setdefault("api-version", api_version)
            ## LOGGING
            logging_obj.pre_call(
                input=data["messages"],
                api_key=azure_client.api_key,
                additional_args={
                    "headers": {"Authorization": f"Bearer {azure_client.api_key}"},
                    "api_base": azure_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )
            response = await azure_client.chat.completions.create(
                **data, timeout=timeout
            )
            # return response
            streamwrapper = CustomStreamWrapper(
                completion_stream=response,
                model=model,
                custom_llm_provider="azure",
                logging_obj=logging_obj,
            )
            return streamwrapper  ## DO NOT make this into an async for ... loop, it will yield an async generator, which won't raise errors if the response fails
        except Exception as e:
            if hasattr(e, "status_code"):
                raise AzureOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise AzureOpenAIError(status_code=500, message=str(e))

    async def aembedding(
        self,
        data: dict,
        model_response: ModelResponse,
        azure_client_params: dict,
        api_key: str,
        input: list,
        client=None,
        logging_obj=None,
        timeout=None,
    ):
        response = None
        try:
            if client is None:
                openai_aclient = AsyncAzureOpenAI(**azure_client_params)
            else:
                openai_aclient = client
            response = await openai_aclient.embeddings.create(**data, timeout=timeout)
            stringified_response = response.model_dump()
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=stringified_response,
            )
            return convert_to_model_response_object(
                response_object=stringified_response,
                model_response_object=model_response,
                response_type="embedding",
            )
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            raise e

    def embedding(
        self,
        model: str,
        input: list,
        api_key: str,
        api_base: str,
        api_version: str,
        timeout: float,
        logging_obj=None,
        model_response=None,
        optional_params=None,
        azure_ad_token: Optional[str] = None,
        client=None,
        aembedding=None,
    ):
        super().embedding()
        exception_mapping_worked = False
        if self._client_session is None:
            self._client_session = self.create_client_session()
        try:
            data = {"model": model, "input": input, **optional_params}
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int):
                raise AzureOpenAIError(
                    status_code=422, message="max retries must be an int"
                )

            # init AzureOpenAI Client
            azure_client_params = {
                "api_version": api_version,
                "azure_endpoint": api_base,
                "azure_deployment": model,
                "http_client": litellm.client_session,
                "max_retries": max_retries,
                "timeout": timeout,
            }
            azure_client_params = select_azure_base_url_or_endpoint(
                azure_client_params=azure_client_params
            )
            if api_key is not None:
                azure_client_params["api_key"] = api_key
            elif azure_ad_token is not None:
                azure_client_params["azure_ad_token"] = azure_ad_token

            ## LOGGING
            logging_obj.pre_call(
                input=input,
                api_key=api_key,
                additional_args={
                    "complete_input_dict": data,
                    "headers": {"api_key": api_key, "azure_ad_token": azure_ad_token},
                },
            )

            if aembedding == True:
                response = self.aembedding(
                    data=data,
                    input=input,
                    logging_obj=logging_obj,
                    api_key=api_key,
                    model_response=model_response,
                    azure_client_params=azure_client_params,
                    timeout=timeout,
                )
                return response
            if client is None:
                azure_client = AzureOpenAI(**azure_client_params)  # type: ignore
            else:
                azure_client = client
            ## COMPLETION CALL
            response = azure_client.embeddings.create(**data, timeout=timeout)  # type: ignore
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data, "api_base": api_base},
                original_response=response,
            )

            return convert_to_model_response_object(response_object=response.model_dump(), model_response_object=model_response, response_type="embedding")  # type: ignore
        except AzureOpenAIError as e:
            exception_mapping_worked = True
            raise e
        except Exception as e:
            if hasattr(e, "status_code"):
                raise AzureOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise AzureOpenAIError(status_code=500, message=str(e))

    async def aimage_generation(
        self,
        data: dict,
        model_response: ModelResponse,
        azure_client_params: dict,
        api_key: str,
        input: list,
        client=None,
        logging_obj=None,
        timeout=None,
    ):
        response = None
        try:
            if client is None:
                client_session = litellm.aclient_session or httpx.AsyncClient(
                    transport=AsyncCustomHTTPTransport(),
                )
                azure_client = AsyncAzureOpenAI(
                    http_client=client_session, **azure_client_params
                )
            else:
                azure_client = client
            ## LOGGING
            logging_obj.pre_call(
                input=data["prompt"],
                api_key=azure_client.api_key,
                additional_args={
                    "headers": {"api_key": azure_client.api_key},
                    "api_base": azure_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )
            response = await azure_client.images.generate(**data, timeout=timeout)
            stringified_response = response.model_dump()
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=stringified_response,
            )
            return convert_to_model_response_object(
                response_object=stringified_response,
                model_response_object=model_response,
                response_type="image_generation",
            )
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            raise e

    def image_generation(
        self,
        prompt: str,
        timeout: float,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        model_response: Optional[litellm.utils.ImageResponse] = None,
        azure_ad_token: Optional[str] = None,
        logging_obj=None,
        optional_params=None,
        client=None,
        aimg_generation=None,
    ):
        exception_mapping_worked = False
        try:
            if model and len(model) > 0:
                model = model
            else:
                model = None
            data = {"model": model, "prompt": prompt, **optional_params}
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int):
                raise AzureOpenAIError(
                    status_code=422, message="max retries must be an int"
                )

            # init AzureOpenAI Client
            azure_client_params = {
                "api_version": api_version,
                "azure_endpoint": api_base,
                "azure_deployment": model,
                "max_retries": max_retries,
                "timeout": timeout,
            }
            azure_client_params = select_azure_base_url_or_endpoint(
                azure_client_params=azure_client_params
            )
            if api_key is not None:
                azure_client_params["api_key"] = api_key
            elif azure_ad_token is not None:
                azure_client_params["azure_ad_token"] = azure_ad_token

            if aimg_generation == True:
                response = self.aimage_generation(data=data, input=input, logging_obj=logging_obj, model_response=model_response, api_key=api_key, client=client, azure_client_params=azure_client_params, timeout=timeout)  # type: ignore
                return response

            if client is None:
                client_session = litellm.client_session or httpx.Client(
                    transport=CustomHTTPTransport(),
                )
                azure_client = AzureOpenAI(http_client=client_session, **azure_client_params)  # type: ignore
            else:
                azure_client = client

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=azure_client.api_key,
                additional_args={
                    "headers": {"api_key": azure_client.api_key},
                    "api_base": azure_client._base_url._uri_reference,
                    "acompletion": False,
                    "complete_input_dict": data,
                },
            )

            ## COMPLETION CALL
            response = azure_client.images.generate(**data, timeout=timeout)  # type: ignore
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response,
            )
            # return response
            return convert_to_model_response_object(response_object=response.model_dump(), model_response_object=model_response, response_type="image_generation")  # type: ignore
        except AzureOpenAIError as e:
            exception_mapping_worked = True
            raise e
        except Exception as e:
            if hasattr(e, "status_code"):
                raise AzureOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise AzureOpenAIError(status_code=500, message=str(e))

    def audio_transcriptions(
        self,
        model: str,
        audio_file: BinaryIO,
        optional_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        client=None,
        azure_ad_token: Optional[str] = None,
        logging_obj=None,
        atranscription: bool = False,
    ):
        data = {"model": model, "file": audio_file, **optional_params}

        # init AzureOpenAI Client
        azure_client_params = {
            "api_version": api_version,
            "azure_endpoint": api_base,
            "azure_deployment": model,
            "timeout": timeout,
        }

        max_retries = optional_params.pop("max_retries", None)

        azure_client_params = select_azure_base_url_or_endpoint(
            azure_client_params=azure_client_params
        )
        if api_key is not None:
            azure_client_params["api_key"] = api_key
        elif azure_ad_token is not None:
            azure_client_params["azure_ad_token"] = azure_ad_token

        if max_retries is not None:
            azure_client_params["max_retries"] = max_retries

        if atranscription == True:
            return self.async_audio_transcriptions(
                audio_file=audio_file,
                data=data,
                model_response=model_response,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                client=client,
                azure_client_params=azure_client_params,
                max_retries=max_retries,
                logging_obj=logging_obj,
            )
        if client is None:
            azure_client = AzureOpenAI(http_client=litellm.client_session, **azure_client_params)  # type: ignore
        else:
            azure_client = client

        ## LOGGING
        logging_obj.pre_call(
            input=f"audio_file_{uuid.uuid4()}",
            api_key=azure_client.api_key,
            additional_args={
                "headers": {"Authorization": f"Bearer {azure_client.api_key}"},
                "api_base": azure_client._base_url._uri_reference,
                "atranscription": True,
                "complete_input_dict": data,
            },
        )

        response = azure_client.audio.transcriptions.create(
            **data, timeout=timeout  # type: ignore
        )
        stringified_response = response.model_dump()
        ## LOGGING
        logging_obj.post_call(
            input=audio_file.name,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=stringified_response,
        )
        hidden_params = {"model": "whisper-1", "custom_llm_provider": "azure"}
        final_response = convert_to_model_response_object(response_object=stringified_response, model_response_object=model_response, hidden_params=hidden_params, response_type="audio_transcription")  # type: ignore
        return final_response

    async def async_audio_transcriptions(
        self,
        audio_file: BinaryIO,
        data: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
        azure_client_params=None,
        max_retries=None,
        logging_obj=None,
    ):
        response = None
        try:
            if client is None:
                async_azure_client = AsyncAzureOpenAI(
                    **azure_client_params,
                    http_client=litellm.aclient_session,
                )
            else:
                async_azure_client = client

            ## LOGGING
            logging_obj.pre_call(
                input=f"audio_file_{uuid.uuid4()}",
                api_key=async_azure_client.api_key,
                additional_args={
                    "headers": {
                        "Authorization": f"Bearer {async_azure_client.api_key}"
                    },
                    "api_base": async_azure_client._base_url._uri_reference,
                    "atranscription": True,
                    "complete_input_dict": data,
                },
            )

            response = await async_azure_client.audio.transcriptions.create(
                **data, timeout=timeout
            )  # type: ignore

            stringified_response = response.model_dump()

            ## LOGGING
            logging_obj.post_call(
                input=audio_file.name,
                api_key=api_key,
                additional_args={
                    "headers": {
                        "Authorization": f"Bearer {async_azure_client.api_key}"
                    },
                    "api_base": async_azure_client._base_url._uri_reference,
                    "atranscription": True,
                    "complete_input_dict": data,
                },
                original_response=stringified_response,
            )
            hidden_params = {"model": "whisper-1", "custom_llm_provider": "azure"}
            response = convert_to_model_response_object(response_object=stringified_response, model_response_object=model_response, hidden_params=hidden_params, response_type="audio_transcription")  # type: ignore
            return response
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                original_response=str(e),
            )
            raise e

    async def ahealth_check(
        self,
        model: Optional[str],
        api_key: str,
        api_base: str,
        api_version: str,
        timeout: float,
        mode: str,
        messages: Optional[list] = None,
        input: Optional[list] = None,
        prompt: Optional[str] = None,
    ):
        client_session = litellm.aclient_session or httpx.AsyncClient(
            transport=AsyncCustomHTTPTransport(),  # handle dall-e-2 calls
        )
        if "gateway.ai.cloudflare.com" in api_base:
            ## build base url - assume api base includes resource name
            if not api_base.endswith("/"):
                api_base += "/"
            api_base += f"{model}"
            client = AsyncAzureOpenAI(
                base_url=api_base,
                api_version=api_version,
                api_key=api_key,
                timeout=timeout,
                http_client=client_session,
            )
            model = None
            # cloudflare ai gateway, needs model=None
        else:
            client = AsyncAzureOpenAI(
                api_version=api_version,
                azure_endpoint=api_base,
                api_key=api_key,
                timeout=timeout,
                http_client=client_session,
            )

            # only run this check if it's not cloudflare ai gateway
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
        else:
            raise Exception("mode not set")
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
