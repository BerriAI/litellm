import asyncio
import json
import os
import types
import uuid
from typing import (
    Any,
    BinaryIO,
    Callable,
    Coroutine,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)

import httpx  # type: ignore
import requests
from openai import AsyncAzureOpenAI, AzureOpenAI
from typing_extensions import overload

import litellm
from litellm import OpenAIConfig
from litellm.caching import DualCache
from litellm.utils import (
    Choices,
    CustomStreamWrapper,
    Message,
    ModelResponse,
    TranscriptionResponse,
    UnsupportedParamsError,
    convert_to_model_response_object,
    get_secret,
)

from ..types.llms.openai import (
    Assistant,
    AssistantEventHandler,
    AssistantStreamManager,
    AssistantToolParam,
    AsyncAssistantEventHandler,
    AsyncAssistantStreamManager,
    AsyncCursorPage,
    HttpxBinaryResponseContent,
    MessageData,
    OpenAICreateThreadParamsMessage,
    OpenAIMessage,
    Run,
    SyncCursorPage,
    Thread,
)
from .base import BaseLLM
from .custom_httpx.azure_dall_e_2 import AsyncCustomHTTPTransport, CustomHTTPTransport

azure_ad_cache = DualCache()


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


class AzureOpenAIConfig:
    """
    Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions

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
            "temperature",
            "n",
            "stream",
            "stop",
            "max_tokens",
            "tools",
            "tool_choice",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "function_call",
            "functions",
            "tools",
            "tool_choice",
            "top_p",
            "logprobs",
            "top_logprobs",
            "response_format",
            "seed",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        api_version: str,  # Y-M-D-{optional}
        drop_params,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params()

        api_version_times = api_version.split("-")
        api_version_year = api_version_times[0]
        api_version_month = api_version_times[1]
        api_version_day = api_version_times[2]
        for param, value in non_default_params.items():
            if param == "tool_choice":
                """
                This parameter requires API version 2023-12-01-preview or later

                tool_choice='required' is not supported as of 2024-05-01-preview
                """
                ## check if api version supports this param ##
                if (
                    api_version_year < "2023"
                    or (api_version_year == "2023" and api_version_month < "12")
                    or (
                        api_version_year == "2023"
                        and api_version_month == "12"
                        and api_version_day < "01"
                    )
                ):
                    if litellm.drop_params == True or (
                        drop_params is not None and drop_params == True
                    ):
                        pass
                    else:
                        raise UnsupportedParamsError(
                            status_code=400,
                            message=f"""Azure does not support 'tool_choice', for api_version={api_version}. Bump your API version to '2023-12-01-preview' or later. This parameter requires 'api_version="2023-12-01-preview"' or later. Azure API Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions""",
                        )
                elif value == "required" and (
                    api_version_year == "2024" and api_version_month <= "05"
                ):  ## check if tool_choice value is supported ##
                    if litellm.drop_params == True or (
                        drop_params is not None and drop_params == True
                    ):
                        pass
                    else:
                        raise UnsupportedParamsError(
                            status_code=400,
                            message=f"Azure does not support '{value}' as a {param} param, for api_version={api_version}. To drop 'tool_choice=required' for calls with this Azure API version, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\nAzure API Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions",
                        )
                else:
                    optional_params["tool_choice"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def get_mapped_special_auth_params(self) -> dict:
        return {"token": "azure_ad_token"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "token":
                optional_params["azure_ad_token"] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-model-availability
        """
        return ["europe", "sweden", "switzerland", "france", "uk"]


class AzureOpenAIAssistantsAPIConfig:
    """
    Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/assistants-reference-messages?tabs=python#create-message
    """

    def __init__(
        self,
    ) -> None:
        pass

    def get_supported_openai_create_message_params(self):
        return [
            "role",
            "content",
            "attachments",
            "metadata",
        ]

    def map_openai_params_create_message_params(
        self, non_default_params: dict, optional_params: dict
    ):
        for param, value in non_default_params.items():
            if param == "role":
                optional_params["role"] = value
            if param == "metadata":
                optional_params["metadata"] = value
            elif param == "content":  # only string accepted
                if isinstance(value, str):
                    optional_params["content"] = value
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message="Azure only accepts content as a string.",
                        status_code=400,
                    )
            elif (
                param == "attachments"
            ):  # this is a v2 param. Azure currently supports the old 'file_id's param
                file_ids: List[str] = []
                if isinstance(value, list):
                    for item in value:
                        if "file_id" in item:
                            file_ids.append(item["file_id"])
                        else:
                            if litellm.drop_params == True:
                                pass
                            else:
                                raise litellm.utils.UnsupportedParamsError(
                                    message="Azure doesn't support {}. To drop it from the call, set `litellm.drop_params = True.".format(
                                        value
                                    ),
                                    status_code=400,
                                )
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message="Invalid param. attachments should always be a list. Got={}, Expected=List. Raw value={}".format(
                            type(value), value
                        ),
                        status_code=400,
                    )
        return optional_params


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


def get_azure_ad_token_from_oidc(azure_ad_token: str):
    azure_client_id = os.getenv("AZURE_CLIENT_ID", None)
    azure_tenant_id = os.getenv("AZURE_TENANT_ID", None)
    azure_authority_host = os.getenv(
        "AZURE_AUTHORITY_HOST", "https://login.microsoftonline.com"
    )

    if azure_client_id is None or azure_tenant_id is None:
        raise AzureOpenAIError(
            status_code=422,
            message="AZURE_CLIENT_ID and AZURE_TENANT_ID must be set",
        )

    oidc_token = get_secret(azure_ad_token)

    if oidc_token is None:
        raise AzureOpenAIError(
            status_code=401,
            message="OIDC token could not be retrieved from secret manager.",
        )

    azure_ad_token_cache_key = json.dumps(
        {
            "azure_client_id": azure_client_id,
            "azure_tenant_id": azure_tenant_id,
            "azure_authority_host": azure_authority_host,
            "oidc_token": oidc_token,
        }
    )

    azure_ad_token_access_token = azure_ad_cache.get_cache(azure_ad_token_cache_key)
    if azure_ad_token_access_token is not None:
        return azure_ad_token_access_token

    req_token = httpx.post(
        f"{azure_authority_host}/{azure_tenant_id}/oauth2/v2.0/token",
        data={
            "client_id": azure_client_id,
            "grant_type": "client_credentials",
            "scope": "https://cognitiveservices.azure.com/.default",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": oidc_token,
        },
    )

    if req_token.status_code != 200:
        raise AzureOpenAIError(
            status_code=req_token.status_code,
            message=req_token.text,
        )

    azure_ad_token_json = req_token.json()
    azure_ad_token_access_token = azure_ad_token_json.get("access_token", None)
    azure_ad_token_expires_in = azure_ad_token_json.get("expires_in", None)

    if azure_ad_token_access_token is None:
        raise AzureOpenAIError(
            status_code=422, message="Azure AD Token access_token not returned"
        )

    if azure_ad_token_expires_in is None:
        raise AzureOpenAIError(
            status_code=422, message="Azure AD Token expires_in not returned"
        )

    azure_ad_cache.set_cache(
        key=azure_ad_token_cache_key,
        value=azure_ad_token_access_token,
        ttl=azure_ad_token_expires_in,
    )

    return azure_ad_token_access_token


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
            if azure_ad_token.startswith("oidc/"):
                azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
            headers["Authorization"] = f"Bearer {azure_ad_token}"
        return headers

    def _get_sync_azure_client(
        self,
        api_version: Optional[str],
        api_base: Optional[str],
        api_key: Optional[str],
        azure_ad_token: Optional[str],
        model: str,
        max_retries: int,
        timeout: Union[float, httpx.Timeout],
        client: Optional[Any],
        client_type: Literal["sync", "async"],
    ):
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
            if azure_ad_token.startswith("oidc/"):
                azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
            azure_client_params["azure_ad_token"] = azure_ad_token
        if client is None:
            if client_type == "sync":
                azure_client = AzureOpenAI(**azure_client_params)  # type: ignore
            elif client_type == "async":
                azure_client = AsyncAzureOpenAI(**azure_client_params)  # type: ignore
        else:
            azure_client = client
            if api_version is not None and isinstance(azure_client._custom_query, dict):
                # set api_version to version passed by user
                azure_client._custom_query.setdefault("api-version", api_version)

        return azure_client

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
        timeout: Union[float, httpx.Timeout],
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
                        if azure_ad_token.startswith("oidc/"):
                            azure_ad_token = get_azure_ad_token_from_oidc(
                                azure_ad_token
                            )

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
                    if azure_ad_token.startswith("oidc/"):
                        azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
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
                if azure_ad_token.startswith("oidc/"):
                    azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
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

            stringified_response = response.model_dump()
            logging_obj.post_call(
                input=data["messages"],
                api_key=api_key,
                original_response=stringified_response,
                additional_args={"complete_input_dict": data},
            )
            return convert_to_model_response_object(
                response_object=stringified_response,
                model_response_object=model_response,
            )
        except AzureOpenAIError as e:
            ## LOGGING
            logging_obj.post_call(
                input=data["messages"],
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            exception_mapping_worked = True
            raise e
        except asyncio.CancelledError as e:
            ## LOGGING
            logging_obj.post_call(
                input=data["messages"],
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            raise AzureOpenAIError(status_code=500, message=str(e))
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=data["messages"],
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
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
            if azure_ad_token.startswith("oidc/"):
                azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
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
                if azure_ad_token.startswith("oidc/"):
                    azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
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
        client: Optional[AsyncAzureOpenAI] = None,
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
                if azure_ad_token.startswith("oidc/"):
                    azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
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

            if aembedding is True:
                response = self.aembedding(
                    data=data,
                    input=input,
                    logging_obj=logging_obj,
                    api_key=api_key,
                    model_response=model_response,
                    azure_client_params=azure_client_params,
                    timeout=timeout,
                    client=client,
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

            ## BASE MODEL CHECK
            if (
                model_response is not None
                and optional_params.get("base_model", None) is not None
            ):
                model_response._hidden_params["model"] = optional_params.pop(
                    "base_model"
                )

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
                if azure_ad_token.startswith("oidc/"):
                    azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
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
        max_retries: int,
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

        azure_client_params = select_azure_base_url_or_endpoint(
            azure_client_params=azure_client_params
        )
        if api_key is not None:
            azure_client_params["api_key"] = api_key
        elif azure_ad_token is not None:
            if azure_ad_token.startswith("oidc/"):
                azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
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

    def audio_speech(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        organization: Optional[str],
        max_retries: int,
        timeout: Union[float, httpx.Timeout],
        azure_ad_token: Optional[str] = None,
        aspeech: Optional[bool] = None,
        client=None,
    ) -> HttpxBinaryResponseContent:

        max_retries = optional_params.pop("max_retries", 2)

        if aspeech is not None and aspeech is True:
            return self.async_audio_speech(
                model=model,
                input=input,
                voice=voice,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                max_retries=max_retries,
                timeout=timeout,
                client=client,
            )  # type: ignore

        azure_client: AzureOpenAI = self._get_sync_azure_client(
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            model=model,
            max_retries=max_retries,
            timeout=timeout,
            client=client,
            client_type="sync",
        )  # type: ignore

        response = azure_client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore
            input=input,
            **optional_params,
        )
        return response

    async def async_audio_speech(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        max_retries: int,
        timeout: Union[float, httpx.Timeout],
        client=None,
    ) -> HttpxBinaryResponseContent:

        azure_client: AsyncAzureOpenAI = self._get_sync_azure_client(
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            model=model,
            max_retries=max_retries,
            timeout=timeout,
            client=client,
            client_type="async",
        )  # type: ignore

        response = await azure_client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore
            input=input,
            **optional_params,
        )

        return response

    def get_headers(
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
    ) -> dict:
        client_session = litellm.client_session or httpx.Client(
            transport=CustomHTTPTransport(),  # handle dall-e-2 calls
        )
        if "gateway.ai.cloudflare.com" in api_base:
            ## build base url - assume api base includes resource name
            if not api_base.endswith("/"):
                api_base += "/"
            api_base += f"{model}"
            client = AzureOpenAI(
                base_url=api_base,
                api_version=api_version,
                api_key=api_key,
                timeout=timeout,
                http_client=client_session,
            )
            model = None
            # cloudflare ai gateway, needs model=None
        else:
            client = AzureOpenAI(
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

        if messages is None:
            messages = [{"role": "user", "content": "Hey"}]
        try:
            completion = client.chat.completions.with_raw_response.create(
                model=model,  # type: ignore
                messages=messages,  # type: ignore
            )
        except Exception as e:
            raise e
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

        if completion.headers.get("x-ms-region", None) is not None:
            response["x-ms-region"] = completion.headers["x-ms-region"]

        return response

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
    ) -> dict:
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

        if completion.headers.get("x-ms-region", None) is not None:
            response["x-ms-region"] = completion.headers["x-ms-region"]

        return response


class AzureAssistantsAPI(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def get_azure_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI] = None,
    ) -> AzureOpenAI:
        received_args = locals()
        if client is None:
            data = {}
            for k, v in received_args.items():
                if k == "self" or k == "client":
                    pass
                elif k == "api_base" and v is not None:
                    data["azure_endpoint"] = v
                elif v is not None:
                    data[k] = v
            azure_openai_client = AzureOpenAI(**data)  # type: ignore
        else:
            azure_openai_client = client

        return azure_openai_client

    def async_get_azure_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI] = None,
    ) -> AsyncAzureOpenAI:
        received_args = locals()
        if client is None:
            data = {}
            for k, v in received_args.items():
                if k == "self" or k == "client":
                    pass
                elif k == "api_base" and v is not None:
                    data["azure_endpoint"] = v
                elif v is not None:
                    data[k] = v

            azure_openai_client = AsyncAzureOpenAI(**data)
            # azure_openai_client = AsyncAzureOpenAI(**data)  # type: ignore
        else:
            azure_openai_client = client

        return azure_openai_client

    ### ASSISTANTS ###

    async def async_get_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
    ) -> AsyncCursorPage[Assistant]:
        azure_openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

        response = await azure_openai_client.beta.assistants.list()

        return response

    # fmt: off

    @overload
    def get_assistants(
        self, 
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        aget_assistants: Literal[True], 
    ) -> Coroutine[None, None, AsyncCursorPage[Assistant]]:
        ...

    @overload
    def get_assistants(
        self, 
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        aget_assistants: Optional[Literal[False]], 
    ) -> SyncCursorPage[Assistant]: 
        ...

    # fmt: on

    def get_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        aget_assistants=None,
    ):
        if aget_assistants is not None and aget_assistants == True:
            return self.async_get_assistants(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )
        azure_openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            api_version=api_version,
        )

        response = azure_openai_client.beta.assistants.list()

        return response

    ### MESSAGES ###

    async def a_add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI] = None,
    ) -> OpenAIMessage:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        a_add_message: Optional[bool] = None,
    ):
        if a_add_message is not None and a_add_message == True:
            return self.a_add_message(
                thread_id=thread_id,
                message_data=message_data,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )
        openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI] = None,
    ) -> AsyncCursorPage[OpenAIMessage]:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        aget_messages: Literal[True],
    ) -> Coroutine[None, None, AsyncCursorPage[OpenAIMessage]]:
        ...

    @overload
    def get_messages(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        aget_messages: Optional[Literal[False]],
    ) -> SyncCursorPage[OpenAIMessage]:
        ...

    # fmt: on

    def get_messages(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        aget_messages=None,
    ):
        if aget_messages is not None and aget_messages == True:
            return self.async_get_messages(
                thread_id=thread_id,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )
        openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
    ) -> Thread:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client: Optional[AsyncAzureOpenAI],
        acreate_thread: Literal[True],
    ) -> Coroutine[None, None, Thread]:
        ...

    @overload
    def create_thread(
        self,
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client: Optional[AzureOpenAI],
        acreate_thread: Optional[Literal[False]],
    ) -> Thread:
        ...

    # fmt: on

    def create_thread(
        self,
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client=None,
        acreate_thread=None,
    ):
        """
        Here's an example:
        ```
        from litellm.llms.openai import OpenAIAssistantsAPI, MessageData

        # create thread
        message: MessageData = {"role": "user", "content": "Hey, how's it going?"}
        openai_api.create_thread(messages=[message])
        ```
        """
        if acreate_thread is not None and acreate_thread == True:
            return self.async_create_thread(
                metadata=metadata,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                messages=messages,
            )
        azure_openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

        data = {}
        if messages is not None:
            data["messages"] = messages  # type: ignore
        if metadata is not None:
            data["metadata"] = metadata  # type: ignore

        message_thread = azure_openai_client.beta.threads.create(**data)  # type: ignore

        return Thread(**message_thread.dict())

    async def async_get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
    ) -> Thread:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        aget_thread: Literal[True],
    ) -> Coroutine[None, None, Thread]:
        ...

    @overload
    def get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        aget_thread: Optional[Literal[False]],
    ) -> Thread:
        ...

    # fmt: on

    def get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        aget_thread=None,
    ):
        if aget_thread is not None and aget_thread == True:
            return self.async_get_thread(
                thread_id=thread_id,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )
        openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

        response = openai_client.beta.threads.retrieve(thread_id=thread_id)

        return Thread(**response.dict())

    # def delete_thread(self):
    #     pass

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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
    ) -> Run:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
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
        client: AsyncAzureOpenAI,
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
        client: AzureOpenAI,
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        arun_thread: Literal[True],
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        arun_thread: Optional[Literal[False]],
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
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        arun_thread=None,
        event_handler: Optional[AssistantEventHandler] = None,
    ):
        if arun_thread is not None and arun_thread == True:
            if stream is not None and stream == True:
                azure_client = self.async_get_azure_client(
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    azure_ad_token=azure_ad_token,
                    timeout=timeout,
                    max_retries=max_retries,
                    client=client,
                )
                return self.async_run_thread_stream(
                    client=azure_client,
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
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )
        openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

        if stream is not None and stream == True:
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
