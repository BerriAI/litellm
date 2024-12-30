import asyncio
import json
import os
import time
from typing import Any, Callable, List, Literal, Optional, Union

import httpx  # type: ignore
from openai import AsyncAzureOpenAI, AzureOpenAI

import litellm
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    LlmProviders,
    ModelResponse,
)
from litellm.utils import (
    CustomStreamWrapper,
    convert_to_model_response_object,
    get_secret,
    modify_url,
)

from ...types.llms.openai import HttpxBinaryResponseContent
from ..base import BaseLLM
from .common_utils import AzureOpenAIError, process_azure_headers

azure_ad_cache = DualCache()


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
                            if litellm.drop_params is True:
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

    client = litellm.module_level_client
    req_token = client.post(
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


def _check_dynamic_azure_params(
    azure_client_params: dict,
    azure_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]],
) -> bool:
    """
    Returns True if user passed in client params != initialized azure client

    Currently only implemented for api version
    """
    if azure_client is None:
        return True

    dynamic_params = ["api_version"]
    for k, v in azure_client_params.items():
        if k in dynamic_params and k == "api_version":
            if v is not None and v != azure_client._custom_query["api-version"]:
                return True

    return False


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

    def make_sync_azure_openai_chat_completion_request(
        self,
        azure_client: AzureOpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
    ):
        """
        Helper to:
        - call chat.completions.create.with_raw_response when litellm.return_response_headers is True
        - call chat.completions.create by default
        """
        try:
            raw_response = azure_client.chat.completions.with_raw_response.create(
                **data, timeout=timeout
            )

            headers = dict(raw_response.headers)
            response = raw_response.parse()
            return headers, response
        except Exception as e:
            raise e

    async def make_azure_openai_chat_completion_request(
        self,
        azure_client: AsyncAzureOpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
    ):
        """
        Helper to:
        - call chat.completions.create.with_raw_response when litellm.return_response_headers is True
        - call chat.completions.create by default
        """
        try:
            raw_response = await azure_client.chat.completions.with_raw_response.create(
                **data, timeout=timeout
            )

            headers = dict(raw_response.headers)
            response = raw_response.parse()
            return headers, response
        except Exception as e:
            raise e

    def completion(  # noqa: PLR0915
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        api_key: str,
        api_base: str,
        api_version: str,
        api_type: str,
        azure_ad_token: str,
        dynamic_params: bool,
        print_verbose: Callable,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
        optional_params,
        litellm_params,
        logger_fn,
        acompletion: bool = False,
        headers: Optional[dict] = None,
        client=None,
    ):
        if headers:
            optional_params["extra_headers"] = headers
        try:
            if model is None or messages is None:
                raise AzureOpenAIError(
                    status_code=422, message="Missing model or messages"
                )

            max_retries = optional_params.pop("max_retries", 2)
            json_mode: Optional[bool] = optional_params.pop("json_mode", False)

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
                data = litellm.AzureOpenAIConfig().transform_request(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    headers=headers or {},
                )

            if acompletion is True:
                if optional_params.get("stream", False):
                    return self.async_streaming(
                        logging_obj=logging_obj,
                        api_base=api_base,
                        dynamic_params=dynamic_params,
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
                        dynamic_params=dynamic_params,
                        timeout=timeout,
                        client=client,
                        logging_obj=logging_obj,
                        convert_tool_call_to_json_mode=json_mode,
                    )
            elif "stream" in optional_params and optional_params["stream"] is True:
                return self.streaming(
                    logging_obj=logging_obj,
                    api_base=api_base,
                    dynamic_params=dynamic_params,
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

                if (
                    client is None
                    or not isinstance(client, AzureOpenAI)
                    or dynamic_params
                ):
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
                if not isinstance(azure_client, AzureOpenAI):
                    raise AzureOpenAIError(
                        status_code=500,
                        message="azure_client is not an instance of AzureOpenAI",
                    )

                headers, response = self.make_sync_azure_openai_chat_completion_request(
                    azure_client=azure_client, data=data, timeout=timeout
                )
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
                    convert_tool_call_to_json_mode=json_mode,
                    _response_headers=headers,
                )
        except AzureOpenAIError as e:
            raise e
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise AzureOpenAIError(
                status_code=status_code, message=str(e), headers=error_headers
            )

    async def acompletion(
        self,
        api_key: str,
        api_version: str,
        model: str,
        api_base: str,
        data: dict,
        timeout: Any,
        dynamic_params: bool,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        azure_ad_token: Optional[str] = None,
        convert_tool_call_to_json_mode: Optional[bool] = None,
        client=None,  # this is the AsyncAzureOpenAI
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
                "http_client": litellm.aclient_session,
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
            if client is None or dynamic_params:
                azure_client = AsyncAzureOpenAI(**azure_client_params)
            else:
                azure_client = client

            ## LOGGING
            logging_obj.pre_call(
                input=data["messages"],
                api_key=azure_client.api_key,
                additional_args={
                    "headers": {
                        "api_key": api_key,
                        "azure_ad_token": azure_ad_token,
                    },
                    "api_base": azure_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )

            headers, response = await self.make_azure_openai_chat_completion_request(
                azure_client=azure_client,
                data=data,
                timeout=timeout,
            )
            logging_obj.model_call_details["response_headers"] = headers

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
                hidden_params={"headers": headers},
                _response_headers=headers,
                convert_tool_call_to_json_mode=convert_tool_call_to_json_mode,
            )
        except AzureOpenAIError as e:
            ## LOGGING
            logging_obj.post_call(
                input=data["messages"],
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
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
        dynamic_params: bool,
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

        if client is None or dynamic_params:
            azure_client = AzureOpenAI(**azure_client_params)
        else:
            azure_client = client
        ## LOGGING
        logging_obj.pre_call(
            input=data["messages"],
            api_key=azure_client.api_key,
            additional_args={
                "headers": {
                    "api_key": api_key,
                    "azure_ad_token": azure_ad_token,
                },
                "api_base": azure_client._base_url._uri_reference,
                "acompletion": True,
                "complete_input_dict": data,
            },
        )
        headers, response = self.make_sync_azure_openai_chat_completion_request(
            azure_client=azure_client, data=data, timeout=timeout
        )
        streamwrapper = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="azure",
            logging_obj=logging_obj,
            stream_options=data.get("stream_options", None),
            _response_headers=process_azure_headers(headers),
        )
        return streamwrapper

    async def async_streaming(
        self,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        api_key: str,
        api_version: str,
        dynamic_params: bool,
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
                "http_client": litellm.aclient_session,
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
            if client is None or dynamic_params:
                azure_client = AsyncAzureOpenAI(**azure_client_params)
            else:
                azure_client = client
            ## LOGGING
            logging_obj.pre_call(
                input=data["messages"],
                api_key=azure_client.api_key,
                additional_args={
                    "headers": {
                        "api_key": api_key,
                        "azure_ad_token": azure_ad_token,
                    },
                    "api_base": azure_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )

            headers, response = await self.make_azure_openai_chat_completion_request(
                azure_client=azure_client,
                data=data,
                timeout=timeout,
            )
            logging_obj.model_call_details["response_headers"] = headers

            # return response
            streamwrapper = CustomStreamWrapper(
                completion_stream=response,
                model=model,
                custom_llm_provider="azure",
                logging_obj=logging_obj,
                stream_options=data.get("stream_options", None),
                _response_headers=headers,
            )
            return streamwrapper  ## DO NOT make this into an async for ... loop, it will yield an async generator, which won't raise errors if the response fails
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise AzureOpenAIError(
                status_code=status_code, message=str(e), headers=error_headers
            )

    async def aembedding(
        self,
        data: dict,
        model_response: EmbeddingResponse,
        azure_client_params: dict,
        input: list,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        client: Optional[AsyncAzureOpenAI] = None,
        timeout=None,
    ):
        response = None
        try:
            if client is None:
                openai_aclient = AsyncAzureOpenAI(**azure_client_params)
            else:
                openai_aclient = client
            raw_response = await openai_aclient.embeddings.with_raw_response.create(
                **data, timeout=timeout
            )
            headers = dict(raw_response.headers)
            response = raw_response.parse()
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
                hidden_params={"headers": headers},
                _response_headers=process_azure_headers(headers),
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
        api_base: str,
        api_version: str,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        model_response: EmbeddingResponse,
        optional_params: dict,
        api_key: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        max_retries: Optional[int] = None,
        client=None,
        aembedding=None,
        headers: Optional[dict] = None,
    ) -> EmbeddingResponse:
        if headers:
            optional_params["extra_headers"] = headers
        if self._client_session is None:
            self._client_session = self.create_client_session()
        try:
            data = {"model": model, "input": input, **optional_params}
            if max_retries is None:
                max_retries = litellm.DEFAULT_MAX_RETRIES
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
            if aembedding:
                azure_client_params["http_client"] = litellm.aclient_session
            else:
                azure_client_params["http_client"] = litellm.client_session
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
                return self.aembedding(  # type: ignore
                    data=data,
                    input=input,
                    logging_obj=logging_obj,
                    api_key=api_key,
                    model_response=model_response,
                    azure_client_params=azure_client_params,
                    timeout=timeout,
                    client=client,
                )
            if client is None:
                azure_client = AzureOpenAI(**azure_client_params)  # type: ignore
            else:
                azure_client = client
            ## COMPLETION CALL
            raw_response = azure_client.embeddings.with_raw_response.create(**data, timeout=timeout)  # type: ignore
            headers = dict(raw_response.headers)
            response = raw_response.parse()
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data, "api_base": api_base},
                original_response=response,
            )

            return convert_to_model_response_object(response_object=response.model_dump(), model_response_object=model_response, response_type="embedding", _response_headers=process_azure_headers(headers))  # type: ignore
        except AzureOpenAIError as e:
            raise e
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise AzureOpenAIError(
                status_code=status_code, message=str(e), headers=error_headers
            )

    async def make_async_azure_httpx_request(
        self,
        client: Optional[AsyncHTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        api_base: str,
        api_version: str,
        api_key: str,
        data: dict,
        headers: dict,
    ) -> httpx.Response:
        """
        Implemented for azure dall-e-2 image gen calls

        Alternative to needing a custom transport implementation
        """
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            async_handler = get_async_httpx_client(
                llm_provider=LlmProviders.AZURE,
                params=_params,
            )
        else:
            async_handler = client  # type: ignore

        if (
            "images/generations" in api_base
            and api_version
            in [  # dall-e-3 starts from `2023-12-01-preview` so we should be able to avoid conflict
                "2023-06-01-preview",
                "2023-07-01-preview",
                "2023-08-01-preview",
                "2023-09-01-preview",
                "2023-10-01-preview",
            ]
        ):  # CREATE + POLL for azure dall-e-2 calls

            api_base = modify_url(
                original_url=api_base, new_path="/openai/images/generations:submit"
            )

            data.pop(
                "model", None
            )  # REMOVE 'model' from dall-e-2 arg https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#request-a-generated-image-dall-e-2-preview
            response = await async_handler.post(
                url=api_base,
                data=json.dumps(data),
                headers=headers,
            )
            if "operation-location" in response.headers:
                operation_location_url = response.headers["operation-location"]
            else:
                raise AzureOpenAIError(status_code=500, message=response.text)
            response = await async_handler.get(
                url=operation_location_url,
                headers=headers,
            )

            await response.aread()

            timeout_secs: int = 120
            start_time = time.time()
            if "status" not in response.json():
                raise Exception(
                    "Expected 'status' in response. Got={}".format(response.json())
                )
            while response.json()["status"] not in ["succeeded", "failed"]:
                if time.time() - start_time > timeout_secs:

                    raise AzureOpenAIError(
                        status_code=408, message="Operation polling timed out."
                    )

                await asyncio.sleep(int(response.headers.get("retry-after") or 10))
                response = await async_handler.get(
                    url=operation_location_url,
                    headers=headers,
                )
                await response.aread()

            if response.json()["status"] == "failed":
                error_data = response.json()
                raise AzureOpenAIError(status_code=400, message=json.dumps(error_data))

            result = response.json()["result"]
            return httpx.Response(
                status_code=200,
                headers=response.headers,
                content=json.dumps(result).encode("utf-8"),
                request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
            )
        return await async_handler.post(
            url=api_base,
            json=data,
            headers=headers,
        )

    def make_sync_azure_httpx_request(
        self,
        client: Optional[HTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        api_base: str,
        api_version: str,
        api_key: str,
        data: dict,
        headers: dict,
    ) -> httpx.Response:
        """
        Implemented for azure dall-e-2 image gen calls

        Alternative to needing a custom transport implementation
        """
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler = HTTPHandler(**_params, client=litellm.client_session)  # type: ignore
        else:
            sync_handler = client  # type: ignore

        if (
            "images/generations" in api_base
            and api_version
            in [  # dall-e-3 starts from `2023-12-01-preview` so we should be able to avoid conflict
                "2023-06-01-preview",
                "2023-07-01-preview",
                "2023-08-01-preview",
                "2023-09-01-preview",
                "2023-10-01-preview",
            ]
        ):  # CREATE + POLL for azure dall-e-2 calls

            api_base = modify_url(
                original_url=api_base, new_path="/openai/images/generations:submit"
            )

            data.pop(
                "model", None
            )  # REMOVE 'model' from dall-e-2 arg https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#request-a-generated-image-dall-e-2-preview
            response = sync_handler.post(
                url=api_base,
                data=json.dumps(data),
                headers=headers,
            )
            if "operation-location" in response.headers:
                operation_location_url = response.headers["operation-location"]
            else:
                raise AzureOpenAIError(status_code=500, message=response.text)
            response = sync_handler.get(
                url=operation_location_url,
                headers=headers,
            )

            response.read()

            timeout_secs: int = 120
            start_time = time.time()
            if "status" not in response.json():
                raise Exception(
                    "Expected 'status' in response. Got={}".format(response.json())
                )
            while response.json()["status"] not in ["succeeded", "failed"]:
                if time.time() - start_time > timeout_secs:
                    raise AzureOpenAIError(
                        status_code=408, message="Operation polling timed out."
                    )

                time.sleep(int(response.headers.get("retry-after") or 10))
                response = sync_handler.get(
                    url=operation_location_url,
                    headers=headers,
                )
                response.read()

            if response.json()["status"] == "failed":
                error_data = response.json()
                raise AzureOpenAIError(status_code=400, message=json.dumps(error_data))

            result = response.json()["result"]
            return httpx.Response(
                status_code=200,
                headers=response.headers,
                content=json.dumps(result).encode("utf-8"),
                request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
            )
        return sync_handler.post(
            url=api_base,
            json=data,
            headers=headers,
        )

    def create_azure_base_url(
        self, azure_client_params: dict, model: Optional[str]
    ) -> str:
        api_base: str = azure_client_params.get(
            "azure_endpoint", ""
        )  # "https://example-endpoint.openai.azure.com"
        if api_base.endswith("/"):
            api_base = api_base.rstrip("/")
        api_version: str = azure_client_params.get("api_version", "")
        if model is None:
            model = ""

        if "/openai/deployments/" in api_base:
            base_url_with_deployment = api_base
        else:
            base_url_with_deployment = api_base + "/openai/deployments/" + model

        base_url_with_deployment += "/images/generations"
        base_url_with_deployment += "?api-version=" + api_version

        return base_url_with_deployment

    async def aimage_generation(
        self,
        data: dict,
        model_response: ModelResponse,
        azure_client_params: dict,
        api_key: str,
        input: list,
        logging_obj: LiteLLMLoggingObj,
        headers: dict,
        client=None,
        timeout=None,
    ) -> litellm.ImageResponse:
        response: Optional[dict] = None
        try:
            # response = await azure_client.images.generate(**data, timeout=timeout)
            api_base: str = azure_client_params.get(
                "api_base", ""
            )  # "https://example-endpoint.openai.azure.com"
            if api_base.endswith("/"):
                api_base = api_base.rstrip("/")
            api_version: str = azure_client_params.get("api_version", "")
            img_gen_api_base = self.create_azure_base_url(
                azure_client_params=azure_client_params, model=data.get("model", "")
            )

            ## LOGGING
            logging_obj.pre_call(
                input=data["prompt"],
                api_key=api_key,
                additional_args={
                    "complete_input_dict": data,
                    "api_base": img_gen_api_base,
                    "headers": headers,
                },
            )
            httpx_response: httpx.Response = await self.make_async_azure_httpx_request(
                client=None,
                timeout=timeout,
                api_base=img_gen_api_base,
                api_version=api_version,
                api_key=api_key,
                data=data,
                headers=headers,
            )
            response = httpx_response.json()

            stringified_response = response
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=stringified_response,
            )
            return convert_to_model_response_object(  # type: ignore
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
        optional_params: dict,
        logging_obj: LiteLLMLoggingObj,
        headers: dict,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        model_response: Optional[ImageResponse] = None,
        azure_ad_token: Optional[str] = None,
        client=None,
        aimg_generation=None,
    ) -> ImageResponse:
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

            if aimg_generation is True:
                return self.aimage_generation(data=data, input=input, logging_obj=logging_obj, model_response=model_response, api_key=api_key, client=client, azure_client_params=azure_client_params, timeout=timeout, headers=headers)  # type: ignore

            img_gen_api_base = self.create_azure_base_url(
                azure_client_params=azure_client_params, model=data.get("model", "")
            )

            ## LOGGING
            logging_obj.pre_call(
                input=data["prompt"],
                api_key=api_key,
                additional_args={
                    "complete_input_dict": data,
                    "api_base": img_gen_api_base,
                    "headers": headers,
                },
            )
            httpx_response: httpx.Response = self.make_sync_azure_httpx_request(
                client=None,
                timeout=timeout,
                api_base=img_gen_api_base,
                api_version=api_version or "",
                api_key=api_key or "",
                data=data,
                headers=headers,
            )
            response = httpx_response.json()

            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response,
            )
            # return response
            return convert_to_model_response_object(response_object=response, model_response_object=model_response, response_type="image_generation")  # type: ignore
        except AzureOpenAIError as e:
            raise e
        except Exception as e:
            error_code = getattr(e, "status_code", None)
            if error_code is not None:
                raise AzureOpenAIError(status_code=error_code, message=str(e))
            else:
                raise AzureOpenAIError(status_code=500, message=str(e))

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
        return HttpxBinaryResponseContent(response=response.response)

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

        azure_response = await azure_client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore
            input=input,
            **optional_params,
        )

        return HttpxBinaryResponseContent(response=azure_response.response)

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
        client_session = litellm.client_session or httpx.Client()
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
