from typing import Any, Callable, Optional

from openai import AsyncAzureOpenAI, AzureOpenAI

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import prompt_factory
from litellm.utils import CustomStreamWrapper, ModelResponse, TextCompletionResponse

from ...openai.completion.transformation import OpenAITextCompletionConfig
from ..common_utils import AzureOpenAIError, BaseAzureLLM

openai_text_completion_config = OpenAITextCompletionConfig()


def select_azure_base_url_or_endpoint(azure_client_params: dict):
    azure_endpoint = azure_client_params.get("azure_endpoint", None)
    if azure_endpoint is not None:
        # see : https://github.com/openai/openai-python/blob/3d61ed42aba652b547029095a7eb269ad4e1e957/src/openai/lib/azure.py#L192
        if "/openai/deployments" in azure_endpoint:
            # this is base_url, not an azure_endpoint
            azure_client_params["base_url"] = azure_endpoint
            azure_client_params.pop("azure_endpoint")

    return azure_client_params


class AzureTextCompletion(BaseAzureLLM):
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
        azure_ad_token_provider: Optional[Callable],
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
        try:
            if model is None or messages is None:
                raise AzureOpenAIError(
                    status_code=422, message="Missing model or messages"
                )

            max_retries = optional_params.pop("max_retries", 2)
            prompt = prompt_factory(
                messages=messages, model=model, custom_llm_provider="azure_text"
            )

            azure_client_params = self.initialize_azure_sdk_client(
                litellm_params=litellm_params or {},
                api_key=api_key,
                model_name=model,
                api_version=api_version,
                api_base=api_base,
            )

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

                data = {"model": None, "prompt": prompt, **optional_params}
            else:
                data = {
                    "model": model,  # type: ignore
                    "prompt": prompt,
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
                        azure_client_params=azure_client_params,
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
                        max_retries=max_retries,
                        azure_client_params=azure_client_params,
                    )
            elif "stream" in optional_params and optional_params["stream"] is True:
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
                    azure_client_params=azure_client_params,
                )
            else:
                ## LOGGING
                logging_obj.pre_call(
                    input=prompt,
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

                raw_response = azure_client.completions.with_raw_response.create(
                    **data, timeout=timeout
                )
                response = raw_response.parse()
                stringified_response = response.model_dump()
                ## LOGGING
                logging_obj.post_call(
                    input=prompt,
                    api_key=api_key,
                    original_response=stringified_response,
                    additional_args={
                        "headers": headers,
                        "api_version": api_version,
                        "api_base": api_base,
                    },
                )
                return (
                    openai_text_completion_config.convert_to_chat_model_response_object(
                        response_object=TextCompletionResponse(**stringified_response),
                        model_response_object=model_response,
                    )
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
        model_response: ModelResponse,
        logging_obj: Any,
        max_retries: int,
        azure_ad_token: Optional[str] = None,
        client=None,  # this is the AsyncAzureOpenAI
        azure_client_params: dict = {},
    ):
        response = None
        try:
            # init AzureOpenAI Client
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
                input=data["prompt"],
                api_key=azure_client.api_key,
                additional_args={
                    "headers": {"Authorization": f"Bearer {azure_client.api_key}"},
                    "api_base": azure_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )
            raw_response = await azure_client.completions.with_raw_response.create(
                **data, timeout=timeout
            )
            response = raw_response.parse()
            return openai_text_completion_config.convert_to_chat_model_response_object(
                response_object=response.model_dump(),
                model_response_object=model_response,
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
        azure_client_params: dict = {},
    ):
        max_retries = data.pop("max_retries", 2)
        if not isinstance(max_retries, int):
            raise AzureOpenAIError(
                status_code=422, message="max retries must be an int"
            )
        # init AzureOpenAI Client
        if client is None:
            azure_client = AzureOpenAI(**azure_client_params)
        else:
            azure_client = client
            if api_version is not None and isinstance(azure_client._custom_query, dict):
                # set api_version to version passed by user
                azure_client._custom_query.setdefault("api-version", api_version)
        ## LOGGING
        logging_obj.pre_call(
            input=data["prompt"],
            api_key=azure_client.api_key,
            additional_args={
                "headers": {"Authorization": f"Bearer {azure_client.api_key}"},
                "api_base": azure_client._base_url._uri_reference,
                "acompletion": True,
                "complete_input_dict": data,
            },
        )
        raw_response = azure_client.completions.with_raw_response.create(
            **data, timeout=timeout
        )
        response = raw_response.parse()
        streamwrapper = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="azure_text",
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
        azure_client_params: dict = {},
    ):
        try:
            # init AzureOpenAI Client
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
                input=data["prompt"],
                api_key=azure_client.api_key,
                additional_args={
                    "headers": {"Authorization": f"Bearer {azure_client.api_key}"},
                    "api_base": azure_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )
            raw_response = await azure_client.completions.with_raw_response.create(
                **data, timeout=timeout
            )
            response = raw_response.parse()
            # return response
            streamwrapper = CustomStreamWrapper(
                completion_stream=response,
                model=model,
                custom_llm_provider="azure_text",
                logging_obj=logging_obj,
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
