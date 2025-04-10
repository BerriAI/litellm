from typing import Any, Callable, Optional

from openai import AsyncAzureOpenAI, AzureOpenAI

from litellm.litellm_core_utils.prompt_templates.factory import prompt_factory
from litellm.utils import CustomStreamWrapper, ModelResponse, TextCompletionResponse

from ...openai.completion.transformation import OpenAITextCompletionConfig
from ..common_utils import AzureOpenAIError, BaseAzureLLM

openai_text_completion_config = OpenAITextCompletionConfig()


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

            ### CHECK IF CLOUDFLARE AI GATEWAY ###
            ### if so - set the model as part of the base url
            if "gateway.ai.cloudflare.com" in api_base:
                ## build base url - assume api base includes resource name
                client = self._init_azure_client_for_cloudflare_ai_gateway(
                    api_key=api_key,
                    api_version=api_version,
                    api_base=api_base,
                    model=model,
                    client=client,
                    max_retries=max_retries,
                    timeout=timeout,
                    azure_ad_token=azure_ad_token,
                    azure_ad_token_provider=azure_ad_token_provider,
                    acompletion=acompletion,
                )

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
                        litellm_params=litellm_params,
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
                        litellm_params=litellm_params,
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
                azure_client = self.get_azure_openai_client(
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    client=client,
                    litellm_params=litellm_params,
                    _is_async=False,
                    model=model,
                )

                if not isinstance(azure_client, AzureOpenAI):
                    raise AzureOpenAIError(
                        status_code=500,
                        message="azure_client is not an instance of AzureOpenAI",
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
        litellm_params: dict = {},
    ):
        response = None
        try:
            # init AzureOpenAI Client
            # setting Azure client
            azure_client = self.get_azure_openai_client(
                api_version=api_version,
                api_base=api_base,
                api_key=api_key,
                model=model,
                _is_async=True,
                client=client,
                litellm_params=litellm_params,
            )
            if not isinstance(azure_client, AsyncAzureOpenAI):
                raise AzureOpenAIError(
                    status_code=500,
                    message="azure_client is not an instance of AsyncAzureOpenAI",
                )

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
        litellm_params: dict = {},
    ):
        max_retries = data.pop("max_retries", 2)
        if not isinstance(max_retries, int):
            raise AzureOpenAIError(
                status_code=422, message="max retries must be an int"
            )
        # init AzureOpenAI Client
        azure_client = self.get_azure_openai_client(
            api_version=api_version,
            api_base=api_base,
            api_key=api_key,
            model=model,
            _is_async=False,
            client=client,
            litellm_params=litellm_params,
        )
        if not isinstance(azure_client, AzureOpenAI):
            raise AzureOpenAIError(
                status_code=500,
                message="azure_client is not an instance of AzureOpenAI",
            )

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
        litellm_params: dict = {},
    ):
        try:
            # init AzureOpenAI Client
            azure_client = self.get_azure_openai_client(
                api_version=api_version,
                api_base=api_base,
                api_key=api_key,
                model=model,
                _is_async=True,
                client=client,
                litellm_params=litellm_params,
            )
            if not isinstance(azure_client, AsyncAzureOpenAI):
                raise AzureOpenAIError(
                    status_code=500,
                    message="azure_client is not an instance of AsyncAzureOpenAI",
                )
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
