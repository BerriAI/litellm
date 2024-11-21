from typing import Callable, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.watsonx import WatsonXAIEndpoint, WatsonXAPIParams
from litellm.types.utils import CustomStreamingDecoder, ModelResponse

from ...openai_like.chat.handler import OpenAILikeChatHandler
from ..common_utils import WatsonXAIError, _get_api_params


class WatsonXChatHandler(OpenAILikeChatHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare_url(
        self, model: str, api_params: WatsonXAPIParams, stream: Optional[bool]
    ) -> str:
        if model.startswith("deployment/"):
            if api_params.get("space_id") is None:
                raise WatsonXAIError(
                    status_code=401,
                    url=api_params["url"],
                    message="Error: space_id is required for models called using the 'deployment/' endpoint. Pass in the space_id as a parameter or set it in the WX_SPACE_ID environment variable.",
                )
            deployment_id = "/".join(model.split("/")[1:])
            endpoint = (
                WatsonXAIEndpoint.DEPLOYMENT_CHAT_STREAM.value
                if stream is True
                else WatsonXAIEndpoint.DEPLOYMENT_CHAT.value
            )
            endpoint = endpoint.format(deployment_id=deployment_id)
        else:
            endpoint = (
                WatsonXAIEndpoint.CHAT_STREAM.value
                if stream is True
                else WatsonXAIEndpoint.CHAT.value
            )
        base_url = httpx.URL(api_params["url"])
        base_url = base_url.join(endpoint)
        full_url = str(
            base_url.copy_add_param(key="version", value=api_params["api_version"])
        )

        return full_url

    def _prepare_payload(
        self, model: str, api_params: WatsonXAPIParams, stream: Optional[bool]
    ) -> dict:
        payload: dict = {}
        if model.startswith("deployment/"):
            return payload
        payload["model_id"] = model
        payload["project_id"] = api_params["project_id"]
        return payload

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
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
        fake_stream: bool = False,
    ):
        api_params = _get_api_params(optional_params, print_verbose=print_verbose)

        if headers is None:
            headers = {}
        headers.update(
            {
                "Authorization": f"Bearer {api_params['token']}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        stream: Optional[bool] = optional_params.get("stream", False)

        ## get api url and payload
        api_base = self._prepare_url(model=model, api_params=api_params, stream=stream)
        watsonx_auth_payload = self._prepare_payload(
            model=model, api_params=api_params, stream=stream
        )
        optional_params.update(watsonx_auth_payload)

        return super().completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
            timeout=timeout,
            client=client,
            custom_endpoint=True,
            streaming_decoder=streaming_decoder,
        )
