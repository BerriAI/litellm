from typing import List, Dict, Literal, Tuple, Callable, Optional, TYPE_CHECKING, Union
import urllib

from aiohttp import ClientSession
from functools import cached_property
from typing import Optional

from litellm.types.utils import GenericStreamingChunk
import httpx

import litellm

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.types.utils import ModelResponse
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.databricks.streaming_utils import ModelResponseIterator


from ..credentials import get_token_creator
from ...base import BaseLLM


class GenAIHubOrchestrationError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class SAPModelResponseIterator(ModelResponseIterator):

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        chunk = chunk["final_result"]
        return super().chunk_parser(chunk)


class GenAIHubOrchestration(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self._access_token_data = {}
        self.token_creator, self.base_url, self.resource_group = get_token_creator()


    @property
    def headers(self) -> Dict:
        access_token = self.token_creator()
        # headers for completions and embeddings requests
        headers = {
            "Authorization": access_token,
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json",
        }
        return headers

    @cached_property
    def deployment_url(self) -> str:
        with httpx.Client(timeout=30) as client:
            valid_deployments = []
            deployments = client.get(self.base_url + '/lm/deployments', headers=self.headers).json()
            for deployment in deployments.get('resources', []):
                if deployment['scenarioId'] == 'orchestration':
                    config_details = client.get(self.base_url + f'/lm/configurations/{deployment["configurationId"]}', headers=self.headers).json()
                    if config_details["executableId"] == 'orchestration':
                        valid_deployments.append((deployment["deploymentUrl"], deployment["createdAt"]))
            return sorted(valid_deployments, key=lambda x: x[1], reverse=True)[0][0]


    def validate_environment(
        self,
        endpoint_type: Literal["chat_completions", "embeddings"]
    ) -> Tuple[str, dict]:
        if endpoint_type == "chat_completions":
            api_base = "{}/v2/completion".format(self.deployment_url)
        elif endpoint_type == "embeddings":
            api_base = "{}/v2/embeddings".format(self.deployment_url)

        return api_base, self.headers


    def encode_model_id(self, model_id: str) -> str:
        """
        Double encode the model ID to ensure it matches the expected double-encoded format.
        Args:
            model_id (str): The model ID to encode.
        Returns:
            str: The double-encoded model ID.
        """
        return urllib.parse.quote(model_id, safe="")  # type: ignore

    async def _async_streaming(
        self,
        config,
        model: str,
        api_base: str,
        headers: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
        shared_session: Optional["ClientSession"] = None,
        client: Optional[AsyncHTTPHandler] = None
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            if client is None or not isinstance(client, AsyncHTTPHandler):
                client = litellm.AsyncHTTPHandler(shared_session=shared_session)
            data = config
            response = await client.post(
                url=api_base,
                headers=headers,
                json=data,
                timeout=timeout,
                stream=True
            )
            response.raise_for_status()
            completion_stream = SAPModelResponseIterator(
                streaming_response=response.aiter_lines(), sync_stream=False
            )
        except httpx.HTTPStatusError as err:
            raise GenAIHubOrchestrationError(
                status_code=err.response.status_code,
                message=err.response.text
            )
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(
                status_code=408, message="Timeout error occurred."
            )

        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            logging_obj=logging_obj,
            # custom_llm_provider="sap",
            stream_options={},
            make_call=None,
        )

    async def _async_completion(
        self,
        config,
        model: str,
        api_base: str,
        headers: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
        client: Optional[AsyncHTTPHandler] = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            if client is None or not isinstance(client, AsyncHTTPHandler):
                client = litellm.AsyncHTTPHandler(shared_session=shared_session)
            response = await client.post(
                url=api_base,
                headers=headers,
                json=config,
                timeout=timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise GenAIHubOrchestrationError(
                status_code=err.response.status_code,
                message=err.response.text
            )
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(
                status_code=408, message="Timeout error occurred."
            )
        return litellm.GenAIHubOrchestrationConfig()._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
        )

    def _streaming(
        self,
        config,
        model: str,
        api_base: str,
        headers: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            if client is None or not isinstance(client, HTTPHandler):
                client = litellm.module_level_client
            data = config
            # data["stream"] = {"enabled": True}
            response = client.post(
                url=api_base,
                headers=headers,
                json=data,
                stream=True,
                timeout=timeout)
            response.raise_for_status()
            completion_stream = SAPModelResponseIterator(
                streaming_response=response.iter_lines(), sync_stream=True
            )
        except httpx.HTTPStatusError as err:
            raise GenAIHubOrchestrationError(
                status_code=err.response.status_code,
                message=err.response.text
            )
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(
                status_code=408, message="Timeout error occurred."
            )

        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            logging_obj=logging_obj,
            # custom_llm_provider="sap",
            stream_options={},
            make_call=None,
        )

    def _complete(
        self,
        config,
        model: str,
        api_base: str,
        headers: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            if client is None or not isinstance(client, HTTPHandler):
                client = litellm.module_level_client
            response = client.post(
                url=api_base,
                headers=headers,
                json=config,
                timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise GenAIHubOrchestrationError(
                status_code=err.response.status_code,
                message=err.response.text
            )
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(
                status_code=408, message="Timeout error occurred."
            )
        return litellm.GenAIHubOrchestrationConfig()._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
        )

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj: LiteLLMLoggingObject,
        optional_params: dict,
        acompletion: bool,
        headers: Optional[Dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        litellm_params: Optional[Dict] = None,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        shared_session: Optional["ClientSession"] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        **kwargs
    ):
        stream = optional_params.get("stream", None)

        api_base, headers = self.validate_environment(
            endpoint_type="chat_completions"
        )

        config = litellm.GenAIHubOrchestrationConfig()._transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        ## COMPLETION CALL
        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": config,
                "api_base": api_base,
                "headers": headers,
            },
        )

        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion and not stream:
            return self._async_completion(
                config=config,
                model=model,
                api_base=api_base,
                headers=headers,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                shared_session=shared_session,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )
        elif acompletion and stream is not None and stream is True:
            return self._async_streaming(
                config=config,
                model=model,
                api_base=api_base,
                headers=headers,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )

        elif stream is not None and stream is True:
            return self._streaming(
                config=config,
                model=model,
                api_base=api_base,
                headers=headers,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )
        else:
            return self._complete(
                config=config,
                model=model,
                api_base=api_base,
                headers=headers,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )
