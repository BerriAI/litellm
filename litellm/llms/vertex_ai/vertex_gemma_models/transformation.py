"""
Transformation logic for Vertex AI Gemma Models

Handles the custom request/response format:
- Request: Wraps messages in 'instances' with @requestFormat: "chatCompletions"
- Response: Extracts data from 'predictions' wrapper

The actual message transformation reuses OpenAIGPTConfig since Gemma uses OpenAI-compatible format.
"""

from typing import Any, Callable, Dict, List, Optional, Union, cast

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


class VertexGemmaConfig(OpenAIGPTConfig):
    """
    Configuration and transformation class for Vertex AI Gemma models

    Extends OpenAIGPTConfig to wrap/unwrap the instances/predictions format
    used by Vertex AI's Gemma deployment endpoint.
    """

    def __init__(self) -> None:
        super().__init__()

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Vertex AI Gemma models do not support streaming.
        Return True to enable fake streaming on the client side.
        """
        return True

    def _handle_fake_stream_response(
        self,
        model_response: ModelResponse,
        stream: bool,
    ) -> Union[ModelResponse, Any]:
        """
        Helper method to return fake stream iterator if streaming is requested.

        Args:
            model_response: The completed model response
            stream: Whether streaming was requested

        Returns:
            MockResponseIterator if stream=True, otherwise the model_response
        """
        if stream:
            from litellm.llms.base_llm.base_model_iterator import MockResponseIterator

            return MockResponseIterator(model_response=model_response)
        return model_response

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request to Vertex Gemma format.

        Uses parent class to create OpenAI-compatible request, then wraps it
        in the Vertex Gemma instances format.
        """
        # Get the base OpenAI request from parent class
        openai_request = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Remove params not needed/supported by Vertex Gemma
        openai_request.pop("model", None)
        openai_request.pop("stream", None)  # Streaming not supported, will be faked client-side
        openai_request.pop("stream_options", None)  # Stream options not supported
        # Vertex Gemma's chatCompletions wrapper does not understand
        # `context_management` (an Anthropic/Responses API concept). Strip it
        # so the upstream endpoint does not 400 on the unknown field.
        openai_request.pop("context_management", None)

        # Wrap in Vertex Gemma format
        return {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    **openai_request,
                }
            ]
        }

    def _unwrap_predictions_response(
        self,
        response_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Unwrap the Vertex Gemma predictions format to OpenAI format.

        Vertex Gemma wraps the OpenAI-compatible response in a 'predictions' field.
        This method extracts it so the parent class can process it normally.
        """
        if "predictions" not in response_json:
            raise BaseLLMException(
                status_code=422,
                message="Invalid response format: missing 'predictions' field",
            )

        return response_json["predictions"]

    @staticmethod
    def _sync_post(
        client: HTTPHandler | httpx.Client | None,
        api_base: str,
        headers: dict,
        request_data: dict,
        timeout: float | httpx.Timeout | None,
    ) -> httpx.Response:
        if isinstance(client, HTTPHandler):
            return client.post(
                url=api_base,
                headers=headers,
                json=request_data,
                timeout=timeout,
            )
        if isinstance(client, httpx.Client):
            if timeout is None:
                return client.post(
                    url=api_base,
                    headers=headers,
                    json=request_data,
                )
            return client.post(
                url=api_base,
                headers=headers,
                json=request_data,
                timeout=timeout,
            )
        return _get_httpx_client().post(
            url=api_base,
            headers=headers,
            json=request_data,
            timeout=timeout,
        )

    @staticmethod
    async def _async_post(
        client: AsyncHTTPHandler | httpx.AsyncClient | None,
        api_base: str,
        headers: dict,
        request_data: dict,
        timeout: float | httpx.Timeout | None,
    ) -> httpx.Response:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
        from litellm.types.utils import LlmProviders

        if isinstance(client, AsyncHTTPHandler):
            return await client.post(
                url=api_base,
                headers=headers,
                json=request_data,
                timeout=timeout,
            )
        if isinstance(client, httpx.AsyncClient):
            if timeout is None:
                return await client.post(
                    url=api_base,
                    headers=headers,
                    json=request_data,
                )
            return await client.post(
                url=api_base,
                headers=headers,
                json=request_data,
                timeout=timeout,
            )
        return await get_async_httpx_client(llm_provider=LlmProviders.VERTEX_AI).post(
            url=api_base,
            headers=headers,
            json=request_data,
            timeout=timeout,
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        api_key: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        acompletion: bool,
        litellm_params: dict,
        logger_fn: Optional[Callable] = None,
        client: HTTPHandler | AsyncHTTPHandler | httpx.Client | httpx.AsyncClient | None = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        encoding=None,
        custom_llm_provider: str = "vertex_ai",
    ):
        """
        Make completion request to Vertex Gemma endpoint.
        Supports both sync and async requests with fake streaming.
        """
        if acompletion:
            async_client = client if isinstance(client, (AsyncHTTPHandler, httpx.AsyncClient)) else None
            return self._async_completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=api_key,
                model_response=model_response,
                print_verbose=print_verbose,
                logging_obj=logging_obj,
                optional_params=optional_params,
                litellm_params=litellm_params,
                client=async_client,
                timeout=timeout,
                encoding=encoding,
            )
        else:
            sync_client = client if isinstance(client, (HTTPHandler, httpx.Client)) else None
            return self._sync_completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=api_key,
                model_response=model_response,
                print_verbose=print_verbose,
                logging_obj=logging_obj,
                optional_params=optional_params,
                litellm_params=litellm_params,
                client=sync_client,
                timeout=timeout,
                encoding=encoding,
            )

    def _sync_completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        litellm_params: dict,
        client: HTTPHandler | httpx.Client | None = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        encoding: Any = None,
    ):
        """Synchronous completion request"""
        from litellm.utils import convert_to_model_response_object

        # Check if streaming is requested (will be faked)
        stream = optional_params.get("stream", False)

        # Transform the request using parent class methods
        request_data = self.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params.copy(),
            litellm_params=litellm_params,
            headers={},
        )

        # Set up headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Log the request
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": request_data,
                "api_base": api_base,
            },
        )

        # Make the HTTP request
        response = self._sync_post(
            client=client,
            api_base=api_base,
            headers=headers,
            request_data=request_data,
            timeout=timeout,
        )

        if response.status_code != 200:
            raise BaseLLMException(
                status_code=response.status_code,
                message=f"Request failed: {response.text}",
            )

        response_json = response.json()

        # Unwrap predictions to get OpenAI-compatible response
        openai_response = self._unwrap_predictions_response(response_json)

        # Use litellm's standard response converter
        model_response = cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=openai_response,
                model_response_object=model_response,
                _response_headers={},
            ),
        )

        # Ensure model is set correctly
        model_response.model = model

        # Log the response
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        # Return fake stream iterator if streaming was requested
        return self._handle_fake_stream_response(model_response=model_response, stream=stream)

    async def _async_completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        litellm_params: dict,
        client: Optional[Union[AsyncHTTPHandler, httpx.AsyncClient]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        encoding: Any = None,
    ):
        """Asynchronous completion request"""
        from litellm.utils import convert_to_model_response_object

        # Check if streaming is requested (will be faked)
        stream = optional_params.get("stream", False)

        # Transform the request using parent class async methods
        request_data = await self.async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params.copy(),
            litellm_params=litellm_params,
            headers={},
        )

        # Set up headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Log the request
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": request_data,
                "api_base": api_base,
            },
        )

        # Make the HTTP request
        response = await self._async_post(
            client=client,
            api_base=api_base,
            headers=headers,
            request_data=request_data,
            timeout=timeout,
        )

        if response.status_code != 200:
            raise BaseLLMException(
                status_code=response.status_code,
                message=f"Request failed: {response.text}",
            )

        response_json = response.json()

        # Unwrap predictions to get OpenAI-compatible response
        openai_response = self._unwrap_predictions_response(response_json)

        # Use litellm's standard response converter
        model_response = cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=openai_response,
                model_response_object=model_response,
                _response_headers={},
            ),
        )

        # Ensure model is set correctly
        model_response.model = model

        # Log the response
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        # Return fake stream iterator if streaming was requested
        return self._handle_fake_stream_response(model_response=model_response, stream=stream)
