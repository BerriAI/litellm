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
        
        # Remove 'model' from the request as it's not needed in the instance
        openai_request.pop("model", None)
        
        # Wrap in Vertex Gemma format
        return {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    **openai_request,
                }
            ]
        }
    
    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Async version of transform_request.
        """
        # Get the base OpenAI request from parent class
        openai_request = await super().async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        # Remove 'model' from the request as it's not needed in the instance
        openai_request.pop("model", None)
        
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
        client: Optional[httpx.Client] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        encoding=None,
        custom_llm_provider: str = "vertex_ai",
    ):
        """
        Make completion request to Vertex Gemma endpoint.
        Supports both sync and async requests.
        """
        # Handle streaming
        stream = optional_params.get("stream", False)
        if stream:
            raise BaseLLMException(
                status_code=400,
                message="Streaming is not yet supported for Vertex AI Gemma models",
            )

        if acompletion:
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
                timeout=timeout,
                encoding=encoding,
            )
        else:
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
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding: Any,
    ):
        """Synchronous completion request"""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        from litellm.utils import convert_to_model_response_object

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
        http_handler = HTTPHandler(concurrent_limit=1)
        response = http_handler.post(
            url=api_base,
            headers=headers,
            json=request_data,
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
        
        return model_response

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
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding: Any,
    ):
        """Asynchronous completion request"""
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
        from litellm.utils import convert_to_model_response_object

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
        http_handler = AsyncHTTPHandler(concurrent_limit=1)
        response = await http_handler.post(
            url=api_base,
            headers=headers,
            json=request_data,
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
        
        return model_response

