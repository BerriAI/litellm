"""
Recraft AI Image Generation Configuration

This module provides configuration for Recraft AI's image generation API.
"""

import json
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object


class RecraftImageGenerationConfig(BaseImageGenerationConfig):
    """
    Recraft AI image generation configuration
    
    Supports Recraft V2 and V3 models with various styles and parameters.
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI-compatible parameters for Recraft image generation.
        
        Args:
            model: The model name
            
        Returns:
            List of supported parameter names
        """
        return ["n", "response_format", "size", "user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Recraft-specific parameters.
        
        Args:
            non_default_params: Non-default parameters
            optional_params: Optional parameters
            model: Model name
            drop_params: Whether to drop unsupported params
            
        Returns:
            Mapped parameters for Recraft API
        """
        supported_params = self.get_supported_openai_params(model)
        
        # Map OpenAI params to Recraft params
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
            elif k == "style":
                # Map OpenAI style to Recraft style
                optional_params["style"] = v
            elif k == "substyle": 
                # Recraft-specific parameter
                optional_params["substyle"] = v
            elif k == "negative_prompt":
                # Recraft-specific parameter  
                optional_params["negative_prompt"] = v
            elif k == "controls":
                # Recraft-specific parameter
                optional_params["controls"] = v
            elif k == "text_layout":
                # Recraft V3 specific parameter
                optional_params["text_layout"] = v
            elif k == "style_id":
                # Custom style reference
                optional_params["style_id"] = v
            elif drop_params:
                pass
            else:
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. "
                    f"Supported parameters are {supported_params}. "
                    f"Set drop_params=True to drop unsupported parameters."
                )

        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for Recraft API requests.
        
        Args:
            api_base: Base API URL
            api_key: API key
            model: Model name
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            stream: Whether streaming is enabled
            
        Returns:
            Complete URL for the request
        """
        if api_base:
            return f"{api_base.rstrip('/')}/images/generations"
        else:
            return "https://external.api.recraft.ai/v1/images/generations"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate and set up the environment for Recraft API calls.
        
        Args:
            headers: Request headers
            model: Model name
            messages: Messages (not used for image generation)
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            api_key: API key
            api_base: API base URL
            
        Returns:
            Updated headers
        """
        if api_key is None:
            api_key = (
                get_secret_str("RECRAFT_API_KEY")
                or get_secret_str("RECRAFT_API_TOKEN")
                or litellm_params.get("api_key")
            )
            
        if api_key is None:
            raise ValueError(
                "Recraft API key is required. Set RECRAFT_API_KEY environment variable "
                "or pass api_key parameter."
            )
            
        headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        
        return headers

    def transform_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to Recraft API format.
        
        Args:
            model: Model name
            prompt: Image generation prompt
            optional_params: Optional parameters
            headers: Request headers
            
        Returns:
            Request data for Recraft API
        """
        # Set default model if not specified
        if not model or model == "dall-e-2":  # Default mapping
            model = "recraftv3"
            
        data = {
            "prompt": prompt,
            "model": model,
        }
        
        # Add optional parameters
        if "n" in optional_params:
            data["n"] = optional_params["n"]
        if "size" in optional_params:
            data["size"] = optional_params["size"]
        if "response_format" in optional_params:
            data["response_format"] = optional_params["response_format"]
        if "style" in optional_params:
            data["style"] = optional_params["style"]
        if "substyle" in optional_params:
            data["substyle"] = optional_params["substyle"]
        if "negative_prompt" in optional_params:
            data["negative_prompt"] = optional_params["negative_prompt"]
        if "controls" in optional_params:
            data["controls"] = optional_params["controls"]
        if "text_layout" in optional_params:
            data["text_layout"] = optional_params["text_layout"]
        if "style_id" in optional_params:
            data["style_id"] = optional_params["style_id"]
        if "user" in optional_params:
            data["user"] = optional_params["user"]
            
        return data

    def transform_response(
        self,
        model: str,
        raw_response: Union[httpx.Response, Dict],
        model_response: ImageResponse,
        logging_obj: Any,
        api_key: str,
        prompt: str,
        optional_params: dict,
    ) -> ImageResponse:
        """
        Transform Recraft API response to LiteLLM format.
        
        Args:
            model: Model name
            raw_response: Raw response from Recraft
            model_response: Model response object to populate
            logging_obj: Logging object
            api_key: API key used
            prompt: Original prompt
            optional_params: Optional parameters used
            
        Returns:
            Populated ImageResponse object
        """
        if isinstance(raw_response, httpx.Response):
            response_json = raw_response.json()
        else:
            response_json = raw_response

        # Log the API call
        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            original_response=response_json,
            additional_args={"complete_input_dict": optional_params},
        )

        # Transform to OpenAI-compatible format
        return convert_to_model_response_object(
            response_object=response_json,
            model_response_object=model_response,
            response_type="image_generation",
        )

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ImageResponse:
        """
        Async image generation for Recraft.
        
        Args:
            model: Model name
            prompt: Image generation prompt
            api_key: API key
            api_base: API base URL
            model_response: Model response object
            optional_params: Optional parameters
            logging_obj: Logging object
            timeout: Request timeout
            client: HTTP client
            
        Returns:
            Generated image response
        """
        headers = {}
        headers = self.validate_environment(
            headers=headers,
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params={"api_key": api_key},
            api_key=api_key,
            api_base=api_base,
        )
        
        url = self.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params={},
            stream=False,
        )
        
        data = self.transform_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            headers=headers,
        )
        
        # Log the pre-call
        logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )
        
        if client is None:
            timeout_val = timeout if timeout else 60.0
            if isinstance(timeout_val, httpx.Timeout):
                client = AsyncHTTPHandler(timeout=timeout_val)
            else:
                client = AsyncHTTPHandler(timeout=httpx.Timeout(timeout_val))
        
        response = await client.post(
            url=url,
            headers=headers,
            data=json.dumps(data),
        )
        
        if response.status_code != 200:
            error_message = f"Recraft API Error: {response.status_code} {response.text}"
            raise Exception(error_message)
        
        return self.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            prompt=prompt,
            optional_params=optional_params,
        )

    def image_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ImageResponse:
        """
        Sync image generation for Recraft.
        
        Args:
            model: Model name
            prompt: Image generation prompt
            api_key: API key
            api_base: API base URL
            model_response: Model response object
            optional_params: Optional parameters
            logging_obj: Logging object
            timeout: Request timeout
            client: HTTP client
            
        Returns:
            Generated image response
        """
        headers = {}
        headers = self.validate_environment(
            headers=headers,
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params={"api_key": api_key},
            api_key=api_key,
            api_base=api_base,
        )
        
        url = self.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params={},
            stream=False,
        )
        
        data = self.transform_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            headers=headers,
        )
        
        # Log the pre-call
        logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )
        
        if client is None:
            timeout_val = timeout if timeout else 60.0
            if isinstance(timeout_val, httpx.Timeout):
                client = HTTPHandler(timeout=timeout_val)
            else:
                client = HTTPHandler(timeout=httpx.Timeout(timeout_val))
        
        response = client.post(
            url=url,
            headers=headers,
            data=json.dumps(data),
        )
        
        if response.status_code != 200:
            error_message = f"Recraft API Error: {response.status_code} {response.text}"
            raise Exception(error_message)
        
        return self.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            prompt=prompt,
            optional_params=optional_params,
        )