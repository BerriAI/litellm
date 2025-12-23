"""
Image generation handler for JSON-configured providers.

Handles image generation requests using JSON-configured providers with
automatic transformations and cost tracking.
"""

import os
from typing import Any, Dict, Optional

import httpx
from openai.types.image import Image

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.json_providers.cost_tracker import CostTracker
from litellm.llms.json_providers.sdk_provider_registry import SDKProviderRegistry
from litellm.llms.json_providers.transformation_engine import TransformationEngine
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ImageResponse


class JSONProviderImageGeneration:
    """
    Image generation handler for JSON-configured providers.
    
    This class handles the complete flow:
    1. Load provider configuration
    2. Transform request from LiteLLM format to provider format
    3. Make API call with proper authentication
    4. Transform response from provider format to LiteLLM format
    5. Calculate and track costs
    """

    @staticmethod
    async def aimage_generation(
        prompt: str,
        model: str,
        provider_config_name: str,
        optional_params: Optional[Dict] = None,
        **kwargs
    ) -> ImageResponse:
        """
        Generate images using JSON-configured provider (async).
        
        Args:
            prompt: Text prompt for image generation
            model: Model name (e.g., 'imagen-3.0-fast-generate-001')
            provider_config_name: Name of provider in sdk_providers.json
            optional_params: Additional parameters (n, size, aspect_ratio, etc.)
            **kwargs: Additional keyword arguments
        
        Returns:
            ImageResponse with generated images and cost tracking
        """
        # Load provider configuration
        config = SDKProviderRegistry.get(provider_config_name)
        if not config:
            raise ValueError(
                f"Provider configuration '{provider_config_name}' not found. "
                f"Available providers: {SDKProviderRegistry.list_providers('image_generation')}"
            )

        verbose_logger.debug(f"Using JSON-configured provider: {provider_config_name}")

        # Prepare LiteLLM parameters
        litellm_params = {
            "prompt": prompt,
            "model": model,
            "n": optional_params.get("n", 1) if optional_params else 1,
            "size": optional_params.get("size", "1024x1024") if optional_params else "1024x1024",
        }
        
        # Add all optional params
        if optional_params:
            litellm_params.update(optional_params)

        verbose_logger.debug(f"LiteLLM parameters: {litellm_params}")

        # Transform request to provider format
        try:
            request_body = TransformationEngine.transform_request(
                litellm_params, config.transformations["request"]
            )
            verbose_logger.debug(f"Transformed request body: {request_body}")
        except Exception as e:
            verbose_logger.error(f"Request transformation failed: {e}")
            raise

        # Build API URL
        api_base = (
            os.getenv(config.api_base_env) if config.api_base_env else None
        )
        api_base = api_base or config.api_base

        endpoint_config = config.endpoints.get("generate")
        if not endpoint_config:
            raise ValueError(f"No 'generate' endpoint configured for {provider_config_name}")

        # Replace {model} placeholder in path
        url = api_base + endpoint_config.path.format(model=model)
        verbose_logger.debug(f"API URL: {url}")

        # Get authentication
        auth_config = config.authentication
        api_key = get_secret_str(auth_config.env_var)
        
        if not api_key:
            raise ValueError(
                f"Required environment variable '{auth_config.env_var}' not set. "
                f"Set it with your {provider_config_name} API key."
            )

        # Build request headers and params
        headers = {"Content-Type": "application/json"}
        params = {}

        if auth_config.type == "query_param":
            params[auth_config.param_name or "key"] = api_key
        elif auth_config.type == "bearer_token":
            headers["Authorization"] = f"Bearer {api_key}"
        elif auth_config.type == "custom_header":
            headers[auth_config.header_name or "Authorization"] = api_key

        verbose_logger.debug(f"Request headers: {list(headers.keys())}")
        verbose_logger.debug(f"Query params: {list(params.keys())}")

        # Make API request
        try:
            async_client = get_async_httpx_client(
                llm_provider="json_provider",
                params={"timeout": kwargs.get("timeout", 60.0)}
            )
            
            response = await async_client.post(
                url,
                json=request_body,
                headers=headers,
                params=params,
                timeout=kwargs.get("timeout", 60.0),
            )
            
            response.raise_for_status()
            provider_response = response.json()
            
            verbose_logger.debug(f"Provider response status: {response.status_code}")
            
        except httpx.HTTPStatusError as e:
            verbose_logger.error(f"HTTP error from provider: {e}")
            verbose_logger.error(f"Response body: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            raise
        except Exception as e:
            verbose_logger.error(f"Request to provider failed: {e}")
            raise

        # Transform response to LiteLLM format
        try:
            transformed = TransformationEngine.transform_response(
                provider_response, config.transformations["response"]
            )
            verbose_logger.debug(f"Transformed response: {list(transformed.keys())}")
        except Exception as e:
            verbose_logger.error(f"Response transformation failed: {e}")
            raise

        # Build ImageResponse
        images = []
        image_data = transformed.get("images", [])
        
        # Ensure images is a list
        if not isinstance(image_data, list):
            image_data = [image_data]
        
        for img_data in image_data:
            images.append(
                Image(
                    b64_json=img_data,
                    url=None,
                    revised_prompt=transformed.get("revised_prompt"),
                )
            )

        verbose_logger.debug(f"Created {len(images)} image objects")

        image_response = ImageResponse(
            created=int(response.headers.get("date", "0")) if hasattr(response, "headers") else 0,
            data=images,
        )

        # Calculate and add cost
        cost = CostTracker.calculate_image_generation_cost(
            image_response, model, config.cost_tracking
        )
        
        image_response = CostTracker.add_cost_to_response(image_response, cost)
        
        verbose_logger.info(
            f"Generated {len(images)} images using {provider_config_name}/{model}, cost: ${cost:.4f}"
        )

        return image_response

    @staticmethod
    def image_generation(
        prompt: str,
        model: str,
        provider_config_name: str,
        optional_params: Optional[Dict] = None,
        **kwargs
    ) -> ImageResponse:
        """
        Generate images using JSON-configured provider (sync).
        
        This is a synchronous wrapper around aimage_generation.
        
        Args:
            prompt: Text prompt for image generation
            model: Model name
            provider_config_name: Name of provider in sdk_providers.json
            optional_params: Additional parameters
            **kwargs: Additional keyword arguments
        
        Returns:
            ImageResponse with generated images
        """
        import asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            JSONProviderImageGeneration.aimage_generation(
                prompt, model, provider_config_name, optional_params, **kwargs
            )
        )
