"""
Chat completion handler for JSON-configured providers.

Handles chat completion requests using JSON-configured providers with
automatic transformations and cost tracking.
"""

import os
from typing import Any, Dict, List, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, HTTPHandler
from litellm.llms.json_providers.cost_tracker import CostTracker
from litellm.llms.json_providers.sdk_provider_registry import SDKProviderRegistry
from litellm.llms.json_providers.transformation_engine import TransformationEngine
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ModelResponse, Usage, Choices, Message


class JSONProviderCompletion:
    """
    Chat completion handler for JSON-configured providers.
    
    Accepts native provider format, sends to API, transforms response to LiteLLM format.
    """

    @staticmethod
    async def acompletion(
        model: str,
        provider_config_name: str,
        request_body: Dict[str, Any],
        **kwargs
    ) -> ModelResponse:
        """
        Create chat completion using JSON-configured provider (async).
        
        Accepts NATIVE provider format (e.g., OpenAI format), sends to API,
        then transforms response to LiteLLM format.
        
        Args:
            model: Model name (e.g., 'gpt-4o-mini')
            provider_config_name: Name of provider in sdk_providers.json
            request_body: Native provider request format (e.g., {messages: [...], temperature: 0.7})
            **kwargs: Additional keyword arguments (timeout, etc.)
        
        Returns:
            ModelResponse with completion and cost tracking
        
        Example (OpenAI):
            request_body = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
        """
        # Load provider configuration
        config = SDKProviderRegistry.get(provider_config_name)
        if not config:
            raise ValueError(
                f"Provider configuration '{provider_config_name}' not found. "
                f"Available providers: {SDKProviderRegistry.list_providers('chat')}"
            )

        verbose_logger.debug(f"Using JSON-configured provider: {provider_config_name}")
        verbose_logger.debug(f"Native request body: {request_body}")

        # Build API URL
        api_base = (
            os.getenv(config.api_base_env) if config.api_base_env else None
        )
        api_base = api_base or config.api_base

        endpoint_config = config.endpoints.get("chat")
        if not endpoint_config:
            raise ValueError(f"No 'chat' endpoint configured for {provider_config_name}")

        url = api_base + endpoint_config.path
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

        # Make API request
        try:
            async_client = get_async_httpx_client(
                llm_provider="json_provider",
                params={"timeout": kwargs.get("timeout", 600.0)}
            )
            
            response = await async_client.post(
                url,
                json=request_body,
                headers=headers,
                params=params,
                timeout=kwargs.get("timeout", 600.0),
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

        # Transform response from provider format to LiteLLM format
        try:
            transformed = TransformationEngine.transform_response(
                provider_response, config.transformations["response"]
            )
            verbose_logger.debug(f"Transformed response (Provider -> LiteLLM): {list(transformed.keys())}")
        except Exception as e:
            verbose_logger.error(f"Response transformation (Provider -> LiteLLM) failed: {e}")
            raise

        # Build ModelResponse
        try:
            # Extract usage
            usage_data = transformed.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )

            # Extract choices
            choices_data = transformed.get("choices", [])
            choices = []
            for choice in choices_data:
                message_data = choice.get("message", {})
                message = Message(
                    content=message_data.get("content"),
                    role=message_data.get("role", "assistant"),
                    tool_calls=message_data.get("tool_calls"),
                    function_call=message_data.get("function_call")
                )
                choices.append(
                    Choices(
                        finish_reason=choice.get("finish_reason"),
                        index=choice.get("index", 0),
                        message=message
                    )
                )

            model_response = ModelResponse(
                id=transformed.get("id", "chatcmpl-default"),
                choices=choices,
                created=transformed.get("created", 0),
                model=transformed.get("model", model),
                object="chat.completion",
                system_fingerprint=transformed.get("system_fingerprint"),
                usage=usage
            )

            verbose_logger.debug(f"Created ModelResponse with {len(choices)} choices")

        except Exception as e:
            verbose_logger.error(f"Error building ModelResponse: {e}")
            raise

        # Calculate and add cost
        cost = CostTracker.calculate_completion_cost(
            model_response, model, config.cost_tracking
        )
        
        model_response = CostTracker.add_cost_to_response(model_response, cost)
        
        verbose_logger.info(
            f"Completion using {provider_config_name}/{model}, "
            f"tokens: {usage.prompt_tokens}+{usage.completion_tokens}, "
            f"cost: ${cost:.6f}"
        )

        return model_response

    @staticmethod
    def completion(
        model: str,
        provider_config_name: str,
        request_body: Dict[str, Any],
        **kwargs
    ) -> ModelResponse:
        """
        Create chat completion using JSON-configured provider (sync).
        
        Accepts NATIVE provider format.
        
        Args:
            model: Model name
            provider_config_name: Name of provider in sdk_providers.json
            request_body: Native provider request format
            **kwargs: Additional keyword arguments
        
        Returns:
            ModelResponse with completion
        """
        import asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            JSONProviderCompletion.acompletion(
                model, provider_config_name, request_body, **kwargs
            )
        )
