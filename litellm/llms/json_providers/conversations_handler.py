"""
Conversations handler for JSON-configured providers.

Handles conversation creation/management using JSON-configured providers with
automatic transformations.
"""

import os
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.json_providers.sdk_provider_registry import SDKProviderRegistry
from litellm.llms.json_providers.transformation_engine import TransformationEngine
from litellm.secret_managers.main import get_secret_str


class JSONProviderConversations:
    """
    Conversations handler for JSON-configured providers.
    
    Accepts native provider format, sends to API, transforms response to LiteLLM format.
    """

    @staticmethod
    async def acreate_conversation(
        provider_config_name: str,
        request_body: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create conversation using JSON-configured provider (async).
        
        Accepts NATIVE provider format (e.g., OpenAI Conversations format), sends to API,
        then transforms response to LiteLLM format.
        
        Args:
            provider_config_name: Name of provider in sdk_providers.json
            request_body: Native provider request format
            **kwargs: Additional keyword arguments (timeout, etc.)
        
        Returns:
            Dict with conversation details
        
        Example (OpenAI Conversations):
            request_body = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
                ],
                "metadata": {"user_id": "123"}
            }
        """
        # Load provider configuration
        config = SDKProviderRegistry.get(provider_config_name)
        if not config:
            raise ValueError(
                f"Provider configuration '{provider_config_name}' not found. "
                f"Available providers: {SDKProviderRegistry.list_providers('conversations')}"
            )

        verbose_logger.debug(f"Using JSON-configured provider: {provider_config_name}")
        verbose_logger.debug(f"Native request body: {request_body}")

        # Build API URL
        api_base = (
            os.getenv(config.api_base_env) if config.api_base_env else None
        )
        api_base = api_base or config.api_base

        endpoint_config = config.endpoints.get("create")
        if not endpoint_config:
            raise ValueError(f"No 'create' endpoint configured for {provider_config_name}")

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

        # Transform response from provider format to LiteLLM format
        try:
            transformed = TransformationEngine.transform_response(
                provider_response, config.transformations["response"]
            )
            verbose_logger.debug(f"Transformed response (Provider -> LiteLLM): {list(transformed.keys())}")
        except Exception as e:
            verbose_logger.error(f"Response transformation (Provider -> LiteLLM) failed: {e}")
            raise

        verbose_logger.info(
            f"Created conversation using {provider_config_name}, id: {transformed.get('id', 'N/A')}"
        )

        return transformed

    @staticmethod
    def create_conversation(
        provider_config_name: str,
        request_body: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create conversation using JSON-configured provider (sync).
        
        Accepts NATIVE provider format.
        
        Args:
            provider_config_name: Name of provider in sdk_providers.json
            request_body: Native provider request format
            **kwargs: Additional keyword arguments
        
        Returns:
            Dict with conversation details
        """
        import asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            JSONProviderConversations.acreate_conversation(
                provider_config_name, request_body, **kwargs
            )
        )
