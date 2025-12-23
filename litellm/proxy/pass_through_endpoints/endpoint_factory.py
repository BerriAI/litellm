"""
Factory for generating pass-through endpoint handlers from JSON configuration.

This module provides a declarative way to create pass-through endpoints without
writing boilerplate Python code. Simply define endpoints in JSON and they're
automatically registered with all the proper auth, streaming, and routing logic.
"""

import os
from typing import Any, Callable, Dict, Optional

import httpx
from fastapi import Depends, FastAPI, Request, Response

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.secret_managers.main import get_secret_str

from .endpoint_config_registry import EndpointConfig, EndpointConfigRegistry
from .pass_through_endpoints import create_pass_through_route


class PassthroughEndpointFactory:
    """
    Factory for creating pass-through endpoint handlers from JSON configuration.

    This class handles:
    - Authentication header generation
    - API key retrieval from environment
    - Target URL construction
    - Streaming detection
    - Query parameter handling
    """

    @staticmethod
    def create_auth_headers(config: EndpointConfig, api_key: str) -> Dict[str, str]:
        """
        Generate authentication headers based on endpoint configuration.

        Args:
            config: Endpoint configuration
            api_key: API key to use for authentication

        Returns:
            Dictionary of headers to include in the request
        """
        if config.auth.type == "bearer_token":
            header_format = config.auth.header_format or "Bearer {api_key}"
            return {"Authorization": header_format.format(api_key=api_key)}
        elif config.auth.type == "custom_header":
            header_name = config.auth.header_name or "Authorization"
            header_format = config.auth.header_format or "{api_key}"
            return {header_name: header_format.format(api_key=api_key)}
        elif config.auth.type == "query_param":
            # Query params are handled separately in get_query_params
            return {}
        elif config.auth.type == "custom_handler":
            # Custom handlers provide their own headers
            return {}
        return {}

    @staticmethod
    def get_api_key_from_env(config: EndpointConfig) -> Optional[str]:
        """
        Get API key from environment variable.

        Uses LiteLLM's secret manager to support various secret backends
        (env vars, AWS Secrets Manager, etc.)

        Args:
            config: Endpoint configuration

        Returns:
            API key string if found, None otherwise
        """
        try:
            return get_secret_str(config.auth.env_var)
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Failed to get API key from {config.auth.env_var}: {e}"
            )
            return None

    @staticmethod
    def get_target_url(config: EndpointConfig) -> str:
        """
        Get target base URL with environment variable override support.

        Priority:
        1. Environment variable (if target_base_url_env is set)
        2. Static target_base_url
        3. Template target_base_url_template

        Args:
            config: Endpoint configuration

        Returns:
            Target base URL string

        Raises:
            ValueError: If no target URL is configured
        """
        # Check environment variable override first
        if config.target_base_url_env:
            env_url = os.getenv(config.target_base_url_env)
            if env_url:
                verbose_proxy_logger.debug(
                    f"Using base URL from env var {config.target_base_url_env}: {env_url}"
                )
                return env_url

        # Use static URL if available
        if config.target_base_url:
            return config.target_base_url

        # Use template (requires dynamic substitution later)
        if config.target_base_url_template:
            return config.target_base_url_template

        raise ValueError(
            f"No target URL configured for {config.provider_slug}. "
            f"Set {config.target_base_url_env} environment variable or define target_base_url in config."
        )

    @staticmethod
    async def detect_streaming(
        config: EndpointConfig, request: Request, endpoint: str
    ) -> bool:
        """
        Detect if a request should be treated as streaming based on configuration.

        Supports multiple detection methods:
        - request_body_field: Check a field in the request body (e.g., "stream": true)
        - url_contains: Check if URL contains a pattern (e.g., "streamGenerateContent")
        - header: Check Accept header for streaming content type
        - none: No streaming support

        Args:
            config: Endpoint configuration
            request: FastAPI request object
            endpoint: Endpoint path

        Returns:
            True if request is streaming, False otherwise
        """
        if config.streaming.detection_method == "request_body_field":
            # Check request body for streaming field
            try:
                request_body = await _read_request_body(request)
                field_name = config.streaming.field_name or "stream"
                is_streaming = request_body.get(field_name, False)
                verbose_proxy_logger.debug(
                    f"Streaming detection (body field '{field_name}'): {is_streaming}"
                )
                return bool(is_streaming)
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"Failed to check request body for streaming: {e}"
                )
                return False
        elif config.streaming.detection_method == "url_contains":
            pattern = config.streaming.pattern or "stream"
            is_streaming = pattern in endpoint
            verbose_proxy_logger.debug(
                f"Streaming detection (URL contains '{pattern}'): {is_streaming}"
            )
            return is_streaming
        elif config.streaming.detection_method == "header":
            accept_header = request.headers.get("accept", "")
            is_streaming = "text/event-stream" in accept_header
            verbose_proxy_logger.debug(
                f"Streaming detection (Accept header): {is_streaming}"
            )
            return is_streaming
        elif config.streaming.detection_method == "none":
            return False

        verbose_proxy_logger.warning(
            f"Unknown streaming detection method: {config.streaming.detection_method}"
        )
        return False

    @staticmethod
    def get_query_params(
        config: EndpointConfig, request: Request, api_key: Optional[str]
    ) -> Optional[Dict[str, str]]:
        """
        Get query parameters to include in the request.

        Handles special case where API key is passed as query param.

        Args:
            config: Endpoint configuration
            request: FastAPI request object
            api_key: API key to include

        Returns:
            Dictionary of query params or None
        """
        if config.auth.type == "query_param" and api_key:
            query_params = dict(request.query_params)
            param_name = config.auth.param_name or "key"
            query_params[param_name] = api_key
            return query_params

        if config.features.custom_query_params:
            return dict(request.query_params)

        return None

    @classmethod
    def create_endpoint_handler(cls, config: EndpointConfig) -> Callable:
        """
        Create an endpoint handler function from configuration.

        This generates a FastAPI-compatible async endpoint handler that:
        1. Validates authentication
        2. Constructs target URL
        3. Adds auth headers/params
        4. Detects streaming
        5. Creates pass-through route
        6. Returns response

        Args:
            config: Endpoint configuration

        Returns:
            Async function that handles endpoint requests
        """

        async def handler(
            endpoint: str,
            request: Request,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        ):
            """
            Dynamically generated endpoint handler.

            Args:
                endpoint: Path captured from route (e.g., "v1/chat/completions")
                request: FastAPI request object
                fastapi_response: FastAPI response object
                user_api_key_dict: Authenticated user information

            Returns:
                Response from the target API
            """
            verbose_proxy_logger.debug(
                f"Handling request for {config.provider_slug}: {endpoint}"
            )

            # Get base URL
            try:
                base_target_url = cls.get_target_url(config)
            except ValueError as e:
                raise Exception(str(e))

            # Build full target URL
            encoded_endpoint = httpx.URL(endpoint).path
            if not encoded_endpoint.startswith("/"):
                encoded_endpoint = "/" + encoded_endpoint

            base_url = httpx.URL(base_target_url)

            # Join paths properly
            if not base_url.path or base_url.path == "/":
                updated_url = base_url.copy_with(path=encoded_endpoint)
            else:
                base_path = base_url.path.rstrip("/")
                clean_path = encoded_endpoint.lstrip("/")
                full_path = f"{base_path}/{clean_path}"
                updated_url = base_url.copy_with(path=full_path)

            # Get API key and create headers
            api_key = cls.get_api_key_from_env(config)
            if api_key is None and not config.features.custom_auth_handler:
                raise Exception(
                    f"Required '{config.auth.env_var}' in environment to make "
                    f"pass-through calls to {config.provider_slug}. "
                    f"Set this environment variable with your {config.provider_slug} API key."
                )

            custom_headers = cls.create_auth_headers(config, api_key or "")

            # Handle query params for auth
            query_params = cls.get_query_params(config, request, api_key)

            # Detect streaming
            is_streaming = await cls.detect_streaming(config, request, endpoint)

            # Build final target URL
            target = str(updated_url)
            if is_streaming and config.streaming.query_param_suffix:
                target += config.streaming.query_param_suffix
                verbose_proxy_logger.debug(
                    f"Added streaming query param: {config.streaming.query_param_suffix}"
                )

            verbose_proxy_logger.debug(
                f"Final target URL: {target}, streaming: {is_streaming}"
            )

            # Create pass-through route
            endpoint_func = create_pass_through_route(
                endpoint=endpoint,
                target=target,
                custom_headers=custom_headers,
                _forward_headers=config.features.forward_headers,
                merge_query_params=config.features.merge_query_params,
                is_streaming_request=is_streaming,
                query_params=query_params,
            )

            # Execute and return
            return await endpoint_func(request, fastapi_response, user_api_key_dict)

        return handler

    @classmethod
    def register_endpoint_from_config(cls, app: FastAPI, config: EndpointConfig):
        """
        Register an endpoint route on FastAPI app from configuration.

        Args:
            app: FastAPI application instance
            config: Endpoint configuration

        Raises:
            Exception: If registration fails
        """
        verbose_proxy_logger.info(
            f"Registering JSON-configured endpoint: {config.provider_slug} at {config.route_prefix}"
        )

        handler = cls.create_endpoint_handler(config)

        # Setup dependencies
        dependencies = []
        if config.features.require_litellm_auth:
            dependencies = [Depends(user_api_key_auth)]

        # Register route
        app.add_api_route(
            path=config.route_prefix,
            endpoint=handler,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            tags=config.tags,
            dependencies=dependencies,
        )

        verbose_proxy_logger.debug(
            f"Successfully registered {config.provider_slug} endpoint"
        )

    @classmethod
    def register_all_endpoints(cls, app: FastAPI):
        """
        Register all endpoints from the registry on the FastAPI app.

        This is typically called during app startup to register all
        JSON-configured endpoints.

        Args:
            app: FastAPI application instance
        """
        providers = EndpointConfigRegistry.list_providers()

        if not providers:
            verbose_proxy_logger.info(
                "No JSON-configured endpoints found, skipping registration"
            )
            return

        verbose_proxy_logger.info(
            f"Registering {len(providers)} JSON-configured endpoints"
        )

        for provider_slug in providers:
            config = EndpointConfigRegistry.get(provider_slug)
            if config:
                try:
                    cls.register_endpoint_from_config(app, config)
                except Exception as e:
                    verbose_proxy_logger.error(
                        f"Failed to register {provider_slug}: {e}", exc_info=True
                    )
            else:
                verbose_proxy_logger.warning(
                    f"Config not found for provider: {provider_slug}"
                )

        verbose_proxy_logger.info(
            f"Completed registration of {len(providers)} JSON-configured endpoints"
        )
