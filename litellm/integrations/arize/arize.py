"""
arize AI is OTEL compatible

this file has Arize ai specific helper functions
"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm.integrations.arize import _utils
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import (
    DynamicLoggingCache,
)
from litellm.types.integrations.arize import ArizeConfig
from litellm.types.services import ServiceLoggerPayload
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    Span = Union[_Span, Any]
else:
    Protocol = Any
    Span = Any


class ArizeLogger(OpenTelemetry):
    @property
    def callback_name(self) -> str:
        return "arize"
    
    def __init__(self, space_key: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Arize OTEL logger
        """
        from litellm.integrations.opentelemetry import OpenTelemetryConfig
        arize_config = ArizeLogger.get_arize_config(
            space_key=space_key,
            api_key=api_key,
        )
        if arize_config.endpoint is None:
            raise ValueError(
                "No valid endpoint found for Arize, please set 'ARIZE_ENDPOINT' to your GRPC endpoint or 'ARIZE_HTTP_ENDPOINT' to your HTTP endpoint"
            )
        otel_config = OpenTelemetryConfig(
            exporter=arize_config.protocol,
            endpoint=arize_config.endpoint,
            headers=f"space_id={arize_config.space_key},api_key={arize_config.api_key}"
        )

        super().__init__(config=otel_config, callback_name="arize")
    
    def set_attributes(self, span: Span, kwargs, response_obj: Optional[Any]):
        ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
        return

    @staticmethod
    def set_arize_attributes(span: Span, kwargs, response_obj):
        _utils.set_attributes(span, kwargs, response_obj)
        return

    @staticmethod
    def get_arize_config(
        space_key: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> ArizeConfig:
        """
        Helper function to get Arize configuration.

        Returns:
            ArizeConfig: A Pydantic model containing Arize configuration.

        Raises:
            ValueError: If required environment variables are not set.
        """
        space_key = space_key or os.environ.get("ARIZE_SPACE_KEY")
        api_key = api_key or os.environ.get("ARIZE_API_KEY")

        grpc_endpoint = os.environ.get("ARIZE_ENDPOINT")
        http_endpoint = os.environ.get("ARIZE_HTTP_ENDPOINT")

        endpoint = None
        protocol: Protocol = "otlp_grpc"

        if grpc_endpoint:
            protocol = "otlp_grpc"
            endpoint = grpc_endpoint
        elif http_endpoint:
            protocol = "otlp_http"
            endpoint = http_endpoint
        else:
            protocol = "otlp_grpc"
            endpoint = "https://otlp.arize.com/v1"

        return ArizeConfig(
            space_key=space_key,
            api_key=api_key,
            protocol=protocol,
            endpoint=endpoint,
        )

    async def async_service_success_hook(
        self,
        payload: ServiceLoggerPayload,
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ):
        """Arize is used mainly for LLM I/O tracing, sending router+caching metrics adds bloat to arize logs"""
        pass

    async def async_service_failure_hook(
        self,
        payload: ServiceLoggerPayload,
        error: Optional[str] = "",
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[float, datetime]] = None,
        event_metadata: Optional[dict] = None,
    ):
        """Arize is used mainly for LLM I/O tracing, sending router+caching metrics adds bloat to arize logs"""
        pass

    def create_litellm_proxy_request_started_span(
        self,
        start_time: datetime,
        headers: dict,
    ):
        """Arize is used mainly for LLM I/O tracing, sending Proxy Server Request adds bloat to arize logs"""
        pass
    

    @staticmethod
    def construct_dynamic_arize_headers(
        standard_callback_dynamic_params: StandardCallbackDynamicParams
    ):
        """
        Construct dynamic Arize headers from standard callback dynamic params

        Returns:
            dict: A dictionary of dynamic Arize headers
        """
        dynamic_headers = {}
        if standard_callback_dynamic_params.get("arize_space_key"):
            dynamic_headers["space_key"] = standard_callback_dynamic_params.get(
                "arize_space_key"
            )
        if standard_callback_dynamic_params.get("arize_api_key"):
            dynamic_headers["api_key"] = standard_callback_dynamic_params.get(
                "arize_api_key"
            )
        
        if standard_callback_dynamic_params.get("arize_space_id"):
            dynamic_headers["arize-space-id"] = standard_callback_dynamic_params.get(
                "arize_space_id"
            )
        return dynamic_headers
    

    @staticmethod
    def standard_callback_dynamic_params_contains_dynamic_arize_headers(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
    ):
        """
        Check if the standard callback dynamic params contains dynamic Arize headers
        """
        return (
            standard_callback_dynamic_params.get("arize_api_key") or 
            standard_callback_dynamic_params.get("arize_space_id") or 
            standard_callback_dynamic_params.get("arize_space_key")
        )
    
    @staticmethod
    def transform_standard_callback_dynamic_params_to_arize_credentials(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
    ):
        """
        Get Arize credentials from standard callback dynamic params
        """
        return {
            "space_key": (
                standard_callback_dynamic_params.get("arize_space_id") or 
                # space key is the deprecated field name arize uses
                standard_callback_dynamic_params.get("arize_space_key")
            ),
            "api_key": standard_callback_dynamic_params.get("arize_api_key"),
        }

    @staticmethod
    def get_dynamic_arize_logger(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
        original_callback: CustomLogger,
    ) -> CustomLogger:
        """
        Get a dynamic Arize logger if the standard callback dynamic params contains dynamic Arize headers

        If no dynamic headers, return the original callback
        """
        if ArizeLogger.standard_callback_dynamic_params_contains_dynamic_arize_headers(standard_callback_dynamic_params):
            arize_credentials = ArizeLogger.transform_standard_callback_dynamic_params_to_arize_credentials(standard_callback_dynamic_params)
            #########################################################
            # check if arize logger is already cached
            #########################################################
            arize_otel_logger = in_memory_dynamic_logger_cache.get_cache(
                credentials=arize_credentials,
                service_name="arize",
            )
            if arize_otel_logger is not None:
                return arize_otel_logger
            

            #########################################################
            # If not cached, create a new arize logger + set in cache
            #########################################################
            arize_otel_logger = ArizeLogger(
                space_key=arize_credentials.get("space_key"),
                api_key=arize_credentials.get("api_key"),
            )
            in_memory_dynamic_logger_cache.set_cache(
                credentials=arize_credentials,
                service_name="arize",
                logging_obj=arize_otel_logger,
            )
            return arize_otel_logger
        
        #########################################################
        # If no dynamic headers, return the original callback
        #########################################################
        return original_callback
        
